"""
clinical_rag.py
---------------
Retrieval-Augmented Generation (RAG) pipeline over a CGM patient cohort.

Each patient's clinical statistics are embedded with sentence-transformers and
stored in a FAISS vector index.  A natural-language query retrieves the most
relevant patients, and an LLM (OpenAI / Anthropic — or a deterministic fallback)
synthesises a cohort-level answer.

Demonstrates: embeddings, FAISS vector search, RAG architecture, LLM integration
with graceful degradation, clinical NLP.  Educational only — not medical advice.

Install:
    pip install sentence-transformers faiss-cpu openai anthropic

Usage:
    python ml/clinical_rag.py --data sample_data/CGM\ Records
    python ml/clinical_rag.py --query "which patients have the most hypoglycemic events?"
    OPENAI_API_KEY=... python ml/clinical_rag.py --query "who needs immediate attention?"
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Patient document builder
# ---------------------------------------------------------------------------

def compute_patient_stats(subject_id: int, folder: Path) -> dict:
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        return {}
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    cgm = pd.to_numeric(df["CGM"], errors="coerce").dropna()
    if len(cgm) == 0:
        return {}
    hypo_events = int(cgm.lt(70).astype(int).diff().eq(1).sum())
    return {
        "subject_id": subject_id,
        "n_readings": int(len(cgm)),
        "mean_glucose": round(float(cgm.mean()), 1),
        "std_glucose": round(float(cgm.std()), 1),
        "cv_pct": round(float(cgm.std() / cgm.mean() * 100), 1),
        "time_in_range_pct": round(float(((cgm >= 70) & (cgm <= 180)).mean() * 100), 1),
        "time_below_70_pct": round(float((cgm < 70).mean() * 100), 1),
        "time_above_180_pct": round(float((cgm > 180).mean() * 100), 1),
        "time_below_54_pct": round(float((cgm < 54).mean() * 100), 1),
        "n_hypo_events": hypo_events,
        "min_glucose": round(float(cgm.min()), 1),
        "max_glucose": round(float(cgm.max()), 1),
    }


def stats_to_document(s: dict) -> str:
    """Convert a patient stats dict into a plain-language document for embedding."""
    sid = s["subject_id"]
    tir_status = "meets" if s["time_in_range_pct"] >= 70 else "does not meet"
    hypo_status = "within" if s["time_below_70_pct"] < 4 else "exceeds"
    cv_status = "stable" if s["cv_pct"] < 36 else "elevated"
    risk_level = (
        "high" if s["time_below_54_pct"] > 1 or s["n_hypo_events"] > 10
        else "moderate" if s["time_below_70_pct"] >= 4
        else "low"
    )
    return (
        f"Patient {sid}: time in range {s['time_in_range_pct']}% ({tir_status} the 70% ADA goal). "
        f"Hypoglycemia: {s['time_below_70_pct']}% time below 70 mg/dL ({hypo_status} 4% safety target), "
        f"{s['n_hypo_events']} discrete hypoglycemic events, "
        f"{s['time_below_54_pct']}% below 54 mg/dL (severe). "
        f"Mean glucose {s['mean_glucose']} mg/dL, CV {s['cv_pct']}% ({cv_status}). "
        f"Hyperglycemia: {s['time_above_180_pct']}% above 180 mg/dL. "
        f"Overall hypoglycemia risk: {risk_level}."
    )


# ---------------------------------------------------------------------------
# Embedding + FAISS index
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
    import faiss as _faiss
    _VECTOR_AVAILABLE = True
except ImportError:
    _VECTOR_AVAILABLE = False


class PatientVectorStore:
    """
    FAISS-backed vector store of per-patient clinical documents.

    Degrades gracefully to keyword-TF scoring when sentence-transformers /
    faiss-cpu are not installed.  Install with:
        pip install sentence-transformers faiss-cpu
    """

    def __init__(self):
        self.docs: list[str] = []
        self.stats: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self.index = None
        self._model = None

    def _encoder(self):
        if self._model is None:
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def build(self, all_stats: list[dict]) -> None:
        self.stats = all_stats
        self.docs = [stats_to_document(s) for s in all_stats]
        if _VECTOR_AVAILABLE:
            embeddings = self._encoder().encode(self.docs, normalize_embeddings=True)
            self._embeddings = embeddings.astype("float32")
            dim = embeddings.shape[1]
            self.index = _faiss.IndexFlatIP(dim)
            self.index.add(self._embeddings)
            print(f"[RAG] Indexed {len(self.docs)} patients with FAISS  (dim={dim})")
        else:
            print("[RAG] sentence-transformers/faiss not installed — using keyword fallback.")
            print("      Install with: pip install sentence-transformers faiss-cpu")
            self.index = None

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if _VECTOR_AVAILABLE and self.index is not None:
            _faiss.write_index(self.index, str(path / "patients.faiss"))
        with open(path / "docs.json", "w") as f:
            json.dump({"docs": self.docs, "stats": self.stats}, f, indent=2)
        print(f"[RAG] Index saved to {path}")

    def load(self, path: Path) -> None:
        with open(path / "docs.json") as f:
            data = json.load(f)
        self.docs = data["docs"]
        self.stats = data["stats"]
        faiss_path = path / "patients.faiss"
        if _VECTOR_AVAILABLE and faiss_path.exists():
            self.index = _faiss.read_index(str(faiss_path))
            print(f"[RAG] Loaded FAISS index with {len(self.docs)} patients from {path}")
        else:
            self.index = None
            print(f"[RAG] Loaded {len(self.docs)} patient docs (keyword fallback) from {path}")

    def _keyword_score(self, query: str, doc: str) -> float:
        """Simple overlap score for fallback retrieval."""
        q_tokens = set(query.lower().split())
        d_tokens = set(doc.lower().split())
        return len(q_tokens & d_tokens) / (len(q_tokens) + 1e-9)

    def retrieve(self, query: str, k: int = 5) -> list[tuple[float, dict, str]]:
        if _VECTOR_AVAILABLE and self.index is not None:
            q_emb = self._encoder().encode([query], normalize_embeddings=True).astype("float32")
            scores, idxs = self.index.search(q_emb, k)
            return [(float(scores[0][i]), self.stats[idxs[0][i]], self.docs[idxs[0][i]])
                    for i in range(k) if idxs[0][i] >= 0]
        # Keyword fallback
        scored = [(self._keyword_score(query, doc), i)
                  for i, doc in enumerate(self.docs)]
        scored.sort(reverse=True)
        return [(score, self.stats[i], self.docs[i]) for score, i in scored[:k]]


# ---------------------------------------------------------------------------
# LLM layer
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 400) -> str | None:
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI] {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return r.content[0].text.strip()
        except Exception as e:
            print(f"[Anthropic] {e}")

    return None


SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical data analyst reviewing continuous glucose monitoring (CGM)
    data for a cohort of 25 Type 1 Diabetes patients.  Answer questions factually
    using only the retrieved patient records provided.  Be concise and clinical.
    Do not give dosing instructions or diagnoses.  Flag educational-only status.
""")


def rag_answer(store: PatientVectorStore, query: str, k: int = 5) -> str:
    results = store.retrieve(query, k=k)
    context = "\n".join(
        f"[Score {score:.3f}] {doc}" for score, _, doc in results
    )
    user_prompt = f"Query: {query}\n\nRetrieved patient records:\n{context}\n\nAnswer:"

    llm_out = call_llm(SYSTEM_PROMPT, user_prompt)
    if llm_out:
        return f"[LLM answer]\n{llm_out}\n\n(Educational only — not medical advice.)"

    # Deterministic fallback: summarise the top-k retrieved docs
    lines = ["[Retrieval-only answer — no LLM API key detected]\n"]
    for rank, (score, s, _) in enumerate(results, 1):
        lines.append(
            f"{rank}. Patient {s['subject_id']}: TIR {s['time_in_range_pct']}%, "
            f"hypo events {s['n_hypo_events']}, CV {s['cv_pct']}%  (similarity {score:.3f})"
        )
    lines.append("\n(Educational only — not medical advice.)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "Which patients have the most hypoglycemic events and need immediate attention?",
    "Which patients meet the ADA time-in-range goal of 70%?",
    "Who has the most unstable glucose variability?",
    "Summarise the overall cohort glycemic control.",
]


def main() -> None:
    ap = argparse.ArgumentParser(description="Clinical RAG over CGM cohort")
    ap.add_argument("--data", default="sample_data/CGM Records",
                    help="Root folder containing Subject N/ subfolders")
    ap.add_argument("--index", default="ml/outputs/rag_index",
                    help="Where to save / load the FAISS index")
    ap.add_argument("--rebuild", action="store_true",
                    help="Force rebuild the index even if cached")
    ap.add_argument("--query", default=None,
                    help="Natural-language query (default: run all demo queries)")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    index_path = Path(args.index)
    store = PatientVectorStore()

    # Build or load index
    if args.rebuild or not (index_path / "patients.faiss").exists():
        data_root = Path(args.data)
        all_stats = []
        for subj_dir in sorted(data_root.glob("Subject *")):
            try:
                sid = int(subj_dir.name.split()[-1])
            except ValueError:
                continue
            s = compute_patient_stats(sid, subj_dir)
            if s:
                all_stats.append(s)
        if not all_stats:
            print(f"No patient data found in {data_root}. Generating synthetic cohort…")
            rng = np.random.default_rng(42)
            for sid in range(1, 26):
                tir = float(rng.uniform(45, 85))
                below70 = float(rng.uniform(1, 12))
                cv = float(rng.uniform(20, 50))
                all_stats.append({
                    "subject_id": sid,
                    "n_readings": 11000,
                    "mean_glucose": round(float(rng.uniform(130, 185)), 1),
                    "std_glucose": round(cv * 1.6, 1),
                    "cv_pct": round(cv, 1),
                    "time_in_range_pct": round(tir, 1),
                    "time_below_70_pct": round(below70, 1),
                    "time_above_180_pct": round(100 - tir - below70, 1),
                    "time_below_54_pct": round(float(rng.uniform(0, 2.5)), 1),
                    "n_hypo_events": int(rng.integers(0, 25)),
                    "min_glucose": round(float(rng.uniform(38, 60)), 1),
                    "max_glucose": round(float(rng.uniform(280, 380)), 1),
                })
        store.build(all_stats)
        store.save(index_path)
    else:
        store.load(index_path)

    # Answer queries
    queries = [args.query] if args.query else DEFAULT_QUERIES
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print("-" * 60)
        print(rag_answer(store, q, k=args.top_k))


if __name__ == "__main__":
    main()
