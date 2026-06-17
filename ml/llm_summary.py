"""
llm_summary.py
--------------
Generate a clinician-readable narrative from a subject's glucose statistics.
Provider-agnostic LLM integration: if an API key is present (OpenAI or Anthropic),
it calls the model; otherwise it falls back to a deterministic, template-based
generator so the feature always runs offline.

Demonstrates: LLM integration with graceful degradation + prompt construction over
structured clinical metrics. Educational only.

Usage:
    python ml/llm_summary.py --data "sample_data/CGM Records" --subject 1
    OPENAI_API_KEY=... python ml/llm_summary.py --subject 1     # uses the live model
"""
import argparse, os, textwrap
from pathlib import Path
import numpy as np, pandas as pd


def compute_stats(df):
    cgm = df["CGM"].astype(float)
    return {
        "n_readings": int(len(cgm)),
        "mean_glucose": round(float(cgm.mean()), 1),
        "cv_pct": round(float(cgm.std() / cgm.mean() * 100), 1),
        "time_in_range_pct": round(float(((cgm >= 70) & (cgm <= 180)).mean() * 100), 1),
        "time_below_70_pct": round(float((cgm < 70).mean() * 100), 1),
        "time_above_180_pct": round(float((cgm > 180).mean() * 100), 1),
        "n_hypo_events": int((cgm < 70).astype(int).diff().eq(1).sum()),
    }


def build_prompt(sid, s):
    return textwrap.dedent(f"""\
        You are a clinical diabetes educator. In 4-5 plain-language sentences, summarize
        this Type 1 Diabetes patient's continuous glucose data for a care-team note.
        Mention time-in-range vs. the >70% goal, hypoglycemia exposure (goal <4% below
        70 mg/dL), variability (CV goal <36%), and one actionable suggestion. Do not give
        a diagnosis or dosing instructions.

        Subject {sid} metrics:
        - Time in range (70-180): {s['time_in_range_pct']}%
        - Time below 70: {s['time_below_70_pct']}%  ({s['n_hypo_events']} hypo events)
        - Time above 180: {s['time_above_180_pct']}%
        - Mean glucose: {s['mean_glucose']} mg/dL
        - Glucose variability (CV): {s['cv_pct']}%
        """)


def call_llm(prompt):
    """Try OpenAI, then Anthropic, else return None."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            r = OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}], temperature=0.3)
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI unavailable: {e}]")
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model="claude-3-5-haiku-latest", max_tokens=300,
                messages=[{"role": "user", "content": prompt}])
            return r.content[0].text.strip()
        except Exception as e:
            print(f"[Anthropic unavailable: {e}]")
    return None


def template_summary(sid, s):
    tir = s["time_in_range_pct"]; below = s["time_below_70_pct"]; cv = s["cv_pct"]
    tir_txt = ("meets the >70% goal" if tir >= 70 else "is below the >70% goal")
    hypo_txt = ("within the <4% target" if below < 4 else "above the <4% safety target")
    cv_txt = ("stable (CV <36%)" if cv < 36 else "elevated (CV >=36%)")
    action = ("reduce hypoglycemia exposure by reviewing basal rates and pre-meal timing"
              if below >= 4 else
              "focus on raising time-in-range by tightening post-meal control")
    return (f"Subject {sid} spent {tir}% of readings in target range, which {tir_txt}. "
            f"Time below 70 mg/dL was {below}% across {s['n_hypo_events']} hypoglycemic "
            f"events, {hypo_txt}. Mean glucose was {s['mean_glucose']} mg/dL with a "
            f"coefficient of variation of {cv}%, indicating glycemic variability that is "
            f"{cv_txt}. As a next step, the care team might {action}. "
            f"(Educational summary — not medical advice.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--subject", type=int, default=1)
    a = ap.parse_args()
    folder = Path(a.data) / f"Subject {a.subject}"
    df = pd.concat([pd.read_csv(c, parse_dates=["EventDateTime"]) for c in folder.glob("*.csv")])
    s = compute_stats(df)
    prompt = build_prompt(a.subject, s)
    out = call_llm(prompt)
    mode = "LLM" if out else "offline template"
    if not out:
        out = template_summary(a.subject, s)
    print(f"--- Clinical summary (Subject {a.subject}) [{mode}] ---\n{out}")


if __name__ == "__main__":
    main()
