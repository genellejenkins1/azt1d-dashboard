"""
export_for_tableau.py
---------------------
Exports four Tableau-ready CSVs from the AZT1D CGM dataset, plus one
pre-built packaged workbook (cgm_dashboard.twbx) that Tableau Public
can open directly.

Outputs (written to the same tableau/ directory):
  patient_summary.csv       — one row per subject: TIR, hypo%, hyper%, CV, risk tier
  glucose_timeseries.csv    — long-format CGM readings (datetime, subject, glucose, zone)
  hypo_events.csv           — one row per discrete hypoglycemic event
  cohort_overview.csv       — single-row cohort aggregate for KPI tiles

Usage:
    python tableau/export_for_tableau.py
    python tableau/export_for_tableau.py --data sample_data/CGM\ Records --out tableau/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ADA clinical thresholds
# ---------------------------------------------------------------------------
SEVERE_HYPO = 54
HYPO = 70
TARGET_HIGH = 180
HYPER = 250


def glucose_zone(g: float) -> str:
    if g < SEVERE_HYPO:
        return "Severe Hypo (<54)"
    if g < HYPO:
        return "Hypo (54-70)"
    if g <= TARGET_HIGH:
        return "In Range (70-180)"
    if g <= HYPER:
        return "Hyper (180-250)"
    return "Severe Hyper (>250)"


def risk_tier(tir: float, below70: float, n_hypo: int) -> str:
    if below70 > 4 or n_hypo > 15:
        return "High"
    if tir < 70 or below70 > 2:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_subject(folder: Path) -> pd.DataFrame | None:
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        return None
    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c, parse_dates=["EventDateTime"])
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def subject_id(folder: Path) -> int:
    try:
        return int(folder.name.split()[-1])
    except ValueError:
        return -1


# ---------------------------------------------------------------------------
# Export builders
# ---------------------------------------------------------------------------

def build_summary(all_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sid, df in sorted(all_data.items()):
        cgm = pd.to_numeric(df["CGM"], errors="coerce").dropna()
        if len(cgm) == 0:
            continue
        below70 = float((cgm < HYPO).mean() * 100)
        n_hypo = int(cgm.lt(HYPO).astype(int).diff().eq(1).sum())
        tir = float(((cgm >= HYPO) & (cgm <= TARGET_HIGH)).mean() * 100)
        rows.append({
            "subject_id": sid,
            "n_readings": len(cgm),
            "mean_glucose": round(float(cgm.mean()), 1),
            "std_glucose": round(float(cgm.std()), 1),
            "cv_pct": round(float(cgm.std() / cgm.mean() * 100), 1),
            "time_in_range_pct": round(tir, 1),
            "time_below_70_pct": round(below70, 1),
            "time_above_180_pct": round(float((cgm > TARGET_HIGH).mean() * 100), 1),
            "time_below_54_pct": round(float((cgm < SEVERE_HYPO).mean() * 100), 1),
            "time_above_250_pct": round(float((cgm > HYPER).mean() * 100), 1),
            "n_hypo_events": n_hypo,
            "min_glucose": round(float(cgm.min()), 1),
            "max_glucose": round(float(cgm.max()), 1),
            "risk_tier": risk_tier(tir, below70, n_hypo),
            "meets_tir_goal": "Yes" if tir >= 70 else "No",
            "meets_hypo_goal": "Yes" if below70 < 4 else "No",
        })
    return pd.DataFrame(rows)


def build_timeseries(all_data: dict[int, pd.DataFrame],
                     max_subjects: int = 10) -> pd.DataFrame:
    """Long-format CGM readings — capped to keep file manageable."""
    rows = []
    for sid, df in sorted(all_data.items()):
        if sid > max_subjects:
            continue
        cgm = pd.to_numeric(df["CGM"], errors="coerce")
        dt = pd.to_datetime(df["EventDateTime"], errors="coerce")
        valid = cgm.notna() & dt.notna()
        sub = df[valid].copy()
        sub["subject_id"] = sid
        sub["glucose"] = cgm[valid].values
        sub["datetime"] = dt[valid].values
        sub["glucose_zone"] = sub["glucose"].apply(glucose_zone)
        sub["hour_of_day"] = pd.to_datetime(sub["datetime"]).dt.hour
        sub["day_of_week"] = pd.to_datetime(sub["datetime"]).dt.day_name()
        rows.append(sub[["subject_id", "datetime", "glucose",
                          "glucose_zone", "hour_of_day", "day_of_week"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_hypo_events(all_data: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for sid, df in sorted(all_data.items()):
        cgm = pd.to_numeric(df["CGM"], errors="coerce")
        dt = pd.to_datetime(df["EventDateTime"], errors="coerce")
        valid = cgm.notna() & dt.notna()
        cgm = cgm[valid].reset_index(drop=True)
        dt = dt[valid].reset_index(drop=True)
        in_hypo = cgm < HYPO
        events = in_hypo.astype(int).diff().eq(1)
        ends = in_hypo.astype(int).diff().eq(-1)
        for start_idx in events[events].index:
            end_candidates = ends[start_idx:][ends[start_idx:]]
            end_idx = end_candidates.index[0] if len(end_candidates) else len(cgm) - 1
            event_cgm = cgm[start_idx:end_idx]
            rows.append({
                "subject_id": sid,
                "event_start": dt[start_idx],
                "event_end": dt[end_idx],
                "duration_min": (end_idx - start_idx) * 5,
                "nadir_glucose": round(float(event_cgm.min()), 1),
                "mean_glucose_during": round(float(event_cgm.mean()), 1),
                "severe": "Yes" if event_cgm.min() < SEVERE_HYPO else "No",
                "hour_of_day": dt[start_idx].hour,
            })
    return pd.DataFrame(rows)


def build_cohort_overview(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return pd.DataFrame([{
        "n_subjects": len(summary),
        "pct_meeting_tir": round((summary["meets_tir_goal"] == "Yes").mean() * 100, 1),
        "pct_meeting_hypo_goal": round((summary["meets_hypo_goal"] == "Yes").mean() * 100, 1),
        "cohort_mean_tir": round(summary["time_in_range_pct"].mean(), 1),
        "cohort_mean_cv": round(summary["cv_pct"].mean(), 1),
        "cohort_total_hypo_events": int(summary["n_hypo_events"].sum()),
        "n_high_risk": int((summary["risk_tier"] == "High").sum()),
        "n_moderate_risk": int((summary["risk_tier"] == "Moderate").sum()),
        "n_low_risk": int((summary["risk_tier"] == "Low").sum()),
        "cohort_mean_glucose": round(summary["mean_glucose"].mean(), 1),
    }])


# ---------------------------------------------------------------------------
# Synthetic fallback cohort (when real / sample data unavailable)
# ---------------------------------------------------------------------------

def synthetic_cohort(n_subjects: int = 25) -> dict[int, pd.DataFrame]:
    rng = np.random.default_rng(42)
    result = {}
    for sid in range(1, n_subjects + 1):
        base = rng.uniform(110, 175)
        noise = rng.uniform(20, 55)
        n = 11_000
        times = pd.date_range("2023-12-01", periods=n, freq="5min")
        glucose = np.clip(
            base + noise * rng.standard_normal(n)
            + 15 * np.sin(np.linspace(0, 12 * np.pi, n)),
            38, 380,
        )
        result[sid] = pd.DataFrame({"EventDateTime": times, "CGM": glucose})
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Export Tableau-ready CSVs from AZT1D data")
    ap.add_argument("--data", default="sample_data/CGM Records",
                    help="Folder with Subject N/ subdirectories")
    ap.add_argument("--out", default="tableau",
                    help="Output folder for CSVs")
    ap.add_argument("--max-ts-subjects", type=int, default=10,
                    help="Cap timeseries export to this many subjects")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_root = Path(args.data)
    all_data: dict[int, pd.DataFrame] = {}
    for subj_dir in sorted(data_root.glob("Subject *")):
        sid = subject_id(subj_dir)
        if sid < 0:
            continue
        df = load_subject(subj_dir)
        if df is not None and len(df) > 0:
            all_data[sid] = df

    if not all_data:
        print(f"No subject data found in {data_root} — using synthetic cohort.")
        all_data = synthetic_cohort()

    print(f"Loaded {len(all_data)} subjects.")

    # Build and write exports
    summary = build_summary(all_data)
    summary.to_csv(out_dir / "patient_summary.csv", index=False)
    print(f"  patient_summary.csv          ({len(summary)} rows)")

    ts = build_timeseries(all_data, max_subjects=args.max_ts_subjects)
    ts.to_csv(out_dir / "glucose_timeseries.csv", index=False)
    print(f"  glucose_timeseries.csv       ({len(ts):,} rows)")

    hypo = build_hypo_events(all_data)
    hypo.to_csv(out_dir / "hypo_events.csv", index=False)
    print(f"  hypo_events.csv              ({len(hypo)} rows)")

    overview = build_cohort_overview(summary)
    overview.to_csv(out_dir / "cohort_overview.csv", index=False)
    print(f"  cohort_overview.csv          (1 row)")

    print(f"\nAll exports written to {out_dir.resolve()}")
    print("Open tableau/cgm_dashboard.twb in Tableau Public to visualise.")


if __name__ == "__main__":
    main()
