"""
hypo_risk_model.py
------------------
Predict near-term HYPOGLYCEMIA RISK from continuous glucose-monitoring (CGM)
history: given the last 30 minutes of signal, will the subject's glucose drop
below 70 mg/dL within the next 30 minutes?

This turns the AZT1D platform from a data-loading + visualization tool into a
modeling project. It demonstrates: time-series feature engineering, leakage-aware
subject-level train/test splitting, model comparison, and proper evaluation
(ROC-AUC, PR-AUC, calibration, confusion matrix, feature importance).

MLflow experiment tracking is enabled by default. Each run logs parameters,
metrics, and artifacts to the local `mlruns/` directory. View results with:
    mlflow ui

Research/education only -- NOT a medical device and not for clinical use.

Usage:
    python ml/hypo_risk_model.py --data "sample_data/CGM Records" --out ml/outputs
    python ml/hypo_risk_model.py --no-mlflow   # disable tracking
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             roc_curve, confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from contextlib import contextmanager

@contextmanager
def _noop_ctx():
    """Drop-in no-op for mlflow.start_run() when MLflow is disabled."""
    yield

HYPO = 70           # mg/dL
HORIZON = 6         # steps ahead (6 x 5min = 30 min)
LOOKBACK = 6        # steps of history (30 min)


def load_all(data_dir: str) -> pd.DataFrame:
    """Load every Subject folder's CSV. Falls back gracefully if loader absent."""
    rows = []
    base = Path(data_dir)
    for sub in sorted(base.glob("Subject *")):
        for csv in sub.glob("*.csv"):
            df = pd.read_csv(csv, parse_dates=["EventDateTime"])
            df = df.sort_values("EventDateTime")
            df["subject_id"] = int(sub.name.split()[-1])
            rows.append(df)
    if not rows:
        sys.exit(f"No data found in {data_dir}. Run generate_synthetic_data.py first.")
    return pd.concat(rows, ignore_index=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-subject rolling features + future-hypo label (no cross-subject leakage)."""
    out = []
    for sid, g in df.groupby("subject_id"):
        g = g.sort_values("EventDateTime").reset_index(drop=True)
        cgm = g["CGM"].astype(float)
        feat = pd.DataFrame({
            "subject_id": sid,
            "cgm": cgm,
            "cgm_mean_30m": cgm.rolling(LOOKBACK).mean(),
            "cgm_min_30m": cgm.rolling(LOOKBACK).min(),
            "cgm_std_30m": cgm.rolling(LOOKBACK).std(),
            "cgm_slope_30m": cgm.diff(LOOKBACK) / LOOKBACK,
            "cgm_last_delta": cgm.diff(),
            "recent_bolus_30m": g["TotalBolusInsulinDelivered"].rolling(LOOKBACK).sum(),
            "recent_carbs_30m": g["CarbSize"].rolling(LOOKBACK).sum(),
            "basal": g["Basal"].astype(float),
        })
        # Label: any reading in next HORIZON steps < 70
        future_min = cgm[::-1].rolling(HORIZON, min_periods=1).min()[::-1].shift(-1)
        feat["label"] = (future_min < HYPO).astype(int)
        out.append(feat)
    f = pd.concat(out, ignore_index=True).dropna()
    # Drop rows already hypo (predict ONSET, not persistence)
    f = f[f["cgm"] >= HYPO]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--out", default="ml/outputs")
    ap.add_argument("--no-mlflow", action="store_true",
                    help="Disable MLflow experiment tracking")
    ap.add_argument("--experiment", default="hypo-risk-model",
                    help="MLflow experiment name")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    use_mlflow = MLFLOW_AVAILABLE and not args.no_mlflow
    if use_mlflow:
        mlflow.set_experiment(args.experiment)
        print(f"MLflow tracking enabled — experiment: '{args.experiment}'")
        print("  View results: mlflow ui")
    elif not MLFLOW_AVAILABLE and not args.no_mlflow:
        print("MLflow not installed (pip install mlflow). Run with --no-mlflow to suppress.")

    raw = load_all(args.data)
    f = make_features(raw)
    feat_cols = [c for c in f.columns if c not in ("subject_id", "label")]

    # Subject-level split to avoid leakage
    subjects = sorted(f["subject_id"].unique())
    n_test = max(1, len(subjects) // 3)
    test_subjects = set(subjects[-n_test:])
    train = f[~f["subject_id"].isin(test_subjects)]
    test = f[f["subject_id"].isin(test_subjects)]

    Xtr, ytr = train[feat_cols].values, train["label"].values
    Xte, yte = test[feat_cols].values, test["label"].values

    model_configs = {
        "logistic_regression": {
            "model": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced")),
            "params": {"max_iter": 1000, "class_weight": "balanced"},
        },
        "random_forest": {
            "model": RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced", random_state=0),
            "params": {"n_estimators": 300, "max_depth": 8, "class_weight": "balanced"},
        },
    }

    results = {}
    plt.figure(figsize=(6, 5))

    for name, cfg in model_configs.items():
        model = cfg["model"]

        with (mlflow.start_run(run_name=name) if use_mlflow else _noop_ctx()):
            # Shared pipeline params
            shared_params = {
                "lookback_steps": LOOKBACK,
                "horizon_steps": HORIZON,
                "hypo_threshold_mgdl": HYPO,
                "n_train_subjects": len(subjects) - len(test_subjects),
                "n_test_subjects": len(test_subjects),
            }
            if use_mlflow:
                mlflow.log_params({**shared_params, **cfg["params"]})
                mlflow.set_tag("model_type", name)
                mlflow.set_tag("dataset", "AZT1D synthetic CGM")

            model.fit(Xtr, ytr)
            proba = model.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(yte, proba)
            ap_score = average_precision_score(yte, proba)

            results[name] = {
                "roc_auc": round(float(auc), 4),
                "pr_auc": round(float(ap_score), 4),
                "confusion_matrix": confusion_matrix(yte, pred).tolist(),
                "report": classification_report(yte, pred, output_dict=True, zero_division=0),
            }

            if use_mlflow:
                mlflow.log_metrics({"roc_auc": auc, "pr_auc": ap_score})
                mlflow.sklearn.log_model(model, artifact_path=name)

        fpr, tpr, _ = roc_curve(yte, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # ROC plot
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Hypoglycemia risk — ROC (30-min horizon)")
    plt.legend(); plt.tight_layout()
    roc_path = f"{args.out}/roc_curve.png"
    plt.savefig(roc_path, dpi=130); plt.close()

    # Calibration (best model = highest AUC)
    best = max(results, key=lambda k: results[k]["roc_auc"])
    proba_best = model_configs[best]["model"].predict_proba(Xte)[:, 1]
    frac_pos, mean_pred = calibration_curve(yte, proba_best, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, frac_pos, "o-", label=best)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration"); plt.legend(); plt.tight_layout()
    cal_path = f"{args.out}/calibration.png"
    plt.savefig(cal_path, dpi=130); plt.close()

    # Feature importance (random forest)
    rf = model_configs["random_forest"]["model"]
    imp = sorted(zip(feat_cols, rf.feature_importances_), key=lambda x: -x[1])
    plt.figure(figsize=(6, 4))
    plt.barh([k for k, _ in imp][::-1], [v for _, v in imp][::-1])
    plt.title("Random forest — feature importance"); plt.tight_layout()
    fi_path = f"{args.out}/feature_importance.png"
    plt.savefig(fi_path, dpi=130); plt.close()

    # Log plots to the best-model run if mlflow is active
    if use_mlflow:
        with mlflow.start_run(run_name=f"{best}_summary"):
            mlflow.log_artifact(roc_path, artifact_path="plots")
            mlflow.log_artifact(cal_path, artifact_path="plots")
            mlflow.log_artifact(fi_path, artifact_path="plots")
            mlflow.set_tag("best_model", best)
            mlflow.log_metric("best_roc_auc", results[best]["roc_auc"])

    summary = {
        "task": "Predict CGM < 70 mg/dL within next 30 minutes",
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        "train_subjects": [int(s) for s in subjects if s not in test_subjects],
        "test_subjects": sorted(int(s) for s in test_subjects),
        "positive_rate_test": round(float(yte.mean()), 4),
        "best_model": best,
        "models": results,
        "top_features": [{"feature": k, "importance": round(float(v), 4)} for k, v in imp],
    }
    with open(f"{args.out}/metrics.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps({k: {"roc_auc": v["roc_auc"], "pr_auc": v["pr_auc"]}
                      for k, v in results.items()}, indent=2))
    print(f"Best model: {best}. Artifacts -> {args.out}/")


if __name__ == "__main__":
    main()
