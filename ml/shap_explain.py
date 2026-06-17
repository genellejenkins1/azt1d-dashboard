"""
shap_explain.py
---------------
Explain the hypoglycemia-risk model with SHAP. Answers "WHY did the model flag this
window as high risk?" -- global feature importance and per-prediction attributions.
Interpretability is essential for clinical ML trust.

Usage:
    python ml/shap_explain.py --data "sample_data/CGM Records" --out ml/outputs
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from hypo_risk_model import load_all, make_features  # reuse the same pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sample_data/CGM Records")
    ap.add_argument("--out", default="ml/outputs")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    f = make_features(load_all(a.data))
    feat_cols = [c for c in f.columns if c not in ("subject_id", "label")]
    X, y = f[feat_cols], f["label"]

    model = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   class_weight="balanced", random_state=0).fit(X, y)

    expl = shap.TreeExplainer(model)
    sample = X.sample(min(800, len(X)), random_state=0)
    sv = expl.shap_values(sample)
    # Normalize across SHAP versions to a 2D (n_samples, n_features) positive-class array
    if isinstance(sv, list):
        sv_pos = sv[1]
    else:
        sv_pos = np.asarray(sv)
        if sv_pos.ndim == 3:          # (n, features, classes) -> positive class
            sv_pos = sv_pos[:, :, -1]

    shap.summary_plot(sv_pos, sample, show=False, plot_type="bar")
    plt.tight_layout(); plt.savefig(f"{a.out}/shap_importance_bar.png", dpi=130); plt.close()

    shap.summary_plot(sv_pos, sample, show=False)
    plt.tight_layout(); plt.savefig(f"{a.out}/shap_beeswarm.png", dpi=130); plt.close()

    mean_abs = np.abs(sv_pos.reshape(len(sample), -1)).mean(0)
    order = np.argsort(mean_abs)[::-1]
    print("Top risk drivers (mean |SHAP|):")
    for i in order[:6]:
        print(f"  {feat_cols[i]:20s} {mean_abs[i]:.4f}")
    print(f"Plots -> {a.out}/shap_importance_bar.png, shap_beeswarm.png")


if __name__ == "__main__":
    main()
