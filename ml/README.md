# AZT1D — Hypoglycemia Risk Model (ML slice)

A modeling layer on top of the AZT1D dashboard: predict **near-term hypoglycemia**
(CGM < 70 mg/dL within the next 30 minutes) from the last 30 minutes of glucose,
insulin, and carb signal.

## Why this exists
The base project is a strong data-loading + visualization platform. This slice adds
the missing modeling story — feature engineering, leakage-aware evaluation, model
comparison, and calibration — so the project demonstrates end-to-end data science.

## Run it (no real data needed)
```bash
pip install numpy pandas scikit-learn matplotlib
python ml/generate_synthetic_data.py --subjects 6 --days 14 --out "sample_data/CGM Records"   # synthetic, schema-accurate
python ml/hypo_risk_model.py --data "sample_data/CGM Records" --out ml/outputs
```

## What it does
- **Features (per subject, no cross-subject leakage):** rolling 30-min mean / min /
  std / slope of CGM, last-step delta, recent bolus and carb sums, basal rate.
- **Label:** any CGM reading below 70 mg/dL in the next 30 minutes. Rows already in
  hypoglycemia are dropped so the model predicts *onset*, not persistence.
- **Split:** subject-level train/test (held-out subjects), so performance reflects
  generalization to new people, not memorized individuals.
- **Models:** balanced logistic regression vs. random forest.
- **Evaluation:** ROC-AUC, PR-AUC, confusion matrix, calibration curve, RF feature
  importance. All written to `ml/outputs/` (`metrics.json` + PNGs).

## Results (synthetic demo run, 6 subjects / 14 days)
| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic regression | ~0.985 | ~0.90 |
| Random forest | ~0.983 | ~0.87 |

> ⚠️ These numbers are on **synthetic** data with cleaner dynamics than reality, so
> they are optimistic. The point is a working, correctly-evaluated pipeline; on the
> real AZT1D data expect lower, more realistic AUC. Re-run on the DVC-pulled data to
> report true performance.

## Not a medical device
Research and education only. Do not use for clinical decisions.
