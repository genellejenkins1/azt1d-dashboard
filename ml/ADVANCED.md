# AZT1D — Advanced ML / AI modules

Built on top of the base data platform to demonstrate a broader data-science stack.
All run on the synthetic demo data (`ml/generate_synthetic_data.py`); re-run on the
real DVC-pulled data for true performance. **Research/education only — not clinical.**

Model results from `hypo_risk_model.py` (subjects 1–4 train, 5–6 test) are saved to
`ml/outputs/` and displayed live in the **Risk Analysis** tab of the Streamlit dashboard.

| Module | File | Skills showcased | Status |
|---|---|---|---|
| Hypoglycemia-risk classifier | `hypo_risk_model.py` | scikit-learn, subject-level train/test split, ROC-AUC 0.985, PR-AUC 0.895, MLflow | runs |
| Counterfactual what-if simulator | `counterfactual_simulator.py` | response modeling + autoregressive simulation | runs |
| SHAP explainability | `shap_explain.py` | model interpretability / responsible AI | runs |
| LLM clinical-summary generator | `llm_summary.py` | LLM integration (OpenAI/Anthropic) + offline fallback | runs (offline) |
| Glucose forecasting (LSTM) | `forecast_lstm.py` | PyTorch, sequence modeling, multivariate forecasting | code complete, compiles |
| RAG clinical chatbot | `clinical_rag.py` | sentence-transformers, FAISS vector store, LLM, keyword fallback | runs (offline) |
| FHIR R4 export | `fhir_export.py` | HL7 FHIR R4 Bundle, LOINC 15074-8, ADA interpretation codes | runs |
| SAS clinical analysis | `cgm_analysis.sas` | PROC MEANS/FREQ/UNIVARIATE/REG/LOGISTIC/SGPLOT | submit in SAS Studio |

## Quick start
```bash
pip install -r requirements.txt
python ml/generate_synthetic_data.py --subjects 6 --days 14 --out "sample_data/CGM Records"
python ml/hypo_risk_model.py
python ml/counterfactual_simulator.py --subject 1 --bolus-scale 1.3
python ml/shap_explain.py
python ml/llm_summary.py --subject 1      # set OPENAI_API_KEY/ANTHROPIC_API_KEY for a live model
python ml/forecast_lstm.py --epochs 15    # PyTorch LSTM vs. persistence baseline
```

## Notes on results (synthetic demo)
- **SHAP** surfaces recent insulin (bolus) and recent carbs as the dominant near-term
  hypoglycemia drivers — clinically sensible.
- **Counterfactual** rolls a learned one-step response model forward; increasing the
  meal bolus raises hypoglycemia exposure, as expected.
- **Forecasting** reports LSTM MAE against a persistence baseline so the deep model is
  judged against a fair reference.
- **LLM summary** degrades gracefully: no key → deterministic guideline-referenced
  narrative; with a key → live model.
