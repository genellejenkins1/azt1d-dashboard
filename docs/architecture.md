# AZT1D Dashboard — Architecture & Data Flow

```mermaid
flowchart LR
    subgraph Data
        A[(AZT1D raw CSVs\n25 subjects, 5-min CGM)]:::data
        DVC[DVC data versioning]:::infra
        SYN[generate_synthetic_data.py\nschema-accurate demo data]:::infra
    end
    A --> DVC
    DVC --> L
    SYN -. demo mode .-> L

    subgraph Pipeline
        L[SubjectDataLoader\nvalidate + preprocess]:::code
        F[Derived clinical features\nglucose zones · TIR · hypo/hyper flags]:::code
    end
    L --> F

    subgraph Dashboard["Streamlit Dashboard (app.py)"]
        T1[Glucose Visualization\nCGM time-series · insulin · carbs]:::app
        T2[Clinical Metrics\nADA stat cards · TIR · CV · TBR]:::app
        T3[Risk Analysis\nADA goals · zone chart · ML results]:::app
        T4[Parameter Adjustment\ncounterfactual TIR explorer]:::app
    end
    F --> T1 & T2 & T3 & T4

    subgraph ML["ML / AI Layer"]
        HM[hypo_risk_model.py\nLogReg + RF · ROC-AUC 0.985\nMLflow tracked]:::ml
        SH[shap_explain.py\nSHAP beeswarm + importance]:::ml
        LS[forecast_lstm.py\nPyTorch LSTM forecaster]:::ml
        CF[counterfactual_simulator.py\nautoregressive what-if]:::ml
        LLM[llm_summary.py\nOpenAI/Anthropic + offline fallback]:::ml
        RAG[clinical_rag.py\nFAISS vector store · RAG chatbot]:::ml
        FHIR[fhir_export.py\nFHIR R4 Bundle · LOINC 15074-8]:::ml
        SAS[cgm_analysis.sas\nPROC MEANS/REG/LOGISTIC/SGPLOT]:::ml
    end
    F --> HM & SH & LS & CF & LLM & RAG & FHIR & SAS
    HM --> OUT[ml/outputs/\nmetrics.json · ROC · SHAP · calibration]:::out
    OUT --> T3

    subgraph Reporting
        EDA[notebooks/cgm_eda.ipynb\n22-cell EDA · Plotly · Pandas]:::report
        TAB[tableau/cgm_dashboard.twb\nTableau Public workbook]:::report
        QMD[report/cgm_analysis.qmd\nR/Tidyverse · Quarto · CI-rendered]:::report
    end
    F --> EDA & TAB & QMD

    U([Researcher / clinician]):::user
    T1 & T2 & T3 & T4 --> U

    classDef data   fill:#e8f0fe,stroke:#4285f4;
    classDef infra  fill:#fef7e0,stroke:#f9ab00;
    classDef code   fill:#e6f4ea,stroke:#34a853;
    classDef app    fill:#fce8e6,stroke:#ea4335;
    classDef ml     fill:#f3e8fd,stroke:#a142f4;
    classDef out    fill:#f1f3f4,stroke:#5f6368;
    classDef report fill:#fff8e1,stroke:#fbc02d;
    classDef user   fill:#e0f7fa,stroke:#00838f;
```

## Layers

| Layer | Components | Key technologies |
|---|---|---|
| **Data** | AZT1D raw CSVs + DVC pointer; synthetic generator for demo | DVC, NumPy, Pandas |
| **Pipeline** | `SubjectDataLoader` — schema validation, ADA feature engineering | Pandas, Pytest (49 tests) |
| **Dashboard** | 4-tab Streamlit app — visualization, clinical metrics, risk analysis, counterfactual | Streamlit, Plotly |
| **ML / AI** | Hypo-risk classifier, SHAP, LSTM forecaster, counterfactual simulator, LLM narrative, RAG chatbot, FHIR export, SAS analysis | scikit-learn, PyTorch, SHAP, sentence-transformers, FAISS, MLflow, HL7 FHIR |
| **Reporting** | EDA notebook, Tableau workbook, Quarto/R report (CI-rendered) | Jupyter, Tableau, Quarto, R/Tidyverse |

## Key design decisions

- **Subject-level train/test split** in `hypo_risk_model.py` — subjects 1–4 train, 5–6 test — prevents temporal and cross-subject data leakage.
- **Graceful degradation** throughout: RAG falls back to keyword scoring without FAISS; LLM modules fall back to deterministic templates without an API key; dashboard ML panel loads pre-computed artifacts and shows an info message if they are absent.
- **Real data never deployed** — `data/raw/` is gitignored and DVC-tracked; Streamlit Cloud runs on synthetic demo data only.
- **CI** runs pytest + R/Quarto render on every push via GitHub Actions.
