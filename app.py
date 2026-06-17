"""
AZT1D Dashboard - Streamlit Application

A comprehensive dashboard for visualizing and analyzing Type 1 Diabetes data
from the AZT1D 2025 dataset.

Authors: Naif A. Ganadily, Genelle Jenkins, Toshika Talele
Prof. Hassan and PhD Student Saman, Arizona State University
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

import json
from pathlib import Path

from src.data_loader import SubjectDataLoader
from src.counterfactual import (
    DEFAULT_LOWER,
    DEFAULT_UPPER,
    apply_uniform_offset,
    compute_range_metrics,
    compute_offset_counterfactual_metrics,
)

# Page configuration
st.set_page_config(
    page_title="AZT1D Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional stylesheet
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* Main content padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* ── Page header ── */
    .page-header {
        background: linear-gradient(135deg, #0a2342 0%, #1a4d8f 100%);
        border-radius: 10px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .page-header-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .page-header-sub {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.70);
        margin: 0.3rem 0 0 0;
    }
    .page-header-badge {
        background: rgba(255,255,255,0.12);
        color: #fff;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.25);
        white-space: nowrap;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e8ecf0;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        margin-bottom: 0.5rem;
    }
    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #7a8899;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.4rem;
    }
    .stat-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #0a2342;
        line-height: 1.1;
    }
    .stat-note {
        font-size: 0.72rem;
        color: #9aa5b1;
        margin-top: 0.3rem;
    }
    .stat-card.good  { border-top: 3px solid #198754; }
    .stat-card.warn  { border-top: 3px solid #fd7e14; }
    .stat-card.bad   { border-top: 3px solid #dc3545; }
    .stat-card.info  { border-top: 3px solid #1a4d8f; }

    /* ── Section label ── */
    .section-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #9aa5b1;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 1.4rem 0 0.6rem 0;
    }

    /* ── Risk badge ── */
    .risk-banner {
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .risk-banner.low    { background: #d1f0e0; border-left: 5px solid #198754; }
    .risk-banner.moderate { background: #fde8ce; border-left: 5px solid #fd7e14; }
    .risk-banner.high   { background: #fad7da; border-left: 5px solid #dc3545; }
    .risk-tier-text { font-size: 1.35rem; font-weight: 700; }
    .risk-tier-text.low      { color: #145232; }
    .risk-tier-text.moderate { color: #7a3800; }
    .risk-tier-text.high     { color: #7c1a22; }
    .risk-sub { font-size: 0.82rem; color: #555; margin-top: 0.2rem; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #e8ecf0;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.82rem;
        font-weight: 600;
        color: #7a8899;
        padding: 0.6rem 1.2rem;
        border-radius: 0;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #1a4d8f !important;
        border-bottom: 2px solid #1a4d8f !important;
        background: transparent !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f7f9fc;
        border-right: 1px solid #e8ecf0;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.72rem;
        font-weight: 700;
        color: #9aa5b1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stDateInput label {
        font-size: 0.8rem;
        color: #4a5568;
    }

    /* Sidebar info block */
    .sidebar-info {
        background: #eef2f8;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        font-size: 0.8rem;
        color: #4a5568;
        line-height: 1.6;
    }
    .sidebar-info strong { color: #0a2342; }

    /* ── Dataframe / table ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #e8ecf0 !important;
        border-radius: 8px !important;
        overflow: hidden;
    }

    /* ── Quick stat chips (tab1) ── */
    .chip-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.5rem; }
    .chip {
        background: #eef2f8;
        border-radius: 8px;
        padding: 0.7rem 1.1rem;
        font-size: 0.82rem;
        color: #0a2342;
        font-weight: 500;
        flex: 1;
        min-width: 140px;
    }
    .chip strong { font-size: 1.1rem; font-weight: 700; display: block; margin-bottom: 1px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_data_loader():
    """Load and cache the data loader instance."""
    return SubjectDataLoader()


@st.cache_data
def load_subject_data(subject_id):
    """Load and cache subject data."""
    loader = load_data_loader()
    return loader.load_subject(subject_id)


@st.cache_data
def get_subject_summary(subject_id):
    """Get and cache subject summary statistics."""
    loader = load_data_loader()
    return loader.get_subject_summary(subject_id)


def create_glucose_plot(df, title="Glucose Levels Over Time"):
    """
    Create an interactive glucose time-series plot with insulin and carb overlays.
    
    Similar to the GlySim dashboard style.
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        subplot_titles=("Continuous Glucose Monitor (CGM)", "Insulin & Carbohydrates"),
        vertical_spacing=0.18,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )
    
    # CGM trace (main plot)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['CGM'],
            name='CGM',
            line=dict(color='#2E86AB', width=2),
            mode='lines',
            hovertemplate='<b>Time:</b> %{x}<br><b>Glucose:</b> %{y:.0f} mg/dL<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Target range bands
    fig.add_hrect(
        y0=70, y1=180,
        line_width=0,
        fillcolor="green",
        opacity=0.1,
        annotation_text="Target Range",
        annotation_position="right",
        row=1, col=1
    )
    
    # Hypoglycemia threshold
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="orange",
        annotation_text="Hypoglycemia",
        annotation_position="right",
        row=1, col=1
    )
    
    # Hyperglycemia threshold
    fig.add_hline(
        y=180,
        line_dash="dash",
        line_color="red",
        annotation_text="Hyperglycemia",
        annotation_position="right",
        row=1, col=1
    )
    
    # Basal insulin (line plot on bottom)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Basal'],
            name='Basal Insulin',
            line=dict(color='#A23B72', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(162, 59, 114, 0.2)',
            hovertemplate='<b>Time:</b> %{x}<br><b>Basal:</b> %{y:.2f} U<extra></extra>'
        ),
        row=2, col=1,
        secondary_y=False
    )
    
    # Bolus events (scatter on bottom)
    bolus_events = df[df['has_bolus'] == 1]
    if len(bolus_events) > 0:
        fig.add_trace(
            go.Scatter(
                x=bolus_events.index,
                y=bolus_events['TotalBolusInsulinDelivered'],
                name='Bolus Insulin',
                mode='markers',
                marker=dict(
                    color='#F18F01',
                    size=10,
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Time:</b> %{x}<br><b>Bolus:</b> %{y:.2f} U<extra></extra>'
            ),
            row=2, col=1,
            secondary_y=False
        )
    
    # Meal events (scatter on bottom, secondary axis)
    meal_events = df[df['has_carbs'] == 1]
    if len(meal_events) > 0:
        fig.add_trace(
            go.Scatter(
                x=meal_events.index,
                y=meal_events['CarbSize'],
                name='Carbohydrates',
                mode='markers',
                marker=dict(
                    color='#06A77D',
                    size=12,
                    symbol='diamond',
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>Time:</b> %{x}<br><b>Carbs:</b> %{y:.0f} g<extra></extra>'
            ),
            row=2, col=1,
            secondary_y=True
        )
    
    # Update axes
    fig.update_xaxes(title_text="Date & Time", row=2, col=1)
    fig.update_yaxes(title_text="Glucose (mg/dL)", row=1, col=1, range=[40, 400])
    fig.update_yaxes(title_text="Insulin (U)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Carbs (g)", row=2, col=1, secondary_y=True)
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center', font=dict(size=15, color="#0a2342")),
        height=700,
        hovermode='x unified',
        showlegend=True,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fafbfc",
        font=dict(family="Inter, sans-serif", color="#4a5568"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f4", linecolor="#e8ecf0")
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f4", linecolor="#e8ecf0")
    
    return fig


def display_clinical_metrics(summary):
    """Display key clinical metrics in a grid."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Mean Glucose",
            value=f"{summary['mean_glucose']:.1f} mg/dL",
            delta=None
        )
        st.metric(
            label="Time in Range",
            value=f"{summary['time_in_range']:.1f}%",
            delta=None,
            help="Target range: 70-180 mg/dL"
        )
    
    with col2:
        st.metric(
            label="CV (Variability)",
            value=f"{summary['cv_glucose']:.1f}%",
            delta=None,
            help="Coefficient of Variation - lower is better"
        )
        st.metric(
            label="Time Below Range",
            value=f"{summary['time_below_range']:.1f}%",
            delta=None,
            help="Hypoglycemia (<70 mg/dL)"
        )
    
    with col3:
        st.metric(
            label="Days of Data",
            value=f"{summary['days_of_data']}",
            delta=None
        )
        st.metric(
            label="Time Above Range",
            value=f"{summary['time_above_range']:.1f}%",
            delta=None,
            help="Hyperglycemia (>180 mg/dL)"
        )
    
    with col4:
        st.metric(
            label="Total Bolus Insulin",
            value=f"{summary['total_bolus_insulin']:.1f} U",
            delta=None
        )
        st.metric(
            label="Total Meals",
            value=f"{summary['n_meals']}",
            delta=None
        )


_OUTPUTS = Path("ml/outputs")


@st.cache_data
def _load_model_metrics() -> dict:
    path = _OUTPUTS / "metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _render_risk_tab(summary: dict, df: pd.DataFrame) -> None:
    """Hypoglycemia risk analysis panel derived from ADA clinical thresholds."""
    st.subheader("Hypoglycemia Risk Analysis")
    st.markdown(
        "Risk tier is computed from three ADA-aligned clinical targets: "
        "Time in Range ≥ 70%, Time Below Range < 4%, and Glucose CV < 36%. "
        "Each unmet target adds one point to the risk score."
    )

    tbr   = summary.get("time_below_range", 0.0)
    tir   = summary.get("time_in_range", 100.0)
    cv    = summary.get("cv_glucose", 0.0)
    mean_g = summary.get("mean_glucose", 0.0)

    # ADA goal flags
    tir_ok  = tir  >= 70.0
    tbr_ok  = tbr  <  4.0
    cv_ok   = cv   < 36.0
    score   = sum([not tir_ok, not tbr_ok, not cv_ok])

    if score == 0:
        tier, color = "Low", "green"
    elif score == 1:
        tier, color = "Moderate", "orange"
    else:
        tier, color = "High", "red"

    tier_cls = tier.lower()
    st.markdown(
        f"""
        <div class="risk-banner {tier_cls}">
            <div>
                <div class="risk-tier-text {tier_cls}">Risk Tier: {tier}</div>
                <div class="risk-sub">ADA clinical targets missed: {score} of 3 &nbsp;&mdash;&nbsp;
                TIR {"met" if tir_ok else "not met"} &nbsp;·&nbsp;
                TBR {"met" if tbr_ok else "not met"} &nbsp;·&nbsp;
                CV {"met" if cv_ok else "not met"}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">ADA Clinical Targets</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "Met" if tir_ok else "Not Met"
        st.metric("Time in Range (≥ 70%)", f"{tir:.1f}%", delta=status,
                  delta_color="normal" if tir_ok else "inverse")
    with col2:
        status = "Met" if tbr_ok else "Not Met"
        st.metric("Time Below Range (< 4%)", f"{tbr:.1f}%", delta=status,
                  delta_color="normal" if tbr_ok else "inverse")
    with col3:
        status = "Met" if cv_ok else "Not Met"
        st.metric("Glucose CV (< 36%)", f"{cv:.1f}%", delta=status,
                  delta_color="normal" if cv_ok else "inverse")

    st.markdown("---")
    st.markdown('<div class="section-label">Glucose Zone Distribution</div>', unsafe_allow_html=True)
    zone_map = {
        "Severe Hypo (<54)":   (df["CGM"] < 54).mean() * 100,
        "Hypo (54–70)":        ((df["CGM"] >= 54) & (df["CGM"] < 70)).mean() * 100,
        "In Range (70–180)":   ((df["CGM"] >= 70) & (df["CGM"] <= 180)).mean() * 100,
        "Hyper (180–250)":     ((df["CGM"] > 180) & (df["CGM"] <= 250)).mean() * 100,
        "Severe Hyper (>250)": (df["CGM"] > 250).mean() * 100,
    }
    zone_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#ff7f0e", "#d62728"]

    fig_zone = go.Figure(go.Bar(
        x=list(zone_map.keys()),
        y=list(zone_map.values()),
        marker_color=zone_colors,
        text=[f"{v:.1f}%" for v in zone_map.values()],
        textposition="outside",
    ))
    fig_zone.update_layout(
        yaxis=dict(title="% of Readings", range=[0, 100]),
        xaxis_title="Glucose Zone",
        height=380,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_zone, use_container_width=True)

    st.markdown('<div class="section-label">Diurnal Glucose Pattern — Mean by Hour of Day</div>', unsafe_allow_html=True)
    hourly = df.copy()
    hourly["hour"] = hourly.index.hour
    hourly_mean = hourly.groupby("hour")["CGM"].mean().reset_index()

    fig_diurnal = go.Figure()
    fig_diurnal.add_trace(go.Scatter(
        x=hourly_mean["hour"],
        y=hourly_mean["CGM"],
        mode="lines+markers",
        line=dict(color="#2E86AB", width=2),
        marker=dict(size=6),
        name="Mean CGM",
    ))
    fig_diurnal.add_hrect(y0=70, y1=180, line_width=0,
                          fillcolor="green", opacity=0.08,
                          annotation_text="Target Range")
    fig_diurnal.update_layout(
        xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=2),
        yaxis=dict(title="Mean Glucose (mg/dL)"),
        height=340,
        margin=dict(t=20),
    )
    st.plotly_chart(fig_diurnal, use_container_width=True)

    # ── ML Model Results (from hypo_risk_model.py — proper train/test split) ──
    st.markdown("---")
    st.markdown('<div class="section-label">ML Hypoglycemia Prediction Model</div>', unsafe_allow_html=True)
    st.markdown(
        "Trained on subjects 1–4 (9,660 windows), evaluated on held-out subjects 5–6 (6,354 windows). "
        "Task: predict CGM < 70 mg/dL within the next 30 minutes from 30-min rolling CGM features "
        "plus recent insulin and carbohydrate events. "
        "See `ml/hypo_risk_model.py` for full pipeline with MLflow tracking."
    )

    metrics = _load_model_metrics()
    if not metrics:
        st.info("Run `python ml/hypo_risk_model.py` to generate model metrics.")
    else:
        best = metrics.get("best_model", "logistic_regression")
        m = metrics["models"][best]
        report = m["report"]
        cm = m["confusion_matrix"]

        # Key metric cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f'<div class="stat-card good"><div class="stat-label">ROC-AUC</div>'
                f'<div class="stat-value">{m["roc_auc"]:.3f}</div>'
                f'<div class="stat-note">Held-out subjects 5–6</div></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="stat-card good"><div class="stat-label">PR-AUC</div>'
                f'<div class="stat-value">{m["pr_auc"]:.3f}</div>'
                f'<div class="stat-note">Precision-recall · imbalanced</div></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="stat-card info"><div class="stat-label">Hypo Recall</div>'
                f'<div class="stat-value">{report["1"]["recall"]:.1%}</div>'
                f'<div class="stat-note">True positive rate on hypo class</div></div>',
                unsafe_allow_html=True,
            )
        with c4:
            pos_rate = metrics.get("positive_rate_test", 0)
            st.markdown(
                f'<div class="stat-card info"><div class="stat-label">Hypo Rate (Test)</div>'
                f'<div class="stat-value">{pos_rate:.1%}</div>'
                f'<div class="stat-note">Class imbalance in held-out set</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-label" style="margin-top:1.2rem;">Model Artifacts</div>', unsafe_allow_html=True)
        col_roc, col_feat = st.columns(2)
        with col_roc:
            roc_path = _OUTPUTS / "roc_curve.png"
            if roc_path.exists():
                st.image(str(roc_path), caption=f"ROC Curve — {best.replace('_', ' ').title()} (AUC {m['roc_auc']:.3f})", use_container_width=True)
        with col_feat:
            feat_path = _OUTPUTS / "feature_importance.png"
            if feat_path.exists():
                st.image(str(feat_path), caption="Feature Importance — Random Forest", use_container_width=True)

        # Confusion matrix as a small Plotly heatmap
        st.markdown('<div class="section-label" style="margin-top:1rem;">Confusion Matrix — Held-Out Subjects</div>', unsafe_allow_html=True)
        cm_labels = ["No Hypo", "Hypo"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=cm_labels,
            y=cm_labels,
            colorscale=[[0, "#eef2f8"], [1, "#1a4d8f"]],
            showscale=False,
            text=[[str(v) for v in row] for row in cm],
            texttemplate="%{text}",
            textfont=dict(size=18, color="white"),
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=280,
            margin=dict(t=10, b=40, l=60, r=20),
            paper_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif"),
        )
        col_cm, col_top = st.columns([1, 1])
        with col_cm:
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_top:
            st.markdown('<div class="section-label" style="margin-top:0;">Top Predictive Features</div>', unsafe_allow_html=True)
            top_feats = metrics.get("top_features", [])
            if top_feats:
                feat_names = [f["feature"].replace("_", " ") for f in top_feats[:6]]
                feat_vals  = [f["importance"] for f in top_feats[:6]]
                fig_top = go.Figure(go.Bar(
                    x=feat_vals,
                    y=feat_names,
                    orientation="h",
                    marker_color="#1a4d8f",
                    text=[f"{v:.1%}" for v in feat_vals],
                    textposition="outside",
                ))
                fig_top.update_layout(
                    height=280,
                    margin=dict(t=10, b=10, l=10, r=60),
                    xaxis=dict(title="Importance", showgrid=True, gridcolor="#eef0f4"),
                    yaxis=dict(autorange="reversed"),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#fafbfc",
                    font=dict(family="Inter, sans-serif", size=12),
                )
                st.plotly_chart(fig_top, use_container_width=True)

    st.caption(
        "Model trained with proper subject-level train/test split to prevent data leakage. "
        "Risk scoring uses ADA consensus targets (Diabetes Care 2019). "
        "Educational tool only — not a medical device."
    )


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown(
        """
        <div class="page-header">
            <div>
                <div class="page-header-title">AZT1D Dashboard</div>
                <div class="page-header-sub">Continuous Glucose Monitoring &amp; Clinical Analytics Platform</div>
            </div>
            <div class="page-header-badge">Arizona State University &nbsp;·&nbsp; Type 1 Diabetes Research</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Initialize data loader
    try:
        loader = load_data_loader()
        available_subjects = loader.get_available_subjects()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure DVC data is pulled: `dvc pull`")
        return
    
    # Sidebar controls
    st.sidebar.header("Data Selection")
    
    # Subject selector
    selected_subject = st.sidebar.selectbox(
        "Select Subject",
        options=available_subjects,
        index=0,
        help="Choose a subject to visualize"
    )
    
    # Load subject data
    with st.spinner(f"Loading Subject {selected_subject} data..."):
        df = load_subject_data(selected_subject)
        summary = get_subject_summary(selected_subject)
    
    # Date range selector
    st.sidebar.subheader("Date Range")
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, min_date + timedelta(days=3)),
        min_value=min_date,
        max_value=max_date,
        help="Select up to 7 days for optimal visualization"
    )
    
    # Filter data by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df.index.date >= start_date) & (df.index.date <= end_date)
        df_filtered = df[mask]
        
        if len(df_filtered) == 0:
            st.warning("No data available for selected date range.")
            return
    else:
        df_filtered = df
    
    # Display subject info
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div class="sidebar-info">
            <strong>Subject {selected_subject}</strong><br>
            {summary['start_date']} — {summary['end_date']}<br>
            {summary['n_records']:,} CGM records &nbsp;·&nbsp; {summary['days_of_data']} days
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["Glucose Visualization", "Clinical Metrics", "Risk Analysis", "Parameter Adjustment"])
    
    with tab1:
        st.subheader("Continuous Glucose Monitor & Insulin/Carb Events")
        
        # Create and display plot
        fig = create_glucose_plot(
            df_filtered,
            title=f"Subject {selected_subject} - Glucose Profile"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats for filtered date range
        tir_filtered = (df_filtered['in_target_range'].sum() / len(df_filtered)) * 100
        mean_filtered = df_filtered['CGM'].mean()
        st.markdown('<div class="section-label">Selected Range Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="chip-row">
                <div class="chip"><strong>{len(df_filtered):,}</strong>CGM Readings</div>
                <div class="chip"><strong>{mean_filtered:.1f} mg/dL</strong>Mean Glucose</div>
                <div class="chip"><strong>{tir_filtered:.1f}%</strong>Time in Range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with tab2:
        st.subheader("Clinical Performance Metrics")

        tir_v  = summary['time_in_range']
        tbr_v  = summary['time_below_range']
        tar_v  = summary['time_above_range']
        cv_v   = summary['cv_glucose']
        mean_v = summary['mean_glucose']

        tir_cls = "good" if tir_v >= 70 else ("warn" if tir_v >= 50 else "bad")
        tbr_cls = "good" if tbr_v < 4   else ("warn" if tbr_v < 10  else "bad")
        cv_cls  = "good" if cv_v  < 36  else "warn"

        st.markdown('<div class="section-label">Glycemic Control</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="stat-card {tir_cls}"><div class="stat-label">Time in Range</div><div class="stat-value">{tir_v:.1f}%</div><div class="stat-note">Target ≥ 70% &nbsp;(70–180 mg/dL)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-card {tbr_cls}"><div class="stat-label">Time Below Range</div><div class="stat-value">{tbr_v:.1f}%</div><div class="stat-note">Target &lt; 4% &nbsp;(&lt;70 mg/dL)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-card info"><div class="stat-label">Time Above Range</div><div class="stat-value">{tar_v:.1f}%</div><div class="stat-note">Target &lt; 25% &nbsp;(&gt;180 mg/dL)</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="stat-card {cv_cls}"><div class="stat-label">Glucose CV</div><div class="stat-value">{cv_v:.1f}%</div><div class="stat-note">Target &lt; 36%</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Overall Statistics</div>', unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(f'<div class="stat-card info"><div class="stat-label">Mean Glucose</div><div class="stat-value">{mean_v:.1f}</div><div class="stat-note">mg/dL</div></div>', unsafe_allow_html=True)
        with c6:
            st.markdown(f'<div class="stat-card info"><div class="stat-label">Days of Data</div><div class="stat-value">{summary["days_of_data"]}</div><div class="stat-note">days observed</div></div>', unsafe_allow_html=True)
        with c7:
            st.markdown(f'<div class="stat-card info"><div class="stat-label">Total Bolus Insulin</div><div class="stat-value">{summary["total_bolus_insulin"]:.0f}</div><div class="stat-note">units delivered</div></div>', unsafe_allow_html=True)
        with c8:
            st.markdown(f'<div class="stat-card info"><div class="stat-label">Total Meals</div><div class="stat-value">{summary["n_meals"]}</div><div class="stat-note">carb events recorded</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Additional statistics table
        st.subheader("Detailed Statistics")
        
        stats_df = pd.DataFrame({
            'Metric': [
                'Mean Glucose',
                'Median Glucose',
                'Std Deviation',
                'Coefficient of Variation',
                'Time in Range (70-180)',
                'Time Below Range (<70)',
                'Time Above Range (>180)',
                'Total Insulin (Bolus)',
                'Total Insulin (Basal)',
                'Total Carbohydrates',
                'Number of Bolus Events',
                'Number of Meals'
            ],
            'Value': [
                f"{summary['mean_glucose']:.1f} mg/dL",
                f"{df['CGM'].median():.1f} mg/dL",
                f"{summary['std_glucose']:.1f} mg/dL",
                f"{summary['cv_glucose']:.1f}%",
                f"{summary['time_in_range']:.1f}%",
                f"{summary['time_below_range']:.1f}%",
                f"{summary['time_above_range']:.1f}%",
                f"{summary['total_bolus_insulin']:.1f} U",
                f"{summary['total_basal_insulin']:.1f} U",
                f"{summary['total_carbs']:.0f} g",
                f"{summary['n_bolus_events']}",
                f"{summary['n_meals']}"
            ]
        })
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    with tab3:
        _render_risk_tab(summary, df)

    with tab4:
        st.subheader("Counterfactual Metrics (Target Range & CGM Bias)")

        st.markdown(
            """\
This panel implements *metric-level* counterfactuals only. We do **not**
simulate physiology or insulin delivery. Instead, we:

1. Change the definition of the target range [L, U] and recompute
   time-in-range metrics.
2. Apply a uniform CGM bias (g'_t = g_t + b) and recompute the same metrics.

This is appropriate for exploring how clinical metrics depend on reporting
conventions and sensor bias, without making unverifiable claims about how
therapy decisions would have changed.
"""
        )

        st.markdown("### Parameters")
        col_l, col_u, col_b = st.columns(3)
        with col_l:
            cf_lower = st.number_input(
                "Target range lower (mg/dL)",
                min_value=40.0,
                max_value=140.0,
                value=float(DEFAULT_LOWER),
                step=1.0,
            )
        with col_u:
            cf_upper = st.number_input(
                "Target range upper (mg/dL)",
                min_value=120.0,
                max_value=300.0,
                value=float(DEFAULT_UPPER),
                step=1.0,
            )
        with col_b:
            cgm_offset = st.slider(
                "CGM bias (mg/dL)",
                min_value=-40,
                max_value=40,
                value=0,
                step=1,
                help="Uniform additive offset applied to CGM values.",
            )

        if cf_upper <= cf_lower + 10.0:
            st.warning(
                "Upper bound should be at least 10 mg/dL above the lower bound "
                "for a meaningful target range."
            )
        else:
            # Baseline metrics under canonical ADA-style range
            baseline_metrics = compute_range_metrics(
                df_filtered["CGM"], lower=DEFAULT_LOWER, upper=DEFAULT_UPPER
            )

            # Counterfactual 1: new target range but same glucose trace
            range_only_metrics = compute_range_metrics(
                df_filtered["CGM"], lower=cf_lower, upper=cf_upper
            )

            # Counterfactual 2: new range + uniform CGM offset
            offset_metrics = compute_offset_counterfactual_metrics(
                df_filtered["CGM"],
                offset_mgdl=float(cgm_offset),
                lower=cf_lower,
                upper=cf_upper,
            )

            # Summarise in a compact table
            summary_df = pd.DataFrame(
                [
                    {
                        "Scenario": "Observed (70-180)",
                        **baseline_metrics.as_dict(),
                    },
                    {
                        "Scenario": f"Range-only [{cf_lower:.0f}, {cf_upper:.0f}]",
                        **range_only_metrics.as_dict(),
                    },
                    {
                        "Scenario": f"Range+Offset [{cf_lower:.0f}, {cf_upper:.0f}], bias {cgm_offset:+d}",
                        **offset_metrics.as_dict(),
                    },
                ]
            )

            st.markdown("### Time-in-Range Style Metrics")
            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("### Glucose Trace With Optional CGM Bias")
            cf_glucose = apply_uniform_offset(df_filtered["CGM"], cgm_offset)

            fig_cf = go.Figure()
            fig_cf.add_trace(
                go.Scatter(
                    x=df_filtered.index,
                    y=df_filtered["CGM"],
                    name="Observed CGM",
                    line=dict(color="#2E86AB", width=2),
                )
            )
            if cgm_offset != 0:
                fig_cf.add_trace(
                    go.Scatter(
                        x=df_filtered.index,
                        y=cf_glucose,
                        name=f"CGM + {cgm_offset:+d} mg/dL",
                        line=dict(color="#A23B72", width=2, dash="dash"),
                    )
                )

            # Visualise the counterfactual target band
            fig_cf.add_hrect(
                y0=cf_lower,
                y1=cf_upper,
                line_width=0,
                fillcolor="green",
                opacity=0.08,
            )

            fig_cf.update_yaxes(title_text="Glucose (mg/dL)")
            fig_cf.update_xaxes(title_text="Date & Time")
            fig_cf.update_layout(
                height=450,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            st.plotly_chart(fig_cf, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("AZT1D Dashboard · v0.0.4 · Arizona State University")


if __name__ == "__main__":
    main()