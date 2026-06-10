"""
comparison_tab.py
-----------------
Comparative model analysis + active model selector.

Model names (per spec):
  Eye:
    A1 - EAR Threshold   (rule-based)
    B1 - MobileNetV2     (CNN)
    C1 - Custom CNN      (CNN)

  Posture:
    A2 - Angle-Based     (rule-based)
    B2 - BlazePose DNN   (DNN)
    C2 - Custom LSTM     (LSTM)

IMPORTANT: The comparison charts and summary table display metrics ONLY
for the currently selected models — not all six at once.

Selected models are stored in st.session_state:
  "active_eye_model"     — used by monitoring_tab + inference_server
  "active_posture_model" — used by monitoring_tab + inference_server
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import sys, os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.model_loader import (
    load_selected_eye_model,
    load_selected_posture_model,
    load_selected_results,
    EYE_MODEL_PATHS,
    POSTURE_MODEL_PATHS,
)
from utils.frame_processor import (
    EYE_MODEL_A1, EYE_MODEL_B1, EYE_MODEL_C1,
    POSTURE_MODEL_A2, POSTURE_MODEL_B2, POSTURE_MODEL_C2,
)

# ──────────────────────────────────────────────
# Model lists (in display order)
# ──────────────────────────────────────────────

EYE_MODELS     = [EYE_MODEL_A1, EYE_MODEL_B1, EYE_MODEL_C1]
POSTURE_MODELS = [POSTURE_MODEL_A2, POSTURE_MODEL_B2, POSTURE_MODEL_C2]

EYE_COLORS     = {"A1 - EAR Threshold": "#f39c12",
                  "B1 - MobileNetV2":   "#2ecc71",
                  "C1 - Custom CNN":    "#3498db"}
POSTURE_COLORS = {"A2 - Angle-Based":   "#f39c12",
                  "B2 - BlazePose DNN": "#2ecc71",
                  "C2 - Custom LSTM":   "#e74c3c"}

DEFAULT_EYE_MODEL     = EYE_MODEL_C1
DEFAULT_POSTURE_MODEL = POSTURE_MODEL_C2

CHART_LAYOUT = dict(
    plot_bgcolor="#1e2130",
    paper_bgcolor="#1e2130",
    font=dict(color="#e0e0e0", size=11),
    margin=dict(t=55, b=40, l=50, r=20),
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _eye_color(name):    return EYE_COLORS.get(name,     "#3498db")
def _posture_color(name): return POSTURE_COLORS.get(name, "#e74c3c")


def _accuracy_chart(model_name: str, result: dict, color: str, title: str):
    acc = result.get("accuracy", 0) * 100
    f1  = result.get("f1_score",  0) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy (%)", x=[model_name], y=[acc],
        marker_color=color,
        text=[f"{acc:.1f}%"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="F1-Score (%)", x=[model_name], y=[f1],
        marker_color=color + "88",
        text=[f"{f1:.1f}%"], textposition="outside",
    ))
    fig.update_layout(
        title=title, barmode="group",
        yaxis=dict(title="Score (%)", range=[0, 115]),
        xaxis_title="Model",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320, **CHART_LAYOUT,
    )
    return fig


def _latency_chart(model_name: str, result: dict, color: str, title: str):
    lat = result.get("latency_ms", 0)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[lat], y=[model_name], orientation="h",
        marker_color=color,
        text=[f"{lat:.1f} ms"], textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis=dict(title="Latency (ms)"),
        height=220, **CHART_LAYOUT,
    )
    return fig


def _radar_chart(model_name: str, result: dict, color: str, title: str):
    """Single-model radar: Accuracy / F1-Score / Speed."""
    lat   = result.get("latency_ms", 10)
    speed = max(0, (1 - lat / 50) * 100)   # 50 ms = 0 speed score
    vals  = [
        result.get("accuracy", 0) * 100,
        result.get("f1_score",  0) * 100,
        speed,
    ]
    categories = ["Accuracy", "F1-Score", "Speed"]
    vals_closed = vals + [vals[0]]
    cats_closed = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself", name=model_name,
        line_color=color, fillcolor=color + "44",
    ))
    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor="#1e2130",
        ),
        height=320, **CHART_LAYOUT,
    )
    return fig


def _summary_table(eye_model: str, eye_result: dict,
                   posture_model: str, posture_result: dict):
    """Show a compact table for only the selected two models."""
    import pandas as pd
    rows = [
        {
            "Group":          "Eye",
            "Model":          eye_model,
            "Accuracy (%)":   f"{eye_result.get('accuracy', 0) * 100:.1f}",
            "F1-Score (%)":   f"{eye_result.get('f1_score',  0) * 100:.1f}",
            "Latency (ms)":   f"{eye_result.get('latency_ms', 0):.1f}",
            "Type":           "Rule-Based" if eye_model == EYE_MODEL_A1 else "ML Model",
        },
        {
            "Group":          "Posture",
            "Model":          posture_model,
            "Accuracy (%)":   f"{posture_result.get('accuracy', 0) * 100:.1f}",
            "F1-Score (%)":   f"{posture_result.get('f1_score',  0) * 100:.1f}",
            "Latency (ms)":   f"{posture_result.get('latency_ms', 0):.1f}",
            "Type":           "Rule-Based" if posture_model == POSTURE_MODEL_A2 else "ML Model",
        },
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Model selector widget
# ──────────────────────────────────────────────

def _model_selector():
    st.subheader("Active Model Selection")
    st.caption(
        "Select one eye model and one posture model. "
        "Only the selected models will be loaded and run — "
        "unselected models consume no memory or processing power."
    )

    if "active_eye_model" not in st.session_state:
        st.session_state["active_eye_model"] = DEFAULT_EYE_MODEL
    if "active_posture_model" not in st.session_state:
        st.session_state["active_posture_model"] = DEFAULT_POSTURE_MODEL

    col_eye, col_posture = st.columns(2)

    with col_eye:
        st.markdown("**Eye Strain Model**")
        for m in EYE_MODELS:
            is_rule = (m == EYE_MODEL_A1)
            badge   = " *(rule-based)*" if is_rule else " *(ML model)*"
            st.caption(m + badge)
        chosen_eye = st.radio(
            "Eye model",
            options=EYE_MODELS,
            index=EYE_MODELS.index(st.session_state["active_eye_model"])
                  if st.session_state["active_eye_model"] in EYE_MODELS else 2,
            key="cmp_eye_radio",
            label_visibility="collapsed",
        )
        if chosen_eye == EYE_MODEL_A1:
            st.info("A1 — EAR threshold rule. No .h5 model needed.")
        else:
            path = EYE_MODEL_PATHS.get(chosen_eye)
            if path and os.path.exists(path):
                st.success(f"{chosen_eye} — model file found ✓")
            else:
                st.warning(f"{chosen_eye} — .h5 not found. Place it at:\n`{path}`")

    with col_posture:
        st.markdown("**Posture Model**")
        for m in POSTURE_MODELS:
            is_rule = (m == POSTURE_MODEL_A2)
            badge   = " *(rule-based)*" if is_rule else " *(ML model)*"
            st.caption(m + badge)
        chosen_posture = st.radio(
            "Posture model",
            options=POSTURE_MODELS,
            index=POSTURE_MODELS.index(st.session_state["active_posture_model"])
                  if st.session_state["active_posture_model"] in POSTURE_MODELS else 2,
            key="cmp_posture_radio",
            label_visibility="collapsed",
        )
        if chosen_posture == POSTURE_MODEL_A2:
            st.info("A2 — angle threshold rule. No .h5 model needed.")
        else:
            path = POSTURE_MODEL_PATHS.get(chosen_posture)
            if path and os.path.exists(path):
                st.success(f"{chosen_posture} — model file found ✓")
            else:
                st.warning(f"{chosen_posture} — .h5 not found. Place it at:\n`{path}`")

    if st.button("Apply Selection", type="primary", use_container_width=True,
                 key="cmp_apply_btn"):
        st.session_state["active_eye_model"]     = chosen_eye
        st.session_state["active_posture_model"] = chosen_posture

        # Tell the in-process inference sidecar immediately —
        # this also unloads the previously active models.
        try:
            from inference_server import inference_state
            inference_state.set_active_models(chosen_eye, chosen_posture)
        except Exception:
            pass

        st.success(
            f"Active models updated — "
            f"**{chosen_eye}** (eye) + **{chosen_posture}** (posture). "
            f"Unselected models have been unloaded. "
            f"Switch to the Live Monitoring tab to see results."
        )


# ──────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────

def render_comparison_tab():
    st.header("Comparative Model Analysis")

    # Read the currently active selection (defaults applied if first visit)
    if "active_eye_model" not in st.session_state:
        st.session_state["active_eye_model"] = DEFAULT_EYE_MODEL
    if "active_posture_model" not in st.session_state:
        st.session_state["active_posture_model"] = DEFAULT_POSTURE_MODEL

    active_eye     = st.session_state["active_eye_model"]
    active_posture = st.session_state["active_posture_model"]

    # Load results ONLY for selected models
    selected_results = load_selected_results(active_eye, active_posture)
    eye_result     = selected_results[active_eye]
    posture_result = selected_results[active_posture]

    ec = _eye_color(active_eye)
    pc = _posture_color(active_posture)

    st.info(
        f"Charts and metrics below are for the **selected** models only: "
        f"**{active_eye}** (eye) and **{active_posture}** (posture). "
        f"Change selection below and click *Apply* to update.",
        icon="ℹ️"
    )

    # ── Accuracy ─────────────────────────────────────────
    st.subheader("Accuracy and F1-Score")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.plotly_chart(
            _accuracy_chart(active_eye, eye_result, ec, "Eye Strain Model"),
            use_container_width=True,
        )
    with ac2:
        st.plotly_chart(
            _accuracy_chart(active_posture, posture_result, pc, "Posture Model"),
            use_container_width=True,
        )

    # ── Latency ──────────────────────────────────────────
    st.subheader("Inference Latency")
    lc1, lc2 = st.columns(2)
    with lc1:
        st.plotly_chart(
            _latency_chart(active_eye, eye_result, ec, "Eye Model Latency"),
            use_container_width=True,
        )
    with lc2:
        st.plotly_chart(
            _latency_chart(active_posture, posture_result, pc, "Posture Model Latency"),
            use_container_width=True,
        )

    # ── Radar ────────────────────────────────────────────
    st.subheader("Multi-Metric Radar")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.plotly_chart(
            _radar_chart(active_eye, eye_result, ec, "Eye Model Profile"),
            use_container_width=True,
        )
    with rc2:
        st.plotly_chart(
            _radar_chart(active_posture, posture_result, pc, "Posture Model Profile"),
            use_container_width=True,
        )

    st.divider()

    # ── Summary table (selected models only) ─────────────
    st.subheader("Performance Summary (Selected Models)")
    _summary_table(active_eye, eye_result, active_posture, posture_result)

    st.divider()

    # ── Model selector ────────────────────────────────────
    _model_selector()
