import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

import sys
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from database.db_manager import get_health_metrics


# Chart theme settings shared across all plots
CHART_THEME = dict(
    plot_bgcolor="#1e2130",
    paper_bgcolor="#1e2130",
    font=dict(color="#e0e0e0", size=11),
    margin=dict(t=50, b=40, l=50, r=20),
)


def load_metrics_dataframe(user_id: int, hours: int) -> pd.DataFrame:
    rows = get_health_metrics(user_id, hours)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("datetime")

    # Boolean flags for easier aggregation
    df["eye_strained"] = (df["eye_status"] == "Strained").astype(int)
    df["is_slouching"] = (df["posture_status"] == "Slouching").astype(int)

    return df


def build_health_score_chart(df: pd.DataFrame, hours: int) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["health_score"],
        mode="lines",
        name="Health Score",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.15)",
    ))

    # Draw threshold lines
    fig.add_hline(y=75, line_dash="dot", line_color="#2ecc71", annotation_text="Good (75)", annotation_position="right")
    fig.add_hline(y=50, line_dash="dot", line_color="#f39c12", annotation_text="Warning (50)", annotation_position="right")

    fig.update_layout(
        title=f"Health Score - Last {hours} Hour{'s' if hours > 1 else ''}",
        xaxis_title="Time",
        yaxis=dict(title="Score (0-100)", range=[0, 105]),
        height=320,
        **CHART_THEME,
    )
    return fig


def build_ear_chart(df: pd.DataFrame, hours: int) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["ear_value"],
        mode="lines",
        name="EAR",
        line=dict(color="#9b59b6", width=1.5),
    ))

    # Shade the strained zone below the EAR threshold
    fig.add_hrect(
        y0=0, y1=0.21,
        fillcolor="rgba(231, 76, 60, 0.1)",
        layer="below",
        line_width=0,
        annotation_text="Strained zone",
        annotation_position="top right",
    )
    fig.add_hline(y=0.21, line_dash="dash", line_color="#e74c3c",
                  annotation_text="EAR threshold")

    fig.update_layout(
        title=f"Eye Aspect Ratio (EAR) - Last {hours} Hour{'s' if hours > 1 else ''}",
        xaxis_title="Time",
        yaxis=dict(title="EAR Value", range=[0, 0.5]),
        height=300,
        **CHART_THEME,
    )
    return fig


def build_posture_chart(df: pd.DataFrame, hours: int) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["datetime"],
        y=df["posture_angle"],
        mode="lines",
        name="Neck Tilt Angle",
        line=dict(color="#e67e22", width=1.5),
    ))

    # Shade the slouching zone above the threshold
    fig.add_hrect(
        y0=20, y1=df["posture_angle"].max() + 5,
        fillcolor="rgba(231, 76, 60, 0.08)",
        layer="below",
        line_width=0,
        annotation_text="Slouching zone",
        annotation_position="top right",
    )
    fig.add_hline(y=20, line_dash="dash", line_color="#e74c3c",
                  annotation_text="Slouch threshold (20 deg)")

    fig.update_layout(
        title=f"Neck Tilt Angle - Last {hours} Hour{'s' if hours > 1 else ''}",
        xaxis_title="Time",
        yaxis=dict(title="Angle (degrees)"),
        height=300,
        **CHART_THEME,
    )
    return fig


def build_status_heatmap(df: pd.DataFrame) -> go.Figure:
    
    df_resampled = (
        df.set_index("datetime")[["eye_strained", "is_slouching", "health_score"]]
        .resample("5min")
        .mean()
        .ffill()
        .reset_index()
    )

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Eye Strain Rate (5-min avg)", "Slouching Rate (5-min avg)"),
    )

    fig.add_trace(
        go.Scatter(
            x=df_resampled["datetime"],
            y=df_resampled["eye_strained"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Eye strain %",
            line=dict(color="#9b59b6"),
            fillcolor="rgba(155, 89, 182, 0.3)",
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_resampled["datetime"],
            y=df_resampled["is_slouching"] * 100,
            mode="lines",
            fill="tozeroy",
            name="Slouching %",
            line=dict(color="#e74c3c"),
            fillcolor="rgba(231, 76, 60, 0.3)",
        ),
        row=2, col=1,
    )

    fig.update_yaxes(title_text="Strain %", range=[0, 105], row=1, col=1)
    fig.update_yaxes(title_text="Slouching %", range=[0, 105], row=2, col=1)
    fig.update_layout(
        height=380,
        showlegend=False,
        **CHART_THEME,
    )
    return fig


def render_summary_stats(df: pd.DataFrame):
    if df.empty:
        return

    avg_health = df["health_score"].mean()
    strain_pct = df["eye_strained"].mean() * 100
    slouch_pct = df["is_slouching"].mean() * 100
    avg_ear = df["ear_value"].mean()

    col1, col2, col3, col4 = st.columns(4)

    def kpi(col, label, value, unit="", color="#3498db"):
        col.markdown(
            f"""
            <div style="background:#1e2130; border-radius:8px; padding:14px; text-align:center;">
                <p style="font-size:11px; color:#aaa; margin:0;">{label}</p>
                <p style="font-size:22px; font-weight:bold; color:{color}; margin:4px 0;">
                    {value}{unit}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    kpi(col1, "Avg Health Score", f"{avg_health:.0f}", "/100",
        "#2ecc71" if avg_health >= 75 else "#f39c12")
    kpi(col2, "Eye Strain Time", f"{strain_pct:.0f}", "%",
        "#9b59b6")
    kpi(col3, "Slouching Time", f"{slouch_pct:.0f}", "%",
        "#e74c3c" if slouch_pct > 30 else "#f39c12")
    kpi(col4, "Avg EAR", f"{avg_ear:.3f}", "",
        "#2ecc71" if avg_ear > 0.21 else "#e74c3c")


def render_analytics_tab(user_id: int):
    st.header("Analytics Dashboard")

    # Time window selector
    time_col, refresh_col = st.columns([3, 1])
    with time_col:
        window_hours = st.select_slider(
            "Time window",
            options=[1, 6, 24],
            value=1,
            key="analytics_window",
            format_func=lambda x: f"Last {x} hour{'s' if x > 1 else ''}",
        )
    with refresh_col:
        if st.button("Refresh Data", use_container_width=True, key="analytics_refresh"):
            st.rerun()

    df = load_metrics_dataframe(user_id, window_hours)

    if df.empty:
        st.info(
            "No data recorded yet for this time window. "
            "Start a monitoring session to begin collecting metrics."
        )
        return

    st.caption(f"Showing {len(df)} data points from the last {window_hours} hour(s).")

    # Summary KPIs
    render_summary_stats(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Health score trend
    st.plotly_chart(
        build_health_score_chart(df, window_hours), use_container_width=True
    )

    # Eye and Posture side by side
    eye_col, posture_col = st.columns(2)
    with eye_col:
        st.plotly_chart(build_ear_chart(df, window_hours), use_container_width=True)
    with posture_col:
        st.plotly_chart(
            build_posture_chart(df, window_hours), use_container_width=True
        )

    # Rate heatmap (only meaningful if enough data points)
    if len(df) >= 10:
        st.subheader("Strain and Slouching Rate Over Time")
        st.plotly_chart(build_status_heatmap(df), use_container_width=True)

    # Raw data expander
    with st.expander("View Raw Data"):
        display_cols = [
            "datetime", "eye_status", "ear_value",
            "posture_status", "posture_angle", "health_score",
        ]
        st.dataframe(
            df[display_cols].tail(200),
            use_container_width=True,
            hide_index=True,
        )
