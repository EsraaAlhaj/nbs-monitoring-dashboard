"""
Shared visual styling for all pages: CSS injection, banners, and a common
Plotly layout template so every chart in the app looks consistent.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config.settings import COLORS, DEMO_DATA_BANNER, ESTIMATE_DISCLAIMER

APP_TITLE = "Urban NBS Microclimate Monitoring Platform"
APP_SUBTITLE = "Proposed pilot: University of Jordan | Scalable urban heat monitoring and decision-support platform"


def inject_base_css() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {COLORS['background']};
        }}
        h1, h2, h3, h4 {{
            color: {COLORS['text_primary']};
            font-weight: 600;
        }}
        [data-testid="stMetric"] {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['grid']};
            border-radius: 10px;
            padding: 14px 16px 10px 16px;
        }}
        [data-testid="stMetricLabel"] {{
            color: {COLORS['text_muted']};
        }}
        .kpi-card {{
            background-color: {COLORS['surface']};
            border: 1px solid {COLORS['grid']};
            border-radius: 10px;
            padding: 14px 16px;
            min-height: 128px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
        }}
        .kpi-card-label {{
            color: {COLORS['text_muted']};
            font-size: 0.82rem;
            line-height: 1.35;
            min-height: 2.3em;
        }}
        .kpi-card-value {{
            color: {COLORS['text_primary']};
            font-size: 1.7rem;
            font-weight: 600;
            line-height: 1.2;
            margin-top: 2px;
        }}
        .kpi-card-footer {{
            color: {COLORS['text_muted']};
            font-size: 0.8rem;
            line-height: 1.3;
            margin-top: auto;
            padding-top: 8px;
        }}
        section[data-testid="stSidebar"] {{
            background-color: {COLORS['surface']};
            border-right: 1px solid {COLORS['grid']};
        }}
        .app-header {{
            padding: 6px 0 2px 0;
            border-bottom: 2px solid {COLORS['grid']};
            margin-bottom: 14px;
        }}
        .app-header h1 {{
            margin-bottom: 0;
            font-size: 1.65rem;
        }}
        .app-header p {{
            color: {COLORS['text_muted']};
            margin-top: 2px;
            font-size: 0.95rem;
        }}
        .demo-banner {{
            background-color: #FFF6E6;
            border: 1px solid {COLORS['accent']};
            color: #6B4A0A;
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 16px;
            font-size: 0.92rem;
        }}
        .estimate-note {{
            background-color: #EEF4FB;
            border-left: 4px solid {COLORS['reference_site']};
            color: {COLORS['text_primary']};
            border-radius: 4px;
            padding: 8px 14px;
            margin: 10px 0;
            font-size: 0.88rem;
        }}
        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(page_name: str) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE} · {page_name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner() -> None:
    st.markdown(
        f'<div class="demo-banner">&#9888; {DEMO_DATA_BANNER}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, footer: str = "") -> None:
    """
    Fixed-height KPI card (label top, value middle, footer pinned to a
    reserved bottom area). Call once per st.columns() cell so a row of
    cards shares equal width (from the column) and equal height/alignment
    (from the shared .kpi-card CSS), whether or not each card has footer
    text.
    """
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-card-label">{label}</div>
            <div class="kpi-card-value">{value}</div>
            <div class="kpi-card-footer">{footer if footer else '&nbsp;'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_estimate_note(extra: str = "") -> None:
    text = ESTIMATE_DISCLAIMER + (f" {extra}" if extra else "")
    st.markdown(f'<div class="estimate-note">{text}</div>', unsafe_allow_html=True)


def apply_common_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    resolved_title = title or fig.layout.title.text
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(color=COLORS["text_primary"], family="sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=50, b=10),
        colorway=[COLORS["green_site"], COLORS["reference_site"], COLORS["accent"], COLORS["critical"]],
    )
    if resolved_title:
        fig.update_layout(title=resolved_title)
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig
