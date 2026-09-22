"""UTCI Analysis page: heat-stress category classification, green vs. reference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    COLORS,
    SIMULATION_INTERVAL_MINUTES,
    SIMULATION_NUM_DAYS,
    SIMULATION_START_DATE,
    STATION_PAIRS,
    STATIONS,
    UTCI_CATEGORIES,
    UTCI_CATEGORY_COLORS,
)
from src.cached_data import load_readings
from src.comfort import classify_utci_series, classify_utci_value
from src.statistics_utils import build_pair_comparison
from src.styling import APP_TITLE, apply_common_layout, inject_base_css, render_demo_banner, render_estimate_note, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — UTCI Analysis", layout="wide")
inject_base_css()
render_page_header("UTCI Analysis")
render_demo_banner()

st.markdown(
    "The Universal Thermal Climate Index (UTCI) combines air temperature, humidity, wind "
    "speed and radiant heat into a single value that reflects perceived outdoor thermal "
    "stress, classified into ten standard categories from extreme cold to extreme heat stress."
)
render_estimate_note(
    "UTCI is calculated from an estimated Mean Radiant Temperature (not a field measurement) "
    "— treat category boundaries here as indicative rather than precise."
)

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)
CATEGORY_ORDER = [cat[2] for cat in UTCI_CATEGORIES]
INTERVAL_HOURS = SIMULATION_INTERVAL_MINUTES / 60.0

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
filter_col1, filter_col2 = st.columns([1.2, 1.8])
with filter_col1:
    pair_id = st.selectbox(
        "Site pair",
        options=list(STATION_PAIRS.keys()),
        format_func=lambda k: STATION_PAIRS[k]["label"],
    )
with filter_col2:
    date_range = st.date_input(
        "Period",
        value=(sim_start.date(), sim_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = sim_start.date(), sim_end.date()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

pair = STATION_PAIRS[pair_id]
green_meta = STATIONS[pair["green"]]
ref_meta = STATIONS[pair["reference"]]

readings = load_readings(
    station_ids=(green_meta["station_id"], ref_meta["station_id"]),
    start=start_ts.isoformat(),
    end=end_ts.isoformat(),
)

if readings.empty:
    st.warning("No data available for the selected period.")
    st.stop()

comparison = build_pair_comparison(readings, pair_id, variable="utci_c")
comparison = comparison.dropna(subset=["green", "reference"], how="all")

if comparison.empty:
    st.warning("No UTCI data available for the selected period.")
    st.stop()

st.caption(f"Comparing **{green_meta['name']}** (green) against **{ref_meta['name']}** (reference).")

# ---------------------------------------------------------------------------
# Period summary
# ---------------------------------------------------------------------------
st.subheader("Period Summary")

avg_green = comparison["green"].mean()
avg_ref = comparison["reference"].mean()
latest_row = comparison.sort_values("timestamp").iloc[-1]
latest_green_cat = classify_utci_value(latest_row["green"]) if pd.notna(latest_row["green"]) else None
latest_ref_cat = classify_utci_value(latest_row["reference"]) if pd.notna(latest_row["reference"]) else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Average UTCI — Green", f"{avg_green:.1f} °C" if pd.notna(avg_green) else "—")
m2.metric("Average UTCI — Reference", f"{avg_ref:.1f} °C" if pd.notna(avg_ref) else "—")
m3.metric(
    "Latest UTCI — Green",
    f"{latest_row['green']:.1f} °C" if pd.notna(latest_row["green"]) else "—",
    latest_green_cat,
    delta_color="off",
)
m4.metric(
    "Latest UTCI — Reference",
    f"{latest_row['reference']:.1f} °C" if pd.notna(latest_row["reference"]) else "—",
    latest_ref_cat,
    delta_color="off",
)

# ---------------------------------------------------------------------------
# UTCI over time
# ---------------------------------------------------------------------------
st.subheader("UTCI Over Time")
fig_ts = go.Figure()
fig_ts.add_trace(
    go.Scatter(
        x=comparison["timestamp"], y=comparison["green"],
        name=green_meta["name"], line=dict(color=COLORS["green_site"], width=1.6),
    )
)
fig_ts.add_trace(
    go.Scatter(
        x=comparison["timestamp"], y=comparison["reference"],
        name=ref_meta["name"], line=dict(color=COLORS["reference_site"], width=1.6),
    )
)
fig_ts.add_hline(y=26, line_dash="dot", line_color=COLORS["accent"], annotation_text="Moderate heat stress ≥ 26°C")
fig_ts = apply_common_layout(fig_ts)
fig_ts.update_yaxes(title="UTCI (°C)")
fig_ts.update_xaxes(title="Time")
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Category classification
# ---------------------------------------------------------------------------


def _category_hours(series: pd.Series) -> pd.Series:
    classified = classify_utci_series(series.dropna())
    counts = classified.value_counts().reindex(CATEGORY_ORDER, fill_value=0)
    return (counts * INTERVAL_HOURS).round(1)


green_hours = _category_hours(comparison["green"])
ref_hours = _category_hours(comparison["reference"])
present_categories = [c for c in CATEGORY_ORDER if green_hours.get(c, 0) > 0 or ref_hours.get(c, 0) > 0]

st.subheader("Heat-Stress Category Distribution")
fig_dist = go.Figure()
for cat in present_categories:
    fig_dist.add_trace(
        go.Bar(
            y=["Green", "Reference"],
            x=[green_hours.get(cat, 0), ref_hours.get(cat, 0)],
            name=cat,
            orientation="h",
            marker_color=UTCI_CATEGORY_COLORS.get(cat, "#999999"),
        )
    )
fig_dist.update_layout(barmode="stack")
fig_dist = apply_common_layout(fig_dist)
fig_dist.update_xaxes(title="Hours")
fig_dist.update_yaxes(title="")
st.plotly_chart(fig_dist, use_container_width=True)
st.caption("Share of the selected period each site spent in each UTCI thermal-stress category.")

st.subheader("Hours per Category — Green vs. Reference")
fig_cat = go.Figure()
fig_cat.add_trace(
    go.Bar(
        x=present_categories, y=[green_hours.get(c, 0) for c in present_categories],
        name=green_meta["name"], marker_color=COLORS["green_site"],
    )
)
fig_cat.add_trace(
    go.Bar(
        x=present_categories, y=[ref_hours.get(c, 0) for c in present_categories],
        name=ref_meta["name"], marker_color=COLORS["reference_site"],
    )
)
fig_cat.update_layout(barmode="group")
fig_cat = apply_common_layout(fig_cat)
fig_cat.update_xaxes(title="UTCI thermal-stress category")
fig_cat.update_yaxes(title="Hours")
st.plotly_chart(fig_cat, use_container_width=True)

total_green_hours = float(green_hours.sum())
total_ref_hours = float(ref_hours.sum())
category_table = pd.DataFrame(
    {
        "Category": CATEGORY_ORDER,
        "Green — Hours": [green_hours.get(c, 0) for c in CATEGORY_ORDER],
        "Green — % of Period": [
            round(100 * green_hours.get(c, 0) / total_green_hours, 1) if total_green_hours else 0.0
            for c in CATEGORY_ORDER
        ],
        "Reference — Hours": [ref_hours.get(c, 0) for c in CATEGORY_ORDER],
        "Reference — % of Period": [
            round(100 * ref_hours.get(c, 0) / total_ref_hours, 1) if total_ref_hours else 0.0
            for c in CATEGORY_ORDER
        ],
    }
)
st.dataframe(category_table, use_container_width=True, hide_index=True)
st.caption(
    "Hours per thermal-stress category over the selected period, at the "
    f"{SIMULATION_INTERVAL_MINUTES}-minute recording resolution. Categories with zero hours for "
    "both sites are included for reference to the full standard UTCI scale."
)
