"""Statistical Analysis page: summary statistics, scatterplot, boxplot, histogram."""

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
    STATIONS,
    VARIABLES,
)
from src.cached_data import load_readings
from src.statistics_utils import attach_station_metadata, summary_statistics
from src.styling import APP_TITLE, apply_common_layout, inject_base_css, render_demo_banner, render_estimate_note, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — Statistical Analysis", layout="wide")
inject_base_css()
render_page_header("Statistical Analysis")
render_demo_banner()

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)

ALL_STATION_IDS = [meta["station_id"] for meta in STATIONS.values()]
STATION_NAME_BY_ID = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
SITE_COLOR_MAP = {"green": COLORS["green_site"], "reference": COLORS["reference_site"]}

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
row1_col1, row1_col2, row1_col3 = st.columns([1.8, 1.1, 1.1])
with row1_col1:
    selected_stations = st.multiselect(
        "Stations",
        options=ALL_STATION_IDS,
        default=ALL_STATION_IDS,
        format_func=lambda sid: STATION_NAME_BY_ID.get(sid, sid),
    )
with row1_col2:
    variable = st.selectbox(
        "Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=0,
    )
with row1_col3:
    x_variable = st.selectbox(
        "Scatter X-axis variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=list(VARIABLES.keys()).index("solar_radiation_wm2"),
    )

row2_col1, row2_col2 = st.columns([1.6, 1.6])
with row2_col1:
    date_range = st.date_input(
        "Period",
        value=(sim_start.date(), sim_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
    )
with row2_col2:
    hour_range = st.slider("Hour of day (local)", min_value=0, max_value=23, value=(0, 23))

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = sim_start.date(), sim_end.date()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

if not selected_stations:
    st.warning("Select at least one station.")
    st.stop()

readings = load_readings(
    station_ids=tuple(selected_stations),
    start=start_ts.isoformat(),
    end=end_ts.isoformat(),
)

if readings.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

readings = readings[readings["timestamp"].dt.hour.between(hour_range[0], hour_range[1])]
readings = attach_station_metadata(readings)

if readings.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

label = VARIABLES[variable]["label"]
unit = VARIABLES[variable]["unit"]
x_label = VARIABLES[x_variable]["label"]
x_unit = VARIABLES[x_variable]["unit"]

st.caption(f"{len(readings):,} records match the current filters.")

if variable in ("mean_radiant_temp_c", "utci_c") or x_variable in ("mean_radiant_temp_c", "utci_c"):
    render_estimate_note()

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
st.subheader("Summary Statistics")
stats = summary_statistics(readings, variable)
if stats.empty:
    st.info("No valid values for this variable in the current selection.")
else:
    stats["station_name"] = stats["station_id"].map(STATION_NAME_BY_ID)
    stats = stats[["station_id", "station_name", "count", "mean", "std", "min", "p25", "median", "p75", "max"]]
    stats.columns = ["Station ID", "Station", "Count", "Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption(f"Units: {unit}")

# ---------------------------------------------------------------------------
# Scatterplot
# ---------------------------------------------------------------------------
st.subheader(f"Scatterplot — {x_label} vs. {label}")
scatter_df = readings.dropna(subset=[x_variable, variable])
fig_scatter = go.Figure()
for site_type, color in SITE_COLOR_MAP.items():
    subset = scatter_df[scatter_df["site_type"] == site_type]
    if subset.empty:
        continue
    fig_scatter.add_trace(
        go.Scattergl(
            x=subset[x_variable], y=subset[variable],
            mode="markers",
            name=site_type.capitalize(),
            marker=dict(color=color, size=5, opacity=0.55),
        )
    )
fig_scatter = apply_common_layout(fig_scatter)
fig_scatter.update_xaxes(title=f"{x_label} ({x_unit})")
fig_scatter.update_yaxes(title=f"{label} ({unit})")
st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Boxplot
# ---------------------------------------------------------------------------
st.subheader(f"Distribution by Station — {label}")
fig_box = go.Figure()
for station_id in selected_stations:
    subset = readings[readings["station_id"] == station_id]
    if subset[variable].dropna().empty:
        continue
    site_type = subset["site_type"].iloc[0]
    fig_box.add_trace(
        go.Box(
            y=subset[variable],
            name=STATION_NAME_BY_ID.get(station_id, station_id),
            marker_color=SITE_COLOR_MAP.get(site_type, COLORS["text_muted"]),
        )
    )
fig_box = apply_common_layout(fig_box)
fig_box.update_yaxes(title=f"{label} ({unit})")
fig_box.update_layout(showlegend=False)
st.plotly_chart(fig_box, use_container_width=True)

# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------
st.subheader(f"Histogram — {label}")
fig_hist = go.Figure()
for site_type, color in SITE_COLOR_MAP.items():
    subset = readings[readings["site_type"] == site_type]
    values = subset[variable].dropna()
    if values.empty:
        continue
    fig_hist.add_trace(
        go.Histogram(
            x=values, name=site_type.capitalize(),
            marker_color=color, opacity=0.6, nbinsx=40,
        )
    )
fig_hist.update_layout(barmode="overlay")
fig_hist = apply_common_layout(fig_hist)
fig_hist.update_xaxes(title=f"{label} ({unit})")
fig_hist.update_yaxes(title="Count")
st.plotly_chart(fig_hist, use_container_width=True)
