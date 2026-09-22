"""Data Quality page: completeness, time gaps, implausible readings, stuck sensors."""

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
    STATUS_COLORS,
)
from src.cached_data import load_readings
from src.quality_checks import (
    completeness_by_station,
    detect_implausible_readings,
    detect_stuck_sensors,
    detect_time_gaps,
    station_status_summary,
)
from src.styling import APP_TITLE, apply_common_layout, inject_base_css, render_demo_banner, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — Data Quality", layout="wide")
inject_base_css()
render_page_header("Data Quality")
render_demo_banner()

st.markdown(
    "Diagnostics on the monitoring network itself: how complete each station's record is, "
    "where data is missing, and which readings look implausible or suspiciously static. "
    "A handful of issues have been deliberately seeded into this synthetic dataset so these "
    "checks have something real to detect."
)

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)

STATION_NAME_BY_ID = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
ALL_STATION_IDS = list(STATION_NAME_BY_ID.keys())

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
filter_col1, filter_col2 = st.columns([1.8, 1.4])
with filter_col1:
    selected_stations = st.multiselect(
        "Stations",
        options=ALL_STATION_IDS,
        default=ALL_STATION_IDS,
        format_func=lambda sid: STATION_NAME_BY_ID.get(sid, sid),
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

completeness = completeness_by_station(readings, start_ts, end_ts)
gaps = detect_time_gaps(readings)
implausible = detect_implausible_readings(readings)
stuck = detect_stuck_sensors(readings)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
st.subheader("Overview")
o1, o2, o3, o4, o5 = st.columns(5)
o1.metric("Records in selection", f"{len(readings):,}")
o2.metric(
    "Average completeness",
    f"{completeness['completeness_pct'].mean():.1f}%" if not completeness.empty else "—",
)
o3.metric("Time gaps detected", f"{len(gaps):,}")
o4.metric("Implausible readings", f"{len(implausible):,}")
o5.metric("Stuck-sensor runs", f"{len(stuck):,}")

# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------
st.subheader("Completeness by Station")
completeness_display = completeness.copy()
completeness_display["station_name"] = completeness_display["station_id"].map(STATION_NAME_BY_ID)

fig_completeness = go.Figure()
fig_completeness.add_trace(
    go.Bar(
        x=completeness_display["station_name"],
        y=completeness_display["completeness_pct"],
        marker_color=[
            COLORS["green_site"] if v >= 95 else (COLORS["accent"] if v >= 80 else COLORS["critical"])
            for v in completeness_display["completeness_pct"]
        ],
        text=completeness_display["completeness_pct"].map(lambda v: f"{v:.1f}%"),
        textposition="outside",
    )
)
fig_completeness.add_hline(y=100, line_dash="dot", line_color=COLORS["text_muted"])
fig_completeness = apply_common_layout(fig_completeness)
fig_completeness.update_yaxes(title="Completeness (%)", range=[0, 105])
fig_completeness.update_xaxes(title="")
st.plotly_chart(fig_completeness, use_container_width=True)

completeness_table = completeness_display.rename(
    columns={
        "station_id": "Station ID",
        "station_name": "Station",
        "expected_records": "Expected Records",
        "actual_records": "Actual Records",
        "complete_records": "Complete Records",
        "completeness_pct": "Completeness (%)",
    }
)[["Station ID", "Station", "Expected Records", "Actual Records", "Complete Records", "Completeness (%)"]]
st.dataframe(completeness_table, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Station status overview
# ---------------------------------------------------------------------------
st.subheader("Station Status Overview")
status_summary = station_status_summary(readings)
status_summary["station_name"] = status_summary["station_id"].map(STATION_NAME_BY_ID)

fig_status = go.Figure()
for status in sorted(status_summary["station_status"].unique()):
    subset = status_summary[status_summary["station_status"] == status]
    fig_status.add_trace(
        go.Bar(
            x=subset["station_name"], y=subset["share_pct"],
            name=str(status).capitalize(),
            marker_color=STATUS_COLORS.get(str(status).lower(), "#888888"),
        )
    )
fig_status.update_layout(barmode="stack")
fig_status = apply_common_layout(fig_status)
fig_status.update_yaxes(title="Share of records (%)", range=[0, 100])
fig_status.update_xaxes(title="")
st.plotly_chart(fig_status, use_container_width=True)

# ---------------------------------------------------------------------------
# Time gaps
# ---------------------------------------------------------------------------
st.subheader("Missing Data — Time Gaps")
if gaps.empty:
    st.success("No time gaps larger than the expected interval were found in the current selection.")
else:
    gaps_display = gaps.copy()
    gaps_display["station_name"] = gaps_display["station_id"].map(STATION_NAME_BY_ID)
    gaps_display = gaps_display.rename(
        columns={
            "station_name": "Station", "gap_start": "Gap Start",
            "gap_end": "Gap End", "gap_minutes": "Gap (minutes)",
        }
    )[["Station", "Gap Start", "Gap End", "Gap (minutes)"]]
    st.dataframe(gaps_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Implausible readings
# ---------------------------------------------------------------------------
st.subheader("Implausible Readings")
if implausible.empty:
    st.success("No implausible readings were found in the current selection.")
else:
    implausible_display = implausible.copy()
    implausible_display["station_name"] = implausible_display["station_id"].map(STATION_NAME_BY_ID)
    implausible_display = implausible_display.rename(
        columns={
            "station_name": "Station", "timestamp": "Timestamp", "variable": "Variable",
            "value": "Value", "valid_min": "Valid Min", "valid_max": "Valid Max",
        }
    )[["Station", "Timestamp", "Variable", "Value", "Valid Min", "Valid Max"]]
    st.dataframe(implausible_display, use_container_width=True, hide_index=True)
    st.caption("Values outside the physically plausible range configured for each variable.")

# ---------------------------------------------------------------------------
# Stuck sensors
# ---------------------------------------------------------------------------
st.subheader("Long-Duration Stuck Readings")
if stuck.empty:
    st.success("No stuck-sensor runs were found in the current selection.")
else:
    stuck_display = stuck.copy()
    stuck_display["station_name"] = stuck_display["station_id"].map(STATION_NAME_BY_ID)
    stuck_display["duration_hours"] = (stuck_display["run_length"] * SIMULATION_INTERVAL_MINUTES / 60.0).round(1)
    stuck_display = stuck_display.rename(
        columns={
            "station_name": "Station", "variable": "Variable", "value": "Stuck Value",
            "run_start": "Run Start", "run_end": "Run End", "run_length": "Run Length (records)",
            "duration_hours": "Duration (hours)",
        }
    )[["Station", "Variable", "Stuck Value", "Run Start", "Run End", "Run Length (records)", "Duration (hours)"]]
    st.dataframe(stuck_display, use_container_width=True, hide_index=True)
    st.caption(
        f"Runs of {SIMULATION_INTERVAL_MINUTES}-minute readings where a variable stayed exactly "
        "constant for an unusually long time — a classic symptom of a frozen sensor."
    )
