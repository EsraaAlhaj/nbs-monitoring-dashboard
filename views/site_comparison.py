"""Site Comparison page: green vs. reference site, for a selected pair and period."""

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
    VARIABLES,
)
from src.cached_data import load_readings
from src.statistics_utils import build_pair_comparison, count_cooler_hours, hottest_periods, hourly_profile
from src.styling import APP_TITLE, apply_common_layout, inject_base_css, render_demo_banner, render_estimate_note, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — Site Comparison", layout="wide")
inject_base_css()
render_page_header("Site Comparison")
render_demo_banner()

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
filter_col1, filter_col2, filter_col3 = st.columns([1.2, 1.6, 1.2])
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
with filter_col3:
    variable = st.selectbox(
        "Climate variable (time series)",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=0,
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

st.caption(f"Comparing **{green_meta['name']}** (green) against **{ref_meta['name']}** (reference).")

# ---------------------------------------------------------------------------
# Period summary for the selected variable
# ---------------------------------------------------------------------------
comparison = build_pair_comparison(readings, pair_id, variable=variable)
unit = VARIABLES[variable]["unit"]
label = VARIABLES[variable]["label"]

avg_green = comparison["green"].mean()
avg_ref = comparison["reference"].mean()

# Cooling Hours is always based on air temperature (regardless of the variable
# selected above, per spec) — computed once here and reused by the "Cooling
# Performance Summary" section further down the page.
temp_comparison = build_pair_comparison(readings, pair_id, variable="air_temperature_c")
cooler_stats = count_cooler_hours(temp_comparison)

st.subheader("Period Summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{label} — Green (avg)", f"{avg_green:.1f} {unit}" if pd.notna(avg_green) else "—")
m2.metric(f"{label} — Reference (avg)", f"{avg_ref:.1f} {unit}" if pd.notna(avg_ref) else "—")
if pd.notna(avg_green) and pd.notna(avg_ref):
    m3.metric("Average difference (Green − Reference)", f"{(avg_green - avg_ref):+.2f} {unit}")
else:
    m3.metric("Average difference (Green − Reference)", "—")
m4.metric("Cooling Hours", f"{cooler_stats['cooler_hours']:.1f} h")

if variable in ("mean_radiant_temp_c", "utci_c"):
    render_estimate_note()

# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------
st.subheader(f"{label} Over Time")
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
fig_ts = apply_common_layout(fig_ts)
fig_ts.update_yaxes(title=f"{label} ({unit})")
fig_ts.update_xaxes(title="Time")
st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------------------------------------------------------------
# Temperature difference over time (always air temperature, per spec)
# ---------------------------------------------------------------------------
st.subheader("Temperature Difference Over Time (Green − Reference)")
fig_delta = go.Figure()
fig_delta.add_trace(
    go.Scatter(
        x=temp_comparison["timestamp"], y=temp_comparison["delta"],
        fill="tozeroy", name="Δ Air Temperature",
        line=dict(color=COLORS["green_site"], width=1.2),
        fillcolor=COLORS["green_site_fill"],
    )
)
fig_delta.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"])
fig_delta = apply_common_layout(fig_delta)
fig_delta.update_yaxes(title="Temperature difference (°C)")
fig_delta.update_xaxes(title="Time")
st.plotly_chart(fig_delta, use_container_width=True)
st.caption("Negative values indicate the green site was cooler than the reference site at that moment.")

# ---------------------------------------------------------------------------
# Average performance by hour of day
# ---------------------------------------------------------------------------
st.subheader("Average Temperature Difference by Hour of Day")
profile = hourly_profile(temp_comparison)
fig_hour = go.Figure()
fig_hour.add_trace(
    go.Bar(x=profile["hour"], y=profile["delta"], name="Mean Δ Temperature", marker_color=COLORS["green_site"])
)
fig_hour.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"])
fig_hour = apply_common_layout(fig_hour)
fig_hour.update_xaxes(title="Hour of day (local)", dtick=1)
fig_hour.update_yaxes(title="Mean temperature difference (°C)")
st.plotly_chart(fig_hour, use_container_width=True)
st.caption(
    "This profile is expected to show the cooling effect strengthening through the sunlit hours "
    "(shading and evapotranspiration) rather than remaining a fixed offset."
)

# ---------------------------------------------------------------------------
# Cooling performance summary
# ---------------------------------------------------------------------------
st.subheader("Cooling Performance Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Records where green was cooler", f"{cooler_stats['cooler_records']:,}", f"{cooler_stats['cooler_share_pct']:.1f}% of records")
c2.metric("Equivalent hours cooler", f"{cooler_stats['cooler_hours']:.1f} h")
c3.metric("Mean temperature difference", f"{cooler_stats['mean_delta_c']:+.2f} °C" if pd.notna(cooler_stats['mean_delta_c']) else "—")
c4.metric("Maximum cooling observed", f"{cooler_stats['max_cooling_c']:.2f} °C" if pd.notna(cooler_stats['max_cooling_c']) else "—")

# ---------------------------------------------------------------------------
# Hottest periods analysis
# ---------------------------------------------------------------------------
st.subheader("Hottest Periods — Cooling Effect at Peak Heat")
hottest = hottest_periods(temp_comparison, top_n=10)
if hottest.empty:
    st.info("Not enough data in the selected period to identify hottest periods.")
else:
    hottest_display = hottest.rename(
        columns={
            "timestamp": "Timestamp",
            "reference": "Reference (°C)",
            "green": "Green (°C)",
            "cooling_effect_c": "Cooling Effect (°C)",
        }
    )
    st.dataframe(hottest_display, use_container_width=True, hide_index=True)
    st.caption(
        "The ten timestamps with the highest reference-site air temperature in the selected "
        "period, and how much cooler the green site was at that same moment."
    )
