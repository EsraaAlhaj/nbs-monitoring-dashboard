"""
Overview page (Streamlit multipage entry, routed via app.py's st.navigation).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    COLORS,
    MONITORING_AREA_PROVISIONAL_NOTE,
    SIMULATION_INTERVAL_MINUTES,
    SIMULATION_NUM_DAYS,
    SIMULATION_START_DATE,
    STATION_PAIRS,
    STATIONS,
    VARIABLES,
)
from src.cached_data import load_readings, load_stations
from src.quality_checks import completeness_by_station
from src.statistics_utils import (
    build_pair_comparison,
    count_cooler_hours,
    daily_profile,
    hottest_periods,
    hourly_profile,
)
from src.styling import (
    APP_TITLE,
    apply_common_layout,
    inject_base_css,
    render_demo_banner,
    render_estimate_note,
    render_kpi_card,
    render_page_header,
)

st.set_page_config(page_title=f"{APP_TITLE} — Overview", layout="wide")
inject_base_css()
render_page_header("Overview")
render_demo_banner()

stations_df = load_stations()

st.markdown(f"**Current pilot: {len(stations_df)} stations | Platform designed for network expansion**")

st.markdown(
    "This prototype illustrates how paired micro-climate monitoring stations "
    "could track the observed temperature difference between Nature-based "
    "Solutions (vegetated green infrastructure) sites and nearby unplanted "
    "reference sites on the University of Jordan campus. Two monitoring areas "
    "are shown below."
)

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
st.divider()

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)

f1, f2, f3, f4 = st.columns(4)
with f1:
    st.selectbox(
        "Monitoring area",
        options=["University of Jordan"],
        index=0,
        help="This pilot currently covers one campus. Additional areas will appear here as the monitoring network expands.",
    )
with f2:
    pair_id = st.selectbox(
        "Site pair",
        options=list(STATION_PAIRS.keys()),
        format_func=lambda k: STATION_PAIRS[k]["label"],
    )
with f3:
    variable = st.selectbox(
        "Climate variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=0,
    )
with f4:
    date_range = st.date_input(
        "Date range",
        value=(sim_start.date(), sim_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
    )

st.caption(MONITORING_AREA_PROVISIONAL_NOTE)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = sim_start.date(), sim_end.date()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
num_days = max((end_date - start_date).days + 1, 1)

pair = STATION_PAIRS[pair_id]
green_meta = STATIONS[pair["green"]]
ref_meta = STATIONS[pair["reference"]]

filtered_readings = load_readings(
    station_ids=(green_meta["station_id"], ref_meta["station_id"]),
    start=start_ts.isoformat(),
    end=end_ts.isoformat(),
)

if filtered_readings.empty:
    st.warning("No data available for the selected filters.")
else:
    # KPIs and the two smaller "cooling" charts below are always based on
    # air temperature (regardless of the Climate variable filter), matching
    # the pattern established on the Site Comparison page. The Climate
    # variable filter only changes the large paired-comparison chart.
    temp_comparison = build_pair_comparison(filtered_readings, pair_id, variable="air_temperature_c")
    cooler_stats = count_cooler_hours(temp_comparison)

    # "Hottest-period" = the hottest 10% of valid reference-site observations
    # in the selected period (not a fixed count), so the KPI scales sensibly
    # with whatever date range is selected.
    n_valid_reference = int(temp_comparison["reference"].notna().sum())
    hottest_top_n = max(1, round(n_valid_reference * 0.10))
    hottest = hottest_periods(temp_comparison, top_n=hottest_top_n)

    # Unified convention across the whole page: cooling difference =
    # Reference − Green. Positive means the green site measured cooler.
    # count_cooler_hours() returns Green − Reference internally (shared with
    # other pages), so the sign is flipped here only, at display time.
    avg_cooling_diff = -cooler_stats["mean_delta_c"] if pd.notna(cooler_stats["mean_delta_c"]) else float("nan")
    peak_period_diff = hottest["cooling_effect_c"].mean() if not hottest.empty else float("nan")

    completeness = completeness_by_station(filtered_readings, start_ts, end_ts)
    data_availability_pct = completeness["completeness_pct"].mean() if not completeness.empty else float("nan")
    cooling_hours_per_day = cooler_stats["cooler_hours"] / num_days if num_days else float("nan")

    # -----------------------------------------------------------------
    # KPI cards
    # -----------------------------------------------------------------
    st.subheader("Key Indicators")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Average cooling difference",
            f"{avg_cooling_diff:+.2f} °C" if pd.notna(avg_cooling_diff) else "—",
            help="Reference minus green air temperature, averaged over the selected period. Positive means the green site measured cooler.",
        )
    with k2:
        render_kpi_card(
            "Cooling hours/day",
            f"{cooling_hours_per_day:.1f} h" if pd.notna(cooling_hours_per_day) else "—",
            help="Average number of hours per day the green site measured cooler than the reference site.",
        )
    with k3:
        render_kpi_card(
            "Hottest-period difference",
            f"{peak_period_diff:+.2f} °C" if pd.notna(peak_period_diff) else "—",
            help="Reference minus green air temperature, averaged over the hottest 10% of reference-site readings in the selected period. Positive means the green site measured cooler during the hottest conditions.",
        )
    with k4:
        render_kpi_card(
            "Data availability",
            f"{data_availability_pct:.1f}%" if pd.notna(data_availability_pct) else "—",
            help="Average share of expected 15-minute readings actually received, across the two selected stations.",
        )
    st.caption(
        "Cooling difference = Reference − Green air temperature; positive values mean the green site "
        "measured cooler. Hottest-period figure uses the hottest 10% of reference-site readings in the "
        "selected period."
    )

    # -----------------------------------------------------------------
    # Paired comparison + supporting charts
    # -----------------------------------------------------------------
    st.subheader("Paired Site Comparison")
    variable_comparison = build_pair_comparison(filtered_readings, pair_id, variable=variable)
    variable_unit = VARIABLES[variable]["unit"]
    variable_label = VARIABLES[variable]["label"]

    chart_left, chart_right = st.columns([2, 1])
    with chart_left:
        fig_main = go.Figure()
        fig_main.add_trace(
            go.Scatter(
                x=variable_comparison["timestamp"], y=variable_comparison["green"],
                name=green_meta["name"], line=dict(color=COLORS["green_site"], width=1.6),
            )
        )
        fig_main.add_trace(
            go.Scatter(
                x=variable_comparison["timestamp"], y=variable_comparison["reference"],
                name=ref_meta["name"], line=dict(color=COLORS["reference_site"], width=1.6),
            )
        )
        fig_main = apply_common_layout(fig_main, title=f"{variable_label} — {pair['label']}")
        fig_main.update_yaxes(title=f"{variable_label} ({variable_unit})")
        fig_main.update_xaxes(title="Time")
        fig_main.update_layout(height=460)
        st.plotly_chart(fig_main, use_container_width=True)
        if variable in ("mean_radiant_temp_c", "utci_c"):
            render_estimate_note()

    with chart_right:
        hour_profile = hourly_profile(temp_comparison)
        fig_hour = go.Figure()
        fig_hour.add_trace(
            go.Bar(x=hour_profile["hour"], y=-hour_profile["delta"], marker_color=COLORS["green_site"])
        )
        fig_hour.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"])
        fig_hour = apply_common_layout(fig_hour, title="Cooling by Time of Day")
        fig_hour.update_xaxes(title="Hour of day", dtick=4)
        fig_hour.update_yaxes(title="Cooling effect (°C)")
        fig_hour.update_layout(height=210, showlegend=False)
        st.plotly_chart(fig_hour, use_container_width=True)

        day_profile = daily_profile(temp_comparison)
        fig_day = go.Figure()
        fig_day.add_trace(
            go.Bar(x=day_profile["date"], y=-day_profile["delta"], marker_color=COLORS["green_site"])
        )
        fig_day.add_hline(y=0, line_dash="dot", line_color=COLORS["text_muted"])
        fig_day = apply_common_layout(fig_day, title="Daily Cooling Performance")
        fig_day.update_xaxes(title="Date")
        fig_day.update_yaxes(title="Cooling effect (°C)")
        fig_day.update_layout(height=210, showlegend=False)
        st.plotly_chart(fig_day, use_container_width=True)

        st.caption(
            "Cooling effect = Reference − Green air temperature; positive values mean the "
            "green site measured cooler."
        )

    # -----------------------------------------------------------------
    # From monitoring to decision support
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("From Monitoring to Decision Support")
    step_cols = st.columns(4)
    steps = [
        "Compare green and reference conditions",
        "Identify when cooling occurs",
        "Assess performance during the hottest periods",
        "Support maintenance, improvement and replication decisions",
    ]
    for i, (col, step) in enumerate(zip(step_cols, steps), start=1):
        with col:
            with st.container(border=True):
                st.markdown(f"**Step {i}**")
                st.markdown(step)
