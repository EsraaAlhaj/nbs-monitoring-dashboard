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
    SIMULATION_INTERVAL_MINUTES,
    SIMULATION_NUM_DAYS,
    SIMULATION_START_DATE,
    STATION_PAIRS,
    STATIONS,
    STATUS_COLORS,
    VARIABLES,
)
from src.cached_data import load_latest_readings, load_readings, load_stations
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
    render_page_header,
)

st.set_page_config(page_title=f"{APP_TITLE} — Overview", layout="wide")
inject_base_css()
render_page_header("Overview")
render_demo_banner()

stations_df = load_stations()
latest_df = load_latest_readings()

st.markdown(f"**Current pilot: {len(stations_df)} stations | Platform designed for network expansion**")

st.markdown(
    "This prototype illustrates how paired micro-climate monitoring stations "
    "could track the observed temperature difference between Nature-based "
    "Solutions (vegetated green infrastructure) sites and nearby unplanted "
    "reference sites on the University of Jordan campus. Two station pairs "
    "are shown below."
)

if latest_df.empty:
    st.warning(
        "No readings found in the database yet. Run `python scripts/generate_data.py` "
        "to generate the synthetic demonstration dataset, then reload this page."
    )
    st.stop()

merged = stations_df.merge(
    latest_df.drop(columns=["reading_id"], errors="ignore"), on="station_id", how="left"
)


def _display_status(raw_status: str) -> str:
    """Overview-only display label: 'active' reads as 'Simulated' (this is a
    demonstration pilot, not a live sensor network); other statuses are
    shown as-is. Does not affect the underlying station_status values or
    STATUS_COLORS lookups used elsewhere in the app."""
    raw_lower = str(raw_status).lower()
    if raw_lower == "active":
        return "Simulated"
    return str(raw_status).capitalize()


DISPLAY_STATUS_COLORS = {
    "Simulated": STATUS_COLORS["active"],
    "Maintenance": STATUS_COLORS["maintenance"],
    "Offline": STATUS_COLORS["offline"],
    "No data": "#888888",
}

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
    hottest = hottest_periods(temp_comparison, top_n=10)
    peak_period_diff = (-hottest["cooling_effect_c"]).mean() if not hottest.empty else float("nan")
    completeness = completeness_by_station(filtered_readings, start_ts, end_ts)
    data_availability_pct = completeness["completeness_pct"].mean() if not completeness.empty else float("nan")
    cooling_hours_per_day = cooler_stats["cooler_hours"] / num_days if num_days else float("nan")

    # -----------------------------------------------------------------
    # KPI cards
    # -----------------------------------------------------------------
    st.subheader("Key Indicators")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Average air-temperature difference",
        f"{cooler_stats['mean_delta_c']:+.2f} °C" if pd.notna(cooler_stats["mean_delta_c"]) else "—",
    )
    k2.metric(
        "Cooling hours per day",
        f"{cooling_hours_per_day:.1f} h" if pd.notna(cooling_hours_per_day) else "—",
    )
    k3.metric(
        "Peak-period temperature difference",
        f"{peak_period_diff:+.2f} °C" if pd.notna(peak_period_diff) else "—",
    )
    k4.metric(
        "Data availability",
        f"{data_availability_pct:.1f}%" if pd.notna(data_availability_pct) else "—",
    )
    st.caption(
        "Air-temperature figures are Green − Reference (negative means the green site measured cooler). "
        "Peak-period figure covers the ten highest reference-site readings in the selected period. "
        "Data availability is the average share of expected readings received for these two stations."
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
            "Positive values indicate the green site measured cooler than the reference "
            "site (based on air temperature)."
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

st.divider()

# ---------------------------------------------------------------------------
# Station map
# ---------------------------------------------------------------------------
st.subheader("Monitoring Stations")

has_coordinates = merged["latitude"].notna().any() and merged["longitude"].notna().any()

if not has_coordinates:
    st.info("Station locations will be added following field verification and university approval.")
else:
    fig_map = go.Figure()
    for site_type, color, label in [
        ("green", COLORS["green_site"], "NBS (vegetated) site"),
        ("reference", COLORS["reference_site"], "Reference (unplanted) site"),
    ]:
        subset = merged[
            (merged["site_type"] == site_type) & merged["latitude"].notna() & merged["longitude"].notna()
        ]
        if subset.empty:
            continue
        hover_text = [
            f"<b>{row['name']}</b><br>Status: {_display_status(row.get('station_status'))}"
            f"<br>Last update: {row['timestamp']}"
            for _, row in subset.iterrows()
        ]
        fig_map.add_trace(
            go.Scattermapbox(
                lat=subset["latitude"],
                lon=subset["longitude"],
                mode="markers",
                marker=dict(size=18, color=color),
                name=label,
                text=hover_text,
                hoverinfo="text",
            )
        )

    valid_coords = merged.dropna(subset=["latitude", "longitude"])
    center_lat = float(valid_coords["latitude"].mean())
    center_lon = float(valid_coords["longitude"].mean())
    fig_map.update_layout(
        mapbox=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=13.5),
        height=420,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.75)"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

# ---------------------------------------------------------------------------
# Station status table
# ---------------------------------------------------------------------------
st.subheader("Station Status")

status_table = merged[["name", "site_type", "station_status", "timestamp"]].copy()
status_table["station_status"] = status_table["station_status"].fillna("no data")
status_table.columns = ["Station", "Site Type", "Status", "Last Update"]
status_table["Site Type"] = status_table["Site Type"].str.capitalize()
status_table["Status"] = status_table["Status"].apply(_display_status)


def _status_style(val: str) -> str:
    color = DISPLAY_STATUS_COLORS.get(str(val), "#888888")
    return f"background-color: {color}26; color: {color}; font-weight: 600;"


st.dataframe(
    status_table.style.map(_status_style, subset=["Status"]),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Pair summary cards
# ---------------------------------------------------------------------------
st.subheader("Site Pairs — Demonstration Site Comparison")

pair_cols = st.columns(len(STATION_PAIRS))
for col, (card_pair_id, card_pair) in zip(pair_cols, STATION_PAIRS.items()):
    card_green_meta = STATIONS[card_pair["green"]]
    card_ref_meta = STATIONS[card_pair["reference"]]
    green_row = merged[merged["station_id"] == card_green_meta["station_id"]]
    ref_row = merged[merged["station_id"] == card_ref_meta["station_id"]]

    with col:
        st.markdown(f"**{card_pair['label']}**")
        if green_row.empty or ref_row.empty:
            st.info("Station metadata unavailable.")
            continue

        g_temp = green_row["air_temperature_c"].iloc[0]
        r_temp = ref_row["air_temperature_c"].iloc[0]
        g_status = green_row["station_status"].iloc[0]
        r_status = ref_row["station_status"].iloc[0]

        if pd.notna(g_temp) and pd.notna(r_temp):
            delta = g_temp - r_temp
            st.metric(
                label=f"{card_green_meta['name']}",
                value=f"{g_temp:.1f} °C",
                delta=f"{delta:+.1f} °C vs. reference",
                delta_color="inverse",
            )
            st.caption(f"Reference ({card_ref_meta['name']}): {r_temp:.1f} °C")
        else:
            st.info(
                f"Latest reading unavailable — green: {_display_status(g_status)}, "
                f"reference: {_display_status(r_status)}."
            )

st.divider()
st.caption(
    "Use the navigation sidebar to explore Network Trends, Site Comparison, "
    "Thermal & Statistical Analysis, and Data Quality & Downloads."
)
