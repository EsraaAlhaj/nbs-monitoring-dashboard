"""
Thermal & Statistical Analysis page.

Merges the former UTCI Analysis and Statistical Analysis pages into one
navigation entry via st.tabs(), preserving each tab's filters, charts, and
calculations unchanged. Each tab's body lives in its own function so the two
pages' historically same-named local variables (readings, comparison, label,
unit, fig_ts, ...) can't collide now that both run in one script.
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
    UTCI_CATEGORIES,
    UTCI_CATEGORY_COLORS,
    VARIABLES,
)
from src.cached_data import load_readings
from src.comfort import classify_utci_series, classify_utci_value
from src.statistics_utils import attach_station_metadata, build_pair_comparison, summary_statistics
from src.styling import APP_TITLE, apply_common_layout, inject_base_css, render_demo_banner, render_estimate_note, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — Thermal & Statistical Analysis", layout="wide")
inject_base_css()
render_page_header("Thermal & Statistical Analysis")
render_demo_banner()

SIM_START = pd.Timestamp(SIMULATION_START_DATE)
SIM_END = SIM_START + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)


def _render_utci_tab() -> None:
    st.markdown(
        "The Universal Thermal Climate Index (UTCI) combines air temperature, humidity, wind "
        "speed and radiant heat into a single value that reflects perceived outdoor thermal "
        "stress, classified into ten standard categories from extreme cold to extreme heat stress."
    )
    render_estimate_note()

    category_order = [cat[2] for cat in UTCI_CATEGORIES]
    interval_hours = SIMULATION_INTERVAL_MINUTES / 60.0

    filter_col1, filter_col2 = st.columns([1.2, 1.8])
    with filter_col1:
        pair_id = st.selectbox(
            "Site pair",
            options=list(STATION_PAIRS.keys()),
            format_func=lambda k: STATION_PAIRS[k]["label"],
            key="utci_pair",
        )
    with filter_col2:
        date_range = st.date_input(
            "Period",
            value=(SIM_START.date(), SIM_END.date()),
            min_value=SIM_START.date(),
            max_value=SIM_END.date(),
            key="utci_period",
        )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = SIM_START.date(), SIM_END.date()

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
        return

    comparison = build_pair_comparison(readings, pair_id, variable="utci_c")
    comparison = comparison.dropna(subset=["green", "reference"], how="all")

    if comparison.empty:
        st.warning("No UTCI data available for the selected period.")
        return

    st.caption(f"Comparing **{green_meta['name']}** (green) against **{ref_meta['name']}** (reference).")

    # -----------------------------------------------------------------
    # Period summary
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # UTCI over time
    # -----------------------------------------------------------------
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

    # -----------------------------------------------------------------
    # Category classification
    # -----------------------------------------------------------------
    def _category_hours(series: pd.Series) -> pd.Series:
        classified = classify_utci_series(series.dropna())
        counts = classified.value_counts().reindex(category_order, fill_value=0)
        return (counts * interval_hours).round(1)

    green_hours = _category_hours(comparison["green"])
    ref_hours = _category_hours(comparison["reference"])
    present_categories = [c for c in category_order if green_hours.get(c, 0) > 0 or ref_hours.get(c, 0) > 0]

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
            "Category": category_order,
            "Green — Hours": [green_hours.get(c, 0) for c in category_order],
            "Green — % of Period": [
                round(100 * green_hours.get(c, 0) / total_green_hours, 1) if total_green_hours else 0.0
                for c in category_order
            ],
            "Reference — Hours": [ref_hours.get(c, 0) for c in category_order],
            "Reference — % of Period": [
                round(100 * ref_hours.get(c, 0) / total_ref_hours, 1) if total_ref_hours else 0.0
                for c in category_order
            ],
        }
    )
    st.dataframe(category_table, use_container_width=True, hide_index=True)
    st.caption(
        "Hours per thermal-stress category over the selected period, at the "
        f"{SIMULATION_INTERVAL_MINUTES}-minute recording resolution. Categories with zero hours for "
        "both sites are included for reference to the full standard UTCI scale."
    )


def _render_statistical_tab() -> None:
    all_station_ids = [meta["station_id"] for meta in STATIONS.values()]
    station_name_by_id = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
    site_color_map = {"green": COLORS["green_site"], "reference": COLORS["reference_site"]}

    row1_col1, row1_col2, row1_col3 = st.columns([1.8, 1.1, 1.1])
    with row1_col1:
        selected_stations = st.multiselect(
            "Stations",
            options=all_station_ids,
            default=all_station_ids,
            format_func=lambda sid: station_name_by_id.get(sid, sid),
            key="stats_stations",
        )
    with row1_col2:
        variable = st.selectbox(
            "Variable",
            options=list(VARIABLES.keys()),
            format_func=lambda k: VARIABLES[k]["label"],
            index=0,
            key="stats_variable",
        )
    with row1_col3:
        x_variable = st.selectbox(
            "Scatter X-axis variable",
            options=list(VARIABLES.keys()),
            format_func=lambda k: VARIABLES[k]["label"],
            index=list(VARIABLES.keys()).index("solar_radiation_wm2"),
            key="stats_x_variable",
        )

    row2_col1, row2_col2 = st.columns([1.6, 1.6])
    with row2_col1:
        date_range = st.date_input(
            "Period",
            value=(SIM_START.date(), SIM_END.date()),
            min_value=SIM_START.date(),
            max_value=SIM_END.date(),
            key="stats_period",
        )
    with row2_col2:
        hour_range = st.slider("Hour of day (local)", min_value=0, max_value=23, value=(0, 23), key="stats_hour_range")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = SIM_START.date(), SIM_END.date()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if not selected_stations:
        st.warning("Select at least one station.")
        return

    readings = load_readings(
        station_ids=tuple(selected_stations),
        start=start_ts.isoformat(),
        end=end_ts.isoformat(),
    )

    if readings.empty:
        st.warning("No data available for the selected filters.")
        return

    readings = readings[readings["timestamp"].dt.hour.between(hour_range[0], hour_range[1])]
    readings = attach_station_metadata(readings)

    if readings.empty:
        st.warning("No data available for the selected filters.")
        return

    label = VARIABLES[variable]["label"]
    unit = VARIABLES[variable]["unit"]
    x_label = VARIABLES[x_variable]["label"]
    x_unit = VARIABLES[x_variable]["unit"]

    st.caption(f"{len(readings):,} records match the current filters.")

    if variable in ("mean_radiant_temp_c", "utci_c") or x_variable in ("mean_radiant_temp_c", "utci_c"):
        render_estimate_note()

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------
    st.subheader("Summary Statistics")
    stats = summary_statistics(readings, variable)
    if stats.empty:
        st.info("No valid values for this variable in the current selection.")
    else:
        stats["station_name"] = stats["station_id"].map(station_name_by_id)
        stats = stats[["station_id", "station_name", "count", "mean", "std", "min", "p25", "median", "p75", "max"]]
        stats.columns = ["Station ID", "Station", "Count", "Mean", "Std Dev", "Min", "P25", "Median", "P75", "Max"]
        st.dataframe(stats, use_container_width=True, hide_index=True)
        st.caption(f"Units: {unit}")

    # -----------------------------------------------------------------
    # Scatterplot (SVG-rendered go.Scatter, not WebGL Scattergl, for
    # cross-browser compatibility)
    # -----------------------------------------------------------------
    st.subheader(f"Scatterplot — {x_label} vs. {label}")
    scatter_df = readings.dropna(subset=[x_variable, variable])
    fig_scatter = go.Figure()
    for site_type, color in site_color_map.items():
        subset = scatter_df[scatter_df["site_type"] == site_type]
        if subset.empty:
            continue
        fig_scatter.add_trace(
            go.Scatter(
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

    # -----------------------------------------------------------------
    # Boxplot
    # -----------------------------------------------------------------
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
                name=station_name_by_id.get(station_id, station_id),
                marker_color=site_color_map.get(site_type, COLORS["text_muted"]),
            )
        )
    fig_box = apply_common_layout(fig_box)
    fig_box.update_yaxes(title=f"{label} ({unit})")
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # -----------------------------------------------------------------
    # Histogram
    # -----------------------------------------------------------------
    st.subheader(f"Histogram — {label}")
    fig_hist = go.Figure()
    for site_type, color in site_color_map.items():
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


tab_utci, tab_stats = st.tabs(["UTCI Analysis", "Statistical Analysis"])
with tab_utci:
    _render_utci_tab()
with tab_stats:
    _render_statistical_tab()
