"""
Data Quality & Downloads page.

Merges the former Data Quality and Downloads pages into one navigation
entry via st.tabs(), preserving each tab's filters, checks, and export
options unchanged. Each tab's body lives in its own function so the two
pages' historically same-named local variables can't collide now that both
run in one script.
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
from src.cached_data import load_readings
from src.quality_checks import (
    completeness_by_station,
    detect_implausible_readings,
    detect_stuck_sensors,
    detect_time_gaps,
    station_status_summary,
)
from src.statistics_utils import build_pair_comparison, count_cooler_hours, hottest_periods
from src.styling import (
    APP_TITLE,
    apply_common_layout,
    inject_base_css,
    render_demo_banner,
    render_kpi_card,
    render_page_header,
)

st.set_page_config(page_title=f"{APP_TITLE} — Data Quality & Downloads", layout="wide")
inject_base_css()
render_page_header("Data Quality & Downloads")
render_demo_banner()

SIM_START = pd.Timestamp(SIMULATION_START_DATE)
SIM_END = SIM_START + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)


def _render_data_quality_tab() -> None:
    st.markdown(
        "Diagnostics on the monitoring network itself: how complete each station's record is, "
        "where data is missing, and which readings look implausible or suspiciously static. "
        "A handful of issues have been deliberately seeded into this synthetic dataset so these "
        "checks have something real to detect."
    )

    station_name_by_id = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
    all_station_ids = list(station_name_by_id.keys())

    filter_col1, filter_col2 = st.columns([1.8, 1.4])
    with filter_col1:
        selected_stations = st.multiselect(
            "Stations",
            options=all_station_ids,
            default=all_station_ids,
            format_func=lambda sid: station_name_by_id.get(sid, sid),
            key="dq_stations",
        )
    with filter_col2:
        date_range = st.date_input(
            "Period",
            value=(SIM_START.date(), SIM_END.date()),
            min_value=SIM_START.date(),
            max_value=SIM_END.date(),
            key="dq_period",
        )

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

    completeness = completeness_by_station(readings, start_ts, end_ts)
    gaps = detect_time_gaps(readings)
    implausible = detect_implausible_readings(readings)
    stuck = detect_stuck_sensors(readings)

    # -----------------------------------------------------------------
    # Overview
    # -----------------------------------------------------------------
    st.subheader("Overview")
    o1, o2, o3, o4, o5 = st.columns(5)
    with o1:
        render_kpi_card("Records in selection", f"{len(readings):,}")
    with o2:
        render_kpi_card(
            "Average completeness",
            f"{completeness['completeness_pct'].mean():.1f}%" if not completeness.empty else "—",
        )
    with o3:
        render_kpi_card("Time gaps detected", f"{len(gaps):,}")
    with o4:
        render_kpi_card("Implausible readings", f"{len(implausible):,}")
    with o5:
        render_kpi_card("Stuck-sensor runs", f"{len(stuck):,}")

    # -----------------------------------------------------------------
    # Completeness
    # -----------------------------------------------------------------
    st.subheader("Completeness by Station")
    completeness_display = completeness.copy()
    completeness_display["station_name"] = completeness_display["station_id"].map(station_name_by_id)

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

    # -----------------------------------------------------------------
    # Station status overview
    # -----------------------------------------------------------------
    st.subheader("Station Status Overview")
    status_summary = station_status_summary(readings)
    status_summary["station_name"] = status_summary["station_id"].map(station_name_by_id)

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

    # -----------------------------------------------------------------
    # Time gaps
    # -----------------------------------------------------------------
    st.subheader("Missing Data — Time Gaps")
    if gaps.empty:
        st.success("No time gaps larger than the expected interval were found in the current selection.")
    else:
        gaps_display = gaps.copy()
        gaps_display["station_name"] = gaps_display["station_id"].map(station_name_by_id)
        gaps_display = gaps_display.rename(
            columns={
                "station_name": "Station", "gap_start": "Gap Start",
                "gap_end": "Gap End", "gap_minutes": "Gap (minutes)",
            }
        )[["Station", "Gap Start", "Gap End", "Gap (minutes)"]]
        st.dataframe(gaps_display, use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------
    # Implausible readings
    # -----------------------------------------------------------------
    st.subheader("Implausible Readings")
    if implausible.empty:
        st.success("No implausible readings were found in the current selection.")
    else:
        implausible_display = implausible.copy()
        implausible_display["station_name"] = implausible_display["station_id"].map(station_name_by_id)
        implausible_display = implausible_display.rename(
            columns={
                "station_name": "Station", "timestamp": "Timestamp", "variable": "Variable",
                "value": "Value", "valid_min": "Valid Min", "valid_max": "Valid Max",
            }
        )[["Station", "Timestamp", "Variable", "Value", "Valid Min", "Valid Max"]]
        st.dataframe(implausible_display, use_container_width=True, hide_index=True)
        st.caption("Values outside the physically plausible range configured for each variable.")

    # -----------------------------------------------------------------
    # Stuck sensors
    # -----------------------------------------------------------------
    st.subheader("Long-Duration Stuck Readings")
    if stuck.empty:
        st.success("No stuck-sensor runs were found in the current selection.")
    else:
        stuck_display = stuck.copy()
        stuck_display["station_name"] = stuck_display["station_id"].map(station_name_by_id)
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


def _render_downloads_tab() -> None:
    st.markdown(
        "Export the currently simulated data for offline analysis or reporting. All downloads "
        "reflect the filters selected on this tab, independent of filters used elsewhere in the app."
    )

    station_name_by_id = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
    all_station_ids = list(station_name_by_id.keys())

    def _to_csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    # -----------------------------------------------------------------
    # Raw data export
    # -----------------------------------------------------------------
    st.subheader("Raw Readings Export")

    raw_col1, raw_col2 = st.columns([1.8, 1.4])
    with raw_col1:
        raw_stations = st.multiselect(
            "Stations",
            options=all_station_ids,
            default=all_station_ids,
            format_func=lambda sid: station_name_by_id.get(sid, sid),
            key="raw_stations",
        )
    with raw_col2:
        raw_date_range = st.date_input(
            "Period",
            value=(SIM_START.date(), SIM_END.date()),
            min_value=SIM_START.date(),
            max_value=SIM_END.date(),
            key="raw_period",
        )

    if isinstance(raw_date_range, tuple) and len(raw_date_range) == 2:
        raw_start_date, raw_end_date = raw_date_range
    else:
        raw_start_date, raw_end_date = SIM_START.date(), SIM_END.date()

    raw_start_ts = pd.Timestamp(raw_start_date)
    raw_end_ts = pd.Timestamp(raw_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    if not raw_stations:
        st.warning("Select at least one station to export.")
    else:
        raw_readings = load_readings(
            station_ids=tuple(raw_stations),
            start=raw_start_ts.isoformat(),
            end=raw_end_ts.isoformat(),
        )
        if raw_readings.empty:
            st.info("No data available for the selected filters.")
        else:
            raw_readings = raw_readings.copy()
            raw_readings["station_name"] = raw_readings["station_id"].map(station_name_by_id)
            st.caption(f"{len(raw_readings):,} records match the current filters.")
            st.dataframe(raw_readings.head(20), use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered raw data (CSV)",
                data=_to_csv_bytes(raw_readings),
                file_name=f"nbs_readings_{raw_start_date}_{raw_end_date}.csv",
                mime="text/csv",
                key="download_raw_readings",
            )

    st.divider()

    # -----------------------------------------------------------------
    # Comparison / results summary export
    # -----------------------------------------------------------------
    st.subheader("Site Comparison Summary Export")

    cmp_col1, cmp_col2, cmp_col3 = st.columns([1.2, 1.6, 1.2])
    with cmp_col1:
        cmp_pair_id = st.selectbox(
            "Site pair",
            options=list(STATION_PAIRS.keys()),
            format_func=lambda k: STATION_PAIRS[k]["label"],
            key="cmp_pair",
        )
    with cmp_col2:
        cmp_date_range = st.date_input(
            "Period",
            value=(SIM_START.date(), SIM_END.date()),
            min_value=SIM_START.date(),
            max_value=SIM_END.date(),
            key="cmp_period",
        )
    with cmp_col3:
        cmp_variable = st.selectbox(
            "Variable",
            options=list(VARIABLES.keys()),
            format_func=lambda k: VARIABLES[k]["label"],
            key="cmp_variable",
        )

    if isinstance(cmp_date_range, tuple) and len(cmp_date_range) == 2:
        cmp_start_date, cmp_end_date = cmp_date_range
    else:
        cmp_start_date, cmp_end_date = SIM_START.date(), SIM_END.date()

    cmp_start_ts = pd.Timestamp(cmp_start_date)
    cmp_end_ts = pd.Timestamp(cmp_end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    cmp_pair = STATION_PAIRS[cmp_pair_id]
    cmp_green_meta = STATIONS[cmp_pair["green"]]
    cmp_ref_meta = STATIONS[cmp_pair["reference"]]

    cmp_readings = load_readings(
        station_ids=(cmp_green_meta["station_id"], cmp_ref_meta["station_id"]),
        start=cmp_start_ts.isoformat(),
        end=cmp_end_ts.isoformat(),
    )

    if cmp_readings.empty:
        st.info("No data available for the selected comparison filters.")
    else:
        comparison = build_pair_comparison(cmp_readings, cmp_pair_id, variable=cmp_variable)
        label = VARIABLES[cmp_variable]["label"]
        comparison_export = comparison.rename(
            columns={
                "green": f"{cmp_green_meta['name']} ({label})",
                "reference": f"{cmp_ref_meta['name']} ({label})",
                "delta": f"Difference - Green minus Reference ({label})",
            }
        )
        st.dataframe(comparison_export.head(20), use_container_width=True, hide_index=True)
        st.download_button(
            "Download comparison time series (CSV)",
            data=_to_csv_bytes(comparison_export),
            file_name=f"nbs_comparison_{cmp_pair_id}_{cmp_variable}_{cmp_start_date}_{cmp_end_date}.csv",
            mime="text/csv",
            key="download_comparison_series",
        )

        temp_comparison = build_pair_comparison(cmp_readings, cmp_pair_id, variable="air_temperature_c")
        cooler_stats = count_cooler_hours(temp_comparison)
        hottest = hottest_periods(temp_comparison, top_n=10)

        summary_row = {
            "pair": cmp_pair["label"],
            "green_station": cmp_green_meta["name"],
            "reference_station": cmp_ref_meta["name"],
            "period_start": str(cmp_start_date),
            "period_end": str(cmp_end_date),
            **cooler_stats,
        }
        summary_df = pd.DataFrame([summary_row])
        st.download_button(
            "Download cooling performance summary (CSV)",
            data=_to_csv_bytes(summary_df),
            file_name=f"nbs_cooling_summary_{cmp_pair_id}_{cmp_start_date}_{cmp_end_date}.csv",
            mime="text/csv",
            key="download_cooling_summary",
        )

        if not hottest.empty:
            hottest_export = hottest.rename(
                columns={
                    "timestamp": "Timestamp",
                    "reference": "Reference (°C)",
                    "green": "Green (°C)",
                    "cooling_effect_c": "Cooling Effect (°C)",
                }
            )
            st.download_button(
                "Download hottest-periods table (CSV)",
                data=_to_csv_bytes(hottest_export),
                file_name=f"nbs_hottest_periods_{cmp_pair_id}_{cmp_start_date}_{cmp_end_date}.csv",
                mime="text/csv",
                key="download_hottest_periods",
            )


tab_quality, tab_downloads = st.tabs(["Data Quality", "Downloads"])
with tab_quality:
    _render_data_quality_tab()
with tab_downloads:
    _render_downloads_tab()
