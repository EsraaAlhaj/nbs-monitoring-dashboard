"""Downloads page: export filtered raw data and comparison summaries as CSV."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import (
    SIMULATION_INTERVAL_MINUTES,
    SIMULATION_NUM_DAYS,
    SIMULATION_START_DATE,
    STATION_PAIRS,
    STATIONS,
    VARIABLES,
)
from src.cached_data import load_readings
from src.statistics_utils import build_pair_comparison, count_cooler_hours, hottest_periods
from src.styling import APP_TITLE, inject_base_css, render_demo_banner, render_page_header

st.set_page_config(page_title=f"{APP_TITLE} — Downloads", layout="wide")
inject_base_css()
render_page_header("Downloads")
render_demo_banner()

st.markdown(
    "Export the currently simulated data for offline analysis or reporting. All downloads "
    "reflect the filters selected on this page, independent of filters used elsewhere in the app."
)

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)
STATION_NAME_BY_ID = {meta["station_id"]: meta["name"] for meta in STATIONS.values()}
ALL_STATION_IDS = list(STATION_NAME_BY_ID.keys())


def _to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Raw data export
# ---------------------------------------------------------------------------
st.subheader("Raw Readings Export")

raw_col1, raw_col2 = st.columns([1.8, 1.4])
with raw_col1:
    raw_stations = st.multiselect(
        "Stations",
        options=ALL_STATION_IDS,
        default=ALL_STATION_IDS,
        format_func=lambda sid: STATION_NAME_BY_ID.get(sid, sid),
        key="raw_stations",
    )
with raw_col2:
    raw_date_range = st.date_input(
        "Period",
        value=(sim_start.date(), sim_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
        key="raw_period",
    )

if isinstance(raw_date_range, tuple) and len(raw_date_range) == 2:
    raw_start_date, raw_end_date = raw_date_range
else:
    raw_start_date, raw_end_date = sim_start.date(), sim_end.date()

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
        raw_readings["station_name"] = raw_readings["station_id"].map(STATION_NAME_BY_ID)
        st.caption(f"{len(raw_readings):,} records match the current filters.")
        st.dataframe(raw_readings.head(20), use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered raw data (CSV)",
            data=_to_csv_bytes(raw_readings),
            file_name=f"nbs_readings_{raw_start_date}_{raw_end_date}.csv",
            mime="text/csv",
        )

st.divider()

# ---------------------------------------------------------------------------
# Comparison / results summary export
# ---------------------------------------------------------------------------
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
        value=(sim_start.date(), sim_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
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
    cmp_start_date, cmp_end_date = sim_start.date(), sim_end.date()

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
