"""
Network Trends page: multi-station / multi-pair time-series explorer.

Stations are sourced dynamically from the database (via src.cached_data,
same as every other page) — nothing here hard-codes station IDs, pair IDs,
or a fixed station count, so the platform can support additional stations
or site pairs later without touching this file.
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
    VARIABLES,
)
from src.cached_data import load_readings, load_stations
from src.styling import (
    APP_TITLE,
    apply_common_layout,
    inject_base_css,
    render_demo_banner,
    render_estimate_note,
    render_page_header,
)

st.set_page_config(page_title=f"{APP_TITLE} — Network Trends", layout="wide")
inject_base_css()
render_page_header("Network Trends")
render_demo_banner()

st.markdown(
    "Compare climate trends across selected stations and monitoring areas "
    "using a common time and value scale."
)

sim_start = pd.Timestamp(SIMULATION_START_DATE)
sim_end = sim_start + pd.Timedelta(days=SIMULATION_NUM_DAYS) - pd.Timedelta(minutes=SIMULATION_INTERVAL_MINUTES)
# Default to a 7-day window so the page opens on a readable slice; the full
# simulated range remains selectable via the date input's min/max below.
default_period_end = min(sim_start + pd.Timedelta(days=6), sim_end)

stations_df = load_stations()
if stations_df.empty:
    st.warning("No stations found in the database yet.")
    st.stop()

stations_df = stations_df.sort_values("name")
ALL_STATION_IDS = stations_df["station_id"].tolist()
STATION_NAME_BY_ID = dict(zip(stations_df["station_id"], stations_df["name"]))
STATION_SITE_TYPE_BY_ID = dict(zip(stations_df["station_id"], stations_df["site_type"]))
STATION_PAIR_BY_ID = dict(zip(stations_df["station_id"], stations_df["pair_id"]))

ALL_STATIONS_OPTION = "All Stations"
RESOLUTION_OPTIONS = {"Raw (15-minute)": None, "Hourly average": "1h", "Daily average": "1D"}

# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------
row1_col1, row1_col2 = st.columns([2.0, 1.2])
with row1_col1:
    station_selection = st.multiselect(
        "Stations",
        options=[ALL_STATIONS_OPTION] + ALL_STATION_IDS,
        default=[ALL_STATIONS_OPTION],
        format_func=lambda sid: sid if sid == ALL_STATIONS_OPTION else STATION_NAME_BY_ID.get(sid, sid),
    )
with row1_col2:
    variable = st.selectbox(
        "Variable",
        options=list(VARIABLES.keys()),
        format_func=lambda k: VARIABLES[k]["label"],
        index=0,
    )

row2_col1, row2_col2, row2_col3 = st.columns([1.6, 1.4, 1.2])
with row2_col1:
    date_range = st.date_input(
        "Period",
        value=(sim_start.date(), default_period_end.date()),
        min_value=sim_start.date(),
        max_value=sim_end.date(),
    )
with row2_col2:
    hour_range = st.slider("Hour of day (local)", min_value=0, max_value=23, value=(0, 23))
with row2_col3:
    resolution_label = st.selectbox("Display resolution", options=list(RESOLUTION_OPTIONS.keys()), index=1)

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = sim_start.date(), sim_end.date()

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

if not station_selection:
    st.warning("Select at least one station, or choose All Stations.")
    st.stop()

if ALL_STATIONS_OPTION in station_selection:
    selected_station_ids = ALL_STATION_IDS
else:
    selected_station_ids = station_selection

readings = load_readings(
    station_ids=tuple(selected_station_ids),
    start=start_ts.isoformat(),
    end=end_ts.isoformat(),
)

if readings.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

readings = readings[readings["timestamp"].dt.hour.between(hour_range[0], hour_range[1])]

if readings.empty:
    st.warning("No data available for the selected hour-of-day window.")
    st.stop()

label = VARIABLES[variable]["label"]
unit = VARIABLES[variable]["unit"]
decimals = VARIABLES[variable]["decimals"]

if variable in ("mean_radiant_temp_c", "utci_c"):
    render_estimate_note()


def _resample_readings(df: pd.DataFrame, variable: str, freq: str | None) -> pd.DataFrame:
    """Per-station mean of `variable`, resampled to `freq` (None = no resampling)."""
    if freq is None:
        return df[["station_id", "timestamp", variable]].copy()
    parts = []
    for station_id, group in df.groupby("station_id"):
        resampled = group.set_index("timestamp")[variable].resample(freq).mean().reset_index()
        resampled["station_id"] = station_id
        parts.append(resampled)
    if not parts:
        return pd.DataFrame(columns=["station_id", "timestamp", variable])
    return pd.concat(parts, ignore_index=True)


resample_freq = RESOLUTION_OPTIONS[resolution_label]
plot_data = _resample_readings(readings, variable, resample_freq).dropna(subset=[variable])

if plot_data.empty:
    st.warning("No valid values for this variable in the current selection.")
    st.stop()

st.caption(
    f"{len(plot_data):,} plotted points across {len(selected_station_ids)} station(s), "
    f"at {resolution_label.lower()} resolution."
)

# ---------------------------------------------------------------------------
# Shared time axis and value scale across every panel below.
# ---------------------------------------------------------------------------
x_range = [plot_data["timestamp"].min(), plot_data["timestamp"].max()]
y_min, y_max = float(plot_data[variable].min()), float(plot_data[variable].max())
y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
y_range = [y_min - y_pad, y_max + y_pad]

SITE_TYPE_COLORS = {"green": COLORS["green_site"], "reference": COLORS["reference_site"]}
SITE_TYPE_SHORT_LABELS = {"green": "Green Site", "reference": "Reference Site"}

# ---------------------------------------------------------------------------
# One independent chart per Monitoring Area (site pair)
# ---------------------------------------------------------------------------
present_pair_ids = sorted({STATION_PAIR_BY_ID[sid] for sid in selected_station_ids if sid in STATION_PAIR_BY_ID})

if not present_pair_ids:
    st.info("No monitoring areas match the current station selection.")

for pair_id in present_pair_ids:
    pair_label = STATION_PAIRS.get(pair_id, {}).get("label", pair_id.replace("_", " ").title())
    pair_station_ids = [sid for sid in selected_station_ids if STATION_PAIR_BY_ID.get(sid) == pair_id]
    if not pair_station_ids:
        continue

    st.subheader(pair_label)
    fig = go.Figure()
    for sid in pair_station_ids:
        site_type = STATION_SITE_TYPE_BY_ID.get(sid)
        color = SITE_TYPE_COLORS.get(site_type, COLORS["text_muted"])
        series = plot_data[plot_data["station_id"] == sid].sort_values("timestamp")
        if series.empty:
            continue
        site_label = SITE_TYPE_SHORT_LABELS.get(site_type, STATION_NAME_BY_ID.get(sid, sid))
        fig.add_trace(
            go.Scatter(
                x=series["timestamp"], y=series[variable],
                mode="lines", name=site_label,
                line=dict(color=color, width=1.6),
                hovertemplate=f"<b>{site_label}</b>: %{{y:.{decimals}f}} {unit}<extra></extra>",
            )
        )

    # Hidden helper trace carrying the Reference-minus-Green difference so
    # the unified hover box (below) shows time + both site values + their
    # difference together. Drawn with a zero-width line (no marker), so it
    # never appears on the chart itself or in the legend — it only
    # contributes its row to the unified hover. Only added when both a green
    # and a reference station of this pair are in the current selection.
    green_sid = next((sid for sid in pair_station_ids if STATION_SITE_TYPE_BY_ID.get(sid) == "green"), None)
    reference_sid = next((sid for sid in pair_station_ids if STATION_SITE_TYPE_BY_ID.get(sid) == "reference"), None)
    if green_sid and reference_sid:
        green_series = plot_data.loc[plot_data["station_id"] == green_sid, ["timestamp", variable]].rename(
            columns={variable: "green"}
        )
        reference_series = plot_data.loc[plot_data["station_id"] == reference_sid, ["timestamp", variable]].rename(
            columns={variable: "reference"}
        )
        paired = pd.merge(green_series, reference_series, on="timestamp", how="inner").sort_values("timestamp")
        if not paired.empty:
            paired["difference"] = paired["reference"] - paired["green"]
            fig.add_trace(
                go.Scatter(
                    x=paired["timestamp"], y=paired["green"],
                    mode="lines", name="Difference",
                    line=dict(width=0, color=COLORS["accent"]),
                    showlegend=False,
                    customdata=paired["difference"],
                    hovertemplate=f"<b>Difference (Reference − Green)</b>: %{{customdata:+.{decimals}f}} {unit}<extra></extra>",
                )
            )

    fig = apply_common_layout(fig)
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title="Time", range=x_range)
    fig.update_yaxes(title=f"{label} ({unit})", range=y_range)
    st.plotly_chart(fig, use_container_width=True)
