"""
Overview page (Streamlit multipage app entry point).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import (
    COLORS,
    STATION_PAIRS,
    STATION_SITING_DISCLAIMER,
    STATIONS,
    STATUS_COLORS,
)
from src.cached_data import load_latest_readings, load_stations
from src.styling import (
    APP_TITLE,
    inject_base_css,
    render_demo_banner,
    render_page_header,
)

st.set_page_config(page_title=f"{APP_TITLE} — Overview", layout="wide")
inject_base_css()
render_page_header("Overview")
render_demo_banner()

st.markdown(
    "This prototype illustrates how paired micro-climate monitoring stations "
    "could track the cooling impact of Nature-based Solutions (vegetated green "
    "infrastructure) compared to nearby unplanted reference sites on the "
    "University of Jordan campus. Two station pairs are shown below."
)

stations_df = load_stations()
latest_df = load_latest_readings()

if latest_df.empty:
    st.warning(
        "No readings found in the database yet. Run `python scripts/generate_data.py` "
        "to generate the synthetic demonstration dataset, then reload this page."
    )
    st.stop()

merged = stations_df.merge(
    latest_df.drop(columns=["reading_id"], errors="ignore"), on="station_id", how="left"
)

# ---------------------------------------------------------------------------
# Station map
# ---------------------------------------------------------------------------
st.subheader("Monitoring Stations")

has_coordinates = merged["latitude"].notna().any() and merged["longitude"].notna().any()

if not has_coordinates:
    st.info(
        "Station coordinates have not been configured yet — real site "
        "selection is pending. Once GPS coordinates are added to "
        "`STATIONS` in config/settings.py, the station map will appear "
        "here automatically."
    )
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
            f"<b>{row['name']}</b><br>Status: {str(row.get('station_status') or 'unknown').title()}"
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

st.caption(f"Note: {STATION_SITING_DISCLAIMER}")

# ---------------------------------------------------------------------------
# Station status table
# ---------------------------------------------------------------------------
st.subheader("Station Status")

status_table = merged[["name", "site_type", "station_status", "timestamp"]].copy()
status_table["station_status"] = status_table["station_status"].fillna("no data")
status_table.columns = ["Station", "Site Type", "Status", "Last Update"]
status_table["Site Type"] = status_table["Site Type"].str.capitalize()
status_table["Status"] = status_table["Status"].str.capitalize()


def _status_style(val: str) -> str:
    color = STATUS_COLORS.get(str(val).lower(), "#888888")
    return f"background-color: {color}26; color: {color}; font-weight: 600;"


st.dataframe(
    status_table.style.map(_status_style, subset=["Status"]),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Pair summary cards
# ---------------------------------------------------------------------------
st.subheader("Site Pairs — Current Cooling Effect")

pair_cols = st.columns(len(STATION_PAIRS))
for col, (pair_id, pair) in zip(pair_cols, STATION_PAIRS.items()):
    green_meta = STATIONS[pair["green"]]
    ref_meta = STATIONS[pair["reference"]]
    green_row = merged[merged["station_id"] == green_meta["station_id"]]
    ref_row = merged[merged["station_id"] == ref_meta["station_id"]]

    with col:
        st.markdown(f"**{pair['label']}**")
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
                label=f"{green_meta['name']}",
                value=f"{g_temp:.1f} °C",
                delta=f"{delta:+.1f} °C vs. reference",
                delta_color="inverse",
            )
            st.caption(f"Reference ({ref_meta['name']}): {r_temp:.1f} °C")
        else:
            st.info(
                f"Latest reading unavailable — green: {str(g_status).title()}, "
                f"reference: {str(r_status).title()}."
            )

st.divider()
st.caption(
    "Use the navigation sidebar to explore Site Comparison, UTCI Analysis, "
    "Statistical Analysis, Data Quality diagnostics, and data Downloads."
)
