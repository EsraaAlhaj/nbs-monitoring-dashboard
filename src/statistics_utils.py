"""
Statistical helpers shared by the Site Comparison and Thermal & Statistical
Analysis pages.

All functions take plain pandas DataFrames (as returned by
src/data_service.py) and return DataFrames/scalars — no Streamlit or
database code here, so these are easy to unit test in isolation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import STATION_PAIRS, STATIONS


def build_pair_comparison(
    readings: pd.DataFrame,
    pair_id: str,
    variable: str = "air_temperature_c",
) -> pd.DataFrame:
    """
    Wide-format DataFrame indexed by timestamp with one column for the green
    site and one for the reference site of the given pair, plus a `delta`
    column (green - reference). Rows where either side is missing are kept
    with NaN so gaps are visible rather than silently dropped.
    """
    pair = STATION_PAIRS[pair_id]
    green_id = STATIONS[pair["green"]]["station_id"]
    ref_id = STATIONS[pair["reference"]]["station_id"]

    subset = readings[readings["station_id"].isin([green_id, ref_id])]
    pivot = subset.pivot_table(index="timestamp", columns="station_id", values=variable)
    pivot = pivot.rename(columns={green_id: "green", ref_id: "reference"})

    for col in ("green", "reference"):
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["delta"] = pivot["green"] - pivot["reference"]
    return pivot.reset_index().sort_values("timestamp")


def hourly_profile(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Average green/reference/delta by hour-of-day (0-23) across the whole period."""
    df = comparison_df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    profile = (
        df.groupby("hour")[["green", "reference", "delta"]]
        .mean()
        .reindex(range(24))
        .reset_index()
    )
    return profile


def count_cooler_hours(comparison_df: pd.DataFrame) -> dict:
    """
    Count how many timestamps (converted to hours of 15-min records) the
    green site was cooler than the reference site, plus summary shares.
    """
    valid = comparison_df.dropna(subset=["delta"])
    total = len(valid)
    cooler = int((valid["delta"] < 0).sum())
    warmer = int((valid["delta"] > 0).sum())
    equal = total - cooler - warmer

    interval_hours = 0.25  # 15-minute records
    return {
        "total_records": total,
        "cooler_records": cooler,
        "warmer_records": warmer,
        "equal_records": equal,
        "cooler_hours": round(cooler * interval_hours, 1),
        "warmer_hours": round(warmer * interval_hours, 1),
        "cooler_share_pct": round(100 * cooler / total, 1) if total else float("nan"),
        "mean_delta_c": round(float(valid["delta"].mean()), 2) if total else float("nan"),
        "max_cooling_c": round(float(-valid["delta"].min()), 2) if total else float("nan"),
    }


def hottest_periods(
    comparison_df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Top-N timestamps ranked by reference-site temperature (the hottest
    conditions observed), showing how much cooler the green site was at
    those same moments.
    """
    valid = comparison_df.dropna(subset=["reference"]).copy()
    valid = valid.sort_values("reference", ascending=False).head(top_n)
    valid["cooling_effect_c"] = -valid["delta"]
    return valid[["timestamp", "reference", "green", "cooling_effect_c"]].reset_index(drop=True)


def summary_statistics(readings: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Per-station descriptive statistics for one variable."""
    if variable not in readings.columns:
        return pd.DataFrame()
    grouped = readings.dropna(subset=[variable]).groupby("station_id")[variable]
    stats = grouped.agg(
        count="count",
        mean="mean",
        std="std",
        min="min",
        p25=lambda s: s.quantile(0.25),
        median="median",
        p75=lambda s: s.quantile(0.75),
        max="max",
    )
    return stats.reset_index().round(2)


def site_type_label(station_id: str) -> str:
    for meta in STATIONS.values():
        if meta["station_id"] == station_id:
            return meta["site_type"]
    return "unknown"


def station_name(station_id: str) -> str:
    for meta in STATIONS.values():
        if meta["station_id"] == station_id:
            return meta["name"]
    return station_id


def attach_station_metadata(readings: pd.DataFrame) -> pd.DataFrame:
    """Add station_name, site_type, pair_id columns to a readings DataFrame."""
    meta_rows = [
        {
            "station_id": m["station_id"],
            "station_name": m["name"],
            "site_type": m["site_type"],
            "pair_id": m["pair_id"],
        }
        for m in STATIONS.values()
    ]
    meta_df = pd.DataFrame(meta_rows)
    return readings.merge(meta_df, on="station_id", how="left")
