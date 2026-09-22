"""
Data-quality diagnostics: completeness, time gaps, implausible readings,
and stuck-sensor (flatline) detection.

Used by the Data Quality page. Kept dependency-free (pandas/numpy only) so
it can be unit tested without Streamlit or the database.
"""

from __future__ import annotations

import pandas as pd

from config.settings import (
    CORE_READING_FIELDS,
    EXPECTED_INTERVAL_MINUTES,
    GAP_THRESHOLD_MINUTES,
    STUCK_SENSOR_MIN_RUN_LENGTH,
    STUCK_SENSOR_ZERO_EXEMPT_FIELDS,
    VARIABLES,
)


def completeness_by_station(
    readings: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Expected record count is derived from the requested period length and
    the configured sampling interval, so this works for any filtered
    sub-range, not just the full 30-day dataset.
    """
    total_minutes = (period_end - period_start).total_seconds() / 60.0
    expected_records = max(int(total_minutes // EXPECTED_INTERVAL_MINUTES) + 1, 1)

    rows = []
    for station_id, group in readings.groupby("station_id"):
        actual_records = len(group)
        non_missing = int((group["is_missing"] == 0).sum())
        rows.append(
            {
                "station_id": station_id,
                "expected_records": expected_records,
                "actual_records": actual_records,
                "complete_records": non_missing,
                "completeness_pct": round(100 * non_missing / expected_records, 1)
                if expected_records
                else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id")


def detect_time_gaps(readings: pd.DataFrame) -> pd.DataFrame:
    """Find gaps between consecutive timestamps per station larger than the threshold."""
    gap_rows = []
    for station_id, group in readings.groupby("station_id"):
        ts = group["timestamp"].sort_values().reset_index(drop=True)
        if len(ts) < 2:
            continue
        diffs_minutes = ts.diff().dt.total_seconds() / 60.0
        gap_idx = diffs_minutes[diffs_minutes > GAP_THRESHOLD_MINUTES].index
        for idx in gap_idx:
            gap_rows.append(
                {
                    "station_id": station_id,
                    "gap_start": ts.iloc[idx - 1],
                    "gap_end": ts.iloc[idx],
                    "gap_minutes": round(diffs_minutes.iloc[idx], 1),
                }
            )
    return pd.DataFrame(gap_rows)


def detect_implausible_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Flag values outside the physically plausible range configured per variable."""
    flags = []
    for field in CORE_READING_FIELDS:
        if field not in readings.columns or field not in VARIABLES:
            continue
        bounds = VARIABLES[field]
        out_of_range = readings[
            readings[field].notna()
            & ((readings[field] < bounds["valid_min"]) | (readings[field] > bounds["valid_max"]))
        ]
        if out_of_range.empty:
            continue
        subset = out_of_range[["station_id", "timestamp", field]].copy()
        subset = subset.rename(columns={field: "value"})
        subset["variable"] = field
        subset["valid_min"] = bounds["valid_min"]
        subset["valid_max"] = bounds["valid_max"]
        flags.append(subset)

    if not flags:
        return pd.DataFrame(
            columns=["station_id", "timestamp", "value", "variable", "valid_min", "valid_max"]
        )
    return pd.concat(flags, ignore_index=True).sort_values(["station_id", "timestamp"])


def detect_stuck_sensors(
    readings: pd.DataFrame,
    min_run_length: int = STUCK_SENSOR_MIN_RUN_LENGTH,
) -> pd.DataFrame:
    """
    Detect runs of `min_run_length` or more consecutive identical, non-null
    values for each variable/station — a classic symptom of a frozen sensor.
    """
    stuck_frames = []
    fields = [f for f in CORE_READING_FIELDS if f in readings.columns]
    result_columns = ["station_id", "variable", "value", "run_start", "run_end", "run_length"]

    for station_id, group in readings.groupby("station_id"):
        group = group.sort_values("timestamp")
        for field in fields:
            series = group[field]
            valid_mask = series.notna()

            same_as_prev = series.eq(series.shift(1)) & valid_mask & valid_mask.shift(1).fillna(False)
            run_id = (~same_as_prev).cumsum()

            # A continuously-varying field rarely repeats between consecutive
            # readings, so run_id can carry tens of thousands of single-row
            # "runs" per station/field. Summarize every run with one vectorized
            # groupby-agg call rather than iterating each run as a Python
            # object — iterating is what made this function slow.
            valid = group.loc[valid_mask, ["timestamp", field]]
            runs = valid.groupby(run_id[valid_mask]).agg(
                value=(field, "first"),
                run_start=("timestamp", "first"),
                run_end=("timestamp", "last"),
                run_length=(field, "size"),
            )
            runs = runs[runs["run_length"] >= min_run_length]
            if field in STUCK_SENSOR_ZERO_EXEMPT_FIELDS:
                runs = runs[runs["value"] != 0]
            if runs.empty:
                continue
            runs["station_id"] = station_id
            runs["variable"] = field
            stuck_frames.append(runs[result_columns])

    if not stuck_frames:
        return pd.DataFrame(columns=result_columns)
    return pd.concat(stuck_frames, ignore_index=True)


def station_status_summary(readings: pd.DataFrame) -> pd.DataFrame:
    """Share of records in each station_status per station."""
    counts = (
        readings.groupby(["station_id", "station_status"])
        .size()
        .rename("records")
        .reset_index()
    )
    totals = counts.groupby("station_id")["records"].transform("sum")
    counts["share_pct"] = (100 * counts["records"] / totals).round(1)
    return counts
