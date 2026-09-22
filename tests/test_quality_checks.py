"""Tests for src/quality_checks.py using small, hand-constructed reading sets."""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import STATIONS
from src.quality_checks import (
    completeness_by_station,
    detect_implausible_readings,
    detect_stuck_sensors,
    detect_time_gaps,
    station_status_summary,
)

STATION_ID = STATIONS["site_a_nbs"]["station_id"]


def _base_readings(n: int, freq: str = "15min") -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-01 00:00", periods=n, freq=freq)
    return pd.DataFrame(
        {
            "station_id": [STATION_ID] * n,
            "timestamp": timestamps,
            "air_temperature_c": [25.0] * n,
            "relative_humidity_pct": [40.0] * n,
            "wind_speed_ms": [2.0] * n,
            "wind_direction_deg": [180.0] * n,
            "solar_radiation_wm2": [100.0] * n,
            "rainfall_mm": [0.0] * n,
            "mean_radiant_temp_c": [26.0] * n,
            "utci_c": [24.0] * n,
            "station_status": ["active"] * n,
            "is_missing": [0] * n,
        }
    )


def test_completeness_by_station_full_period():
    readings = _base_readings(96)  # exactly one day at 15-min resolution
    period_start = pd.Timestamp("2026-07-01 00:00")
    period_end = pd.Timestamp("2026-07-01 23:45")
    result = completeness_by_station(readings, period_start, period_end)
    row = result[result["station_id"] == STATION_ID].iloc[0]
    assert row["expected_records"] == 96
    assert row["actual_records"] == 96
    assert row["completeness_pct"] == 100.0


def test_completeness_by_station_detects_missing_records():
    readings = _base_readings(96)
    readings.loc[readings.index[:10], "is_missing"] = 1
    period_start = pd.Timestamp("2026-07-01 00:00")
    period_end = pd.Timestamp("2026-07-01 23:45")
    result = completeness_by_station(readings, period_start, period_end)
    row = result[result["station_id"] == STATION_ID].iloc[0]
    assert row["complete_records"] == 86
    assert row["completeness_pct"] == pytest.approx(round(100 * 86 / 96, 1))


def test_detect_time_gaps_finds_artificial_gap():
    readings = _base_readings(10)
    # Remove two consecutive records in the middle to create a 45-minute gap (> the 30-min threshold).
    readings = readings.drop(readings.index[5:7]).reset_index(drop=True)
    gaps = detect_time_gaps(readings)
    assert len(gaps) == 1
    assert gaps.iloc[0]["gap_minutes"] == 45.0


def test_detect_time_gaps_no_gap_for_regular_series():
    readings = _base_readings(20)
    gaps = detect_time_gaps(readings)
    assert gaps.empty


def test_detect_implausible_readings_flags_out_of_range_value():
    readings = _base_readings(5)
    readings.loc[readings.index[2], "relative_humidity_pct"] = 104.0
    flagged = detect_implausible_readings(readings)
    assert len(flagged) == 1
    assert flagged.iloc[0]["variable"] == "relative_humidity_pct"
    assert flagged.iloc[0]["value"] == 104.0


def test_detect_implausible_readings_clean_data_returns_empty():
    readings = _base_readings(5)
    flagged = detect_implausible_readings(readings)
    assert flagged.empty


def test_detect_stuck_sensors_respects_minimum_run_length():
    readings = _base_readings(20)
    values = [2.0 + 0.1 * i for i in range(20)]  # all distinct, no accidental runs
    values[2:5] = [9.9, 9.9, 9.9]  # run of length 3 (below threshold)
    values[10:18] = [7.7] * 8  # run of length 8 (meets threshold)
    readings["wind_speed_ms"] = values
    stuck = detect_stuck_sensors(readings, min_run_length=8)
    matches = stuck[stuck["variable"] == "wind_speed_ms"]
    assert len(matches) == 1
    assert matches.iloc[0]["run_length"] == 8
    assert matches.iloc[0]["value"] == 7.7


def test_station_status_summary_shares_sum_to_100():
    readings = _base_readings(10)
    readings.loc[readings.index[:3], "station_status"] = "offline"
    summary = station_status_summary(readings)
    total_share = summary[summary["station_id"] == STATION_ID]["share_pct"].sum()
    assert total_share == pytest.approx(100.0)
