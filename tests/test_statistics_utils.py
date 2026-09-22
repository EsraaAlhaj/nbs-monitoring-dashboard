"""Tests for src/statistics_utils.py using small, hand-constructed reading sets."""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import STATION_PAIRS, STATIONS
from src.statistics_utils import (
    attach_station_metadata,
    build_pair_comparison,
    count_cooler_hours,
    hottest_periods,
    hourly_profile,
    site_type_label,
    station_name,
    summary_statistics,
)

PAIR_ID = "pair_a"
GREEN_ID = STATIONS[STATION_PAIRS[PAIR_ID]["green"]]["station_id"]
REF_ID = STATIONS[STATION_PAIRS[PAIR_ID]["reference"]]["station_id"]


def _make_readings() -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-01 00:00", periods=4, freq="15min")
    rows = []
    # Green is cooler at every timestamp except the last, where it's warmer.
    green_temps = [20.0, 22.0, 30.0, 26.0]
    ref_temps = [21.0, 24.0, 33.0, 24.0]
    for ts, g, r in zip(timestamps, green_temps, ref_temps):
        rows.append({"station_id": GREEN_ID, "timestamp": ts, "air_temperature_c": g})
        rows.append({"station_id": REF_ID, "timestamp": ts, "air_temperature_c": r})
    return pd.DataFrame(rows)


def test_build_pair_comparison_computes_delta():
    readings = _make_readings()
    comparison = build_pair_comparison(readings, PAIR_ID, variable="air_temperature_c")
    assert list(comparison["green"]) == [20.0, 22.0, 30.0, 26.0]
    assert list(comparison["reference"]) == [21.0, 24.0, 33.0, 24.0]
    assert comparison["delta"].tolist() == pytest.approx([-1.0, -2.0, -3.0, 2.0])


def test_count_cooler_hours_counts_and_shares():
    readings = _make_readings()
    comparison = build_pair_comparison(readings, PAIR_ID, variable="air_temperature_c")
    stats = count_cooler_hours(comparison)
    assert stats["total_records"] == 4
    assert stats["cooler_records"] == 3
    assert stats["warmer_records"] == 1
    assert stats["cooler_share_pct"] == pytest.approx(75.0)
    assert stats["cooler_hours"] == pytest.approx(round(3 * 0.25, 1))
    assert stats["max_cooling_c"] == pytest.approx(3.0)


def test_hottest_periods_orders_by_reference_temperature():
    readings = _make_readings()
    comparison = build_pair_comparison(readings, PAIR_ID, variable="air_temperature_c")
    hottest = hottest_periods(comparison, top_n=2)
    assert len(hottest) == 2
    assert hottest.iloc[0]["reference"] == 33.0
    assert hottest.iloc[0]["cooling_effect_c"] == pytest.approx(3.0)
    assert set(hottest["reference"]) == {33.0, 24.0}


def test_hourly_profile_averages_by_hour():
    readings = _make_readings()
    comparison = build_pair_comparison(readings, PAIR_ID, variable="air_temperature_c")
    profile = hourly_profile(comparison)
    assert len(profile) == 24
    hour0 = profile[profile["hour"] == 0]
    # All four records fall within hour 0 (00:00-00:45).
    assert hour0["delta"].iloc[0] == pytest.approx((-1.0 - 2.0 - 3.0 + 2.0) / 4)


def test_summary_statistics_per_station():
    readings = _make_readings()
    stats = summary_statistics(readings, "air_temperature_c")
    green_row = stats[stats["station_id"] == GREEN_ID].iloc[0]
    assert green_row["count"] == 4
    assert green_row["min"] == 20.0
    assert green_row["max"] == 30.0


def test_site_type_label_and_station_name_lookup():
    assert site_type_label(GREEN_ID) == "green"
    assert site_type_label(REF_ID) == "reference"
    assert station_name(GREEN_ID) == STATIONS[STATION_PAIRS[PAIR_ID]["green"]]["name"]


def test_attach_station_metadata_adds_expected_columns():
    readings = _make_readings()
    enriched = attach_station_metadata(readings)
    assert {"station_name", "site_type", "pair_id"}.issubset(enriched.columns)
    assert set(enriched.loc[enriched["station_id"] == GREEN_ID, "site_type"]) == {"green"}
