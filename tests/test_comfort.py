"""Tests for src/comfort.py: MRT proxy, wind height conversion, UTCI, categories."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.comfort import (
    classify_utci_series,
    classify_utci_value,
    compute_utci,
    estimate_mean_radiant_temperature,
    wind_speed_at_10m,
)


def test_mrt_night_is_below_air_temperature():
    mrt = estimate_mean_radiant_temperature(air_temperature_c=22.0, solar_radiation_wm2=0.0, wind_speed_ms=1.0)
    assert mrt < 22.0


def test_mrt_daytime_exceeds_air_temperature_and_scales_with_solar():
    mrt_low_sun = estimate_mean_radiant_temperature(30.0, 300.0, 1.0)
    mrt_high_sun = estimate_mean_radiant_temperature(30.0, 900.0, 1.0)
    assert mrt_low_sun > 30.0
    assert mrt_high_sun > mrt_low_sun


def test_mrt_wind_damps_solar_excess():
    calm = estimate_mean_radiant_temperature(30.0, 800.0, 0.5)
    windy = estimate_mean_radiant_temperature(30.0, 800.0, 8.0)
    assert windy < calm


def test_mrt_accepts_pandas_series_and_preserves_index():
    solar = pd.Series([0.0, 500.0], index=[10, 20])
    mrt = estimate_mean_radiant_temperature(
        pd.Series([20.0, 30.0], index=[10, 20]), solar, pd.Series([1.0, 1.0], index=[10, 20])
    )
    assert isinstance(mrt, pd.Series)
    assert list(mrt.index) == [10, 20]


def test_wind_speed_at_10m_matches_log_law_formula():
    v_measured, height, z0 = 2.0, 3.0, 0.03
    expected = v_measured * math.log(10.0 / z0) / math.log(height / z0)
    result = wind_speed_at_10m(v_measured, height, z0)
    assert result == pytest.approx(expected, rel=1e-9)


def test_wind_speed_at_10m_increases_with_height_conversion():
    # 10 m is always higher than a 3 m mast, so the wind-profile conversion
    # should always scale the measured value up, for any realistic roughness.
    for z0 in (0.03, 0.1, 0.3, 0.5):
        result = wind_speed_at_10m(2.0, 3.0, z0)
        assert result > 2.0


def test_compute_utci_moderate_conditions_close_to_air_temperature():
    # With tr == tdb and light wind/moderate humidity, UTCI should stay in
    # the same neighbourhood as the air temperature (sanity bound, not an
    # exact physical claim).
    utci = compute_utci(air_temperature_c=25.0, mean_radiant_temp_c=25.0, wind_speed_10m_ms=2.0, relative_humidity_pct=50.0)
    value = float(np.asarray(utci))
    assert 15.0 < value < 35.0


def test_compute_utci_hot_radiant_load_increases_utci():
    mild = compute_utci(30.0, 30.0, 2.0, 40.0)
    hot_radiant = compute_utci(30.0, 55.0, 2.0, 40.0)
    assert float(np.asarray(hot_radiant)) > float(np.asarray(mild))


def test_compute_utci_returns_series_for_series_input():
    tdb = pd.Series([25.0, 35.0], index=[5, 6])
    result = compute_utci(
        tdb, pd.Series([25.0, 45.0], index=[5, 6]), pd.Series([2.0, 2.0], index=[5, 6]), pd.Series([40.0, 30.0], index=[5, 6])
    )
    assert isinstance(result, pd.Series)
    assert list(result.index) == [5, 6]


def test_classify_utci_value_matches_category_table():
    assert classify_utci_value(15.0) == "No thermal stress"
    assert classify_utci_value(28.0) == "Moderate heat stress"
    assert classify_utci_value(40.0) == "Very strong heat stress"
    assert classify_utci_value(-5.0) == "Moderate cold stress"


def test_classify_utci_value_handles_missing():
    assert classify_utci_value(None) is None
    assert classify_utci_value(float("nan")) is None


def test_classify_utci_series_matches_classify_utci_value_elementwise():
    values = pd.Series([5.0, 15.0, 27.0, 34.0, 41.0])
    series_result = classify_utci_series(values)
    for value, category in zip(values, series_result):
        assert category == classify_utci_value(value)
