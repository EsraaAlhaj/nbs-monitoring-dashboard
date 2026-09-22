"""Integration tests for src/data_generator.py: reproducibility, shape, and the
core 'cooling effect' and UTCI-consistency requirements."""

from __future__ import annotations

import pandas as pd

from config.settings import RANDOM_SEED, SIMULATION_INTERVAL_MINUTES, SIMULATION_NUM_DAYS, STATIONS
from src.comfort import classify_utci_value
from src.data_generator import generate_dataset
from src.quality_checks import detect_implausible_readings, detect_stuck_sensors


def test_generate_dataset_is_reproducible_for_same_seed():
    first = generate_dataset(seed=RANDOM_SEED)
    second = generate_dataset(seed=RANDOM_SEED)
    pd.testing.assert_frame_equal(first, second)


def test_generate_dataset_different_seeds_differ():
    first = generate_dataset(seed=1)
    second = generate_dataset(seed=2)
    assert not first["air_temperature_c"].equals(second["air_temperature_c"])


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(seed=RANDOM_SEED)
    expected_rows_per_station = SIMULATION_NUM_DAYS * 24 * 60 // SIMULATION_INTERVAL_MINUTES
    assert len(df) == expected_rows_per_station * len(STATIONS)
    assert set(df["station_id"]) == {meta["station_id"] for meta in STATIONS.values()}
    for col in (
        "station_id", "timestamp", "air_temperature_c", "relative_humidity_pct",
        "wind_speed_ms", "wind_direction_deg", "solar_radiation_wm2", "rainfall_mm",
        "mean_radiant_temp_c", "utci_c", "utci_category", "station_status",
        "is_missing", "missing_fields", "is_injected_anomaly",
    ):
        assert col in df.columns


def test_green_sites_are_cooler_on_average_during_daylight():
    df = generate_dataset(seed=RANDOM_SEED)
    daylight = df[(df["solar_radiation_wm2"] > 50) & (df["is_missing"] == 0)]
    for meta in STATIONS.values():
        if meta["site_type"] != "green":
            continue
        pair = meta["pair_id"]
        reference_id = next(
            m["station_id"] for m in STATIONS.values() if m["pair_id"] == pair and m["site_type"] == "reference"
        )
        green_mean = daylight.loc[daylight["station_id"] == meta["station_id"], "air_temperature_c"].mean()
        ref_mean = daylight.loc[daylight["station_id"] == reference_id, "air_temperature_c"].mean()
        assert green_mean < ref_mean


def test_cooling_effect_scales_with_solar_radiation_not_fixed():
    df = generate_dataset(seed=RANDOM_SEED)
    green_id = STATIONS["site_a_nbs"]["station_id"]
    ref_id = STATIONS["site_a_reference"]["station_id"]

    green = df[(df["station_id"] == green_id) & (df["is_missing"] == 0)].set_index("timestamp")
    reference = df[(df["station_id"] == ref_id) & (df["is_missing"] == 0)].set_index("timestamp")
    common_index = green.index.intersection(reference.index)

    delta = reference.loc[common_index, "air_temperature_c"] - green.loc[common_index, "air_temperature_c"]
    solar = green.loc[common_index, "solar_radiation_wm2"]

    correlation = delta.corr(solar)
    assert correlation > 0.3  # cooling effect grows with solar radiation, not a flat offset


def test_utci_category_matches_utci_value_classification():
    df = generate_dataset(seed=RANDOM_SEED)
    sample = df.dropna(subset=["utci_c", "utci_category"]).sample(n=200, random_state=0)
    for _, row in sample.iterrows():
        assert row["utci_category"] == classify_utci_value(row["utci_c"])


def test_injected_anomalies_are_detectable():
    df = generate_dataset(seed=RANDOM_SEED)
    assert df["is_injected_anomaly"].sum() >= 2

    implausible = detect_implausible_readings(df)
    stuck = detect_stuck_sensors(df)
    assert len(implausible) >= 1
    assert len(stuck) >= 1


def test_down_stations_have_all_core_fields_null():
    df = generate_dataset(seed=RANDOM_SEED)
    down = df[df["station_status"].isin(["maintenance", "offline"])]
    core_fields = ["air_temperature_c", "relative_humidity_pct", "wind_speed_ms", "solar_radiation_wm2"]
    assert down[core_fields].isna().all().all()


def test_active_non_missing_rows_have_complete_core_fields():
    df = generate_dataset(seed=RANDOM_SEED)
    clean = df[(df["station_status"] == "active") & (df["is_missing"] == 0)]
    assert clean["air_temperature_c"].notna().all()
    assert clean["utci_c"].notna().all()
