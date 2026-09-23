"""
Synthetic demonstration data generator.

Produces a 30-day, 15-minute-interval synthetic dataset for the four
stations defined in config.settings.STATIONS. ALL VALUES ARE SIMULATED —
see the module-level disclaimers in config.settings and README.md.

Design summary
---------------
1. A single shared "regional" weather backdrop (air temperature, humidity,
   wind, solar radiation, rainfall) is generated once per timestamp — the
   four stations sit within ~1-2 km of each other, so they experience the
   same synoptic weather. Day-to-day variability is layered in at several
   levels so the pattern never repeats identically: Tmax/Tmin/humidity/wind
   baselines drift via a bounded random walk, AND the diurnal curve's
   trough/peak timing itself carries a small day-to-day jitter (see
   TROUGH_HOUR_JITTER_STD_H / PEAK_HOUR_JITTER_STD_H).
2. Each station then gets small independent spatial/sensor noise.
3. "green" sites additionally get a physically-motivated adjustment
   relative to the regional backdrop:
     - shading reduces the solar radiation reaching the sensor and, in
       turn, estimated MRT and air temperature (cooling scales with the
       *current* solar radiation, i.e. it is near zero at night and largest
       around midday — this is the "cooling effect varies by time of day
       and solar radiation" requirement, not a fixed offset);
     - evapotranspiration adds a secondary, smaller cooling + humidity
       term that also scales with solar radiation and dryness;
     - canopy sheltering reduces local wind speed (also expressed as a
       larger aerodynamic roughness length used later for the 10 m wind
       conversion feeding UTCI);
     - the day and night cooling terms each also drift slowly day to day
       (COOLING_STRENGTH_* / NIGHT_COOLING_OFFSET_*), so the green/reference
       gap varies naturally instead of following a fixed model: daytime
       occasionally sees a much weaker (but still generally positive)
       cooling effect, and a minority of nights drift close enough to zero
       to produce a brief, physically plausible "crossing" where the green
       site reads a touch warmer than the reference site.
4. MRT and UTCI are then computed per station from ITS OWN simulated
   air temperature / solar radiation / wind / humidity.
5. Station status (active/maintenance/offline), transient sensor dropouts,
   and a handful of intentionally injected QA anomalies (implausible
   values, a stuck sensor run) are layered on top, so the Data Quality page
   has real issues to detect. All randomness is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.settings import (
    CORE_READING_FIELDS,
    RANDOM_SEED,
    SIMULATION_INTERVAL_MINUTES,
    SIMULATION_NUM_DAYS,
    SIMULATION_START_DATE,
    SITE_TYPE_GREEN,
    STATIONS,
)
from src.comfort import classify_utci_series, compute_utci, estimate_mean_radiant_temperature, wind_speed_at_10m

# ---------------------------------------------------------------------------
# Tunable model constants (kept together here, not scattered through the
# functions below, so the magnitude of every synthetic effect is easy to
# review in one place).
# ---------------------------------------------------------------------------
DAILY_TMAX_MEAN_C, DAILY_TMAX_STD_C = 34.5, 1.6
DAILY_TMAX_STEP_STD_C = 0.8
DAILY_TMAX_BOUNDS_C = (29.0, 41.0)

DAILY_TMIN_MEAN_C, DAILY_TMIN_STD_C = 20.5, 1.3
DAILY_TMIN_STEP_STD_C = 0.6
DAILY_TMIN_BOUNDS_C = (15.0, 24.0)

TROUGH_HOUR, PEAK_HOUR = 5.5, 15.5
# Small day-to-day jitter on when the trough/peak actually falls, so the
# diurnal curve is not the exact same shape every single day (amplitude
# already varies day to day via the Tmax/Tmin random walks above).
TROUGH_HOUR_JITTER_STD_H = 0.4
PEAK_HOUR_JITTER_STD_H = 0.6
TROUGH_HOUR_BOUNDS_H = (4.0, 7.0)
PEAK_HOUR_BOUNDS_H = (14.0, 17.5)
TEMP_NOISE_STD_C = 0.25

DAILY_RH_BASELINE_MEAN_PCT, DAILY_RH_BASELINE_STD_PCT = 38.0, 7.0
DAILY_RH_STEP_STD_PCT = 3.0
DAILY_RH_BOUNDS_PCT = (18.0, 68.0)
RH_DIURNAL_SWING_PCT = 16.0
RH_NOISE_STD_PCT = 1.5

DAILY_WIND_BASELINE_MEAN_MS, DAILY_WIND_BASELINE_STD_MS = 2.1, 0.4
DAILY_WIND_STEP_STD_MS = 0.25
DAILY_WIND_BOUNDS_MS = (0.8, 3.8)
WIND_DIURNAL_AMPLITUDE_MS = 2.0
WIND_NOISE_STD_MS = 0.2
WIND_MIN_MS = 0.15

PREVAILING_WIND_DEG = 290.0
WIND_DIR_DAILY_DRIFT_STD_DEG = 12.0
WIND_DIR_NOISE_STD_DEG = 14.0

SUNRISE_HOUR, SUNSET_HOUR = 5.5, 19.5
PEAK_SOLAR_MEAN_WM2, PEAK_SOLAR_STD_WM2 = 970.0, 20.0
HAZY_DAY_PROBABILITY = 0.10
HAZY_DAY_FACTOR_RANGE = (0.72, 0.90)
SOLAR_NOISE_REL_STD = 0.02

RAIN_EVENT_PROBABILITY = 0.65  # chance the 30-day window contains one brief shower
RAIN_EVENT_MAX_INTERVALS = 4
RAIN_EVENT_INTENSITY_RANGE_MM = (0.2, 3.0)

STATION_SPATIAL_NOISE = {
    "air_temperature_c": 0.15,
    "relative_humidity_pct": 1.0,
    "wind_speed_ms": 0.10,
    "solar_radiation_wm2": 0.01,  # relative
}

# Green-site cooling mechanism magnitudes.
SHADE_MAX_COOLING_C = 2.6
ET_MAX_COOLING_C = 1.1
NIGHT_RESIDUAL_COOLING_MEAN_C = 0.4
NIGHT_RESIDUAL_COOLING_STD_C = 0.3
GREEN_HUMIDITY_BOOST_MAX_PCT = 9.0
GREEN_CANOPY_TRANSMITTANCE = 0.65  # fraction of regional solar reaching a shaded sensor
GREEN_WIND_SHELTER_FACTOR = 0.68

# Day-to-day variability in the daytime cooling *strength*: a slowly
# drifting multiplier (random walk, not iid) on the physical shading + ET
# cooling estimate. Mostly close to 1.0 (the nominal physical estimate) but
# occasionally drifts toward its lower bound for a run of days, giving
# "occasional very small [green/reference] differences" during the day
# without ever fully erasing the expected daytime cooling.
COOLING_STRENGTH_MEAN, COOLING_STRENGTH_STD = 1.0, 0.15
COOLING_STRENGTH_STEP_STD = 0.07
COOLING_STRENGTH_BOUNDS = (0.55, 1.35)

# Night-time cooling gets its own slowly drifting nightly baseline (added to
# NIGHT_RESIDUAL_COOLING_MEAN_C) which can occasionally dip slightly below
# zero — a brief, physically plausible "nighttime crossing" where the green
# site reads a touch warmer than the reference site for a stretch of hours —
# plus smoothed (not iid) intra-night noise, so the effect survives hourly
# averaging instead of cancelling out.
NIGHT_COOLING_OFFSET_STD_C = 0.3
NIGHT_COOLING_OFFSET_STEP_STD_C = 0.12
NIGHT_COOLING_OFFSET_BOUNDS_C = (-0.45, 0.45)

# Station-status / missing-data injection.
MAINTENANCE_WINDOW_HOURS = (2, 5)
OFFLINE_WINDOW_HOURS = (1, 6)
TRANSIENT_DROPOUT_PROBABILITY = 0.004
STUCK_SENSOR_RUN_LENGTH = 12  # 3 hours at 15-min resolution

DEPENDENCY_FIELDS = [
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "solar_radiation_wm2",
]


def _time_index() -> pd.DatetimeIndex:
    start = pd.Timestamp(SIMULATION_START_DATE)
    periods = int(SIMULATION_NUM_DAYS * 24 * 60 / SIMULATION_INTERVAL_MINUTES)
    return pd.date_range(start, periods=periods, freq=f"{SIMULATION_INTERVAL_MINUTES}min")


def _random_walk(rng: np.random.Generator, n_days: int, mean: float, std: float, step_std: float, bounds: tuple[float, float]) -> np.ndarray:
    values = np.empty(n_days)
    values[0] = np.clip(rng.normal(mean, std), *bounds)
    for d in range(1, n_days):
        values[d] = np.clip(values[d - 1] + rng.normal(0, step_std), *bounds)
    return values


def _diurnal_shape(
    hour_frac: np.ndarray,
    trough_hour: np.ndarray | float = TROUGH_HOUR,
    peak_hour: np.ndarray | float = PEAK_HOUR,
) -> np.ndarray:
    """Asymmetric smooth diurnal cycle in [-1, 1]: trough at `trough_hour`,
    peak at `peak_hour` (each defaults to the module-level TROUGH_HOUR /
    PEAK_HOUR). Both may instead be passed as per-timestamp arrays (same
    shape as `hour_frac`) so the trough/peak timing can carry a small
    day-to-day jitter instead of landing at the exact same hour every day."""
    h = np.asarray(hour_frac, dtype=float)
    trough_hour = np.broadcast_to(np.asarray(trough_hour, dtype=float), h.shape)
    peak_hour = np.broadcast_to(np.asarray(peak_hour, dtype=float), h.shape)
    shape = np.empty_like(h)

    rising_span = peak_hour - trough_hour
    falling_span = 24 - peak_hour + trough_hour

    is_rising = (h >= trough_hour) & (h < peak_hour)
    frac_rise = (h[is_rising] - trough_hour[is_rising]) / rising_span[is_rising]
    shape[is_rising] = -np.cos(np.pi * frac_rise)

    is_falling = ~is_rising
    h_shift = np.where(h[is_falling] < trough_hour[is_falling], h[is_falling] + 24, h[is_falling])
    frac_fall = (h_shift - peak_hour[is_falling]) / falling_span[is_falling]
    shape[is_falling] = np.cos(np.pi * frac_fall)

    return shape


def _smooth_noise(rng: np.random.Generator, n: int, std: float, window: int = 3) -> np.ndarray:
    """Gaussian noise passed through a short moving-average filter, then
    rescaled back to `std` (the averaging alone would shrink the variance)."""
    raw = rng.normal(0, std, size=n)
    if window <= 1:
        return raw
    kernel = np.ones(window) / window
    padded = np.pad(raw, (window // 2, window - 1 - window // 2), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed_std = np.std(smoothed)
    if smoothed_std < 1e-9:
        return smoothed
    return smoothed * (std / smoothed_std)


def generate_regional_weather(rng: np.random.Generator) -> pd.DataFrame:
    idx = _time_index()
    n = len(idx)
    n_days = SIMULATION_NUM_DAYS
    day_of_period = ((idx - idx[0]).days).to_numpy()
    hour_frac = idx.hour + idx.minute / 60.0

    daily_tmax = _random_walk(rng, n_days, DAILY_TMAX_MEAN_C, DAILY_TMAX_STD_C, DAILY_TMAX_STEP_STD_C, DAILY_TMAX_BOUNDS_C)
    daily_tmin = _random_walk(rng, n_days, DAILY_TMIN_MEAN_C, DAILY_TMIN_STD_C, DAILY_TMIN_STEP_STD_C, DAILY_TMIN_BOUNDS_C)
    daily_rh_baseline = _random_walk(rng, n_days, DAILY_RH_BASELINE_MEAN_PCT, DAILY_RH_BASELINE_STD_PCT, DAILY_RH_STEP_STD_PCT, DAILY_RH_BOUNDS_PCT)
    daily_wind_baseline = _random_walk(rng, n_days, DAILY_WIND_BASELINE_MEAN_MS, DAILY_WIND_BASELINE_STD_MS, DAILY_WIND_STEP_STD_MS, DAILY_WIND_BOUNDS_MS)
    daily_peak_solar = np.clip(rng.normal(PEAK_SOLAR_MEAN_WM2, PEAK_SOLAR_STD_WM2, size=n_days), 800, 1050)
    hazy_day = rng.random(n_days) < HAZY_DAY_PROBABILITY
    haze_factor = np.where(hazy_day, rng.uniform(*HAZY_DAY_FACTOR_RANGE, size=n_days), 1.0)
    wind_dir_daily = PREVAILING_WIND_DEG + np.cumsum(rng.normal(0, WIND_DIR_DAILY_DRIFT_STD_DEG, size=n_days))
    daily_trough_hour = np.clip(
        TROUGH_HOUR + rng.normal(0, TROUGH_HOUR_JITTER_STD_H, size=n_days), *TROUGH_HOUR_BOUNDS_H
    )
    daily_peak_hour = np.clip(
        PEAK_HOUR + rng.normal(0, PEAK_HOUR_JITTER_STD_H, size=n_days), *PEAK_HOUR_BOUNDS_H
    )

    tmax = daily_tmax[day_of_period]
    tmin = daily_tmin[day_of_period]
    rh_base = daily_rh_baseline[day_of_period]
    wind_base = daily_wind_baseline[day_of_period]
    peak_solar = daily_peak_solar[day_of_period]
    haze = haze_factor[day_of_period]
    wind_dir_base = wind_dir_daily[day_of_period]
    trough_hour = daily_trough_hour[day_of_period]
    peak_hour = daily_peak_hour[day_of_period]

    # Reused for temperature, humidity, and wind below so the whole diurnal
    # cycle shifts together on a given day (physically coherent) rather than
    # each variable jittering independently.
    shape = _diurnal_shape(hour_frac, trough_hour, peak_hour)
    daily_mean = (tmax + tmin) / 2
    daily_half_range = (tmax - tmin) / 2
    air_temp = daily_mean + daily_half_range * shape + _smooth_noise(rng, n, TEMP_NOISE_STD_C)

    rh = rh_base - RH_DIURNAL_SWING_PCT * shape + _smooth_noise(rng, n, RH_NOISE_STD_PCT)
    rh = np.clip(rh, 8.0, 92.0)

    wind_shape = np.clip(shape, 0, None)
    wind_speed = wind_base + WIND_DIURNAL_AMPLITUDE_MS * wind_shape + _smooth_noise(rng, n, WIND_NOISE_STD_MS)
    wind_speed = np.clip(wind_speed, WIND_MIN_MS, None)

    wind_dir = (wind_dir_base + rng.normal(0, WIND_DIR_NOISE_STD_DEG, size=n)) % 360.0

    daylight = (hour_frac >= SUNRISE_HOUR) & (hour_frac <= SUNSET_HOUR)
    solar_shape = np.zeros(n)
    daylen = SUNSET_HOUR - SUNRISE_HOUR
    solar_shape[daylight] = np.clip(
        np.sin(np.pi * (hour_frac[daylight] - SUNRISE_HOUR) / daylen), 0, None
    ) ** 1.1
    solar = peak_solar * solar_shape * haze
    solar_noise = rng.normal(1.0, SOLAR_NOISE_REL_STD, size=n)
    solar = np.where(daylight, np.clip(solar * solar_noise, 0, None), 0.0)

    rainfall = np.zeros(n)
    if rng.random() < RAIN_EVENT_PROBABILITY:
        # A single brief late-day convective shower on a random day.
        event_day = rng.integers(0, n_days)
        candidate_idx = np.where(
            (day_of_period == event_day) & (hour_frac >= 15.0) & (hour_frac <= 20.0)
        )[0]
        if len(candidate_idx) > 0:
            start = rng.choice(candidate_idx)
            duration = rng.integers(1, RAIN_EVENT_MAX_INTERVALS + 1)
            end = min(start + duration, n)
            rainfall[start:end] = rng.uniform(*RAIN_EVENT_INTENSITY_RANGE_MM, size=end - start)

    return pd.DataFrame(
        {
            "timestamp": idx,
            "day_index": day_of_period,
            "hour_frac": hour_frac,
            "air_temperature_c": air_temp,
            "relative_humidity_pct": rh,
            "wind_speed_ms": wind_speed,
            "wind_direction_deg": wind_dir,
            "solar_radiation_wm2": solar,
            "rainfall_mm": rainfall,
        }
    )


def _apply_site_adjustment(regional: pd.DataFrame, station_key: str, rng: np.random.Generator) -> pd.DataFrame:
    meta = STATIONS[station_key]
    n = len(regional)
    site = regional.copy()

    # Independent micro-site / sensor noise, applied to every station.
    site["air_temperature_c"] += rng.normal(0, STATION_SPATIAL_NOISE["air_temperature_c"], n)
    site["relative_humidity_pct"] = np.clip(
        site["relative_humidity_pct"] + rng.normal(0, STATION_SPATIAL_NOISE["relative_humidity_pct"], n), 0, 100
    )
    site["wind_speed_ms"] = np.clip(
        site["wind_speed_ms"] + rng.normal(0, STATION_SPATIAL_NOISE["wind_speed_ms"], n), WIND_MIN_MS, None
    )
    site["solar_radiation_wm2"] = np.clip(
        site["solar_radiation_wm2"] * (1 + rng.normal(0, STATION_SPATIAL_NOISE["solar_radiation_wm2"], n)), 0, None
    )

    if meta["site_type"] == SITE_TYPE_GREEN:
        day_idx = regional["day_index"].to_numpy()
        solar_frac = np.clip(regional["solar_radiation_wm2"].to_numpy() / PEAK_SOLAR_MEAN_WM2, 0, None)
        is_day = regional["solar_radiation_wm2"].to_numpy() > 5.0

        # Shading: sensor sits within/under canopy, so it "sees" less solar
        # radiation directly, AND experiences shading-driven air cooling
        # that scales with how much sun there currently is.
        canopy_noise = rng.normal(1.0, 0.03, n)
        site["solar_radiation_wm2"] = np.where(
            is_day,
            regional["solar_radiation_wm2"].to_numpy() * GREEN_CANOPY_TRANSMITTANCE * canopy_noise,
            site["solar_radiation_wm2"],
        )
        site["solar_radiation_wm2"] = np.clip(site["solar_radiation_wm2"], 0, None)

        # Slowly drifting day-to-day cooling-strength multiplier (see
        # constants above) so the green/reference gap is not a fixed
        # physical constant day after day — most days sit close to the
        # nominal physical estimate, but the effect occasionally weakens for
        # a run of days.
        cooling_strength = _random_walk(
            rng, SIMULATION_NUM_DAYS, COOLING_STRENGTH_MEAN, COOLING_STRENGTH_STD,
            COOLING_STRENGTH_STEP_STD, COOLING_STRENGTH_BOUNDS,
        )[day_idx]

        shading_cooling = SHADE_MAX_COOLING_C * solar_frac
        dryness = 1 - regional["relative_humidity_pct"].to_numpy() / 100.0
        et_cooling = ET_MAX_COOLING_C * (solar_frac ** 0.7) * np.clip(dryness, 0.15, 1.0) ** 0.5
        day_cooling = (shading_cooling + et_cooling) * cooling_strength + rng.normal(0, 0.15, n)

        # Nightly baseline also drifts slowly day to day (independent random
        # walk, can dip slightly below zero) plus smoothed — not
        # independent-per-reading — noise, so an occasional weak night
        # survives hourly averaging as a genuine, visible "nighttime
        # crossing" rather than being cancelled out by resampling.
        nightly_offset = _random_walk(
            rng, SIMULATION_NUM_DAYS, 0.0, NIGHT_COOLING_OFFSET_STD_C,
            NIGHT_COOLING_OFFSET_STEP_STD_C, NIGHT_COOLING_OFFSET_BOUNDS_C,
        )[day_idx]
        night_cooling = (
            NIGHT_RESIDUAL_COOLING_MEAN_C
            + nightly_offset
            + _smooth_noise(rng, n, NIGHT_RESIDUAL_COOLING_STD_C, window=5)
        )

        cooling = np.where(is_day, day_cooling, night_cooling)
        site["air_temperature_c"] = site["air_temperature_c"] - cooling
        site["_cooling_effect_c"] = cooling

        humidity_boost = GREEN_HUMIDITY_BOOST_MAX_PCT * solar_frac * 0.6 + 1.5
        site["relative_humidity_pct"] = np.clip(
            site["relative_humidity_pct"] + humidity_boost + rng.normal(0, 1.0, n), 0, 100
        )

        site["wind_speed_ms"] = np.clip(
            site["wind_speed_ms"] * GREEN_WIND_SHELTER_FACTOR, WIND_MIN_MS, None
        )
    else:
        site["_cooling_effect_c"] = 0.0

    site["station_id"] = meta["station_id"]
    site["station_key"] = station_key
    return site


def _compute_derived_comfort_fields(site: pd.DataFrame, station_key: str) -> pd.DataFrame:
    meta = STATIONS[station_key]
    site = site.copy()

    site["mean_radiant_temp_c"] = estimate_mean_radiant_temperature(
        site["air_temperature_c"], site["solar_radiation_wm2"], site["wind_speed_ms"]
    )

    wind_10m = wind_speed_at_10m(
        site["wind_speed_ms"].to_numpy(),
        meta["measurement_height_m"],
        meta["roughness_length_m"],
    )
    site["_wind_10m_ms"] = wind_10m

    site["utci_c"] = compute_utci(
        site["air_temperature_c"], site["mean_radiant_temp_c"], wind_10m, site["relative_humidity_pct"]
    )
    site["utci_category"] = classify_utci_series(site["utci_c"]).astype(object)
    return site


def _inject_status_and_missingness(site: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(site)
    site = site.copy()
    site["station_status"] = "active"
    site["is_missing"] = 0
    site["missing_fields"] = ""
    site["is_injected_anomaly"] = 0

    # One planned maintenance window.
    maint_start = rng.integers(0, n - 24)
    maint_len = rng.integers(
        int(MAINTENANCE_WINDOW_HOURS[0] * 4), int(MAINTENANCE_WINDOW_HOURS[1] * 4) + 1
    )
    maint_end = min(maint_start + maint_len, n)
    site.loc[site.index[maint_start:maint_end], "station_status"] = "maintenance"

    # One unplanned offline window, not overlapping the maintenance window.
    for _ in range(10):
        off_start = rng.integers(0, n - 24)
        off_len = rng.integers(int(OFFLINE_WINDOW_HOURS[0] * 4), int(OFFLINE_WINDOW_HOURS[1] * 4) + 1)
        off_end = min(off_start + off_len, n)
        if off_end < maint_start or off_start > maint_end:
            site.loc[site.index[off_start:off_end], "station_status"] = "offline"
            break

    down_mask = site["station_status"].isin(["maintenance", "offline"])
    site.loc[down_mask, [*CORE_READING_FIELDS]] = np.nan
    site.loc[down_mask, "is_missing"] = 1
    site.loc[down_mask, "missing_fields"] = "all"

    # Sparse transient single-field dropouts while nominally active.
    active_idx = site.index[~down_mask]
    transient_mask = rng.random(len(active_idx)) < TRANSIENT_DROPOUT_PROBABILITY
    transient_positions = active_idx[transient_mask]
    glitch_fields = [
        "air_temperature_c",
        "relative_humidity_pct",
        "wind_speed_ms",
        "wind_direction_deg",
        "solar_radiation_wm2",
    ]
    for pos in transient_positions:
        n_fields = rng.integers(1, 3)
        chosen = rng.choice(glitch_fields, size=n_fields, replace=False)
        site.loc[pos, list(chosen)] = np.nan
        site.loc[pos, "is_missing"] = 1
        site.loc[pos, "missing_fields"] = ",".join(sorted(chosen))

    # Cascade: if any comfort-model input is missing, MRT/UTCI can't be
    # computed for that row either — a real station could not report them.
    cascade_mask = site[DEPENDENCY_FIELDS].isna().any(axis=1)
    site.loc[cascade_mask, ["mean_radiant_temp_c", "utci_c"]] = np.nan
    site.loc[cascade_mask & ~down_mask, "utci_category"] = np.nan
    site.loc[down_mask, "utci_category"] = np.nan

    return site


def _inject_qa_anomalies(site: pd.DataFrame, rng: np.random.Generator, add_implausible: bool, add_stuck: bool) -> pd.DataFrame:
    """
    Deliberately inject a small number of clearly-artificial anomalies so
    the Data Quality page's detectors have real issues to surface in this
    demonstration. Flagged via is_injected_anomaly for traceability.
    """
    site = site.copy()
    active_idx = site.index[site["station_status"] == "active"]
    if len(active_idx) == 0:
        return site

    if add_implausible:
        bad_rh_pos = rng.choice(active_idx)
        site.loc[bad_rh_pos, "relative_humidity_pct"] = 104.0
        site.loc[bad_rh_pos, "is_injected_anomaly"] = 1

        night_idx = active_idx[site.loc[active_idx, "solar_radiation_wm2"] == 0]
        if len(night_idx) > 0:
            bad_solar_pos = rng.choice(night_idx)
            site.loc[bad_solar_pos, "solar_radiation_wm2"] = -8.0
            site.loc[bad_solar_pos, "is_injected_anomaly"] = 1

    if add_stuck and len(active_idx) > STUCK_SENSOR_RUN_LENGTH + 5:
        start_pos = rng.integers(0, len(active_idx) - STUCK_SENSOR_RUN_LENGTH - 1)
        run_positions = active_idx[start_pos : start_pos + STUCK_SENSOR_RUN_LENGTH]
        stuck_value = float(site.loc[run_positions[0], "wind_speed_ms"])
        site.loc[run_positions, "wind_speed_ms"] = stuck_value
        site.loc[run_positions, "is_injected_anomaly"] = 1

    return site


def generate_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate the full synthetic dataset for all four stations. Deterministic given `seed`."""
    master_rng = np.random.default_rng(seed)
    regional = generate_regional_weather(np.random.default_rng(master_rng.integers(0, 2**32 - 1)))

    station_keys = list(STATIONS.keys())
    all_sites = []
    for i, key in enumerate(station_keys):
        station_rng = np.random.default_rng(master_rng.integers(0, 2**32 - 1))
        site = _apply_site_adjustment(regional, key, station_rng)
        site = _compute_derived_comfort_fields(site, key)
        site = _inject_status_and_missingness(site, station_rng)
        # Inject the demonstration anomalies on just one station each, so
        # the rest of the dataset stays "clean" and the effect is legible.
        site = _inject_qa_anomalies(
            site, station_rng, add_implausible=(i == 0), add_stuck=(i == 2)
        )
        all_sites.append(site)

    full = pd.concat(all_sites, ignore_index=True)
    full = full.drop(columns=["day_index", "hour_frac", "_cooling_effect_c", "_wind_10m_ms", "station_key"], errors="ignore")

    for field, meta in {
        "air_temperature_c": 1,
        "relative_humidity_pct": 1,
        "wind_speed_ms": 2,
        "wind_direction_deg": 0,
        "solar_radiation_wm2": 0,
        "rainfall_mm": 2,
        "mean_radiant_temp_c": 1,
        "utci_c": 1,
    }.items():
        full[field] = full[field].round(meta)

    full = full.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return full
