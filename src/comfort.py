"""
Thermal-comfort estimation: Mean Radiant Temperature (MRT) proxy, UTCI via
the `pythermalcomfort` library, wind height conversion, and UTCI stress
category classification.

IMPORTANT — these are simulated-data estimates, not field measurements
=========================================================================
This project has no globe thermometer, no six-directional radiation
budget instrument, and no live weather stations. `estimate_mean_radiant_
temperature` below is a deliberately simple, transparent PROXY model, not
a validated radiant-flux calculation (e.g. not ISO 7726, not SOLWEIG, not
RayMan). UTCI computed from it inherits the same limitation. Every page
that displays MRT or UTCI must show this caveat to the user — see
config.settings.ESTIMATE_DISCLAIMER and src/styling.render_estimate_note.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pythermalcomfort.models import utci as _pythermalcomfort_utci

from config.settings import UTCI_CATEGORIES

# ---------------------------------------------------------------------------
# Mean Radiant Temperature — simplified proxy
# ---------------------------------------------------------------------------
# Coefficients chosen to reproduce the general order of magnitude reported
# in outdoor thermal-comfort literature for hot, low-latitude, low-wind,
# clear-sky conditions (sunlit Tmrt often 20-30+ degC above Ta at full sun;
# shaded/night Tmrt close to Ta). See e.g. Thorsson et al. (2007) and the
# SOLWEIG model background (Lindberg et al., 2008) for the general pattern
# this proxy is loosely reproducing — NOT for the coefficients themselves,
# which are illustrative, not fitted to any observed dataset.
REFERENCE_PEAK_SOLAR_WM2 = 1000.0
SOLAR_TO_MRT_COEFFICIENT_C = 28.0
WIND_DAMPING_PER_MS = 0.10
NIGHT_LONGWAVE_LOSS_C = 1.5


def estimate_mean_radiant_temperature(
    air_temperature_c,
    solar_radiation_wm2,
    wind_speed_ms,
):
    """
    Simplified MRT proxy from standard meteorological variables.

    Daytime: MRT = Ta + solar-driven excess (scales with global solar
    radiation, damped by wind-driven convective mixing).
    Night (solar ~ 0): MRT = Ta - a small clear-sky longwave loss term.

    Accepts scalars, numpy arrays or pandas Series; returns the same
    container shape as the inputs (ndarray if a Series is passed in,
    since the arithmetic below is vectorised with numpy).
    """
    solar = np.clip(np.asarray(solar_radiation_wm2, dtype=float), 0, None)
    wind = np.clip(np.asarray(wind_speed_ms, dtype=float), 0.1, None)
    air_temp = np.asarray(air_temperature_c, dtype=float)

    solar_excess = (
        SOLAR_TO_MRT_COEFFICIENT_C
        * (solar / REFERENCE_PEAK_SOLAR_WM2)
        / (1 + WIND_DAMPING_PER_MS * wind)
    )
    night_loss = NIGHT_LONGWAVE_LOSS_C / (1 + 0.05 * wind)

    is_night = solar <= 1.0
    mrt = np.where(is_night, air_temp - night_loss, air_temp + solar_excess)

    if isinstance(solar_radiation_wm2, pd.Series):
        return pd.Series(mrt, index=solar_radiation_wm2.index)
    return mrt


# ---------------------------------------------------------------------------
# Wind height conversion (log wind profile)
# ---------------------------------------------------------------------------
def wind_speed_at_10m(v_measured_ms, measurement_height_m: float, roughness_length_m: float):
    """
    Convert a wind speed measured at `measurement_height_m` to its 10 m
    equivalent using the standard logarithmic wind profile:

        v(z2) = v(z1) * ln(z2 / z0) / ln(z1 / z0)

    UTCI is formally defined using wind speed at 10 m above ground, while
    the simulated stations are assumed to carry anemometers at ~3 m
    (config.STATIONS[...]['measurement_height_m']). `roughness_length_m`
    (z0) is station-specific: larger over the vegetated/green sites
    (canopy sheltering) than over open reference ground, which is also
    how canopy sheltering is represented in this model.
    """
    v = np.clip(np.asarray(v_measured_ms, dtype=float), 0.05, None)
    z0 = roughness_length_m
    z1 = measurement_height_m
    ratio = math.log(10.0 / z0) / math.log(z1 / z0)
    v10 = v * ratio

    if isinstance(v_measured_ms, pd.Series):
        return pd.Series(v10, index=v_measured_ms.index)
    return v10


# ---------------------------------------------------------------------------
# UTCI via pythermalcomfort
# ---------------------------------------------------------------------------
# UTCI's formal domain of applicability: wind 0.5-17 m/s @ 10m, MRT-Ta within
# +/-30 degC of Ta (roughly), air temperature within a wide physiological
# range. Inputs are clipped to this domain before the call so that the
# library's own `limit_inputs` guard does not silently return NaN for
# borderline simulated extremes; clipping is documented here rather than
# hidden, and only ever nudges values that are already at the edge of (or
# beyond) UTCI's defined range.
UTCI_VALID_WIND_MIN_MS = 0.5
UTCI_VALID_WIND_MAX_MS = 17.0
UTCI_VALID_TR_TA_DELTA_MAX_C = 30.0


def compute_utci(air_temperature_c, mean_radiant_temp_c, wind_speed_10m_ms, relative_humidity_pct):
    """
    Vectorised UTCI calculation. Accepts pandas Series (recommended, all of
    the same length) and returns a pandas Series of UTCI values in °C.
    """
    tdb = np.asarray(air_temperature_c, dtype=float)
    tr = np.asarray(mean_radiant_temp_c, dtype=float)
    v = np.asarray(wind_speed_10m_ms, dtype=float)
    rh = np.asarray(relative_humidity_pct, dtype=float)

    v = np.clip(v, UTCI_VALID_WIND_MIN_MS, UTCI_VALID_WIND_MAX_MS)
    tr = np.clip(tr, tdb - UTCI_VALID_TR_TA_DELTA_MAX_C, tdb + UTCI_VALID_TR_TA_DELTA_MAX_C)
    rh = np.clip(rh, 0, 100)

    result = _pythermalcomfort_utci(tdb=tdb, tr=tr, v=v, rh=rh, limit_inputs=True)
    values = np.asarray(result.utci, dtype=float)

    if isinstance(air_temperature_c, pd.Series):
        return pd.Series(values, index=air_temperature_c.index)
    return values


# ---------------------------------------------------------------------------
# UTCI thermal-stress categories
# ---------------------------------------------------------------------------
_BIN_EDGES = [cat[0] for cat in UTCI_CATEGORIES] + [UTCI_CATEGORIES[-1][1]]
_BIN_LABELS = [cat[2] for cat in UTCI_CATEGORIES]


def classify_utci_series(utci_values: pd.Series) -> pd.Series:
    """Map UTCI (°C) values to the standard 10-class thermal-stress category."""
    return pd.cut(utci_values, bins=_BIN_EDGES, labels=_BIN_LABELS, right=False)


def classify_utci_value(utci_value: float) -> str | None:
    if utci_value is None or (isinstance(utci_value, float) and math.isnan(utci_value)):
        return None
    for lower, upper, label in UTCI_CATEGORIES:
        if lower <= utci_value < upper:
            return label
    return UTCI_CATEGORIES[-1][2]
