"""
Central configuration for the NbS Impact Monitoring prototype.

This is the ONE place that should be edited when:
  - real, field-verified GPS coordinates become available for the stations;
  - the monitoring network is expanded or stations are renamed/relocated;
  - simulation parameters (period length, interval, random seed) need to change;
  - display units, valid-value ranges, or the colour palette need adjusting.

No coordinates, thresholds, or simulation parameters should be hard-coded
anywhere else in the project — every other module imports from here.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "monitoring.db"

# --------------------------------------------------------------------------
# Data source mode
# --------------------------------------------------------------------------
# "sqlite"  -> reads from the local SQLite database populated by
#              scripts/generate_data.py (synthetic demonstration data).
# "api"     -> placeholder for a future connection to real field-station
#              telemetry. NOT IMPLEMENTED in this prototype — see
#              src/data_service.py (APIDataSource) for the interface that
#              a real integration would need to fulfil.
#
# Pages never talk to the database directly; they always go through
# src/data_service.py, so flipping this flag (once a real API exists) does
# not require any change to app.py or the files in views/.
DATA_SOURCE_MODE = "sqlite"

# --------------------------------------------------------------------------
# IMPORTANT — Station identity & siting: NOT YET FINALISED
# --------------------------------------------------------------------------
# Real site selection has not happened yet. Station names below are GENERIC
# PLACEHOLDERS ("NBS Site A/B", "Reference Site A/B") and `latitude` /
# `longitude` / `elevation_m` are deliberately left as `None` — this
# prototype does not commit to any specific location, building, or
# sub-area, not even an approximate one.
#
# Each "reference" station is intended to be a short, plausible walking
# distance from its paired "NBS" (green/vegetated) station so the two
# experience the same regional weather and differ mainly in local land
# cover (vegetated vs. bare/paved) once real sites are chosen.
#
# When real sites are selected:
#   1. Fill in `latitude`, `longitude`, and `elevation_m` for each station
#      below (do not hard-code coordinates anywhere else in the project).
#   2. Update `name`/`description` if the generic Site A/B labelling should
#      be replaced with real place names.
#   3. Do not site a station at EXCLUDED_LANDMARK below.
STATION_SITING_DISCLAIMER = (
    "Station names and coordinates are generic placeholders — real site "
    "selection has not yet been finalised. Coordinates are intentionally "
    "left unset (None) until field-verified GPS points are available."
)

SITE_TYPE_GREEN = "green"
SITE_TYPE_REFERENCE = "reference"

STATIONS = {
    "site_a_nbs": {
        "station_id": "ST01",
        "name": "NBS Site A",
        "pair_id": "pair_a",
        "site_type": SITE_TYPE_GREEN,
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "measurement_height_m": 3.0,
        "roughness_length_m": 0.30,  # assumed rougher: vegetated canopy/shrub cover
        "description": (
            "Placeholder NbS (vegetated green-infrastructure) site. Exact "
            "location not yet selected — add real name/coordinates here "
            "once available."
        ),
    },
    "site_a_reference": {
        "station_id": "ST02",
        "name": "Reference Site A",
        "pair_id": "pair_a",
        "site_type": SITE_TYPE_REFERENCE,
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "measurement_height_m": 3.0,
        "roughness_length_m": 0.03,  # assumed open, unplanted / paved surface
        "description": (
            "Placeholder reference (non-intervention) site paired with NBS "
            "Site A. Exact location not yet selected — add real "
            "name/coordinates here once available."
        ),
    },
    "site_b_nbs": {
        "station_id": "ST03",
        "name": "NBS Site B",
        "pair_id": "pair_b",
        "site_type": SITE_TYPE_GREEN,
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "measurement_height_m": 3.0,
        "roughness_length_m": 0.35,  # assumed denser vegetation/planting
        "description": (
            "Placeholder NbS (vegetated green-infrastructure) site. Exact "
            "location not yet selected — add real name/coordinates here "
            "once available."
        ),
    },
    "site_b_reference": {
        "station_id": "ST04",
        "name": "Reference Site B",
        "pair_id": "pair_b",
        "site_type": SITE_TYPE_REFERENCE,
        "latitude": None,
        "longitude": None,
        "elevation_m": None,
        "measurement_height_m": 3.0,
        "roughness_length_m": 0.03,  # assumed open, unplanted / paved surface
        "description": (
            "Placeholder reference (non-intervention) site paired with NBS "
            "Site B. Exact location not yet selected — add real "
            "name/coordinates here once available."
        ),
    },
}

STATION_PAIRS = {
    "pair_a": {
        "pair_id": "pair_a",
        "label": "Site Pair A",
        "green": "site_a_nbs",
        "reference": "site_a_reference",
    },
    "pair_b": {
        "pair_id": "pair_b",
        "label": "Site Pair B",
        "green": "site_b_nbs",
        "reference": "site_b_reference",
    },
}

# Do not use this landmark for station placement (explicit project constraint,
# to be respected once real sites within the University of Jordan campus are chosen).
EXCLUDED_LANDMARK = "Clock Tower (Saha tas-Sa'a), central University of Jordan campus"

# --------------------------------------------------------------------------
# Simulation parameters (synthetic demonstration data)
# --------------------------------------------------------------------------
RANDOM_SEED = 42
SIMULATION_START_DATE = "2026-07-01"  # start of the simulated 30-day summer window
SIMULATION_NUM_DAYS = 30
SIMULATION_INTERVAL_MINUTES = 15
TIMEZONE = "Asia/Amman"

# --------------------------------------------------------------------------
# Variable metadata: units, display labels, plausible physical ranges used
# by the data-quality "implausible reading" checks, and default rounding.
# --------------------------------------------------------------------------
VARIABLES = {
    "air_temperature_c": {
        "label": "Air Temperature",
        "unit": "°C",
        "valid_min": -10.0,
        "valid_max": 55.0,
        "decimals": 1,
    },
    "relative_humidity_pct": {
        "label": "Relative Humidity",
        "unit": "%",
        "valid_min": 0.0,
        "valid_max": 100.0,
        "decimals": 1,
    },
    "wind_speed_ms": {
        "label": "Wind Speed",
        "unit": "m/s",
        "valid_min": 0.0,
        "valid_max": 40.0,
        "decimals": 2,
    },
    "wind_direction_deg": {
        "label": "Wind Direction",
        "unit": "°",
        "valid_min": 0.0,
        "valid_max": 360.0,
        "decimals": 0,
    },
    "solar_radiation_wm2": {
        "label": "Solar Radiation",
        "unit": "W/m²",
        "valid_min": 0.0,
        "valid_max": 1400.0,
        "decimals": 0,
    },
    "rainfall_mm": {
        "label": "Rainfall",
        "unit": "mm",
        "valid_min": 0.0,
        "valid_max": 100.0,
        "decimals": 2,
    },
    "mean_radiant_temp_c": {
        "label": "Mean Radiant Temperature (estimated)",
        "unit": "°C",
        "valid_min": -10.0,
        "valid_max": 90.0,
        "decimals": 1,
    },
    "utci_c": {
        "label": "UTCI (estimated)",
        "unit": "°C",
        "valid_min": -50.0,
        "valid_max": 60.0,
        "decimals": 1,
    },
}

# Fields treated as "core sensor readings" for completeness / missing-data
# calculations (station status and QC flag columns are excluded).
CORE_READING_FIELDS = [
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "solar_radiation_wm2",
    "rainfall_mm",
    "mean_radiant_temp_c",
    "utci_c",
]

STATION_STATUSES = ["active", "maintenance", "offline"]

# --------------------------------------------------------------------------
# Data-quality thresholds
# --------------------------------------------------------------------------
# A run of >= this many consecutive identical readings of a variable is
# flagged as a suspected "stuck sensor" (at 15-min resolution, 8 readings
# = 2 hours).
STUCK_SENSOR_MIN_RUN_LENGTH = 8

# For these fields, a run of exactly zero is a normal physical state (no
# sunlight at night; no rain during a dry spell) rather than a sensor
# fault, so zero-valued runs are exempt from stuck-sensor detection. A run
# stuck at any non-zero value for these fields is still flagged.
STUCK_SENSOR_ZERO_EXEMPT_FIELDS = ["solar_radiation_wm2", "rainfall_mm"]

# A gap between two consecutive timestamps longer than this many minutes
# (beyond the expected interval) is reported as a time gap.
EXPECTED_INTERVAL_MINUTES = SIMULATION_INTERVAL_MINUTES
GAP_THRESHOLD_MINUTES = SIMULATION_INTERVAL_MINUTES * 2

# --------------------------------------------------------------------------
# UTCI thermal-stress categories (ISO/ICHB standard UTCI category table).
# Bounds are in degrees C, lower-inclusive / upper-exclusive except the
# final open-ended category.
# --------------------------------------------------------------------------
UTCI_CATEGORIES = [
    (-100.0, -40.0, "Extreme cold stress"),
    (-40.0, -27.0, "Very strong cold stress"),
    (-27.0, -13.0, "Strong cold stress"),
    (-13.0, 0.0, "Moderate cold stress"),
    (0.0, 9.0, "Slight cold stress"),
    (9.0, 26.0, "No thermal stress"),
    (26.0, 32.0, "Moderate heat stress"),
    (32.0, 38.0, "Strong heat stress"),
    (38.0, 46.0, "Very strong heat stress"),
    (46.0, 100.0, "Extreme heat stress"),
]

# --------------------------------------------------------------------------
# Colour palette — calm blue / green, professional. No third-party branding.
# --------------------------------------------------------------------------
COLORS = {
    "green_site": "#2E7D5B",       # calm forest green - vegetated sites
    "reference_site": "#595959",   # dark grey - reference/unplanted sites
    "green_site_fill": "rgba(46, 125, 91, 0.15)",
    "reference_site_fill": "rgba(89, 89, 89, 0.15)",
    "accent": "#F2A93B",           # warm amber accent for alerts/highlights
    "critical": "#C4453A",
    "background": "#F6F9F8",
    "surface": "#FFFFFF",
    "text_primary": "#1F2D2A",
    "text_muted": "#5B6B67",
    "grid": "#E3EAE8",
}

STATUS_COLORS = {
    "active": "#2E7D5B",
    "maintenance": "#F2A93B",
    "offline": "#C4453A",
}

UTCI_CATEGORY_COLORS = {
    "Extreme cold stress": "#08306B",
    "Very strong cold stress": "#2171B5",
    "Strong cold stress": "#6BAED6",
    "Moderate cold stress": "#C6DBEF",
    "Slight cold stress": "#DEEBF7",
    "No thermal stress": "#4CA36F",
    "Moderate heat stress": "#FDD49E",
    "Strong heat stress": "#FC8D59",
    "Very strong heat stress": "#E34A33",
    "Extreme heat stress": "#B30000",
}

# --------------------------------------------------------------------------
# Disclaimers shown throughout the UI
# --------------------------------------------------------------------------
DEMO_DATA_BANNER = (
    "SIMULATED DEMONSTRATION DATA — This prototype displays synthetically "
    "generated data for illustration only. No live field stations are "
    "currently connected."
)

ESTIMATE_DISCLAIMER = (
    "UTCI and Mean Radiant Temperature shown in this prototype are "
    "estimated from simulated weather variables and should be treated as "
    "indicative rather than field-validated measurements."
)
