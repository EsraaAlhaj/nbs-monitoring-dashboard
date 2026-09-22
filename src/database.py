"""
Low-level SQLite access for the NbS Impact Monitoring prototype.

This module knows how the local demonstration database is structured and
how to read/write it. It is deliberately kept "dumb" (plain SQL, no
business logic) so that src/data_service.py can present a stable interface
to the rest of the app regardless of where the data actually comes from.

Nothing in pages/ or app.py should import this module directly — always go
through src/data_service.py.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from config.settings import CORE_READING_FIELDS, DATABASE_DIR, DATABASE_PATH, STATIONS

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    pair_id         TEXT NOT NULL,
    site_type       TEXT NOT NULL,
    latitude        REAL,
    longitude       REAL,
    elevation_m     REAL,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS readings (
    reading_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id              TEXT NOT NULL REFERENCES stations(station_id),
    timestamp                TEXT NOT NULL,
    air_temperature_c       REAL,
    relative_humidity_pct   REAL,
    wind_speed_ms           REAL,
    wind_direction_deg      REAL,
    solar_radiation_wm2     REAL,
    rainfall_mm             REAL,
    mean_radiant_temp_c     REAL,
    utci_c                  REAL,
    utci_category            TEXT,
    station_status            TEXT NOT NULL,
    is_missing                INTEGER NOT NULL DEFAULT 0,
    missing_fields             TEXT,
    is_injected_anomaly        INTEGER NOT NULL DEFAULT 0,
    UNIQUE(station_id, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_readings_station_time
    ON readings (station_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_readings_time
    ON readings (timestamp);
"""

READING_COLUMNS: Sequence[str] = (
    "station_id",
    "timestamp",
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "wind_direction_deg",
    "solar_radiation_wm2",
    "rainfall_mm",
    "mean_radiant_temp_c",
    "utci_c",
    "utci_category",
    "station_status",
    "is_missing",
    "missing_fields",
    "is_injected_anomaly",
)


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DATABASE_PATH
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def connection_scope(db_path: Optional[Path] = None):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None, reset: bool = False) -> None:
    """Create schema (and optionally wipe existing data) and load station metadata."""
    with connection_scope(db_path) as conn:
        if reset:
            conn.executescript(
                "DROP TABLE IF EXISTS readings; DROP TABLE IF EXISTS stations;"
            )
        conn.executescript(SCHEMA)
        _upsert_stations(conn)


def _upsert_stations(conn: sqlite3.Connection) -> None:
    rows = [
        (
            meta["station_id"],
            meta["name"],
            meta["pair_id"],
            meta["site_type"],
            meta["latitude"],
            meta["longitude"],
            meta.get("elevation_m"),
            meta.get("description", ""),
        )
        for meta in STATIONS.values()
    ]
    conn.executemany(
        """
        INSERT INTO stations (station_id, name, pair_id, site_type, latitude,
                               longitude, elevation_m, description)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(station_id) DO UPDATE SET
            name=excluded.name,
            pair_id=excluded.pair_id,
            site_type=excluded.site_type,
            latitude=excluded.latitude,
            longitude=excluded.longitude,
            elevation_m=excluded.elevation_m,
            description=excluded.description
        """,
        rows,
    )


def insert_readings(df: pd.DataFrame, db_path: Optional[Path] = None) -> int:
    """Bulk-insert a readings DataFrame. Returns number of rows inserted."""
    missing_cols = set(READING_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"readings DataFrame is missing columns: {sorted(missing_cols)}")

    records = df[list(READING_COLUMNS)].copy()
    records["timestamp"] = pd.to_datetime(records["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    placeholders = ", ".join(["?"] * len(READING_COLUMNS))
    columns_sql = ", ".join(READING_COLUMNS)

    with connection_scope(db_path) as conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO readings ({columns_sql}) VALUES ({placeholders})",
            records.itertuples(index=False, name=None),
        )
    return len(records)


def fetch_stations(db_path: Optional[Path] = None) -> pd.DataFrame:
    with connection_scope(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM stations", conn)


def fetch_readings(
    station_ids: Optional[Iterable[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    query = "SELECT * FROM readings WHERE 1=1"
    params: list = []

    if station_ids is not None:
        station_ids = list(station_ids)
        if len(station_ids) == 0:
            return pd.DataFrame(columns=["reading_id", *READING_COLUMNS])
        placeholders = ", ".join(["?"] * len(station_ids))
        query += f" AND station_id IN ({placeholders})"
        params.extend(station_ids)

    if start is not None:
        query += " AND timestamp >= ?"
        params.append(start)

    if end is not None:
        query += " AND timestamp <= ?"
        params.append(end)

    query += " ORDER BY station_id, timestamp"

    with connection_scope(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def fetch_latest_readings(db_path: Optional[Path] = None) -> pd.DataFrame:
    """One row per station: its most recent reading."""
    query = """
        SELECT r.*
        FROM readings r
        INNER JOIN (
            SELECT station_id, MAX(timestamp) AS max_ts
            FROM readings
            GROUP BY station_id
        ) latest
        ON r.station_id = latest.station_id AND r.timestamp = latest.max_ts
    """
    with connection_scope(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def row_count(db_path: Optional[Path] = None) -> int:
    with connection_scope(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM readings")
        return int(cur.fetchone()[0])
