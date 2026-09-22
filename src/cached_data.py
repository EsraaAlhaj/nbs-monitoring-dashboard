"""
Streamlit-cache wrappers around src.data_service.

Kept separate from data_service.py so that module can stay framework-
agnostic (importable and testable without Streamlit). Every page should
import its data through THIS module, not data_service directly.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd
import streamlit as st

from src import data_service, database
from src.data_generator import generate_dataset


@st.cache_resource(show_spinner="Preparing demonstration dataset...")
def _ensure_database_ready() -> None:
    """
    A fresh checkout (e.g. a new Streamlit Community Cloud deployment) has
    no database/monitoring.db — it's intentionally not committed to git
    (see README). Build it once per running app process if it's missing or
    empty, so the app is usable immediately without a manual setup step.
    """
    database.init_db()
    if database.row_count() == 0:
        database.insert_readings(generate_dataset())


_ensure_database_ready()


@st.cache_data(ttl=300, show_spinner=False)
def load_stations() -> pd.DataFrame:
    return data_service.get_stations()


@st.cache_data(ttl=300, show_spinner="Loading readings...")
def load_readings(
    station_ids: Optional[tuple[str, ...]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    return data_service.get_readings(station_ids=station_ids, start=start, end=end)


@st.cache_data(ttl=60, show_spinner=False)
def load_latest_readings() -> pd.DataFrame:
    return data_service.get_latest_readings()


def clear_all_caches() -> None:
    load_stations.clear()
    load_readings.clear()
    load_latest_readings.clear()
