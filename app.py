"""
Streamlit multipage entry point — defines navigation only.

Run with:  streamlit run app.py

Each entry below is a self-contained page script under views/ (own imports,
own st.set_page_config call — Streamlit executes each page as its own script
run). Order here is the navigation order shown in the sidebar.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

pg = st.navigation(
    [
        st.Page("views/overview.py", title="Overview", default=True),
        st.Page("views/network_trends.py", title="Network Trends"),
        st.Page("views/site_comparison.py", title="Site Comparison"),
        st.Page("views/thermal_statistical_analysis.py", title="Thermal & Statistical Analysis"),
        st.Page("views/data_quality_downloads.py", title="Data Quality & Downloads"),
    ]
)
pg.run()
