"""Streamlit entry point. `results-tracker ui` runs this file."""

from __future__ import annotations

import streamlit as st

from results_tracker.ui import comparison, overview, run_detail

st.set_page_config(page_title="Results Tracker", page_icon="📊", layout="wide")

pages = [
    st.Page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(comparison.render, title="Comparison", icon="📋", url_path="comparison"),
    st.Page(run_detail.render, title="Run detail", icon="🔍", url_path="run"),
]
st.navigation(pages).run()
