"""Streamlit entry point. `results-tracker ui` runs this file."""

from __future__ import annotations

import streamlit as st

from results_tracker.ui import ablation, comparison, export, overview, run_detail, sweep

st.set_page_config(page_title="Results Tracker", page_icon="📊", layout="wide")

pages = [
    st.Page(overview.render, title="Overview", icon="🏠", url_path="overview", default=True),
    st.Page(comparison.render, title="Comparison", icon="📋", url_path="comparison"),
    st.Page(sweep.render, title="Sweep", icon="📈", url_path="sweep"),
    st.Page(ablation.render, title="Ablation", icon="🧩", url_path="ablation"),
    st.Page(run_detail.render, title="Run detail", icon="🔍", url_path="run"),
    st.Page(export.render, title="Export", icon="📤", url_path="export"),
]
st.navigation(pages).run()
