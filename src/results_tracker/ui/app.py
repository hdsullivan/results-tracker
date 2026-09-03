"""Streamlit entry point. `results-tracker ui` runs this file."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from results_tracker.ui import ablation, comparison, curves, export, overview, paper, run_detail, settings, studies, sweep, tradeoff, visual
from results_tracker.ui.common import db_path

# The database name leads the tab title, so two GUIs on two files are told apart at a glance.
st.set_page_config(page_title=f"{Path(db_path()).name} · Results Tracker", page_icon="📊", layout="wide")

pages = [
    st.Page(overview.render, title="Overview", icon="🏠", default=True),  # served at "/"
    st.Page(comparison.render, title="Comparison", icon="📋", url_path="comparison"),
    st.Page(sweep.render, title="Sweep", icon="📈", url_path="sweep"),
    st.Page(ablation.render, title="Ablation", icon="🧩", url_path="ablation"),
    st.Page(visual.render, title="Visual", icon="🖼️", url_path="visual"),
    st.Page(curves.render, title="Curves", icon="〽️", url_path="curves"),
    st.Page(tradeoff.render, title="Trade-off", icon="⚖️", url_path="tradeoff"),
    st.Page(run_detail.render, title="Run detail", icon="🔍", url_path="run"),
    st.Page(studies.render, title="Studies", icon="🗂️", url_path="studies"),
    st.Page(paper.render, title="Paper", icon="📄", url_path="paper"),
    st.Page(export.render, title="Export", icon="📤", url_path="export"),
    st.Page(settings.render, title="Settings", icon="⚙️", url_path="settings"),
]
st.navigation(pages).run()
