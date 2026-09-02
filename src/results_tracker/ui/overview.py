"""Overview page: what's in the database."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .common import load_catalog, load_records, sidebar_db


def render() -> None:
    st.title("Results Tracker")
    sidebar_db()
    cat = load_catalog()
    recs = load_records()

    failed = [r for r in recs if r["status"] == "failed"]
    running = [r for r in recs if r["status"] == "running"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", len(cat["projects"]))
    c2.metric("Experiments", len(cat["experiments"]))
    c3.metric("Runs", len(recs))
    c4.metric("Failed / running", f"{len(failed)} / {len(running)}")

    if not recs:
        st.info("Empty database. Seed a demo with `results-tracker demo`, or log runs with `results_tracker.log_run`.")
        return

    st.subheader("Experiments")
    per_exp = pd.DataFrame(cat["experiments"])
    run_df = pd.DataFrame([{"experiment": r["experiment"]} for r in recs])
    per_exp["runs"] = per_exp["experiment"].map(run_df["experiment"].value_counts()).fillna(0).astype(int)
    st.dataframe(per_exp, width="stretch", hide_index=True)

    st.subheader("Recent runs")
    recent = sorted(recs, key=lambda r: r["timestamp"] or 0, reverse=True)[:20]
    df = pd.DataFrame(
        [
            {
                "id": r["run_id"], "time": r["timestamp"], "experiment": r["experiment"], "method": r["method"],
                "dataset": r["dataset"], "seed": r["seed"], "status": r["status"],
                **{f"{k}": v for k, v in r["metrics"].items()},
            }
            for r in recent
        ]
    )
    st.dataframe(df, width="stretch", hide_index=True)
