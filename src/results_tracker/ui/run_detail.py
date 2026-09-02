"""Run detail page: config, metrics, config diff against another run, artifacts on disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from .common import fmt_for, load_metric_defs, load_records, select_project_experiment, sidebar_db

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
LOG_EXT = {".log", ".txt", ".out", ".err"}


def run_label(r: dict) -> str:
    parts = [f"#{r['run_id']}"]
    for k in ("method", "dataset", "instance"):
        if r.get(k) is not None:
            parts.append(str(r[k]))
    if r.get("seed") is not None:
        parts.append(f"seed {r['seed']}")
    if r["status"] != "completed":
        parts.append(f"[{r['status']}]")
    return " · ".join(parts)


def show_artifacts(path: str) -> None:
    p = Path(path).expanduser()
    if not p.exists():
        st.warning(f"Artifacts directory not found: `{p}`")
        return
    files = sorted(f for f in p.rglob("*") if f.is_file())
    images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
    logs = [f for f in files if f.suffix.lower() in LOG_EXT]
    st.caption(f"`{p}` · {len(files)} files")
    if images:
        cols = st.columns(min(4, len(images)))
        for i, img in enumerate(images[:24]):
            with cols[i % len(cols)]:
                st.image(str(img), caption=img.relative_to(p).as_posix(), width="stretch")
        if len(images) > 24:
            st.caption(f"... and {len(images) - 24} more images")
    for lf in logs[:3]:
        with st.expander(f"log: {lf.relative_to(p).as_posix()}"):
            try:
                text = lf.read_text(errors="replace")
            except OSError as e:
                text = f"(could not read: {e})"
            tail = text.splitlines()[-200:]
            st.code("\n".join(tail))
    if not images and not logs:
        st.write([f.relative_to(p).as_posix() for f in files[:50]])


def render() -> None:
    st.title("Run detail")
    sidebar_db()
    project, experiment = select_project_experiment()
    if experiment is None:
        return
    recs = load_records(project, experiment)
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return

    by_id = {r["run_id"]: r for r in recs}
    labels = {run_label(r): r["run_id"] for r in recs}
    chosen = st.selectbox("Run", list(labels), key="run_pick")
    run = by_id[labels[chosen]]

    meta = st.columns(5)
    meta[0].metric("Status", run["status"])
    meta[1].metric("Source", run["source"])
    meta[2].metric("Seed", "—" if run["seed"] is None else run["seed"])
    meta[3].metric("Commit", (run["git_commit"] or "—")[:8])
    meta[4].metric("Logged", run["timestamp"].strftime("%Y-%m-%d %H:%M") if run["timestamp"] else "—")
    if run["tags"]:
        st.caption("tags: " + ", ".join(f"`{t}`" for t in run["tags"]))
    if run["notes"]:
        st.caption(run["notes"])

    st.subheader("Metrics")
    if run["metrics"]:
        cols = st.columns(min(6, len(run["metrics"])))
        for i, (k, v) in enumerate(run["metrics"].items()):
            arrow = "↑" if defs.get(k, {}).get("higher_is_better", True) else "↓"
            cols[i % len(cols)].metric(f"{k} {arrow}", "—" if v is None else format(v, fmt_for(defs, k)))
    else:
        st.write("no metrics")

    left, right = st.columns(2)
    with left:
        st.subheader("Config")
        st.json(run["config"], expanded=True)
    with right:
        st.subheader("Compare config with")
        others = {run_label(r): r["run_id"] for r in recs if r["run_id"] != run["run_id"]}
        if others:
            other_label = st.selectbox("Other run", list(others), key="run_other")
            other = by_id[others[other_label]]
            diff = agg.config_diff(run["config"], other["config"])
            if diff:
                st.dataframe(
                    pd.DataFrame([{"key": k, "this run": str(a), "other run": str(b)} for k, (a, b) in diff.items()]),
                    width="stretch", hide_index=True,
                )
            else:
                st.success("Configs are identical.")
            mdiff = {k: (run["metrics"].get(k), other["metrics"].get(k)) for k in set(run["metrics"]) | set(other["metrics"])}
            st.dataframe(
                pd.DataFrame(
                    [
                        {"metric": k, "this run": a, "other run": b,
                         "Δ (this − other)": (a - b) if (a is not None and b is not None) else None}
                        for k, (a, b) in sorted(mdiff.items())
                    ]
                ),
                width="stretch", hide_index=True,
            )
        else:
            st.caption("Only one run in this experiment.")

    st.subheader("Artifacts")
    if run["artifacts_dir"]:
        show_artifacts(run["artifacts_dir"])
    else:
        st.caption("No artifacts directory recorded for this run.")
