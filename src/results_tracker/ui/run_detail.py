"""Run detail page: config, metrics, config diff against another run, artifacts on disk."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from ..api import delete_runs
from .common import db_path, engine_for, fmt_for, load_metric_defs, load_records, select_project_experiment, sidebar_db

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


def show_artifacts(run: dict, other: Optional[dict], defs: dict) -> None:
    """Artifacts as a lab-style figure strip (GT | Measurement | this run [| other run]) plus the raw files."""
    from ..export.figures import figure_bytes, to_grayscale_png
    from ..export.visual import build_panels, guess_roles, list_image_files, reconstruction_figure

    path = run["artifacts_dir"]
    p = Path(path).expanduser()
    if not p.exists():
        st.warning(f"Artifacts directory not found: `{p}`")
        return
    files = sorted(f for f in p.rglob("*") if f.is_file())
    images = list_image_files([path])
    logs = [f for f in files if f.suffix.lower() in LOG_EXT]
    st.caption(f"`{p}` · {len(files)} files")

    if images:
        roles = guess_roles(images)
        none = "— none —"
        with st.expander("Figure options", expanded=False):
            c1, c2, c3 = st.columns(3)
            recon = c1.selectbox("Reconstruction", images, index=images.index(roles["reconstruction"]) if roles["reconstruction"] in images else 0, key="rd_recon")
            ref = c2.selectbox("Ground truth", [none] + images, index=([none] + images).index(roles["reference"]) if roles["reference"] else 0, key="rd_ref")
            meas = c3.selectbox("Measurement", [none] + images, index=([none] + images).index(roles["measurement"]) if roles["measurement"] else 0, key="rd_meas")
            k1, k2, k3, k4 = st.columns(4)
            mode = k1.radio("Mode", ["Reconstruction", "Error map"], horizontal=True, disabled=ref == none, key="rd_mode")
            zoom = k2.checkbox("Zoom inset", value=True, key="rd_zoom", disabled=mode == "Error map")
            include_other = k3.checkbox("Include compared run", value=other is not None and bool(other.get("artifacts_dir")),
                                        disabled=other is None or not other.get("artifacts_dir"), key="rd_other")
            gray = k4.checkbox("Grayscale", value=False, key="rd_gray")
        metrics = [m for m in ("psnr", "ssim") if m in run.get("metrics", {})] or list(run.get("metrics", {}))[:2]
        recs = [run] + ([other] if include_other and other else [])
        panels, ref_panel, problems = build_panels(recs, recon, defs, metrics=metrics,
                                                   reference=None if ref == none else ref,
                                                   measurement=None if meas == none else meas)
        for pr in problems:
            st.warning(pr)
        methods = [pnl for pnl in panels if pnl.kind == "method"]
        for pnl, r in zip(methods, recs):  # two runs of one method would otherwise share a title
            pnl.title = f"{r.get('method_label') or r.get('method') or 'run'} (#{r['run_id']})" if len(recs) > 1 else (r.get("method_label") or r.get("method") or recon)
        meas_panel = next((pnl for pnl in panels if pnl.kind == "measurement"), None)
        if methods:
            try:
                fig, spec = reconstruction_figure(methods, reference=ref_panel, measurement=meas_panel,
                                                  mode="error" if mode == "Error map" else "image", zoom=zoom, width="double")
                png = figure_bytes(fig, "png", dpi=200)
                st.image(to_grayscale_png(png) if gray else png)
                fw, fh = fig.get_size_inches()
                st.caption(f"{fw:.2f} × {fh:.2f} in · IEEE text width · native pixels · shared display range"
                           + (" and error scale" if mode == "Error map" else "") + " · numbers are this run's logged metrics")
                d1, d2 = st.columns(2)
                stem = f"run{run['run_id']}-{Path(recon).stem}"
                d1.download_button("Download PNG (300 dpi)", figure_bytes(fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png")
                d2.download_button("Download PDF", figure_bytes(fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
            except ValueError as e:
                st.error(str(e))
        used = {recon, ref, meas}
        extras = [f for f in images if f not in used]
        if extras:
            with st.expander(f"Other images ({len(extras)})"):
                cols = st.columns(min(4, len(extras)))
                for i, img in enumerate(extras[:24]):
                    with cols[i % len(cols)]:
                        st.image(str(p / img), caption=img, width="stretch")
    for lf in logs[:3]:
        with st.expander(f"log: {lf.relative_to(p).as_posix()}"):
            try:
                text = lf.read_text(errors="replace")
            except OSError as e:
                text = f"(could not read: {e})"
            st.code("\n".join(text.splitlines()[-200:]))
    if not images and not logs:
        st.write([f.relative_to(p).as_posix() for f in files[:50]])


def render() -> None:
    st.title("Run detail")
    sidebar_db()
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    project, experiment = select_project_experiment(prefer="comparison")
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
    ts = run["timestamp"]
    meta[4].metric("Logged", ts.strftime("%Y-%m-%d") if ts else "—", help=ts.strftime("%H:%M:%S") if ts else None)
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
        other = None
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
        show_artifacts(run, other if others else None, defs)
    else:
        st.caption("No artifacts directory recorded for this run.")

    with st.expander("Delete this run"):
        st.warning("Removes the run from the database only; files in the artifacts folder are kept. "
                   "Tables, sweeps and ablations that used this run will change. This cannot be undone.")
        armed = st.checkbox(f"Yes, delete run #{run['run_id']} ({run_label(run)})", key="del_confirm")
        if st.button(f"Delete run #{run['run_id']}", type="primary", disabled=not armed, key="del_btn"):
            n = delete_runs([run["run_id"]], engine=engine_for(db_path()))
            st.cache_data.clear()
            for k in ("run_pick", "run_other", "del_confirm"):
                st.session_state.pop(k, None)
            st.session_state["flash"] = f"Deleted run #{run['run_id']} ({run_label(run)})." if n else "Run was already gone."
            st.rerun()
