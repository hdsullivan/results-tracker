"""Visual comparison page in the lab's IEEE reconstruction-figure style.

Reference | Measurement block, spacer, baselines -> proposed (export/visual.make_visual). Below the figure:
an IEEEtran-style caption, a panel-metrics table that cross-checks the logged PSNR against PSNR recomputed
from the shown image, and the source files, all in the paper look.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import streamlit as st

from .. import aggregate as agg
from ..export.figures import figure_bytes, figure_tex, to_grayscale_png
from ..export.visual import ZOOM_FRACTION, build_panels, guess_roles, list_image_files, make_visual, panel_metrics_rows
from .common import load_metric_defs, load_records, select_project_experiment, sidebar_db
from .tables import figure_caption_html, generic_html

NONE = "— none —"


def _opt(files: list[str], role_file: Optional[str]) -> int:
    return ([NONE] + files).index(role_file) if role_file in files else 0


def render() -> None:
    st.title("Visual comparison")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="comparison")
    if experiment is None:
        return
    recs = agg.completed(load_records(project, experiment))
    defs = load_metric_defs()
    with_art = [r for r in recs if r.get("artifacts_dir")]
    if not with_art:
        st.info("No runs in this experiment have an `artifacts_dir`. Log runs with `artifacts_dir=...` pointing at a "
                "folder that holds the reconstruction images (same file name for every method).")
        return

    with st.sidebar:
        st.markdown("**Sample**")
        datasets = list(dict.fromkeys(r["dataset"] for r in with_art if r.get("dataset") is not None))
        dataset = st.selectbox("Dataset", datasets) if datasets else None
        pool = [r for r in with_art if dataset is None or r.get("dataset") == dataset]
        row_opts = [NONE] + [k for k in ("seed", "instance") if len({r.get(k) for r in pool}) > 1] + \
                   [f"config.{k}" for k in agg.varying_config_keys(pool)]
        row_key = st.selectbox("Rows (one per value)", row_opts, help="e.g. seed, or a config key like K")
        row_key = None if row_key == NONE else row_key
        instances = sorted({r["instance"] for r in pool if r.get("instance") is not None}, key=str)
        instance = st.selectbox("Instance", instances) if instances and row_key != "instance" else None
        pool = [r for r in pool if instance is None or r.get("instance") == instance]
        seeds = sorted({r["seed"] for r in pool if r.get("seed") is not None})
        seed = st.selectbox("Seed", seeds) if seeds and row_key != "seed" else None
        files = list_image_files(r["artifacts_dir"] for r in pool)
        if not files:
            st.warning("No image files found in the artifact folders.")
            return
        st.markdown("**Images**")
        roles = guess_roles(files)
        image = st.selectbox("Reconstruction file", files, index=files.index(roles["reconstruction"]) if roles["reconstruction"] in files else 0)
        reference = st.selectbox("Ground truth", [NONE] + files, index=_opt(files, roles["reference"]))
        measurement = st.selectbox("Measurement / input", [NONE] + files, index=_opt(files, roles["measurement"]))
        kernel = st.selectbox("Kernel / PSF thumbnail", [NONE] + files, index=_opt(files, roles["kernel"]))
        metrics_all = agg.metric_names(pool)
        metrics = st.multiselect("Metrics stamped on panels", metrics_all, default=[m for m in metrics_all if m in ("psnr", "ssim")][:2])

    pool_sel = [r for r in pool if seed is None or r.get("seed") == seed]
    method_names = list(dict.fromkeys(r["method"] for r in agg.select_runs(pool_sel)))
    methods = st.multiselect("Methods (baselines first, proposed last)", method_names, default=method_names)

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.radio("Mode", ["Reconstruction", "Error maps"], horizontal=True, disabled=reference == NONE)
    zoom = c2.checkbox("Zoom inset", value=True, disabled=mode == "Error maps")
    width = c3.selectbox("Width", ["double", "single"], help="double = IEEE text width 7.16 in")
    gray = c4.checkbox("Grayscale preview", value=False)

    zoom_fraction, zoom_center, crop_box = ZOOM_FRACTION, (0.5, 0.5), None
    if zoom and mode == "Reconstruction":
        with st.expander("Zoom box (identical on every panel)", expanded=False):
            explicit = st.checkbox("Explicit pixel box", value=False)
            if explicit:
                k1, k2, k3, k4 = st.columns(4)
                cx = k1.number_input("x", 0, 4096, 30)
                cy = k2.number_input("y", 0, 4096, 30)
                cw = k3.number_input("width", 2, 4096, 32)
                ch = k4.number_input("height", 2, 4096, 32)
                crop_box = (int(cx), int(cy), int(cw), int(ch))
            else:
                k1, k2, k3 = st.columns(3)
                zoom_fraction = k1.slider("Box side (fraction of short side)", 0.1, 0.6, ZOOM_FRACTION, 0.05)
                zoom_center = (k2.slider("Centre x", 0.0, 1.0, 0.5, 0.05), k3.slider("Centre y", 0.0, 1.0, 0.5, 0.05))

    try:
        vr = make_visual(
            recs, defs, experiment=experiment, dataset=dataset, seed=seed, instance=instance, image=image,
            reference=None if reference == NONE else reference, measurement=None if measurement == NONE else measurement,
            kernel=None if kernel == NONE else kernel, methods=methods or None, metrics=metrics,
            mode="error" if mode == "Error maps" else "image", zoom=zoom, zoom_fraction=zoom_fraction,
            zoom_center=zoom_center, crop_box=crop_box, rows=row_key, width=width, auto_roles=False,
        )
    except ValueError as e:
        st.error(str(e))
        return
    for label, why in vr.omitted.items():
        st.warning(f"Not shown: {label} — {why}.")
    for pr in vr.problems:
        st.warning(pr)

    png = figure_bytes(vr.fig, "png", dpi=200)
    st.image(to_grayscale_png(png) if gray else png)
    st.markdown(figure_caption_html(vr.spec.caption_stub(), number=1), unsafe_allow_html=True)
    fw, fh = vr.fig.get_size_inches()
    st.caption(f"{fw:.2f} × {fh:.2f} in · IEEE text width · native pixels · shared display range"
               + (" and error scale" if mode == "Error maps" else ""))

    stem = f"{experiment}-{dataset or 'all'}" + (f"-seed{seed}" if seed is not None else "") + \
           ("-error" if mode == "Error maps" else ("-zoom" if zoom else "")) + "-visual"
    stem = stem.replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download PNG (300 dpi)", figure_bytes(vr.fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png")
    d2.download_button("Download PDF", figure_bytes(vr.fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
    d3.download_button("Download provenance JSON", json.dumps(asdict(vr.spec), indent=2, default=str),
                       file_name=f"{stem}.json", mime="application/json")

    # --- panel metrics: logged vs recomputed from the shown image (single-row figures)
    if not row_key:
        chosen = agg.select_runs(pool_sel, methods=methods or None)
        panels, ref_panel, _ = build_panels(chosen, image, defs, metrics=metrics,
                                            reference=None if reference == NONE else reference)
        headers, rows, warns = panel_metrics_rows(chosen, panels, ref_panel, defs, metrics=metrics or ("psnr",))
        st.subheader("Panel metrics")
        st.markdown(generic_html(headers, rows, number=1, left_cols=1,
                                 caption="Metrics of the shown images: as logged with the run, and PSNR recomputed from the "
                                         "displayed reconstruction against the ground truth (luminance, 10 px border dropped, "
                                         "data range 1). A gap flags a mismatch between the figure and the table numbers."),
                    unsafe_allow_html=True)
        for w in warns:
            st.warning(w)
        if ref_panel is not None and not warns and rows:
            st.success("Logged PSNR matches the displayed images within 0.05 dB for every panel.")

    with st.expander("LaTeX figure snippet"):
        st.code(figure_tex(f"figures/{stem}.pdf", caption=vr.spec.caption_stub(), label="fig:visual", width=width), language="latex")
    with st.expander("Source files"):
        srows = []
        for p in vr.spec.panels:
            pth = Path(p["path"]) if p.get("path") else None
            srows.append([p["title"], p.get("kind", ""), p.get("row", "") or "—", str(pth) if pth else "—"])
        st.markdown(generic_html(["Panel", "Role", "Row", "Path"], srows, left_cols=4,
                                 caption="Every image in the figure and where it came from (also in the provenance JSON)."),
                    unsafe_allow_html=True)
