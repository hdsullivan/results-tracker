"""Visual comparison page in the lab's IEEE reconstruction-figure style.

Reference | Measurement block, spacer, baselines -> proposed (export/visual.make_visual). Several comparisons
can be stacked on one page, each with its own methods, instance, seed, mode and zoom, so "PnP vs ours" and
"ours vs DPIR" sit one above the other. Below every figure: an IEEEtran-style caption, a panel-metrics table
that cross-checks the logged PSNR against PSNR recomputed from the shown image, and the source files.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from .. import aggregate as agg
from ..export.figures import figure_bytes, figure_tex, to_grayscale_png
from ..export.visual import (
    ZOOM_FRACTION,
    build_panels,
    convention_for,
    guess_roles,
    list_image_files,
    make_visual,
    panel_metrics_rows,
)
from .common import load_metric_defs, load_records, select_project_experiment, sidebar_db
from .tables import figure_caption_html, generic_html

NONE = "— none —"
MAX_COMPARISONS = 6


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

    # --- shared across every comparison on the page: the sample set and the image files/roles
    with st.sidebar:
        st.markdown("**Sample**")
        datasets = list(dict.fromkeys(r["dataset"] for r in with_art if r.get("dataset") is not None))
        dataset = st.selectbox("Dataset", datasets) if datasets else None
        pool = [r for r in with_art if dataset is None or r.get("dataset") == dataset]
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
        dr = st.number_input("Data range for float images", min_value=0.0, value=0.0, step=1.0,
                             help="Divisor applied to float/32-bit images, the same for every panel. 0 = images are already in [0, 1]. "
                                  "8-/16-bit images are scaled by their dtype range automatically.")
        data_range = dr if dr > 0 else None
        st.markdown("**Page**")
        n_comparisons = int(st.number_input("Comparisons stacked on this page", min_value=1, max_value=MAX_COMPARISONS, value=1, step=1,
                                            help="Each comparison picks its own methods, instance, seed, mode and zoom; "
                                                 "they share the dataset, image files and metrics above."))

    shared = dict(recs=recs, defs=defs, experiment=experiment, dataset=dataset, pool=pool, image=image,
                  reference=None if reference == NONE else reference, measurement=None if measurement == NONE else measurement,
                  kernel=None if kernel == NONE else kernel, metrics=metrics, data_range=data_range)
    for i in range(1, n_comparisons + 1):
        if n_comparisons > 1:
            st.header(f"Comparison {i}")
        _comparison(i, **shared)
        if i < n_comparisons:
            st.divider()


def _comparison(i: int, *, recs, defs, experiment, dataset, pool, image, reference, measurement, kernel, metrics, data_range) -> None:
    """One figure with its own methods / instance / seed / mode / zoom; widget keys are suffixed with `i`."""
    key = f"vis{i}_"
    s1, s2, s3 = st.columns(3)
    row_opts = [NONE] + [k for k in ("seed", "instance") if len({r.get(k) for r in pool}) > 1] + \
               [f"config.{k}" for k in agg.varying_config_keys(pool)]
    row_key = s1.selectbox("Rows (one per value)", row_opts, key=key + "rows", help="e.g. seed, or a config key like K")
    row_key = None if row_key == NONE else row_key
    instances = sorted({r["instance"] for r in pool if r.get("instance") is not None}, key=str)
    instance = s2.selectbox("Instance", instances, key=key + "instance") if instances and row_key != "instance" else None
    pool_i = [r for r in pool if instance is None or r.get("instance") == instance]
    seeds = sorted({r["seed"] for r in pool_i if r.get("seed") is not None})
    seed = s3.selectbox("Seed", seeds, key=key + "seed") if seeds and row_key != "seed" else None
    pool_sel = [r for r in pool_i if seed is None or r.get("seed") == seed]

    method_names = list(dict.fromkeys(r["method"] for r in agg.select_runs(pool_sel)))
    methods = st.multiselect("Methods (baselines first, proposed last)", method_names, default=method_names, key=key + "methods")

    c1, c2, c3, c4 = st.columns(4)
    mode = c1.radio("Mode", ["Reconstruction", "Error maps"], horizontal=True, disabled=reference is None, key=key + "mode")
    zoom = c2.checkbox("Zoom inset", value=True, disabled=mode == "Error maps", key=key + "zoom")
    width = c3.selectbox("Width", ["double", "single"], help="double = IEEE text width 7.16 in", key=key + "width")
    gray = c4.checkbox("Grayscale preview", value=False, key=key + "gray")

    zoom_fraction, zoom_center, crop_box = ZOOM_FRACTION, (0.5, 0.5), None
    if zoom and mode == "Reconstruction":
        with st.expander("Zoom box (identical on every panel)", expanded=False):
            explicit = st.checkbox("Explicit pixel box", value=False, key=key + "explicit")
            if explicit:
                k1, k2, k3, k4 = st.columns(4)
                cx = k1.number_input("x", 0, 4096, 30, key=key + "cx")
                cy = k2.number_input("y", 0, 4096, 30, key=key + "cy")
                cw = k3.number_input("width", 2, 4096, 32, key=key + "cw")
                ch = k4.number_input("height", 2, 4096, 32, key=key + "ch")
                crop_box = (int(cx), int(cy), int(cw), int(ch))
            else:
                k1, k2, k3 = st.columns(3)
                zoom_fraction = k1.slider("Box side (fraction of short side)", 0.1, 0.6, ZOOM_FRACTION, 0.05, key=key + "zf")
                zoom_center = (k2.slider("Centre x", 0.0, 1.0, 0.5, 0.05, key=key + "zx"),
                               k3.slider("Centre y", 0.0, 1.0, 0.5, 0.05, key=key + "zy"))

    try:
        vr = make_visual(
            recs, defs, experiment=experiment, dataset=dataset, seed=seed, instance=instance, image=image,
            reference=reference, measurement=measurement, kernel=kernel, methods=methods or None, metrics=metrics,
            mode="error" if mode == "Error maps" else "image", zoom=zoom, zoom_fraction=zoom_fraction,
            zoom_center=zoom_center, crop_box=crop_box, rows=row_key, width=width, auto_roles=False, data_range=data_range,
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
    st.markdown(figure_caption_html(vr.spec.caption_stub(), number=i), unsafe_allow_html=True)
    fw, fh = vr.fig.get_size_inches()
    st.caption(f"{fw:.2f} × {fh:.2f} in · IEEE text width · native pixels · shared display range"
               + (" and error scale" if mode == "Error maps" else ""))

    stem = f"{experiment}-{dataset or 'all'}" + (f"-{instance}" if instance is not None else "") + \
           (f"-seed{seed}" if seed is not None else "") + ("-error" if mode == "Error maps" else ("-zoom" if zoom else "")) + \
           (f"-{i}" if i > 1 else "") + "-visual"
    stem = stem.replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download PNG (300 dpi)", figure_bytes(vr.fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png", key=key + "png")
    d2.download_button("Download PDF", figure_bytes(vr.fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf", key=key + "pdf")
    d3.download_button("Download provenance JSON", json.dumps(asdict(vr.spec), indent=2, default=str),
                       file_name=f"{stem}.json", mime="application/json", key=key + "json")

    # --- panel metrics: logged vs recomputed from the shown image (single-row figures)
    if not row_key:
        chosen = agg.select_runs(pool_sel, methods=methods or None)
        panels, ref_panel, _ = build_panels(chosen, image, defs, metrics=metrics, reference=reference, data_range=data_range)
        conv = convention_for(chosen)
        headers, rows, warns = panel_metrics_rows(chosen, panels, ref_panel, defs, metrics=metrics or ("psnr",), convention=conv)
        st.subheader("Panel metrics")
        st.markdown(generic_html(headers, rows, number=i, left_cols=1,
                                 caption="Metrics of the shown images: as logged with the run, and " + conv.describe()
                                         + " against the ground truth. A gap flags a mismatch between the figure and the table."
                                         + (f" Convention recorded by the run: {conv.note}." if conv.note else "")),
                    unsafe_allow_html=True)
        for w in warns:
            st.warning(w)
        if ref_panel is not None and not warns and rows:
            st.success("Logged PSNR" + (" and SSIM" if "ssim" in metrics else "") + " match the displayed images for every panel "
                       "(within 0.05 dB" + (" / 0.005" if "ssim" in metrics else "") + ").")

    with st.expander("LaTeX figure snippet"):
        st.code(figure_tex(f"figures/{stem}.pdf", caption=vr.spec.caption_stub(), label=f"fig:visual{i if i > 1 else ''}", width=width),
                language="latex")
    with st.expander("Source files"):
        srows: list[list[Any]] = []
        for p in vr.spec.panels:
            pth = Path(p["path"]) if p.get("path") else None
            srows.append([p["title"], p.get("kind", ""), p.get("row", "") or "—", str(pth) if pth else "—"])
        st.markdown(generic_html(["Panel", "Role", "Row", "Path"], srows, left_cols=4,
                                 caption="Every image in the figure and where it came from (also in the provenance JSON)."),
                    unsafe_allow_html=True)
