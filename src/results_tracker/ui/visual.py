"""Visual comparison page in the lab's IEEE reconstruction-figure style.

Reference | Measurement block, spacer, baselines -> proposed. Zoom inset, metric stamps, error-map mode,
optional rows (seed / instance / a config key such as K) and an optional kernel thumbnail.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import streamlit as st

from .. import aggregate as agg
from ..export.figures import figure_bytes, figure_tex, to_grayscale_png
from ..export.visual import ZOOM_FRACTION, build_panels, build_rows, list_image_files, reconstruction_figure
from .common import load_metric_defs, load_records, select_project_experiment, sidebar_db

NONE = "— none —"


def _pick(files: list[str], keys: tuple[str, ...]) -> str:
    return next((f for f in files if any(k in f.lower() for k in keys)), NONE)


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
        pool_sel = [r for r in pool if seed is None or r.get("seed") == seed]
        files = list_image_files(r["artifacts_dir"] for r in pool)
        if not files:
            st.warning("No image files found in the artifact folders.")
            return
        st.markdown("**Images**")
        image = st.selectbox("Reconstruction file", files, index=files.index(_pick(files, ("recon",)) if _pick(files, ("recon",)) != NONE else files[0]))
        reference = st.selectbox("Ground truth", [NONE] + files, index=([NONE] + files).index(_pick(files, ("ground_truth", "gt", "reference", "clean"))))
        measurement = st.selectbox("Measurement / input", [NONE] + files, index=([NONE] + files).index(_pick(files, ("measurement", "input", "degraded", "noisy", "observ"))))
        kernel = st.selectbox("Kernel / PSF thumbnail", [NONE] + files, index=([NONE] + files).index(_pick(files, ("kernel", "psf"))))
        metrics_all = agg.metric_names(pool)
        metrics = st.multiselect("Metrics stamped on panels", metrics_all, default=[m for m in metrics_all if m in ("psnr", "ssim")][:2])

    ordered = agg.select_runs(pool_sel)
    method_names = list(dict.fromkeys(r["method"] for r in ordered))
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

    # panels
    ref_arg = None if reference == NONE else reference
    meas_arg = None if measurement == NONE else measurement
    ker_arg = None if kernel == NONE else kernel
    if row_key:
        shown = [r for r in pool if r.get("artifacts_dir")]
        for label, why in agg.omitted_methods(recs, shown, dataset=dataset).items():
            st.warning(f"Not shown: {label} — {why}.")
        rows, problems = build_rows(pool, row_key, image, defs, metrics=metrics, methods=methods, reference=ref_arg)
        aux, ref_panel, probs2 = build_panels(shown[:1], image, defs, reference=ref_arg, measurement=meas_arg, kernel=ker_arg)
        meas_panel = next((p for p in aux if p.kind == "measurement"), None)
        ker_panel = next((p for p in aux if p.kind == "kernel"), None)
        problems += [pr for pr in probs2 if pr.startswith(("reference", "measurement", "kernel"))]
        panel_arg = rows
    else:
        chosen = agg.select_runs(pool_sel, methods=methods)
        for label, why in agg.omitted_methods(recs, chosen, dataset=dataset).items():
            st.warning(f"Not shown: {label} — {why}.")
        panels, ref_panel, problems = build_panels(chosen, image, defs, metrics=metrics, reference=ref_arg,
                                                   measurement=meas_arg, kernel=ker_arg)
        meas_panel = next((p for p in panels if p.kind == "measurement"), None)
        ker_panel = next((p for p in panels if p.kind == "kernel"), None)
        panel_arg = [p for p in panels if p.kind == "method"]
        rows = None
    for pr in problems:
        st.warning(pr)
    if not panel_arg or (rows is not None and not any(r.panels for r in rows)):
        st.error("Nothing to show.")
        return

    try:
        fig, spec = reconstruction_figure(
            panel_arg, reference=ref_panel, measurement=meas_panel, kernel=ker_panel,
            mode="error" if mode == "Error maps" else "image", zoom=zoom, zoom_fraction=zoom_fraction,
            zoom_center=zoom_center, crop_box=crop_box, width=width,
        )
    except ValueError as e:
        st.error(str(e))
        return
    spec.experiment, spec.dataset, spec.instance, spec.seed, spec.image = experiment, dataset, instance, seed, image
    png = figure_bytes(fig, "png", dpi=200)
    st.image(to_grayscale_png(png) if gray else png)
    fw, fh = fig.get_size_inches()
    st.caption(f"{fw:.2f} × {fh:.2f} in · IEEE text width · native pixels · shared display range"
               + (" and error scale" if mode == "Error maps" else ""))

    stem = f"{experiment}-{dataset or 'all'}" + (f"-seed{seed}" if seed is not None else "") + \
           ("-error" if mode == "Error maps" else ("-zoom" if zoom else "")) + "-visual"
    stem = stem.replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download PNG (300 dpi)", figure_bytes(fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png")
    d2.download_button("Download PDF", figure_bytes(fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
    d3.download_button("Download provenance JSON", json.dumps(asdict(spec), indent=2, default=str),
                       file_name=f"{stem}.json", mime="application/json")

    st.subheader("Caption material")
    st.write(spec.caption_stub())
    with st.expander("LaTeX figure snippet"):
        st.code(figure_tex(f"figures/{stem}.pdf", caption=spec.caption_stub(), label="fig:visual", width=width), language="latex")
    with st.expander("Source files"):
        st.write([{"panel": p["title"], "path": p["path"], **({"row": p["row"]} if p.get("row") else {})} for p in spec.panels])
