"""Visual comparison page: reference | measurement | baselines | proposed, with shared crop and error scale."""

from __future__ import annotations

import json
from dataclasses import asdict

import streamlit as st

from .. import aggregate as agg
from ..export.figures import figure_bytes, figure_tex, to_grayscale_png
from ..export.visual import build_panels, list_image_files, reconstruction_figure
from .common import load_metric_defs, load_records, select_project_experiment, sidebar_db


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
        datasets = list(dict.fromkeys(r["dataset"] for r in with_art if r.get("dataset") is not None))  # logging order
        dataset = st.selectbox("Dataset", datasets) if datasets else None
        pool = [r for r in with_art if dataset is None or r.get("dataset") == dataset]
        instances = sorted({r["instance"] for r in pool if r.get("instance") is not None}, key=str)
        instance = st.selectbox("Instance", instances) if instances else None
        pool = [r for r in pool if instance is None or r.get("instance") == instance]
        seeds = sorted({r["seed"] for r in pool if r.get("seed") is not None})
        seed = st.selectbox("Seed", seeds) if seeds else None
        files = list_image_files(r["artifacts_dir"] for r in pool)
        if not files:
            st.warning("No image files found in the artifact folders.")
            return
        st.markdown("**Images**")
        default_img = next((f for f in files if "recon" in f.lower()), files[0])
        image = st.selectbox("Reconstruction file", files, index=files.index(default_img))
        none = "— none —"
        ref_default = next((f for f in files if any(k in f.lower() for k in ("ground_truth", "gt", "reference", "clean"))), none)
        reference = st.selectbox("Reference (ground truth)", [none] + files, index=([none] + files).index(ref_default))
        meas_default = next((f for f in files if any(k in f.lower() for k in ("measurement", "input", "degraded", "noisy", "observ"))), none)
        measurement = st.selectbox("Measurement / input", [none] + files, index=([none] + files).index(meas_default))
        metrics_all = agg.metric_names(pool)
        metrics = st.multiselect("Metrics under panels", metrics_all, default=[m for m in metrics_all if m in ("psnr", "ssim")][:2])

    ordered = agg.select_runs(pool, dataset=dataset, seed=seed, instance=instance)
    method_names = [r["method"] for r in ordered]
    methods = st.multiselect("Methods (baselines first, proposed last; drag to reorder)", method_names, default=method_names)
    chosen = agg.select_runs(pool, dataset=dataset, seed=seed, instance=instance, methods=methods)
    for label, why in agg.omitted_methods(recs, chosen, dataset=dataset).items():
        st.warning(f"Not shown: {label} — {why}.")

    c1, c2, c3, c4, c5 = st.columns(5)
    use_crop = c1.checkbox("Zoom crop", value=True)
    error_maps = c2.checkbox("Error maps", value=reference != none, disabled=reference == none)
    width = c3.selectbox("Width", ["double", "single"])
    gray = c4.checkbox("Grayscale preview", value=False, help="Check that the figure still reads in print.")
    labels = c5.checkbox("(a), (b) labels", value=False)

    panels, ref_panel, problems = build_panels(
        chosen, image, defs, metrics=metrics, reference=None if reference == none else reference,
        measurement=None if measurement == none else measurement,
    )
    for p in problems:
        st.warning(p)
    if not panels:
        st.error("Nothing to show.")
        return
    h, w = panels[0].image.shape[:2]

    crop_box = None
    if use_crop:
        st.caption(f"Image size {w} × {h}. The same crop is applied to every panel.")
        k1, k2, k3, k4 = st.columns(4)
        cw = k3.number_input("crop width", 4, w, min(32, w))
        ch = k4.number_input("crop height", 4, h, min(32, h))
        cx = k1.number_input("crop x", 0, max(0, w - int(cw)), min(w // 3, max(0, w - int(cw))))
        cy = k2.number_input("crop y", 0, max(0, h - int(ch)), min(h // 3, max(0, h - int(ch))))
        crop_box = (int(cx), int(cy), int(cw), int(ch))

    fig, spec = reconstruction_figure(panels, reference=ref_panel, crop_box=crop_box, error_maps=error_maps,
                                      width=width, panel_labels=labels)
    spec.experiment, spec.dataset, spec.instance, spec.seed, spec.image = experiment, dataset, instance, seed, image
    spec.measurement = None if measurement == none else measurement
    png = figure_bytes(fig, "png", dpi=200)
    st.image(to_grayscale_png(png) if gray else png)
    fw, fh = fig.get_size_inches()
    st.caption(f"{fw:.2f} × {fh:.2f} in · nearest-neighbour rendering · shared display range and error scale")

    stem = f"{experiment}-{dataset or 'all'}-seed{seed}-visual".replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download PNG (300 dpi)", figure_bytes(fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png")
    d2.download_button("Download PDF", figure_bytes(fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
    d3.download_button("Download provenance JSON", json.dumps(asdict(spec), indent=2, default=str),
                       file_name=f"{stem}.json", mime="application/json")

    st.subheader("Caption material")
    st.write(spec.caption_stub())
    with st.expander("LaTeX figure snippet"):
        st.code(figure_tex(f"figures/{stem}.png", caption=spec.caption_stub(), label="fig:visual", width=width), language="latex")
    with st.expander("Source files"):
        st.write([{"panel": p["title"], "path": p["path"]} for p in spec.panels])
