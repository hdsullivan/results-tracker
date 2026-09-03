"""Export page: LaTeX tables and IEEE figures for the current experiment, with preview, download and pinning.

Every widget is keyed under `exp_*` so an asset opened from the Paper page (`?asset=tab:main`) can prefill the
page with the options it was pinned with (`prefill_from_asset`); pinning again saves the current options back.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Optional

import streamlit as st

from .. import aggregate as agg
from ..export.bundle import build_bundle
from ..export.csv import runs_csv
from ..export.figures import ablation_figure, comparison_figure, figure_bytes, figure_tex, ieee_preamble, sweep_figure, to_grayscale_png
from ..export.latex import ablation_latex, comparison_latex, provenance_note, sweep_latex, width_hint
from ..export.paper import KIND_PAGE, KIND_TITLES, KINDS
from ..export.visual import ZOOM_FRACTION, guess_roles, list_image_files, make_visual
from .common import (
    active_where, db_path, hib_map, keyed, keyed_multiselect, keyed_radio, keyed_selectbox, load_catalog, load_metric_defs,
    load_records, load_records_union, pin_to_paper, reset_on_experiment_change, select_extra_experiments, select_project_experiment,
    sidebar_db, sidebar_filter, where_cli, where_text,
)
from .tables import ablation_html, comparison_html, generic_html, sweep_html

BUNDLE = "Paper bundle (zip)"
TITLES = [KIND_TITLES[k] for k in KINDS] + [BUNDLE]
KIND_BY_TITLE = {v: k for k, v in KIND_TITLES.items()}
PREFERRED = {"comparison": "comparison-table", "ablation": "ablation-table", "sweep": "sweep-figure"}
NONE = "— none —"
WIDTHS = ["single", "double", "ieee-single", "ieee-double"]
FONTS = ["(default)", "small", "footnotesize", "scriptsize"]


def prefill_from_asset(a) -> dict[str, Any]:
    """Widget states (key -> value) that reproduce an asset's options on this page."""
    o = dict(a.options or {})
    pre: dict[str, Any] = {"exp_kind": KIND_TITLES.get(a.kind, TITLES[0]), "exp_label": a.label, "exp_caption": a.caption or ""}

    def table_opts() -> None:
        env = o["env"] if "env" in o else "table"
        pre.update({"exp_env": env or "none", "exp_std": o.get("std", "pm"), "exp_font": o.get("font") or "(default)"})

    def put(key: str, opt: str, default: Any = None, convert=lambda v: v) -> None:
        if opt in o and o[opt] is not None:
            pre[key] = convert(o[opt])
        elif default is not None:
            pre[key] = default

    k = a.kind
    if k == "comparison-table":
        table_opts()
        put("exp_rows", "rows"); put("exp_cols", "cols", convert=lambda v: v or "none"); put("exp_metrics", "metrics"); put("exp_underline", "underline")
        if o.get("cols", "dataset") is None:
            pre["exp_cols"] = "none"
    elif k == "ablation-table":
        table_opts()
        put("exp_metrics", "metrics"); put("exp_show_delta", "show_delta"); put("exp_settings", "setting_columns")
    elif k in ("sweep-table", "sweep-figure"):
        put("exp_param", "param"); put("exp_metric", "metric"); put("exp_by", "by")
        if k == "sweep-table":
            table_opts()
            put("exp_param_label", "param_label")
        else:
            put("exp_xlabel", "xlabel"); put("exp_ylabel", "ylabel"); put("exp_width", "width"); put("exp_band", "band")
            put("exp_emph", "emphasize"); put("exp_height", "height", convert=float); put("exp_panel", "panel_label")
    elif k == "ablation-figure":
        put("exp_metric", "metric"); put("exp_xlabel", "xlabel"); put("exp_width", "width"); put("exp_panel", "panel_label")
    elif k == "comparison-figure":
        put("exp_metric", "metric"); put("exp_rows", "rows"); put("exp_cols", "cols", convert=lambda v: v or "none")
        put("exp_ylabel", "ylabel"); put("exp_width", "width"); put("exp_emph", "emphasize"); put("exp_zero", "zero_based")
        put("exp_hatch", "hatch"); put("exp_panel", "panel_label")
    elif k == "visual-figure":
        put("exp_vdataset", "dataset"); put("exp_vrows", "rows", default=NONE); put("exp_vseed", "seed")
        pre["exp_vmode"] = "Error maps" if o.get("mode") == "error" else "Reconstruction"
        put("exp_vimage", "image"); put("exp_vref", "reference", default=NONE); put("exp_vmeas", "measurement", default=NONE)
        put("exp_vzoom", "zoom"); put("exp_vzf", "zoom_fraction", convert=float)
        if o.get("zoom_center"):
            pre["exp_vzc"] = float(o["zoom_center"][0])
    return pre


def _latex_block(tex: str, filename: str, preview: Optional[str] = None) -> None:
    if preview:
        st.markdown("**Preview** (how it will look once compiled)")
        st.markdown(preview, unsafe_allow_html=True)
    st.code(tex, language="latex")
    st.download_button("Download .tex", tex, file_name=filename, mime="text/x-tex")
    with st.expander("Preamble packages"):
        st.code(ieee_preamble(), language="latex")
        st.caption("IEEEtran sets caption style (small caps above tables) and column widths; the generated code only "
                   "uses booktabs rules, so it inherits the class's look.")


def _figure_block(fig, stem: str, width: str = "single") -> None:
    png = figure_bytes(fig, "png", dpi=200)
    w, h = fig.get_size_inches()
    gray = st.checkbox("Grayscale preview", value=False, key=f"gray_{stem}",
                       help="IEEE readers may print in black and white; line styles, markers and hatching must carry the identity.")
    st.image(to_grayscale_png(png) if gray else png, caption=f"{w:.2f} × {h:.2f} in at 200 dpi preview; the PDF is vector.")
    c1, c2 = st.columns(2)
    c1.download_button("Download PDF (vector)", figure_bytes(fig, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
    c2.download_button("Download PNG (300 dpi)", figure_bytes(fig, "png", dpi=300), file_name=f"{stem}.png", mime="image/png")
    with st.expander("LaTeX figure snippet"):
        st.code(figure_tex(f"figures/{stem}.pdf", caption="TODO: what is plotted, aggregation unit, uncertainty, n.",
                           label=f"fig:{stem}", width=width), language="latex")


def _common_table_opts():
    c1, c2, c3 = st.columns(3)
    with c1:
        env = keyed_selectbox("Environment", ["table", "table*", "none"], "exp_env", "table", help="none = bare tabular")
    with c2:
        std = keyed_selectbox("Std style", ["pm", "small", "none"], "exp_std", "pm", help="pm: 30.96 ± 0.39 · small: scriptsize ± · none")
    with c3:
        font = keyed_selectbox("Font size", FONTS, "exp_font", "(default)")
    label = keyed(st.text_input, "\\label", "exp_label", "")
    caption = keyed(st.text_area, "Caption (blank = auto-generated)", "exp_caption", "", height=70)
    return (None if env == "none" else env), std, (None if font == "(default)" else font), (label or None), (caption or None)


def render() -> None:
    st.title("Export")
    sidebar_db()
    project, experiment = select_project_experiment()
    if experiment is None:
        return
    extra = select_extra_experiments(project, experiment)
    recs_all = load_records_union(project, [experiment, *extra])
    defs = load_metric_defs()
    if not recs_all:
        st.info("No runs in this experiment.")
        return
    recs_all = sidebar_filter(recs_all)
    recs = agg.completed(recs_all)
    if not recs:
        if not active_where():
            st.info("No completed runs in this experiment.")
        return
    exp_type = recs[0].get("experiment_type") or "comparison"
    metrics_all = agg.metric_names(recs)
    hib = hib_map(defs)
    prov = provenance_note(db_path(), " + ".join([experiment, *extra]), len(recs_all), extra=f"Filter: {where_cli()}" if active_where() else "")
    if extra:
        st.caption(f"Pooling **{experiment}** with {', '.join(f'**{e}**' for e in extra)} ({len(recs_all)} runs); "
                   "use `experiment` or a distinguishing config key as a row or column key.")
    opened = st.session_state.get("opened_asset")
    if opened and opened[1] == experiment:
        st.caption(f"Asset **`{opened[0]}`** opened from the Paper page: the options below are the ones it was pinned with. "
                   "Change them and pin again to update it.")
    if active_where():
        st.caption(f"Filtered to {len(recs_all)} runs: {where_text()} · on the command line add `{where_cli()}`; "
                   "the filter is recorded in the provenance comment of every table.")

    reset_on_experiment_change("exp_", experiment)  # options restart from the new experiment's defaults
    with st.sidebar:
        st.markdown("**Export**")
        title = keyed_radio("What to export", TITLES, "exp_kind", KIND_TITLES[PREFERRED.get(exp_type, "comparison-table")])
    kind = KIND_BY_TITLE.get(title)

    group_keys = agg.grouping_keys(recs)  # experiment (when pooled), method, dataset, ..., config.*, derived.*
    stem = experiment.replace(" ", "_")

    def orders(row_key: str, col_key: Optional[str]) -> dict[str, Any]:
        return dict(row_order=agg.method_order(recs) if row_key == "method" else agg.value_order(recs, row_key),
                    col_order=agg.value_order(recs, col_key) if col_key else None)

    def pin(options: dict[str, Any], label: Optional[str] = None, caption: Optional[str] = None) -> None:
        assert kind is not None
        pin_to_paper({kind: options}, records=recs_all, key="exp_pin", suggested_label=label, caption=caption, extra_experiments=extra)

    if kind in KIND_PAGE:
        st.info(f"{KIND_TITLES[kind]}s are configured and pinned on the **{KIND_PAGE[kind].capitalize()}** page; "
                "`results-tracker export paper` renders them like every other asset.")
        return

    if kind == "comparison-table":
        c1, c2 = st.columns(2)
        with c1:
            row_key = keyed_selectbox("Rows", group_keys, "exp_rows", "method")
        col_opts = ["none"] + [k for k in group_keys if k != row_key]
        with c2:
            col_key = keyed_selectbox("Column groups", col_opts, "exp_cols",
                                      "experiment" if extra else ("dataset" if "dataset" in col_opts else "none"))
        metrics = keyed_multiselect("Metrics", metrics_all, "exp_metrics", metrics_all)
        underline = keyed(st.checkbox, "Underline second best", "exp_underline", True)
        env, std, font, label, caption = _common_table_opts()
        if not metrics:
            st.warning("Pick at least one metric.")
            return
        ck = None if col_key == "none" else col_key
        pt = agg.pivot_table(recs, row_key, ck, metrics=metrics, higher_is_better=hib, **orders(row_key, ck))
        audit = agg.audit_grid(recs_all, [row_key] + ([ck] if ck else []))
        if audit.missing or audit.failed or audit.uneven or audit.coverage:
            st.warning("Audit: " + audit.summary() + (". Missing cells are rendered as `--`." if audit.missing else "."))
            if audit.missing:
                st.write([" · ".join(f"{k}={v}" for k, v in zip(audit.keys, m)) for m in audit.missing[:20]])
            for c in audit.coverage:
                st.warning("Rows are pooled over different " + c)
        else:
            st.success("Audit: " + audit.summary())
        hint = width_hint(pt, std, env, font)
        if hint:
            st.info(hint)
        tex = comparison_latex(pt, defs, caption=caption, label=label, env=env, font=font, std=std,
                               underline_second=underline,
                               row_labels=agg.method_labels(recs, latex=True) if row_key == "method" else None,
                               audit=audit, provenance=prov)
        _latex_block(tex, f"{stem}-table.tex", preview=comparison_html(
            pt, defs, caption=caption, show_std=std != "none", underline_second=underline,
            row_labels=agg.method_labels(recs) if row_key == "method" else None))
        pin({"rows": row_key, "cols": ck, "metrics": metrics, "underline": underline, "env": env, "std": std, "font": font}, label, caption or "")

    elif kind == "ablation-table":
        metrics = keyed_multiselect("Metrics", metrics_all, "exp_metrics", metrics_all)
        c1, c2 = st.columns(2)
        with c1:
            show_delta = keyed(st.checkbox, "Show Δ vs full model", "exp_show_delta", True)
        with c2:
            settings = keyed(st.checkbox, "Per-setting ✓/✗ columns", "exp_settings", True)
        env, std, font, label, caption = _common_table_opts()
        if not metrics:
            st.warning("Pick at least one metric.")
            return
        try:
            rows = agg.ablation_table(recs, metrics=metrics)
        except agg.AmbiguousBaseError as e:
            st.error(str(e))
            return
        if not any(r.is_base for r in rows):
            st.warning("No run matches the base config; deltas will be missing. Tag a run `base`.")
        tex = ablation_latex(rows, metrics, defs, caption=caption, label=label, env=env, font=font, std=std,
                             show_delta=show_delta, setting_columns=settings, provenance=prov)
        _latex_block(tex, f"{stem}-ablation.tex", preview=ablation_html(
            rows, metrics, defs, caption=caption, show_std=std != "none", setting_columns=settings))
        pin({"metrics": metrics, "show_delta": show_delta, "setting_columns": settings, "env": env, "std": std, "font": font}, label, caption or "")

    elif kind in ("sweep-table", "sweep-figure"):
        all_keys = sorted({k for r in recs for k in agg.flatten(r["config"])})
        varying = agg.varying_config_keys(recs)
        if not all_keys:
            st.warning("Runs have no config keys to sweep over.")
            return
        c1, c2, c3 = st.columns(3)
        with c1:
            param = keyed_selectbox("Parameter", all_keys, "exp_param", varying[0] if varying else all_keys[0])
        with c2:
            metric = keyed_selectbox("Metric", metrics_all, "exp_metric", metrics_all[0])
        from .sweep import line_keys

        with c3:
            by = keyed_multiselect("One column/line per", line_keys(recs, param), "exp_by", [],
                                   help="method arm, condition (config.noise, derived.kernel_type, ...)")
        series = {g: s for g, s in agg.sweep_series(recs, param, metric, group_by=by).items() if s}
        if not series:
            st.warning(f"No runs have `{param}` in their config.")
            return
        if kind == "sweep-table":
            param_label = keyed(st.text_input, "Parameter label (LaTeX)", "exp_param_label", param)
            env, std, font, label, caption = _common_table_opts()
            tex = sweep_latex(series, param, metric, defs, caption=caption, label=label, env=env, font=font, std=std,
                              param_label=param_label, provenance=prov)
            _latex_block(tex, f"{stem}-{param}.tex", preview=sweep_html(
                series, param, metric, defs, caption=caption, show_std=std != "none", param_label=param_label))
            pin({"param": param, "metric": metric, "by": by, "param_label": param_label, "env": env, "std": std, "font": font}, label, caption or "")
        else:
            unit = defs.get(metric, {}).get("unit", "")
            c1, c2, c3 = st.columns(3)
            with c1:
                xlabel = keyed(st.text_input, "x label", "exp_xlabel", param)
            with c2:
                ylabel = keyed(st.text_input, "y label", "exp_ylabel", f"{metric} ({unit})" if unit else metric)
            with c3:
                width = keyed_selectbox("Width", WIDTHS, "exp_width", "single",
                                        help="single/double = 5.0/10.5 in (lab convention, LaTeX scales down); ieee-* = literal 3.5/7.16 in")
            c1, c2, c3 = st.columns(3)
            with c1:
                band = keyed(st.checkbox, "Shaded ± std band", "exp_band", True, help="Off: capped error bars.")
            with c2:
                emph = keyed_multiselect("Emphasize (proposed)", [" / ".join(map(str, g)) for g in series if g], "exp_emph", [])
            with c3:
                height = keyed(st.number_input, "Height (in)", "exp_height", 3.1, min_value=1.0, max_value=8.0, step=0.1)
            cap = keyed(st.text_input, "Panel caption (bold, below)", "exp_panel", "", placeholder="a. PSNR vs λ")
            best = {g: agg.best_sweep_value(s, hib.get(metric, True)) for g, s in series.items()}
            fig = sweep_figure(series, param, metric, xlabel=xlabel, ylabel=ylabel, band=band, best_by_group=best,
                               width=width, height=height, emphasize=emph, caption=cap or None)
            _figure_block(fig, f"{stem}-{param}-{metric}", width)
            pin({"param": param, "metric": metric, "by": by, "xlabel": xlabel, "ylabel": ylabel, "width": width, "band": band,
                 "emphasize": emph, "height": height, "panel_label": cap or None})

    elif kind == "ablation-figure":
        c1, c2, c3 = st.columns(3)
        with c1:
            metric = keyed_selectbox("Metric", metrics_all, "exp_metric", metrics_all[0])
        unit = defs.get(metric, {}).get("unit", "")
        with c2:
            xlabel = keyed(st.text_input, "x label", "exp_xlabel", f"$\\Delta$ {metric} vs. full model" + (f" ({unit})" if unit else ""))
        with c3:
            width = keyed_selectbox("Width", WIDTHS, "exp_width", "single")
        cap = keyed(st.text_input, "Panel caption (bold, below)", "exp_panel", "", placeholder="b. Ablation")
        try:
            rows = agg.ablation_table(recs, metrics=[metric])
        except agg.AmbiguousBaseError as e:
            st.error(str(e))
            return
        if not any(r.is_base for r in rows):
            st.warning("No run matches the base config; nothing to plot. Tag a run `base`.")
            return
        d = defs.get(metric, {})
        fig = ablation_figure(rows, metric, higher_is_better=d.get("higher_is_better", True), fmt=d.get("fmt", ".2f"),
                              xlabel=xlabel, width=width, caption=cap or None)
        _figure_block(fig, f"{stem}-ablation-{metric}", width)
        pin({"metric": metric, "xlabel": xlabel, "width": width, "panel_label": cap or None})

    elif kind == "comparison-figure":
        c1, c2, c3 = st.columns(3)
        with c1:
            metric = keyed_selectbox("Metric", metrics_all, "exp_metric", metrics_all[0])
        with c2:
            row_key = keyed_selectbox("Bars", group_keys, "exp_rows", "method")
        col_opts = ["none"] + [k for k in group_keys if k != row_key]
        with c3:
            col_key = keyed_selectbox("x groups", col_opts, "exp_cols", "experiment" if extra else ("dataset" if "dataset" in col_opts else "none"))
        unit = defs.get(metric, {}).get("unit", "")
        c1, c2, c3 = st.columns(3)
        with c1:
            ylabel = keyed(st.text_input, "y label", "exp_ylabel", f"{metric} ({unit})" if unit else metric)
        with c2:
            width = keyed_selectbox("Width", WIDTHS, "exp_width", "single")
        pt = agg.pivot_table(recs, row_key, None if col_key == "none" else col_key, metrics=[metric], higher_is_better=hib,
                             **orders(row_key, None if col_key == "none" else col_key))
        with c3:
            emph = keyed_multiselect("Emphasize (proposed)", [str(r) for r in pt.rows], "exp_emph", [])
        k1, k2, k3 = st.columns(3)
        with k1:
            zero = keyed(st.checkbox, "y axis from 0", "exp_zero", False, help="Default is data-tight; say which in the caption.")
        with k2:
            hatch = keyed(st.checkbox, "Hatch bars", "exp_hatch", False, help="Grayscale print safety.")
        with k3:
            cap = keyed(st.text_input, "Panel caption", "exp_panel", "", placeholder="a. PSNR")
        fig = comparison_figure(pt, metric, ylabel=ylabel, width=width, emphasize=emph, zero_based=zero, hatch=hatch,
                                caption=cap or None, row_labels=agg.method_labels(recs) if row_key == "method" else None)
        _figure_block(fig, f"{stem}-{metric}", width)
        pin({"metric": metric, "rows": row_key, "cols": None if col_key == "none" else col_key, "ylabel": ylabel, "width": width,
             "emphasize": emph, "zero_based": zero, "hatch": hatch, "panel_label": cap or None})

    elif kind == "visual-figure":
        with_art = [r for r in recs if r.get("artifacts_dir")]
        if not with_art:
            st.info("No runs in this experiment have an `artifacts_dir`.")
            return
        c1, c2, c3, c4 = st.columns(4)
        datasets = list(dict.fromkeys(r["dataset"] for r in with_art if r.get("dataset") is not None))
        with c1:
            dataset = keyed_selectbox("Dataset", datasets, "exp_vdataset", datasets[0]) if datasets else None
        pool = [r for r in with_art if dataset is None or r.get("dataset") == dataset]
        seeds = sorted({r["seed"] for r in pool if r.get("seed") is not None})
        row_opts = [NONE] + (["seed"] if len(seeds) > 1 else []) + [f"config.{k}" for k in agg.varying_config_keys(pool)]
        with c2:
            rows_by = keyed_selectbox("Rows", row_opts, "exp_vrows", NONE)
        rows_by = None if rows_by == NONE else rows_by
        with c3:
            seed = keyed_selectbox("Seed", seeds, "exp_vseed", seeds[0]) if seeds and rows_by != "seed" else None
        with c4:
            vmode = keyed_radio("Mode", ["Reconstruction", "Error maps"], "exp_vmode", "Reconstruction", horizontal=True)
        files = list_image_files(r["artifacts_dir"] for r in pool)
        roles = guess_roles(files)
        with st.expander("Files and zoom", expanded=False):
            f1, f2, f3 = st.columns(3)
            with f1:
                image = keyed_selectbox("Reconstruction", files, "exp_vimage", roles["reconstruction"] or (files[0] if files else None))
            with f2:
                reference = keyed_selectbox("Ground truth", [NONE] + files, "exp_vref", roles["reference"] or NONE)
            with f3:
                measurement = keyed_selectbox("Measurement", [NONE] + files, "exp_vmeas", roles["measurement"] or NONE)
            z1, z2, z3 = st.columns(3)
            with z1:
                zoom = keyed(st.checkbox, "Zoom inset", "exp_vzoom", True, disabled=vmode == "Error maps")
            with z2:
                zf = keyed(st.slider, "Box side fraction", "exp_vzf", ZOOM_FRACTION, min_value=0.1, max_value=0.6, step=0.05)
            with z3:
                zc = keyed(st.slider, "Box centre (x = y)", "exp_vzc", 0.5, min_value=0.0, max_value=1.0, step=0.05)
        try:
            vr = make_visual(recs, defs, experiment=experiment, dataset=dataset, seed=seed, image=image,
                             reference=None if reference == NONE else reference, measurement=None if measurement == NONE else measurement,
                             mode="error" if vmode == "Error maps" else "image", zoom=zoom, zoom_fraction=zf, zoom_center=(zc, zc),
                             rows=rows_by, width="double", auto_roles=False)
        except ValueError as e:
            st.error(str(e))
            return
        for label, why in vr.omitted.items():
            st.warning(f"Not shown: {label} — {why}.")
        for pr in vr.problems:
            st.warning(pr)
        vstem = f"{stem}-{dataset or 'all'}" + ("-error" if vmode == "Error maps" else "-visual")
        _figure_block(vr.fig, vstem, "double")
        st.write(vr.spec.caption_stub())
        st.download_button("Download provenance JSON", json.dumps(asdict(vr.spec), indent=2, default=str),
                           file_name=f"{vstem}.json", mime="application/json")
        pin({"dataset": dataset, "seed": seed, "image": image, "reference": None if reference == NONE else reference,
             "measurement": None if measurement == NONE else measurement, "mode": "error" if vmode == "Error maps" else "image",
             "zoom": zoom, "zoom_fraction": zf, "zoom_center": [zc, zc], "rows": rows_by, "width": "double"})

    elif kind == "runs-csv":
        text = runs_csv(recs_all)
        st.caption(f"{len(recs_all)} runs (including failed), config keys and metrics as columns.")
        lines = text.splitlines()
        header = lines[0].split(",")
        body = [ln.split(",") for ln in lines[1:9]]
        st.markdown(generic_html(header, body, caption=f"First {len(body)} of {len(recs_all)} rows of the runs CSV.", left_cols=3),
                    unsafe_allow_html=True)
        st.download_button("Download CSV", text, file_name=f"{stem}-runs.csv", mime="text/csv")
        pin({})

    elif title == BUNDLE:
        cat = load_catalog()
        exps = [e for e in cat["experiments"] if e["project"] == project]
        c1, c2 = st.columns(2)
        width = c1.selectbox("Quantitative figure width", ["single", "double"])
        with_visual = c2.checkbox("Include qualitative image figures", value=True)
        st.caption(f"Regenerates a default table, figure and CSV for every experiment of **{project}** ({len(exps)} experiments), "
                   "with the preamble and a provenance manifest. For the manuscript's own pinned tables and figures use the Paper page "
                   "or `results-tracker export paper`.")
        if st.button("Build bundle", type="primary"):
            experiments = {e["experiment"]: (e["type"], load_records(project, e["experiment"])) for e in exps}
            with st.spinner("Rendering tables and figures…"):
                data, manifest = build_bundle(experiments, defs, project=project, source=db_path(), width=width, visual=with_visual)
            st.session_state["bundle"] = (data, manifest)
        if "bundle" in st.session_state:
            data, manifest = st.session_state["bundle"]
            rows_m = [[m["file"] or "—", m["kind"], m["experiment"], m["runs"], m.get("note", "")] for m in manifest]
            st.markdown(generic_html(["File", "Kind", "Experiment", "Runs", "Note"], rows_m,
                                     caption=f"Contents of the paper bundle for {project}: {sum(1 for m in manifest if m['file'])} files, "
                                             f"{len(data) // 1024} KB.", left_cols=3), unsafe_allow_html=True)
            st.download_button("Download paper bundle (zip)", data, file_name=f"{project}-paper-bundle.zip", mime="application/zip")
