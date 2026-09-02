"""Export page: LaTeX tables and IEEE figures for the current experiment, with preview and download."""

from __future__ import annotations

import streamlit as st

from .. import aggregate as agg
from ..export.csv import runs_csv
from ..export.figures import ablation_figure, comparison_figure, figure_bytes, figure_tex, ieee_preamble, sweep_figure, to_grayscale_png
from ..export.latex import ablation_latex, comparison_latex, provenance_note, sweep_latex, width_hint
from .common import db_path, hib_map, load_metric_defs, load_records, select_project_experiment, sidebar_db

KINDS = ["Comparison table (LaTeX)", "Ablation table (LaTeX)", "Sweep table (LaTeX)",
         "Sweep figure", "Ablation figure", "Comparison figure", "Runs (CSV)"]
PREFERRED = {"comparison": 0, "ablation": 1, "sweep": 3}


def _latex_block(tex: str, filename: str) -> None:
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
    env = c1.selectbox("Environment", ["table", "table*", "none"], help="none = bare tabular")
    std = c2.selectbox("Std style", ["pm", "small", "none"], help="pm: 30.96 ± 0.39 · small: scriptsize ± · none")
    font = c3.selectbox("Font size", ["(default)", "small", "footnotesize", "scriptsize"])
    label = st.text_input("\\label", value="")
    caption = st.text_area("Caption (blank = auto-generated)", value="", height=70)
    return (None if env == "none" else env), std, (None if font == "(default)" else font), (label or None), (caption or None)


def render() -> None:
    st.title("Export")
    sidebar_db()
    project, experiment = select_project_experiment()
    if experiment is None:
        return
    recs_all = load_records(project, experiment)
    recs = agg.completed(recs_all)
    defs = load_metric_defs()
    if not recs:
        st.info("No completed runs in this experiment.")
        return
    exp_type = recs[0].get("experiment_type") or "comparison"
    metrics_all = agg.metric_names(recs)
    hib = hib_map(defs)
    prov = provenance_note(db_path(), experiment, len(recs_all))

    with st.sidebar:
        st.markdown("**Export**")
        kind = st.radio("What to export", KINDS, index=PREFERRED.get(exp_type, 0))

    keys_present = [k for k in ["method", "dataset", "instance", "seed"] if any(r.get(k) is not None for r in recs)]
    config_keys = [f"config.{k}" for k in sorted({k for r in recs for k in agg.flatten(r["config"])})]
    stem = experiment.replace(" ", "_")

    if kind == "Comparison table (LaTeX)":
        c1, c2 = st.columns(2)
        row_key = c1.selectbox("Rows", keys_present + config_keys, index=0)
        col_opts = ["none"] + [k for k in keys_present + config_keys if k != row_key]
        col_key = c2.selectbox("Column groups", col_opts, index=col_opts.index("dataset") if "dataset" in col_opts else 0)
        metrics = st.multiselect("Metrics", metrics_all, default=metrics_all)
        underline = st.checkbox("Underline second best", value=True)
        env, std, font, label, caption = _common_table_opts()
        if not metrics:
            st.warning("Pick at least one metric.")
            return
        ck = None if col_key == "none" else col_key
        pt = agg.pivot_table(recs, row_key, ck, metrics=metrics, higher_is_better=hib)
        audit = agg.audit_grid(recs_all, [row_key] + ([ck] if ck else []))
        if audit.missing or audit.failed or audit.uneven:
            st.warning("Audit: " + audit.summary() + (". Missing cells are rendered as `--`." if audit.missing else "."))
            if audit.missing:
                st.write([" · ".join(f"{k}={v}" for k, v in zip(audit.keys, m)) for m in audit.missing[:20]])
        else:
            st.success("Audit: " + audit.summary())
        hint = width_hint(pt, std, env, font)
        if hint:
            st.info(hint)
        tex = comparison_latex(pt, defs, caption=caption, label=label, env=env, font=font, std=std,
                               underline_second=underline,
                               row_labels=agg.method_labels(recs) if row_key == "method" else None,
                               audit=audit, provenance=prov)
        _latex_block(tex, f"{stem}-table.tex")

    elif kind == "Ablation table (LaTeX)":
        metrics = st.multiselect("Metrics", metrics_all, default=metrics_all)
        c1, c2 = st.columns(2)
        show_delta = c1.checkbox("Show Δ vs full model", value=True)
        settings = c2.checkbox("Per-setting ✓/✗ columns", value=True)
        env, std, font, label, caption = _common_table_opts()
        if not metrics:
            st.warning("Pick at least one metric.")
            return
        rows = agg.ablation_table(recs, metrics=metrics)
        if not any(r.is_base for r in rows):
            st.warning("No run matches the base config; deltas will be missing. Tag a run `base`.")
        tex = ablation_latex(rows, metrics, defs, caption=caption, label=label, env=env, font=font, std=std,
                             show_delta=show_delta, setting_columns=settings, provenance=prov)
        _latex_block(tex, f"{stem}-ablation.tex")

    elif kind in ("Sweep table (LaTeX)", "Sweep figure"):
        all_keys = sorted({k for r in recs for k in agg.flatten(r["config"])})
        varying = agg.varying_config_keys(recs)
        if not all_keys:
            st.warning("Runs have no config keys to sweep over.")
            return
        c1, c2, c3 = st.columns(3)
        param = c1.selectbox("Parameter", all_keys, index=all_keys.index(varying[0]) if varying else 0)
        metric = c2.selectbox("Metric", metrics_all)
        group_opts = [k for k in ["method", "dataset"] if len({r.get(k) for r in recs}) > 1]
        by = c3.multiselect("One column/line per", group_opts, default=[])
        series = {g: s for g, s in agg.sweep_series(recs, param, metric, group_by=by).items() if s}
        if not series:
            st.warning(f"No runs have `{param}` in their config.")
            return
        if kind == "Sweep table (LaTeX)":
            param_label = st.text_input("Parameter label (LaTeX)", value=param)
            env, std, font, label, caption = _common_table_opts()
            tex = sweep_latex(series, param, metric, defs, caption=caption, label=label, env=env, font=font, std=std,
                              param_label=param_label, provenance=prov)
            _latex_block(tex, f"{stem}-{param}.tex")
        else:
            unit = defs.get(metric, {}).get("unit", "")
            c1, c2, c3 = st.columns(3)
            xlabel = c1.text_input("x label", value=param)
            ylabel = c2.text_input("y label", value=f"{metric} ({unit})" if unit else metric)
            width = c3.selectbox("Width", ["single", "double"])
            c1, c2, c3 = st.columns(3)
            band = c1.checkbox("Shaded ± std band", value=False, help="Default: error bars with caps.")
            emph = c2.multiselect("Emphasize", [" / ".join(map(str, g)) for g in series if g], default=[])
            height = c3.number_input("Height (in)", min_value=1.0, max_value=6.0, value=2.2, step=0.1)
            best = {g: agg.best_sweep_value(s, hib.get(metric, True)) for g, s in series.items()}
            fig = sweep_figure(series, param, metric, xlabel=xlabel, ylabel=ylabel, band=band, best_by_group=best,
                               width=width, height=height, emphasize=emph)
            _figure_block(fig, f"{stem}-{param}-{metric}", width)

    elif kind == "Ablation figure":
        c1, c2, c3 = st.columns(3)
        metric = c1.selectbox("Metric", metrics_all)
        unit = defs.get(metric, {}).get("unit", "")
        xlabel = c2.text_input("x label", value=f"$\\Delta$ {metric} vs. full model" + (f" ({unit})" if unit else ""))
        width = c3.selectbox("Width", ["single", "double"])
        rows = agg.ablation_table(recs, metrics=[metric])
        if not any(r.is_base for r in rows):
            st.warning("No run matches the base config; nothing to plot. Tag a run `base`.")
            return
        d = defs.get(metric, {})
        fig = ablation_figure(rows, metric, higher_is_better=d.get("higher_is_better", True), fmt=d.get("fmt", ".2f"),
                              xlabel=xlabel, width=width)
        _figure_block(fig, f"{stem}-ablation-{metric}", width)

    elif kind == "Comparison figure":
        c1, c2, c3 = st.columns(3)
        metric = c1.selectbox("Metric", metrics_all)
        row_key = c2.selectbox("Bars", keys_present + config_keys, index=0)
        col_opts = ["none"] + [k for k in keys_present + config_keys if k != row_key]
        col_key = c3.selectbox("x groups", col_opts, index=col_opts.index("dataset") if "dataset" in col_opts else 0)
        unit = defs.get(metric, {}).get("unit", "")
        c1, c2, c3 = st.columns(3)
        ylabel = c1.text_input("y label", value=f"{metric} ({unit})" if unit else metric)
        width = c2.selectbox("Width", ["single", "double"])
        pt = agg.pivot_table(recs, row_key, None if col_key == "none" else col_key, metrics=[metric], higher_is_better=hib)
        emph = c3.multiselect("Emphasize", [str(r) for r in pt.rows], default=[])
        zero = st.checkbox("y axis from 0", value=False, help="Default is data-tight; say which in the caption.")
        fig = comparison_figure(pt, metric, ylabel=ylabel, width=width, emphasize=emph, zero_based=zero,
                                row_labels=agg.method_labels(recs) if row_key == "method" else None)
        _figure_block(fig, f"{stem}-{metric}", width)

    elif kind == "Runs (CSV)":
        text = runs_csv(recs_all)
        st.caption(f"{len(recs_all)} runs (including failed), config keys and metrics as columns.")
        st.code("\n".join(text.splitlines()[:8]) + ("\n..." if len(recs_all) > 7 else ""), language="text")
        st.download_button("Download CSV", text, file_name=f"{stem}-runs.csv", mime="text/csv")
