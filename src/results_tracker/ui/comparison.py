"""Comparison page: methods x metrics, mean ± std, best in bold."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from ..export.figures import distribution_figure, figure_bytes, figure_tex, to_grayscale_png
from .charts import comparison_bars, distribution_box
from .tables import comparison_html, figure_caption_html, flat_html, generic_html
from .common import (active_where, fmt_for, hib_map, keyed, keyed_multiselect, keyed_selectbox, load_metric_defs, load_records_union,
                     pin_to_paper, reset_on_experiment_change, select_extra_experiments, select_project_experiment, sidebar_db,
                     sidebar_filter, where_text)

BASE_KEYS = ["method", "dataset", "instance", "seed"]


def to_frame(ct: agg.ComparisonTable) -> pd.DataFrame:
    rows = []
    for row in ct.rows:
        d = dict(zip(ct.group_by, row))
        for m in ct.metrics:
            c = ct.cells[row].get(m)
            d[f"{m}_mean"] = c.mean if c else None
            d[f"{m}_std"] = c.std if c else None
            d[f"{m}_n"] = c.n if c else 0
        rows.append(d)
    return pd.DataFrame(rows)


def prefill_from_asset(a) -> dict:
    """Widget states for a `distribution-figure` asset (the only kind this page restores)."""
    o = dict(a.options or {})
    pre = {}
    for key, opt in (("cmp_inst_metric", "metric"), ("cmp_dist_methods", "methods"), ("cmp_dist_points", "points"),
                     ("cmp_dist_width", "width"), ("cmp_dist_panel", "panel_label")):
        if o.get(opt) is not None:
            pre[key] = o[opt]
    return pre


def render() -> None:
    st.title("Comparison")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="comparison")
    if experiment is None:
        return
    reset_on_experiment_change("cmp_", experiment)
    extra = select_extra_experiments(project, experiment)
    recs = load_records_union(project, [experiment, *extra])
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return
    recs = sidebar_filter(recs)
    if not recs:
        return

    all_metrics = agg.metric_names(recs)
    options = agg.grouping_keys(recs)  # experiment (when pooled), method, dataset, ..., config.*, derived.*
    title = experiment + (f" + {', '.join(extra)}" if extra else "")

    with st.sidebar:
        st.markdown("**Table**")
        n_datasets = len({r.get("dataset") for r in recs if r.get("dataset") is not None})
        wanted = ["method"] + (["experiment"] if extra else []) + (["dataset"] if n_datasets > 1 and not extra else [])
        default_keys = [o for o in wanted if o in options] or options[:1]
        group_by = st.multiselect("Rows grouped by", options, default=default_keys,
                                  help="With several datasets the default keeps dataset as a key: pooling over datasets a method was not run on is not a fair comparison.")
        metrics = st.multiselect("Metrics", all_metrics, default=all_metrics)
        show_std = st.checkbox("Show ± std", value=True)
        show_n = st.checkbox("Show n", value=True)

    if not group_by or not metrics:
        st.warning("Pick at least one grouping key and one metric.")
        return

    pool = agg.completed(recs)  # failed runs never enter a results table; they are counted on the Overview
    n_failed = sum(r["status"] == "failed" for r in recs)
    ct = agg.comparison_table(pool, group_by=group_by, metrics=metrics, higher_is_better=hib_map(defs))

    n_runs = len(pool)
    st.caption(f"{title} · {n_runs} completed runs" + (f" ({n_failed} failed excluded)" if n_failed else "")
               + (f" · filter: {where_text()}" if active_where() else "")
               + " · mean ± std over everything not in the row key · **bold** best, <u>underlined</u> second", unsafe_allow_html=True)
    for msg in agg.coverage_audit(pool, group_by, hidden=("dataset", "instance") + (("experiment",) if extra else ())):
        st.warning("Rows are pooled over different " + msg)
    orders = dict(row_order=agg.method_order(pool) if group_by[0] == "method" else agg.value_order(pool, group_by[0]),
                  col_order=agg.value_order(pool, group_by[1]) if len(group_by) == 2 else None)
    if len(group_by) <= 2:
        pt = agg.pivot_table(pool, group_by[0], group_by[1] if len(group_by) == 2 else None, metrics=metrics,
                             higher_is_better=hib_map(defs), **orders)
        st.markdown(comparison_html(pt, defs, show_std=show_std, show_n=show_n,
                                    row_labels=agg.method_labels(pool) if group_by[0] == "method" else None),
                    unsafe_allow_html=True)
    else:
        st.markdown(flat_html(ct, defs, show_std=show_std, show_n=show_n), unsafe_allow_html=True)

    df = to_frame(ct)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), file_name=f"{experiment}-comparison.csv", mime="text/csv")
    if len(group_by) <= 2:
        pin_to_paper({"comparison-table": {"rows": group_by[0], "cols": group_by[1] if len(group_by) == 2 else None,
                                           "metrics": metrics, "std": "pm" if show_std else "none"}},
                     records=recs, key="cmp_pin", extra_experiments=extra)

    st.subheader("Chart")
    metric = st.selectbox("Metric", metrics, key="chart_metric")
    fig = comparison_bars(ct, metric, fmt=fmt_for(defs, metric), unit=defs.get(metric, {}).get("unit", ""))
    st.plotly_chart(fig, theme=None, width="stretch")

    with st.expander("Raw numbers"):
        st.dataframe(df, width="stretch", hide_index=True)

    if any(r.get("instance") is not None for r in pool):
        _per_instance(pool, recs, defs, metrics, experiment, extra)

    if 1 <= len(group_by) <= 2:
        from ..export.latex import comparison_latex

        with st.expander("LaTeX (booktabs)"):
            pt = agg.pivot_table(pool, group_by[0], group_by[1] if len(group_by) == 2 else None, metrics=metrics,
                                 higher_is_better=hib_map(defs), **orders)
            tex = comparison_latex(pt, defs, std="pm" if show_std else "none",
                                   row_labels=agg.method_labels(pool) if group_by[0] == "method" else None)
            st.code(tex, language="latex")
            st.caption("More options (captions, labels, audit, figures) on the Export page.")


def _per_instance(pool: list[dict], recs: list[dict], defs: dict, metrics: list[str], experiment: str, extra: list[str]) -> None:
    """Distribution over instances per method, the instance x method table, and the picker for a qualitative figure."""
    st.subheader("Per instance")
    hib = hib_map(defs)
    c1, c2 = st.columns([1, 3])
    with c1:
        metric = keyed_selectbox("Metric", metrics, "cmp_inst_metric", metrics[0])
    table = agg.instance_table(pool, metric, higher_is_better=hib.get(metric, True))
    if not table.methods:
        st.caption("No per-instance runs with this metric.")
        return
    labels = agg.method_labels(pool)
    with c2:
        shown = keyed_multiselect("Methods", table.methods, "cmp_dist_methods", table.methods, format_func=lambda m: labels.get(m, str(m)))
    shown = [m for m in table.methods if m in shown] or table.methods
    fmt = fmt_for(defs, metric)
    unit = defs.get(metric, {}).get("unit", "")
    st.plotly_chart(distribution_box({m: table.values(m) for m in shown}, metric, ylabel=f"{metric} ({unit})" if unit else metric, labels=labels),
                    theme=None, width="stretch")
    st.markdown(figure_caption_html(
        f"Per-instance {metric} of each method ({len(table.instances)} instances, each averaged over its seeds and conditions): "
        "box = quartiles, whiskers = 1.5 IQR, points = instances. A table's mean ± std hides this spread.", number=2), unsafe_allow_html=True)
    rows = []
    for inst in table.instances:
        best = table.best_method(inst)
        cells = []
        for m in shown:
            s_ = table.stat(inst, m)
            txt = "—" if s_ is None else format(s_.mean, fmt).replace("-", "−")
            cells.append(f"<b>{txt}</b>" if m == best else txt)
        rows.append([str(inst)] + cells)
    with st.expander(f"Instance × method table ({len(rows)} instances)"):
        st.markdown(generic_html(["instance"] + [labels.get(m, str(m)) for m in shown], rows, number=2, left_cols=1,
                                 raw_html_cols=list(range(1, len(shown) + 1)),
                                 caption=f"{metric} per instance (mean over seeds and conditions); best per row in bold."), unsafe_allow_html=True)
    if len(shown) > 1:
        p1, p2 = st.columns(2)
        with p1:
            ours = keyed_selectbox("Ours", shown, "cmp_gain_ours", shown[-1], format_func=lambda m: labels.get(m, str(m)))
        with p2:
            base_opts = [m for m in shown if m != ours] or shown
            baseline = keyed_selectbox("Against", base_opts, "cmp_gain_base", base_opts[0], format_func=lambda m: labels.get(m, str(m)))
        gains = agg.instance_gains(table, ours, baseline)
        if gains:
            sign = "" if hib.get(metric, True) else " (sign flipped: lower is better)"
            grows = [[str(g.instance), format(abs(g.ours), fmt), format(abs(g.baseline), fmt), format(g.gain, "+" + fmt).replace("-", "−")]
                     for g in gains]
            st.markdown(generic_html(["instance", labels.get(ours, str(ours)), labels.get(baseline, str(baseline)), f"gain{sign}"], grows,
                                     number=3, left_cols=1,
                                     caption=f"Instances ranked by the gain of {labels.get(ours, ours)} over {labels.get(baseline, baseline)} "
                                             f"in {metric}: the top rows are candidates for a qualitative figure, the bottom rows the honest "
                                             "failure cases."), unsafe_allow_html=True)
            st.caption(f"Largest gain: **{gains[0].instance}** ({gains[0].gain:+{fmt}}); smallest: **{gains[-1].instance}** ({gains[-1].gain:+{fmt}}).")
    with st.expander("Paper figure (matplotlib, IEEE style)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            width = keyed_selectbox("Width", ["single", "double", "ieee-single", "ieee-double"], "cmp_dist_width", "single")
        with c2:
            points = keyed(st.checkbox, "Show points", "cmp_dist_points", True)
        with c3:
            cap = keyed(st.text_input, "Panel caption", "cmp_dist_panel", "", placeholder="b. Per-image PSNR")
        pf = distribution_figure({m: table.values(m) for m in shown}, metric, ylabel=f"{metric} ({unit})" if unit else metric, width=width,
                                 labels=labels, show_points=points, caption=cap or None)
        g1, g2 = st.columns([3, 1])
        gray = g2.checkbox("Grayscale", value=False, key="cmp_dist_gray")
        png = figure_bytes(pf, "png", dpi=200)
        g1.image(to_grayscale_png(png) if gray else png)
        stem = f"{experiment}-{metric}-distribution".replace(" ", "_")
        g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
        st.code(figure_tex(f"figures/{stem}.pdf", label=f"fig:{stem}", width=width), language="latex")
    pin_to_paper({"distribution-figure": {"metric": metric, "methods": shown, "points": points, "width": width, "panel_label": cap or None}},
                 records=recs, key="cmp_dist_pin", suggested_label=f"fig:{experiment}-{metric}-distribution", extra_experiments=extra)
