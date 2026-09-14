"""Comparison page: methods x metrics, mean ± std, best in bold."""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from ..export.figures import distribution_figure, figure_bytes, figure_tex, to_grayscale_png
from .charts import comparison_bars, distribution_box
from .tables import comparison_html, figure_caption_html, flat_html, generic_html
from .common import (active_where, completed_or_explain, excluded_note, fmt_for, hib_map, keyed, keyed_multiselect, keyed_selectbox,
                     load_metric_defs, load_records_union, page_url, pin_to_paper, reset_on_experiment_change,
                     select_extra_experiments, select_project_experiment, sidebar_db, sidebar_filter, where_text)
from .styling import chart_controls, sidebar_plot_style

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
    if o.get("ylim"):
        pre[f"cmp_dist:{a.experiment}_ylim"] = ",".join(str(v) for v in o["ylim"])
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
    matched = sidebar_filter(recs)
    pool = completed_or_explain(matched, of=recs)  # says why the page is blank instead of leaving it blank
    if not pool:
        return
    recs = matched

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
    style = sidebar_plot_style(project)

    if not group_by or not metrics:
        st.warning("Pick at least one grouping key and one metric.")
        return

    ct = agg.comparison_table(pool, group_by=group_by, metrics=metrics, higher_is_better=hib_map(defs))

    n_runs = len(pool)
    st.caption(f"{title} · {n_runs} completed runs" + excluded_note(recs)
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

    _cell_runs(ct, pool, group_by, metrics, defs, project)

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
    ctl = chart_controls(project, style, key=f"cmp_bars:{experiment}",
                         series=[r[0] for r in ct.rows], series_key=group_by[0],
                         series_labels=agg.method_labels(pool) if group_by[0] == "method" else None,
                         orders=[(" / ".join(group_by[1:]), [" / ".join(map(str, r[1:])) for r in ct.rows])] if len(group_by) > 1 else (),
                         x_name=None, y_name=metric)
    fig = comparison_bars(ct, metric, fmt=fmt_for(defs, metric), unit=defs.get(metric, {}).get("unit", ""),
                          style=ctl.style, ylim=ctl.ylim)
    st.plotly_chart(fig, theme=None, width="stretch")

    with st.expander("Raw numbers"):
        st.dataframe(df, width="stretch", hide_index=True)

    if any(r.get("instance") is not None for r in pool):
        _per_instance(pool, recs, defs, metrics, experiment, extra, project, style)

    if 1 <= len(group_by) <= 2:
        from ..export.latex import comparison_latex

        with st.expander("LaTeX (booktabs)"):
            pt = agg.pivot_table(pool, group_by[0], group_by[1] if len(group_by) == 2 else None, metrics=metrics,
                                 higher_is_better=hib_map(defs), **orders)
            tex = comparison_latex(pt, defs, std="pm" if show_std else "none",
                                   row_labels=agg.method_labels(pool) if group_by[0] == "method" else None)
            st.code(tex, language="latex")
            st.caption("More options (captions, labels, audit, figures) on the Export page.")


def _cell_runs(ct: agg.ComparisonTable, pool: list[dict], group_by: list[str], metrics: list[str], defs: dict,
               project: Optional[str]) -> None:
    """The runs behind one cell of the table above.

    `30.69 ± 0.06` is a mean over runs the reader cannot see; this is where you find out that the spread comes
    from one diverged seed, or that a cell has three runs where its neighbour has six.
    """
    if not ct.rows:
        return
    with st.expander("Runs behind a cell"):
        labels = [ct.row_label(r) for r in ct.rows]
        picked = keyed_selectbox(" / ".join(group_by), labels, "cmp_cell_row", labels[0])
        row = ct.rows[labels.index(picked)]
        runs = [r for r in pool if all(agg.same_value(agg.get_field(r, k), v) for k, v in zip(group_by, row))]
        if not runs:
            st.caption("No run matches this cell.")
            return
        rows = []
        for r in sorted(runs, key=lambda r: (str(r.get("instance") or ""), r.get("seed") is None, r.get("seed") or 0)):
            entry = {
                "run": page_url("run", project=project, experiment=r["experiment"], run=r["run_id"]),
                "id": r["run_id"],
                "instance": str(r["instance"]) if r["instance"] is not None else "—",
                "seed": "—" if r["seed"] is None else str(r["seed"]),  # one type per column: Arrow rejects a mix
            }
            for m in metrics:
                entry[m] = r["metrics"].get(m)
            rows.append(entry)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True,
                     column_config={"run": st.column_config.LinkColumn("open", display_text="open", width="small")})
        for m in metrics:
            cell = ct.cells[row].get(m)
            if cell is not None:
                spread = f" · min {format(cell.min, fmt_for(defs, m))} · max {format(cell.max, fmt_for(defs, m))}" if cell.n > 1 else ""
                st.caption(f"**{m}**: the table shows {cell.format(fmt_for(defs, m))} over these {cell.n} run(s){spread}.")


def _paired(table, ours, baseline, labels: dict, defs: dict, metric: str, fmt: str, unit: str) -> None:
    """How often `ours` actually wins, not just by how much on average.

    A mean ± std says nothing about how many images improved, which is the first thing a reviewer asks of a
    method claimed to be better. The pairs are per instance (seeds and conditions already averaged), and the
    p-values are two-sided.
    """
    pc = agg.paired_comparison(table, ours, baseline)
    if pc is None or pc.n < 2:
        return
    us, them = labels.get(ours, str(ours)), labels.get(baseline, str(baseline))
    st.markdown(f"**Is {us} really better than {them}?**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Instances won", f"{pc.wins} / {pc.n}", help="Instances where the gain is positive; ties are counted separately.")
    c2.metric(f"Median gain ({metric})", format(pc.median, "+" + fmt), help=f"Mean {format(pc.mean, '+' + fmt)}")
    c3.metric("Wilcoxon signed-rank", agg.fmt_p(pc.wilcoxon_p),
              help="Two-sided, over the paired per-instance differences" + ("; exact" if pc.wilcoxon_exact else "; normal approximation (ties or a large sample)"))
    c4.metric("Sign test", agg.fmt_p(pc.sign_p), help="Two-sided exact binomial over wins and losses only — it ignores the size of each gain")
    st.code(pc.sentence(fmt=fmt, unit=unit, labels=labels), language=None)
    st.caption(f"Ready to paste. {pc.wins} win, {pc.losses} lose"
               + (f", {pc.ties} tie" if pc.ties else "") + f" of {pc.n} instances both methods ran. "
               "A p-value here says the ranking is unlikely to be chance *on these instances of this dataset*; it is not "
               "evidence that the ordering holds elsewhere, and comparing several methods against one baseline needs a "
               "multiple-comparison correction the page does not apply.")


def _per_instance(pool: list[dict], recs: list[dict], defs: dict, metrics: list[str], experiment: str, extra: list[str],
                  project: str, style) -> None:
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
    dist = chart_controls(project, style, key=f"cmp_dist:{experiment}", series=shown, series_key="method",
                          series_labels=labels, x_name=None, y_name=metric)
    st.plotly_chart(distribution_box({m: table.values(m) for m in shown}, metric, ylabel=f"{metric} ({unit})" if unit else metric,
                                     labels=labels, style=dist.style, ylim=dist.ylim),
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
            _paired(table, ours, baseline, labels, defs, metric, fmt, unit)
    with st.expander("Paper figure (matplotlib, IEEE style)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            width = keyed_selectbox("Width", ["single", "double", "ieee-single", "ieee-double"], "cmp_dist_width", "single")
        with c2:
            points = keyed(st.checkbox, "Show points", "cmp_dist_points", True)
        with c3:
            cap = keyed(st.text_input, "Panel caption", "cmp_dist_panel", "", placeholder="b. Per-image PSNR")
        pf = distribution_figure({m: table.values(m) for m in shown}, metric, ylabel=f"{metric} ({unit})" if unit else metric, width=width,
                                 labels=labels, show_points=points, caption=cap or None, style=dist.style, ylim=dist.ylim)
        g1, g2 = st.columns([3, 1])
        gray = g2.checkbox("Grayscale", value=False, key="cmp_dist_gray")
        png = figure_bytes(pf, "png", dpi=200)
        g1.image(to_grayscale_png(png) if gray else png)
        stem = f"{experiment}-{metric}-distribution".replace(" ", "_")
        g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
        st.code(figure_tex(f"figures/{stem}.pdf", label=f"fig:{stem}", width=width), language="latex")
    pin_to_paper({"distribution-figure": {"metric": metric, "methods": shown, "points": points, "width": width, "panel_label": cap or None,
                                          **dist.limit_options}},
                 records=recs, key="cmp_dist_pin", suggested_label=f"fig:{experiment}-{metric}-distribution", extra_experiments=extra)
