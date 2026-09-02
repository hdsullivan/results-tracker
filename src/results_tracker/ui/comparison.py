"""Comparison page: methods x metrics, mean ± std, best in bold."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from .charts import comparison_bars
from .common import fmt_for, hib_map, load_metric_defs, load_records, select_project_experiment, sidebar_db

BASE_KEYS = ["method", "dataset", "instance", "seed"]


def markdown_table(ct: agg.ComparisonTable, defs: dict, show_std: bool, show_n: bool) -> str:
    head = [" / ".join(ct.group_by)] + [f"{m} {'↑' if ct.higher_is_better[m] else '↓'}" for m in ct.metrics]
    if show_n:
        head.append("n")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for row in ct.rows:
        cells = [ct.row_label(row)]
        for m in ct.metrics:
            st_ = ct.cells[row].get(m)
            if st_ is None:
                cells.append("—")
                continue
            s = st_.format(fmt_for(defs, m), with_std=show_std)
            if ct.is_best(row, m):
                s = f"**{s}**"
            elif ct.is_second(row, m):
                s = f"<u>{s}</u>"
            cells.append(s)
        if show_n:
            cells.append(str(max((c.n for c in ct.cells[row].values() if c), default=0)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


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


def render() -> None:
    st.title("Comparison")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="comparison")
    if experiment is None:
        return
    recs = load_records(project, experiment)
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return

    config_keys = sorted({k for r in recs for k in agg.flatten(r["config"])})
    all_metrics = agg.metric_names(recs)
    present = [k for k in BASE_KEYS if any(r.get(k) is not None for r in recs)]
    options = present + [f"config.{k}" for k in config_keys]

    with st.sidebar:
        st.markdown("**Table**")
        group_by = st.multiselect("Rows grouped by", options, default=[o for o in ["method"] if o in options] or options[:1])
        metrics = st.multiselect("Metrics", all_metrics, default=all_metrics)
        show_std = st.checkbox("Show ± std", value=True)
        show_n = st.checkbox("Show n", value=True)
        include_failed = st.checkbox("Include failed runs", value=False)

    if not group_by or not metrics:
        st.warning("Pick at least one grouping key and one metric.")
        return

    pool = recs if include_failed else agg.completed(recs)
    ct = agg.comparison_table(pool, group_by=group_by, metrics=metrics, higher_is_better=hib_map(defs))
    if include_failed:
        # comparison_table filters failed runs itself; rebuild with everything if asked
        ct = agg.ComparisonTable(**{**ct.__dict__, "cells": agg.aggregate_metrics(pool, group_by, metrics, only_completed=False)})

    n_runs = len(pool)
    st.caption(f"{experiment} · {n_runs} runs · mean ± std over everything not in the row key · **bold** best, <u>underlined</u> second", unsafe_allow_html=True)
    st.markdown(markdown_table(ct, defs, show_std, show_n), unsafe_allow_html=True)

    df = to_frame(ct)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), file_name=f"{experiment}-comparison.csv", mime="text/csv")

    st.subheader("Chart")
    metric = st.selectbox("Metric", metrics, key="chart_metric")
    fig = comparison_bars(ct, metric, fmt=fmt_for(defs, metric), unit=defs.get(metric, {}).get("unit", ""))
    st.plotly_chart(fig, width="stretch")

    with st.expander("Raw numbers"):
        st.dataframe(df, width="stretch", hide_index=True)

    if 1 <= len(group_by) <= 2:
        from ..export.latex import comparison_latex

        with st.expander("LaTeX (booktabs)"):
            pt = agg.pivot_table(pool, group_by[0], group_by[1] if len(group_by) == 2 else None, metrics=metrics,
                                 higher_is_better=hib_map(defs))
            tex = comparison_latex(pt, defs, std="pm" if show_std else "none",
                                   row_labels=agg.method_labels(pool) if group_by[0] == "method" else None)
            st.code(tex, language="latex")
            st.caption("More options (captions, labels, audit, figures) on the Export page.")
