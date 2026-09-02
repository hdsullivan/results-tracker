"""Ablation page: each config variant vs the full model, with deltas."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from .charts import ablation_deltas
from .common import fmt_for, hib_map, load_metric_defs, load_records, select_project_experiment, sidebar_db
from .run_detail import run_label
from .tables import ablation_html

AUTO = "auto (run tagged 'base', else most common config)"


def render() -> None:
    st.title("Ablation")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="ablation")
    if experiment is None:
        return
    recs = agg.completed(load_records(project, experiment))
    defs = load_metric_defs()
    if not recs:
        st.info("No completed runs in this experiment.")
        return
    metrics_all = agg.metric_names(recs)

    with st.sidebar:
        st.markdown("**Ablation**")
        base_opts = {AUTO: None, **{run_label(r): r["run_id"] for r in recs}}
        base_choice = st.selectbox("Full model (base)", list(base_opts))
        metrics = st.multiselect("Metrics", metrics_all, default=metrics_all)
        relative = st.checkbox("Show Δ as % of base", value=False)

    if not metrics:
        st.warning("Pick at least one metric.")
        return
    rows = agg.ablation_table(recs, base_run_id=base_opts[base_choice], metrics=metrics)
    if not rows:
        st.warning("Nothing to show.")
        return
    base_row = next((r for r in rows if r.is_base), None)
    if base_row is None:
        st.warning("No run matches the base config exactly; deltas are unavailable. Pick a base run in the sidebar.")
    hib = hib_map(defs)

    # --- component matrix: which knobs each variant changed
    keys = sorted({k for r in rows for k in r.diff})
    st.caption(f"{experiment} · {len(recs)} runs · {len(rows)} variants · {len(keys)} ablated settings · "
               f"**bold** = full model")

    st.markdown(ablation_html(rows, metrics, defs, relative=relative), unsafe_allow_html=True)
    st.caption("· settings shown per variant: ✓ on, × off, other values literally")

    # --- chart
    st.subheader("Effect of each change")
    metric = st.selectbox("Metric", metrics, key="abl_metric")
    variants = [r for r in rows if not r.is_base]
    if variants and base_row is not None:
        fig = ablation_deltas(
            [r.label for r in variants],
            [r.delta.get(metric) for r in variants],
            [(r.stats[metric].std if r.stats.get(metric) else 0.0) for r in variants],
            metric, higher_is_better=hib.get(metric, True), fmt=fmt_for(defs, metric),
            unit=defs.get(metric, {}).get("unit", ""),
        )
        st.plotly_chart(fig, theme=None, width="stretch")
        worst = min(variants, key=lambda r: (r.delta.get(metric) or 0) * (1 if hib.get(metric, True) else -1))
        if worst.delta.get(metric) is not None:
            st.caption(f"Largest drop: **{worst.label}** ({worst.delta[metric]:+{fmt_for(defs, metric)}} {metric}).")
    else:
        st.caption("Need a base and at least one variant to chart deltas.")

    from ..export.latex import ablation_latex

    with st.expander("LaTeX (booktabs)"):
        st.code(ablation_latex(rows, metrics, defs), language="latex")
        st.caption("More options on the Export page.")

    # --- export
    out = []
    for r in rows:
        d = {"variant": r.label, "is_base": r.is_base, "n": r.n, **{k: (r.diff[k][1] if k in r.diff else None) for k in keys}}
        for m in metrics:
            st_ = r.stats.get(m)
            d[f"{m}_mean"] = st_.mean if st_ else None
            d[f"{m}_std"] = st_.std if st_ else None
            d[f"{m}_delta"] = r.delta.get(m)
        out.append(d)
    buf = io.StringIO()
    pd.DataFrame(out).to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), file_name=f"{experiment}-ablation.csv", mime="text/csv")
