"""Ablation page: each config variant vs the full model, with deltas."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from ..export.figures import ablation_figure, figure_bytes, figure_tex, to_grayscale_png

from .. import aggregate as agg
from .charts import ablation_deltas
from .common import active_where, fmt_for, hib_map, load_metric_defs, load_records, select_project_experiment, sidebar_db, sidebar_filter, where_text
from .run_detail import run_label
from .tables import ablation_html, figure_caption_html, generic_html

AUTO = "auto (run tagged 'base', else most common config)"


def render() -> None:
    st.title("Ablation")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="ablation")
    if experiment is None:
        return
    recs = load_records(project, experiment)
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return
    recs = agg.completed(sidebar_filter(recs))
    if not recs:
        if not active_where():
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
    try:
        rows = agg.ablation_table(recs, base_run_id=base_opts[base_choice], metrics=metrics)
    except agg.AmbiguousBaseError as e:
        st.error(f"{e}. Pick the full model in the sidebar.")
        return
    if not rows:
        st.warning("Nothing to show.")
        return
    conditions = agg.condition_keys(recs)
    if conditions:
        st.caption("Every arm is pooled over the conditions the full model was repeated on: "
                   + ", ".join(f"`{k}`" for k in conditions) + ".")
    base_row = next((r for r in rows if r.is_base), None)
    if base_row is None:
        st.warning("No run matches the base config exactly; deltas are unavailable. Pick a base run in the sidebar.")
    hib = hib_map(defs)

    # --- component matrix: which knobs each variant changed
    keys = sorted({k for r in rows for k in r.diff})
    st.caption(f"{experiment} · {len(recs)} runs" + (f" · filter: {where_text()}" if active_where() else "")
               + f" · {len(rows)} variants · {len(keys)} ablated settings · "
               f"**bold** = full model")

    st.markdown(ablation_html(rows, metrics, defs, relative=relative, number=1), unsafe_allow_html=True)
    st.caption("· settings shown per variant: ✓ on, × off, other values literally")

    # --- chart
    st.subheader("Effect of each change")
    metric = st.selectbox("Metric", metrics, key="abl_metric")
    variants = [r for r in rows if not r.is_base]
    fmt = fmt_for(defs, metric)
    unit = defs.get(metric, {}).get("unit", "")
    hib_m = hib.get(metric, True)
    effects = agg.ablation_effects(rows, metric, hib_m)
    if variants and base_row is not None:
        fig = ablation_deltas(
            [r.label for r in variants],
            [r.delta.get(metric) for r in variants],
            [(r.stats[metric].std if r.stats.get(metric) else 0.0) for r in variants],
            metric, higher_is_better=hib_m, fmt=fmt, unit=unit,
        )
        st.plotly_chart(fig, theme=None, width="stretch")
        ns = sorted({r.n for r in rows})
        n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
        cap = [f"Change in {metric}{f' ({unit})' if unit else ''} when one setting of the full model is altered "
               f"(variant − full model, mean over {n_txt} runs; error bars: std of the variant). "
               f"Bars to the {'left' if hib_m else 'right'} hurt, to the {'right' if hib_m else 'left'} help. "
               f"Full model: {base_row.stats[metric].format(fmt) if base_row.stats.get(metric) else '—'}."]
        clear = [e for e in effects if e.verdict == "clear" and not e.improves]
        noise = [e for e in effects if e.verdict == "within noise"]
        helps = [e for e in effects if e.improves and e.verdict in ("clear", "likely")]
        if clear:
            cap.append("Clearly needed: " + ", ".join(f"{e.label} ({e.delta:+{fmt}})" for e in clear) + ".")
        if noise:
            cap.append("Within run-to-run noise: " + ", ".join(e.label for e in noise) + ".")
        if helps:
            cap.append("Removing " + ", ".join(e.label for e in helps) + " improves the metric: the full model is not the best configuration here.")
        st.markdown(figure_caption_html(" ".join(cap), number=1), unsafe_allow_html=True)

        st.subheader("Effect sizes")
        erows = []
        for e in effects:
            erows.append([e.label, str(e.n), f"{e.delta:+{fmt}}".replace("-", "−"),
                          "—" if e.rel is None else f"{e.rel * 100:+.1f}%".replace("-", "−"),
                          f"{e.pooled_std:{fmt}}", "—" if e.d is None else f"{e.d:+.1f}".replace("-", "−"),
                          ("helps" if e.improves else "hurts") + f" · {e.verdict}"])
        st.markdown(generic_html(["Change", "n", f"Δ {metric}", "Δ (%)", "pooled std", "d", "verdict"], erows, number=2, left_cols=1,
                                 caption=f"Effect of each change on {metric} relative to the full model. d = Δ / pooled std (Cohen's d "
                                         f"against the full model): |d| ≥ 2 clear, ≥ 1 likely, below 1 within run-to-run noise. "
                                         f"Verdicts need repeated runs; single runs are marked n = 1."),
                    unsafe_allow_html=True)
        worst = effects[0] if effects else None
        if worst is not None and not worst.improves:
            st.caption(f"Largest drop: **{worst.label}** ({worst.delta:+{fmt}} {metric}, {worst.verdict}).")
        with st.expander("Paper figure (matplotlib, IEEE style)"):
            pf = ablation_figure(rows, metric, higher_is_better=hib_m, fmt=fmt,
                                 xlabel=f"$\\Delta$ {metric} vs. full model" + (f" ({unit})" if unit else ""), width="single")
            g1, g2 = st.columns([3, 1])
            gray = g2.checkbox("Grayscale", value=False, key="abl_gray")
            png = figure_bytes(pf, "png", dpi=200)
            g1.image(to_grayscale_png(png) if gray else png)
            g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{experiment}-ablation-{metric}.pdf", mime="application/pdf")
    else:
        st.caption("Need a base and at least one variant to chart deltas.")

    from ..export.latex import ablation_latex

    with st.expander("LaTeX (booktabs table + figure snippet)"):
        st.code(ablation_latex(rows, metrics, defs), language="latex")
        st.code(figure_tex(f"figures/{experiment}-ablation-{metric}.pdf", label=f"fig:{experiment}-ablation", width="single"),
                language="latex")
        st.caption("More options (captions, labels, std style) on the Export page.")

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
