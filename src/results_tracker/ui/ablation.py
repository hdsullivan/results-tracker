"""Ablation page: each config variant vs the full model, with deltas.

One ablation is one method's story, so the page renders a section per method by default (see
`aggregate.ablation_tables`). Pooling methods into a single table would give them one shared base and
would diff config keys only one of them carries -- an `update_order` present on an ADMM method and
absent on a PGM one reads as the ablated setting "- update_order".
"""

from __future__ import annotations

import io
import re

import pandas as pd
import streamlit as st

from ..export.figures import ablation_figure, figure_bytes, figure_tex, to_grayscale_png

from .. import aggregate as agg
from ..plotstyle import VARIANT_KEY
from .charts import ablation_deltas
from .common import (active_where, completed_or_explain, excluded_note, fmt_for, hib_map, keyed_multiselect, load_metric_defs,
                     load_records, pin_to_paper, select_project_experiment, sidebar_db, sidebar_filter, where_text)
from .styling import chart_controls, sidebar_plot_style
from .run_detail import run_label
from .tables import ablation_html, figure_caption_html, generic_html

AUTO = "auto (run tagged 'base', else most common config)"
#: Fields that can carry an ablation of their own. `method` leads and is the default: two methods are
#: two algorithms, each with its own full model. `dataset` is offered because an ablation repeated on a
#: second dataset is a second ablation. `seed` is deliberately absent -- seeds are repetitions every arm
#: pools over, and splitting on one would leave every arm at n = 1.
SPLIT_FIELDS = ("method", "dataset")


def _slug(group: agg.GroupKey) -> str:
    return re.sub(r"\W+", "_", "-".join(str(v) for v in group)) or "all"


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
    all_recs = sidebar_filter(recs)
    completed = completed_or_explain(all_recs, of=recs)
    if not completed:
        return
    recs = completed
    metrics_all = agg.metric_names(recs)
    config_keys = [k[len("config."):] for k in agg.grouping_keys(recs) if k.startswith("config.")]
    split_opts = [f for f in SPLIT_FIELDS if len({r.get(f) for r in recs}) > 1]

    with st.sidebar:
        st.markdown("**Ablation**")
        base_opts = {AUTO: None, **{run_label(r): r["run_id"] for r in recs}}
        base_choice = st.selectbox("Full model (base)", list(base_opts),
                                   help="Only one run is named here; every other section takes the run carrying the "
                                        "same ablated settings, so one pick serves them all.")
        split = keyed_multiselect("Split by", split_opts, f"abl_split:{experiment}",
                                  ["method"] if "method" in split_opts else [],
                                  help="A separate ablation per value — each with its own full model. Methods are "
                                       "split by default: they are different algorithms, not settings of one.")
        conditions = keyed_multiselect(
            "Conditions (pooled over)", config_keys, f"abl_cond:{experiment}", agg.condition_keys(recs),
            help="Config keys the whole ablation was repeated on (blur kernel, noise level …) rather than settings it "
                 "varies. Every arm pools over them instead of splitting on them. Seeded automatically from whatever "
                 "differs between the runs tagged 'base'; set it by hand when nothing is tagged.")
        metrics = st.multiselect("Metrics", metrics_all, default=metrics_all)
        relative = st.checkbox("Show Δ as % of base", value=False)
    style = sidebar_plot_style(project)

    if not metrics:
        st.warning("Pick at least one metric.")
        return
    try:
        tables = agg.ablation_tables(recs, by=split, base_run_id=base_opts[base_choice],
                                     ignore_keys=conditions, metrics=metrics)
    except agg.AmbiguousBaseError as e:
        st.error(f"{e}. Pick the full model in the sidebar.")
        return
    if not tables:
        st.warning("Nothing to show.")
        return

    st.caption(f"{experiment} · {len(recs)} runs" + excluded_note(all_recs)
               + (f" · filter: {where_text()}" if active_where() else "")
               + (f" · {len(tables)} ablations, split by {', '.join(f'`{k}`' for k in split)}" if split else "")
               + (" · pooled over " + ", ".join(f"`{k}`" for k in conditions) if conditions else "")
               + " · **bold** = full model")
    if not conditions and len(agg.varying_config_keys(recs)) > 1:
        st.info("No conditions are set, so every config key counts as an ablated setting. If this experiment was "
                "repeated over a grid (kernel, noise level …), name those keys under **Conditions** in the sidebar.")

    metric = st.selectbox("Metric for the charts", metrics, key="abl_metric")
    hib = hib_map(defs)
    for i, (group, rows) in enumerate(tables.items()):
        if group:
            st.subheader(agg.group_heading(group, recs))
        _section(project, experiment, group, rows, recs, all_recs, metrics, metric, defs, hib,
                 relative=relative, style=style, base_run_id=base_opts[base_choice], conditions=conditions,
                 split=split, number=2 * i + 1)
        if i < len(tables) - 1:
            st.divider()


def _section(project, experiment, group, rows, recs, all_recs, metrics, metric, defs, hib, *,
             relative, style, base_run_id, conditions, split, number) -> None:
    """One ablation: its table, delta chart, effect sizes, paper figure and exports."""
    slug = _slug(group)
    base_row = next((r for r in rows if r.is_base), None)
    if base_row is None:
        st.warning("No run matches the base config exactly; deltas are unavailable. Pick a base run in the sidebar.")
    keys = sorted({k for r in rows for k in r.diff})

    st.markdown(ablation_html(rows, metrics, defs, relative=relative, number=number), unsafe_allow_html=True)
    st.caption(f"{len(rows) - 1} variant(s) · {len(keys)} ablated setting(s) · "
               "settings shown per variant: ✓ on, × off, other values literally")

    variants = [r for r in rows if not r.is_base]
    fmt = fmt_for(defs, metric)
    unit = defs.get(metric, {}).get("unit", "")
    hib_m = hib.get(metric, True)
    effects = agg.ablation_effects(rows, metric, hib_m)
    ctl = None
    if variants and base_row is not None:
        ctl = chart_controls(project, style, key=f"abl_deltas:{experiment}:{slug}", series_key=VARIANT_KEY, colors=False,
                             series=[r.label for r in variants], x_name=f"Δ {metric}", y_name=None)
        fig = ablation_deltas(
            [r.label for r in variants],
            [r.delta.get(metric) for r in variants],
            [(r.stats[metric].std if r.stats.get(metric) else 0.0) for r in variants],
            metric, higher_is_better=hib_m, fmt=fmt, unit=unit, style=ctl.style, xlim=ctl.xlim,
        )
        st.plotly_chart(fig, theme=None, width="stretch", key=f"abl_chart_{slug}")
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
        st.markdown(figure_caption_html(" ".join(cap), number=number), unsafe_allow_html=True)

        erows = []
        for e in effects:
            erows.append([e.label, str(e.n), f"{e.delta:+{fmt}}".replace("-", "−"),
                          "—" if e.rel is None else f"{e.rel * 100:+.1f}%".replace("-", "−"),
                          f"{e.pooled_std:{fmt}}", "—" if e.d is None else f"{e.d:+.1f}".replace("-", "−"),
                          ("helps" if e.improves else "hurts") + f" · {e.verdict}"])
        st.markdown(generic_html(["Change", "n", f"Δ {metric}", "Δ (%)", "pooled std", "d", "verdict"], erows,
                                 number=number + 1, left_cols=1,
                                 caption=f"Effect of each change on {metric} relative to the full model. d = Δ / pooled std (Cohen's d "
                                         f"against the full model): |d| ≥ 2 clear, ≥ 1 likely, below 1 within run-to-run noise. "
                                         f"Verdicts need repeated runs; single runs are marked n = 1."),
                    unsafe_allow_html=True)
        worst = effects[0] if effects else None
        if worst is not None and not worst.improves:
            st.caption(f"Largest drop: **{worst.label}** ({worst.delta:+{fmt}} {metric}, {worst.verdict}).")
        with st.expander("Paper figure (matplotlib, IEEE style)"):
            pf = ablation_figure(rows, metric, higher_is_better=hib_m, fmt=fmt,
                                 xlabel=f"$\\Delta$ {metric} vs. full model" + (f" ({unit})" if unit else ""), width="single",
                                 style=ctl.style, xlim=ctl.xlim)
            g1, g2 = st.columns([3, 1])
            gray = g2.checkbox("Grayscale", value=False, key=f"abl_gray_{slug}")
            png = figure_bytes(pf, "png", dpi=200)
            g1.image(to_grayscale_png(png) if gray else png)
            g2.download_button("Download PDF", figure_bytes(pf, "pdf"), key=f"abl_pdf_{slug}",
                               file_name=f"{experiment}-{slug}-ablation-{metric}.pdf", mime="application/pdf")
    else:
        st.caption("Need a base and at least one variant to chart deltas.")

    from ..export.latex import ablation_latex

    # This section's own base run and its own slice of the split, so a pinned asset re-renders as the
    # ablation shown here rather than as whatever run the sidebar happened to be seeded with.
    section_base = base_row.run_ids[0] if base_row is not None and base_row.run_ids else base_run_id
    common_opts = {"base_run_id": section_base, "ignore_keys": list(conditions)}
    fig_opts = {**common_opts, "metric": metric, "width": "single"}
    if ctl is not None:
        fig_opts["xlim"] = list(ctl.xlim) if ctl.xlim else None
    pin_to_paper({"ablation-table": {**common_opts, "metrics": metrics}, "ablation-figure": fig_opts},
                 records=all_recs, key=f"abl_pin_{slug}",
                 extra_filters=dict(zip(split, group)) if group else None,
                 suggested_label=f"tab:{experiment}-{slug}" if group else None)
    with st.expander("LaTeX (booktabs table + figure snippet)"):
        st.code(ablation_latex(rows, metrics, defs), language="latex")
        st.code(figure_tex(f"figures/{experiment}-{slug}-ablation-{metric}.pdf",
                           label=f"fig:{experiment}-{slug}-ablation", width="single"), language="latex")
        st.caption("More options (captions, labels, std style) on the Export page.")

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
    st.download_button("Download CSV", buf.getvalue(), key=f"abl_csv_{slug}",
                       file_name=f"{experiment}-{slug}-ablation.csv", mime="text/csv")
