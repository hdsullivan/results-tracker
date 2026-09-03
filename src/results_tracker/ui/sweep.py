"""Sweep page: metric vs one swept parameter (lines) or two (heatmap), aggregated over seeds."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from ..export.figures import figure_bytes, figure_tex, sweep_figure, to_grayscale_png
from ..export.latex import sweep_latex

from .. import aggregate as agg
from .charts import is_log_friendly, sweep_heatmap, sweep_lines
from .tables import figure_caption_html, generic_html, sweep_html
from .common import (active_where, fmt_for, load_metric_defs, load_records, pin_to_paper, select_project_experiment, sidebar_db,
                     sidebar_filter, swept_params, where_text)

GROUP_KEYS = ["method", "dataset", "instance"]  # base fields; config.* and derived.* keys are added when they vary


def line_keys(records: list[dict], param_x: str) -> list[str]:
    """Keys a sweep can be split into lines by: varying base fields, config keys (conditions such as noise, or an
    arm's knob such as the denoiser) and derived fields, except the swept parameter itself."""
    return [k for k in agg.grouping_keys(records, base=("method", "dataset", "instance"), varying_only=True)
            if k != f"config.{param_x}"]


def render() -> None:
    st.title("Parameter sweep")
    sidebar_db()
    project, experiment = select_project_experiment(prefer="sweep")
    if experiment is None:
        return
    recs = load_records(project, experiment)
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return
    all_recs = sidebar_filter(recs)
    recs = agg.completed(all_recs)
    if not recs:
        if not active_where():
            st.info("No completed runs in this experiment.")
        return
    filt = f" · filter: {where_text()}" if active_where() else ""

    all_keys = sorted({k for r in recs for k in agg.flatten(r["config"])})
    varying = agg.varying_config_keys(recs)
    metrics = agg.metric_names(recs)
    if not all_keys or not metrics:
        st.warning("Runs need a config with at least one key and at least one metric.")
        return

    with st.sidebar:
        st.markdown("**Sweep**")
        declared = [k for k in swept_params(project, experiment) if k in all_keys]  # what the study said it sweeps
        default_x = declared[0] if declared else (varying[0] if varying else all_keys[0])
        param_x = st.selectbox("Parameter (x)", all_keys, index=all_keys.index(default_x))
        second_opts = ["— none —"] + [k for k in all_keys if k != param_x]
        # a heatmap only by default when nothing declares a 1-D sweep and exactly one other key varies;
        # a recipe sweep also varies its conditions (kernel, noise), which are line splits, not a second axis
        default_y = next((k for k in varying if k != param_x), None) if not declared and len(varying) == 2 else None
        param_y = st.selectbox("Second parameter (heatmap)", second_opts,
                               index=second_opts.index(default_y) if default_y in second_opts else 0,
                               help="Two swept parameters: mean metric per cell. Conditions of a 1-D sweep belong in 'One line per'.")
        metric = st.selectbox("Metric", metrics)
        group_opts = line_keys(recs, param_x)
        group_by = st.multiselect("One line per", group_opts, default=["method"] if "method" in group_opts else [],
                                  help="Split the pooled mean into lines: per method arm, per condition (config.noise, "
                                       "derived.kernel_type, ...). Without a split every value pools all conditions and seeds.")
        show_band = st.checkbox("Shaded ± std band", value=True, help="Off: error bars instead.")

    hib = defs.get(metric, {}).get("higher_is_better", True)
    fmt = fmt_for(defs, metric)
    unit = defs.get(metric, {}).get("unit", "")
    arrow = "↑" if hib else "↓"

    if param_y != "— none —":
        grid = agg.sweep_grid(recs, param_x, param_y, metric)
        if not grid.cells:
            st.warning("No runs have both parameters set.")
            return
        best = grid.best(hib)
        st.caption(f"{experiment} · {len(recs)} runs{filt} · {metric} {arrow} over {param_x} × {param_y} · "
                   f"best at {param_x}={best[0]}, {param_y}={best[1]}: {grid.cells[best].format(fmt)}")
        st.plotly_chart(sweep_heatmap(grid.xs, grid.ys, grid.matrix(), param_x, param_y, metric, fmt, hib, best),
                        theme=None, width="stretch")
        ns = sorted({c.n for c in grid.cells.values()})
        n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
        st.markdown(figure_caption_html(
            f"Mean {metric} over {n_txt} runs per cell as a function of {param_x} (columns) and {param_y} (rows); "
            f"darker is {'better' if hib else 'better (lower)'}. Best cell outlined: {param_x} = {best[0]}, {param_y} = {best[1]} "
            f"({grid.cells[best].format(fmt)}).", number=1), unsafe_allow_html=True)
        headers = [f"{param_y} \\ {param_x}"] + [f"{x:g}" if isinstance(x, float) else str(x) for x in grid.xs]
        rows = []
        for y in grid.ys:
            cells = []
            for x in grid.xs:
                c = grid.cells.get((x, y))
                txt = "—" if c is None else c.format(fmt)
                cells.append(f"<b>{txt}</b>" if (x, y) == best else txt)
            rows.append([f"{y:g}" if isinstance(y, float) else str(y)] + cells)
        st.markdown(generic_html(headers, rows, number=1, left_cols=1, raw_html_cols=list(range(1, len(headers))),
                                 caption=f"{metric} (mean ± std) over {param_x} × {param_y}. Best in bold."), unsafe_allow_html=True)
        df = pd.DataFrame(
            [{param_y: y, **{str(x): (grid.cells[(x, y)].mean if (x, y) in grid.cells else None) for x in grid.xs}}
             for y in grid.ys]
        )
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"{experiment}-{param_x}-{param_y}.csv", mime="text/csv")
        return

    series = agg.sweep_series(recs, param_x, metric, group_by=group_by)
    series = {g: s for g, s in series.items() if s}
    if not series:
        st.warning(f"No runs have `{param_x}` in their config.")
        return
    best = {g: agg.best_sweep_value(s, hib) for g, s in series.items()}
    xs_all = sorted({x for s in series.values() for x, _ in s}, key=lambda x: (isinstance(x, str), x))
    log_default = is_log_friendly(xs_all)
    log_x = st.sidebar.checkbox("Log x axis", value=log_default, disabled=not is_log_friendly(xs_all))

    pooled = [k for k in line_keys(recs, param_x) if k not in group_by and k not in ("instance", "dataset")]
    pool_note = f" · pooled over {', '.join(pooled)}" if pooled else ""
    if len(series) == 1:
        (g, s), = series.items()
        st.caption(f"{experiment} · {len(recs)} runs{filt} · {metric} {arrow} vs {param_x}{pool_note} · "
                   f"best {param_x} = **{best[g]}** ({dict(s)[best[g]].format(fmt)})")
    else:
        st.caption(f"{experiment} · {len(recs)} runs{filt} · {metric} {arrow} vs {param_x}{pool_note} · best per line: " +
                   ", ".join(f"{' / '.join(map(str, g))} → {b}" for g, b in best.items()))

    st.plotly_chart(sweep_lines(series, param_x, metric, fmt, unit, log_x=log_x, band=show_band, best_by_group=best),
                    theme=None, width="stretch")

    # sensitivity: how flat is the optimum?
    plateaus = {g: agg.sweep_plateau(s_, hib) for g, s_ in series.items()}
    ns = sorted({st_.n for s_ in series.values() for _, st_ in s_})
    n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
    parts = [f"{metric}{f' ({unit})' if unit else ''} as a function of {param_x}; mean ± std over {n_txt} runs per value"
             + ("; log axis" if log_x else "") + "."]
    for g, pl in plateaus.items():
        if pl is None:
            continue
        who = (" / ".join(map(str, g)) + ": ") if g else ""
        lo, hi = pl.span
        within = f"within {pl.tolerance:{fmt}} of the best" if pl.tolerance else "at the best"
        parts.append(f"{who}best {param_x} = {pl.best} ({pl.best_stat.format(fmt)}); {metric} stays {within} for "
                     + (f"{param_x} ∈ [{lo}, {hi}]" if lo != hi else f"{param_x} = {lo} only")
                     + f"; the worst value ({param_x} = {pl.worst}) costs {pl.drop:{fmt}}.")
    st.markdown(figure_caption_html(" ".join(parts), number=1), unsafe_allow_html=True)

    st.markdown(sweep_html(series, param_x, metric, defs, number=1), unsafe_allow_html=True)

    st.subheader("Sensitivity")
    srows = []
    for g, pl in plateaus.items():
        if pl is None:
            continue
        lo, hi = pl.span
        srows.append([" / ".join(map(str, g)) if g else metric, str(pl.best), pl.best_stat.format(fmt),
                      f"[{lo}, {hi}]" if lo != hi else str(lo), f"{len(pl.members)} / {len(series[g])}",
                      f"{pl.worst} ({pl.drop:{fmt}} worse)"])
    st.markdown(generic_html(["Line", f"best {param_x}", f"{metric} at best", "plateau", "values on plateau", "worst value"],
                             srows, number=2, left_cols=1,
                             caption=f"Sensitivity to {param_x}: the plateau is the range of values whose mean {metric} is within one "
                                     f"std of the best (or 1% of the range when std is 0). A wide plateau means the choice is forgiving."),
                unsafe_allow_html=True)

    pin_to_paper({"sweep-figure": {"param": param_x, "metric": metric, "by": group_by, "band": show_band, "log_x": log_x, "width": "single"},
                  "sweep-table": {"param": param_x, "metric": metric, "by": group_by}},
                 records=all_recs, key="sweep_pin")
    with st.expander("LaTeX (booktabs table + figure snippet)"):
        st.code(sweep_latex(series, param_x, metric, defs), language="latex")
        st.code(figure_tex(f"figures/{experiment}-{param_x}-{metric}.pdf", label=f"fig:{experiment}-{param_x}", width="single"),
                language="latex")
        st.caption("More options (captions, labels, widths) on the Export page.")
    with st.expander("Paper figure (matplotlib, IEEE style)"):
        pf = sweep_figure(series, param_x, metric, ylabel=f"{metric} ({unit})" if unit else metric, best_by_group=best,
                          band=show_band, log_x=log_x, width="single")
        g1, g2 = st.columns([3, 1])
        gray = g2.checkbox("Grayscale", value=False, key="sweep_gray")
        png = figure_bytes(pf, "png", dpi=200)
        g1.image(to_grayscale_png(png) if gray else png)
        g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{experiment}-{param_x}-{metric}.pdf", mime="application/pdf")

    rows = []
    for g, s in series.items():
        for x, st_ in s:
            rows.append({**dict(zip(group_by, g)), param_x: x, f"{metric}_mean": st_.mean, f"{metric}_std": st_.std, "n": st_.n})
    df = pd.DataFrame(rows)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download CSV", buf.getvalue(), file_name=f"{experiment}-{param_x}-{metric}.csv", mime="text/csv")
