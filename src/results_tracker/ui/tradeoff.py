"""Trade-off page: two metrics against each other (runtime vs PSNR), one series per method joined along a path
key such as the iteration budget K. Baselines and reported numbers get hollow markers: the paper's
parameter-free (filled) vs tuned-by-others (hollow) convention. Pins as a `tradeoff-figure` asset.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .. import aggregate as agg
from ..export.figures import figure_bytes, figure_tex, to_grayscale_png, tradeoff_figure
from .charts import tradeoff_scatter
from .common import (
    active_where, fmt_for, keyed, keyed_multiselect, keyed_selectbox, load_metric_defs, load_records_union, pin_to_paper,
    reset_on_experiment_change, select_extra_experiments, select_project_experiment, sidebar_db, sidebar_filter, where_text,
)
from .tables import figure_caption_html, generic_html

NONE = "— none —"
WIDTHS = ["single", "double", "ieee-single", "ieee-double"]


def prefill_from_asset(a) -> dict[str, Any]:
    o = dict(a.options or {})
    pre: dict[str, Any] = {}
    for key, opt in (("to_x", "x_metric"), ("to_y", "y_metric"), ("to_series", "series"), ("to_logx", "log_x"),
                     ("to_hollow_base", "hollow_baselines"), ("to_hollow", "hollow"), ("to_width", "width"), ("to_panel", "panel_label"),
                     ("to_xlabel", "xlabel"), ("to_ylabel", "ylabel")):
        if opt in o and o[opt] is not None:
            pre[key] = o[opt]
    pre["to_path"] = o.get("path") or NONE
    return pre


def render() -> None:
    st.title("Trade-off")
    sidebar_db()
    project, experiment = select_project_experiment()
    if experiment is None:
        return
    reset_on_experiment_change("to_", experiment)
    extra = select_extra_experiments(project, experiment)
    recs = load_records_union(project, [experiment, *extra])
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return
    recs = agg.completed(sidebar_filter(recs))
    if not recs:
        return
    metrics = agg.metric_names(recs)
    if len(metrics) < 2:
        st.info("A trade-off needs two metrics (e.g. runtime_s and psnr).")
        return
    title = experiment + (f" + {', '.join(extra)}" if extra else "")
    keys = agg.grouping_keys(recs, varying_only=True)

    with st.sidebar:
        st.markdown("**Axes**")
        x_metric = keyed_selectbox("x metric (cost)", metrics, "to_x", next((m for m in metrics if "time" in m or "iter" in m), metrics[-1]))
        y_metric = keyed_selectbox("y metric (quality)", metrics, "to_y", next((m for m in ("psnr", "ssim") if m in metrics), metrics[0]))
        series_key = keyed_selectbox("One series per", keys or ["method"], "to_series", "method")
        path_opts = [NONE] + [k for k in keys if k != series_key]
        path = keyed_selectbox("Points along", path_opts, "to_path", next((k for k in path_opts if k.endswith(".K")), NONE),
                               help="Joined in order within a series: the iteration budget K, a dataset, ...")
        log_x = keyed(st.checkbox, "Log x axis", "to_logx", True)
        hollow_base = keyed(st.checkbox, "Hollow markers for baselines / reported", "to_hollow_base", True)
        hollow_extra = keyed_multiselect("Also hollow", sorted({str(agg.get_field(r, series_key)) for r in recs}), "to_hollow", [])
    path_key = None if path == NONE else path
    pts = agg.tradeoff_points(recs, x_metric, y_metric, series_key=series_key, path_key=path_key)
    if not pts:
        st.warning(f"No run has both `{x_metric}` and `{y_metric}`.")
        return
    hollow = set(hollow_extra) | ({r["method"] for r in recs if r.get("method_is_baseline") or r.get("source") == "reported"}
                                  if hollow_base and series_key == "method" else set())
    labels = agg.method_labels(recs) if series_key == "method" else None
    x_unit, y_unit = defs.get(x_metric, {}).get("unit", ""), defs.get(y_metric, {}).get("unit", "")
    xlabel_default = f"{x_metric} ({x_unit})" if x_unit else x_metric
    ylabel_default = f"{y_metric} ({y_unit})" if y_unit else y_metric
    st.caption(f"{title} · {len(recs)} runs" + (f" · filter: {where_text()}" if active_where() else "")
               + f" · {y_metric} against {x_metric}, one series per {series_key}" + (f", points along {path_key}" if path_key else "")
               + (f" · hollow: {', '.join(map(str, sorted(hollow)))}" if hollow else ""))
    st.plotly_chart(tradeoff_scatter(pts, x_metric, y_metric, x_fmt=fmt_for(defs, x_metric), y_fmt=fmt_for(defs, y_metric),
                                     xlabel=xlabel_default, ylabel=ylabel_default, log_x=log_x, hollow=hollow, labels=labels),
                    theme=None, width="stretch")
    st.markdown(figure_caption_html(
        f"{y_metric} against {x_metric} (mean ± std per point" + (f", one point per {path_key}" if path_key else "") + ")."
        + (" Filled markers: " + ", ".join(str(s) for s in pts if s not in hollow) + "; hollow: " + ", ".join(map(str, sorted(hollow))) + "." if hollow else ""),
        number=1), unsafe_allow_html=True)
    rows = [[(labels or {}).get(s, str(s)), agg.fmt_value(p.label) if path_key else "—", p.x.format(fmt_for(defs, x_metric)),
             p.y.format(fmt_for(defs, y_metric)), p.x.n] for s, plist in pts.items() for p in plist]
    st.markdown(generic_html([series_key, path_key or "", x_metric, y_metric, "n"], rows, number=1, left_cols=2,
                             caption="Every point of the figure."), unsafe_allow_html=True)

    with st.expander("Paper figure (matplotlib, IEEE style)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            width = keyed_selectbox("Width", WIDTHS, "to_width", "single")
        with c2:
            xlabel = keyed(st.text_input, "x label", "to_xlabel", xlabel_default)
        with c3:
            ylabel = keyed(st.text_input, "y label", "to_ylabel", ylabel_default)
        cap = keyed(st.text_input, "Panel caption", "to_panel", "", placeholder="a. Cost vs quality")
        pf = tradeoff_figure(pts, x_metric, y_metric, xlabel=xlabel, ylabel=ylabel, log_x=log_x, hollow=hollow, width=width,
                             labels=labels, caption=cap or None)
        g1, g2 = st.columns([3, 1])
        gray = g2.checkbox("Grayscale", value=False, key="to_gray")
        png = figure_bytes(pf, "png", dpi=200)
        g1.image(to_grayscale_png(png) if gray else png)
        stem = f"{experiment}-{y_metric}-vs-{x_metric}".replace(" ", "_")
        g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
        st.code(figure_tex(f"figures/{stem}.pdf", label=f"fig:{stem}", width=width), language="latex")
    pin_to_paper({"tradeoff-figure": {"x_metric": x_metric, "y_metric": y_metric, "series": series_key, "path": path_key, "log_x": log_x,
                                      "hollow_baselines": hollow_base, "hollow": hollow_extra, "width": width, "xlabel": xlabel,
                                      "ylabel": ylabel, "panel_label": cap or None}},
                 records=recs, key="to_pin", suggested_label=f"fig:{experiment}-tradeoff", extra_experiments=extra)
