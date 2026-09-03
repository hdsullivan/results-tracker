"""Curves page: per-iteration diagnostics (PSNR, step size, noise-level estimate, ...) averaged over runs.

Reads each run's `diagnostics.json` (see curves.py), pools the chosen curve per line (method arm, condition,
derived field), and shows mean ± std against the iteration with the individual runs as an optional overlay.
Pins as a `curves-figure` asset; the print figure follows the lab style.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from .. import aggregate as agg
from ..curves import curve_names, curve_series, normalise, record_curves
from ..export.figures import curves_figure, figure_bytes, figure_tex, to_grayscale_png
from .charts import curves_lines
from .common import (
    active_where, keyed, keyed_multiselect, keyed_selectbox, load_metric_defs, load_records_union, pin_to_paper,
    reset_on_experiment_change, select_extra_experiments, select_project_experiment, sidebar_db, sidebar_filter, where_text,
)
from .tables import figure_caption_html, generic_html

NORMS = {"value": "as recorded", "delta": "minus the first iteration", "ratio": "divided by the first iteration"}
WIDTHS = ["single", "double", "ieee-single", "ieee-double"]


def prefill_from_asset(a) -> dict[str, Any]:
    o = dict(a.options or {})
    pre: dict[str, Any] = {}
    for key, opt in (("cur_curve", "curve"), ("cur_by", "by"), ("cur_norm", "normalise"), ("cur_band", "band"), ("cur_logy", "log_y"),
                     ("cur_width", "width"), ("cur_panel", "panel_label"), ("cur_ylabel", "ylabel")):
        if opt in o and o[opt] is not None:
            pre[key] = o[opt]
    if o.get("guide") is not None:
        pre["cur_guide"] = float(o["guide"])
    return pre


def line_keys(records: list[dict]) -> list[str]:
    return agg.grouping_keys(records, base=("experiment", "method", "dataset", "instance", "seed"), varying_only=True)


def render() -> None:
    st.title("Curves")
    sidebar_db()
    project, experiment = select_project_experiment()
    if experiment is None:
        return
    reset_on_experiment_change("cur_", experiment)
    extra = select_extra_experiments(project, experiment)
    recs = load_records_union(project, [experiment, *extra])
    defs = load_metric_defs()
    if not recs:
        st.info("No runs in this experiment.")
        return
    recs = agg.completed(sidebar_filter(recs))
    if not recs:
        return
    names = curve_names(recs)
    if not names:
        st.info("No completed run of this experiment has curves in its `diagnostics.json`. The recipe runner writes non-scalar "
                "diagnostics (a `curves` dict or bare lists) there for every run; `log_run` users can write the same file into "
                "the run's `artifacts_dir`.")
        return
    title = experiment + (f" + {', '.join(extra)}" if extra else "")

    with st.sidebar:
        st.markdown("**Curve**")
        curve = keyed_selectbox("Curve", names, "cur_curve", "psnr" if "psnr" in names else names[0])
        with_curve = [r for r in recs if curve in record_curves(r)]  # only these runs are plotted, so only their keys vary
        by_opts = line_keys(with_curve)
        by = keyed_multiselect("One line per", by_opts, "cur_by", ["method"] if "method" in by_opts else [],
                               help="method arm, condition (config.noise, derived.kernel_type, ...); the rest is pooled")
        norm = keyed_selectbox("Normalise", list(NORMS), "cur_norm", "value", format_func=NORMS.get)
        band = keyed(st.checkbox, "Shaded ± std band", "cur_band", True)
        members = keyed(st.checkbox, "Show individual runs", "cur_members", False)
        log_y = keyed(st.checkbox, "Log y axis", "cur_logy", False)
        guide = keyed(st.number_input, "Reference line (0 = none)", "cur_guide", 0.0, step=0.5,
                      help="e.g. 1.0 for a ratio such as sigma_hat / sigma")
    series = {g: normalise(cs, norm) for g, cs in curve_series(recs, curve, by).items()}
    if not series:
        st.warning(f"No run has a `{curve}` curve.")
        return
    n_runs = sum(cs.runs for cs in series.values())
    pooled = [k for k in by_opts if k not in by and k not in ("instance", "seed")]
    st.caption(f"{title} · {n_runs} runs with curves" + (f" · filter: {where_text()}" if active_where() else "")
               + f" · {curve} vs iteration ({NORMS[norm]})" + (f" · pooled over {', '.join(pooled)}" if pooled else ""))
    ylabel = curve if norm == "value" else f"{curve} ({NORMS[norm]})"
    st.plotly_chart(curves_lines(series, curve, ylabel=ylabel, band=band, log_y=log_y, guide=guide or None, members=members),
                    theme=None, width="stretch")
    rows = []
    for g, cs in series.items():
        final = cs.final()
        rows.append([" / ".join(map(str, g)) if g else curve, len(cs.mean), cs.runs,
                     "—" if final is None else f"{final:.4g} ± {cs.std[-1]:.3g}", f"{max(cs.mean):.4g}" if cs.mean else "—"])
    lens = sorted({len(cs.mean) for cs in series.values()})
    st.markdown(figure_caption_html(
        f"{curve} against the iteration, mean ± std over runs ({', '.join(f'{cs.runs}' for cs in series.values())} per line)"
        + (f", {NORMS[norm]}" if norm != "value" else "") + f"; {lens[0]}" + (f"–{lens[-1]}" if len(lens) > 1 else "") + " iterations.", number=1),
        unsafe_allow_html=True)
    st.markdown(generic_html(["Line", "iterations", "runs", "final (mean ± std)", "max of mean"], rows, number=1, left_cols=1,
                             caption="Where each line ends."), unsafe_allow_html=True)

    with st.expander("Paper figure (matplotlib, IEEE style)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            width = keyed_selectbox("Width", WIDTHS, "cur_width", "single")
        with c2:
            ylab = keyed(st.text_input, "y label", "cur_ylabel", ylabel)
        with c3:
            cap = keyed(st.text_input, "Panel caption", "cur_panel", "", placeholder="a. PSNR per iteration")
        pf = curves_figure(series, curve, ylabel=ylab, band=band, log_y=log_y, width=width, caption=cap or None, guide=guide or None)
        g1, g2 = st.columns([3, 1])
        gray = g2.checkbox("Grayscale", value=False, key="cur_gray")
        png = figure_bytes(pf, "png", dpi=200)
        g1.image(to_grayscale_png(png) if gray else png)
        stem = f"{experiment}-{curve}".replace(" ", "_")
        g2.download_button("Download PDF", figure_bytes(pf, "pdf"), file_name=f"{stem}.pdf", mime="application/pdf")
        st.code(figure_tex(f"figures/{stem}.pdf", label=f"fig:{stem}", width=width), language="latex")
    pin_to_paper({"curves-figure": {"curve": curve, "by": by, "normalise": norm, "band": band, "log_y": log_y, "guide": guide or None,
                                    "width": width, "ylabel": ylab, "panel_label": cap or None}},
                 records=recs, key="cur_pin", suggested_label=f"fig:{experiment}-{curve}", extra_experiments=extra)
