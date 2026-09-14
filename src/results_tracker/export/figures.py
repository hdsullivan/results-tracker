r"""Paper figures in the lab's IEEE style (ported from adaptivePnP `ablation_utils.set_paper_style`).

- Times-like serif text: 8 pt base, 11 pt axis labels, 9 pt ticks, 11 pt legend, 10.5 pt bold panel captions.
- Full boxed axes frame (0.8 pt) with inward major + minor ticks on all four sides, no grid.
- Bordered legend (black edge, square corners) placed across the top of the figure.
- Solid tab10-style colours with filled circle markers; the proposed method gets a heavier line and
  larger markers; uncertainty as a shaded band (alpha 0.15), error bars on request. More than eight series
  wrap the palette, and the marker shape and dash change with the wrap so a shared hue is never the only
  difference between two lines.
- Bold "(a) ..." captions below each panel via `panel_label`.
- Figure widths 5.0 in (column) / 10.5 in (page): sized for comfortable review and scaled by LaTeX to
  \columnwidth / \textwidth; pass a number of inches for exact IEEE widths (3.5 / 7.16).
- Deterministic: Figure objects only, no pyplot global state; TrueType fonts embedded.

Every size, weight, colour and category order above is a *default*: pass a `plotstyle.PlotStyle` (a
project's, from the GUI) as `style=` to override it, and `xlim=` / `ylim=` to fix an axis range. The
on-screen charts (`ui/charts.py`) read the same style, so the GUI is a preview of these figures.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.ticker
from matplotlib.figure import Figure

from .. import aggregate as agg
from .. import plotstyle
from ..plotstyle import VARIANT_KEY, PlotStyle

# Lab convention (ablation_utils.IEEE_COLUMN_WIDTH / IEEE_PAGE_WIDTH): wider than the literal IEEE
# column so point sizes read as body text once LaTeX scales the figure down.
SINGLE_COL_IN = 5.0
DOUBLE_COL_IN = 10.5
IEEE_SINGLE_COL_IN = 3.5   # literal IEEEtran \columnwidth
IEEE_DOUBLE_COL_IN = 7.16  # literal IEEEtran \textwidth
WIDTHS = {"single": SINGLE_COL_IN, "double": DOUBLE_COL_IN, "ieee-single": IEEE_SINGLE_COL_IN, "ieee-double": IEEE_DOUBLE_COL_IN}

# The defaults of plotstyle.PlotStyle, spelled out (a project's style overrides any of them).
LINE_WIDTH = plotstyle.DEFAULT.line_width
MARKER_SIZE = plotstyle.DEFAULT.marker_size
AXIS_LABEL_SIZE = plotstyle.DEFAULT.axis_label
TICK_LABEL_SIZE = plotstyle.DEFAULT.tick
LEGEND_SIZE = plotstyle.DEFAULT.legend
PANEL_LABEL_SIZE = plotstyle.DEFAULT.panel_label
# STIX mathtext renders visibly smaller than serif body text, so a label that is *entirely* math
# (e.g. r"$\lambda$") gets this size instead of AXIS_LABEL_SIZE (lab rule: AXIS_LABEL_SIZE_MATH).
AXIS_LABEL_SIZE_MATH = plotstyle.DEFAULT.axis_label + plotstyle.DEFAULT.math_bump

# Fixed hue order used across the lab's ablation figures (tab10 subset): blue, red, green, purple, orange,
# brown, gray, pink. Assigned in first-seen order, never re-ranked.
PALETTE = plotstyle.PALETTE
LINESTYLES = list(plotstyle.MPL_LINESTYLES)   # solid for the first eight series; a wrapped hue gets a dash
MARKERS = list(plotstyle.MPL_MARKERS)
HATCHES = ["", "////", "\\\\\\\\", "xxxx", "....", "++++"]
BAR_HATCHES = HATCHES
BAR_FILLS = PALETTE
IMPROVES, HURTS, NEUTRAL = "#1f77b4", "#d62728", "#7f7f7f"
GUIDE_COLOR = "0.55"  # dotted vertical guides (add_iteration_markers)

IEEE_RC = {
    "font.size": 8,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": AXIS_LABEL_SIZE,
    "axes.titlesize": AXIS_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.borderpad": 0.4,
    "hatch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "figure.dpi": 150,
}


def width_in(width: Union[str, float]) -> float:
    if isinstance(width, (int, float)):
        return float(width)
    return WIDTHS[width]


def ieee_rc(style: Optional[PlotStyle] = None) -> dict[str, Any]:
    """IEEE_RC with the style's sizes and weights substituted (the rest of the look is fixed)."""
    s = plotstyle.resolve(style)
    return {**IEEE_RC, "font.size": s.base, "axes.labelsize": s.axis_label, "axes.titlesize": s.axis_label,
            "legend.fontsize": s.legend, "xtick.labelsize": s.tick, "ytick.labelsize": s.tick,
            "lines.linewidth": s.line_width, "lines.markersize": s.marker_size}


def style_map(names: Sequence[Any], emphasize: Iterable[Any] = (), style: Optional[PlotStyle] = None) -> dict[Any, dict[str, Any]]:
    """Fixed colour/linestyle/marker per entity in first-seen order; emphasised ones are a bit thicker.

    A style's `colors` override the palette hue of a named series (the bar fill follows it). Up to eight
    series this is the lab's look exactly: a hue each, solid, filled circles. Beyond that the palette wraps,
    so the shape and the dash change with it -- two lines must never differ only in a hue they share.
    """
    s = plotstyle.resolve(style)
    emph = set(emphasize)
    hues = s.colors_for(names)
    wraps = s.wrap_of(names)
    out: dict[Any, dict[str, Any]] = {}
    for n in names:
        if n in out:
            continue
        i = len(out)
        primary = n in emph
        lw, ms = s.weights(primary)
        wrap = wraps[n]
        out[n] = dict(
            color=hues[n],
            linestyle=LINESTYLES[wrap % len(LINESTYLES)],
            marker=MARKERS[wrap % len(MARKERS)],
            fill=hues[n],
            hatch=BAR_HATCHES[i % len(BAR_HATCHES)],
            linewidth=lw,
            markersize=ms,
            zorder=3 if primary else 2,
        )
    return out


def _apply_limits(ax, xlim: Optional[Sequence[float]] = None, ylim: Optional[Sequence[float]] = None) -> None:
    """Fix an axis range chosen in the GUI (`plotstyle.parse_limits`); None leaves it automatic."""
    if xlim:
        ax.set_xlim(float(xlim[0]), float(xlim[1]))
    if ylim:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))


def _seed_categories(ax, axis: str, categories: Sequence[str]) -> None:
    """Fix the order of a categorical axis. Matplotlib numbers string categories in the order it first sees
    them, so plotting them all (as NaN, invisible) up front pins the order the style asks for."""
    cats = [str(c) for c in categories]
    if len(cats) < 2:
        return
    nan = [float("nan")] * len(cats)
    if axis == "x":
        ax.plot(cats, nan, linestyle="none", marker="")
    else:
        ax.plot(nan, cats, linestyle="none", marker="")


def _label(g: tuple, fallback: str) -> str:
    return " / ".join(map(str, g)) if g else fallback


def _new_figure(width: Union[str, float], height: Optional[float]) -> Figure:
    w = width_in(width)
    h = height if height is not None else w * 0.62
    return Figure(figsize=(w, h))


def _target(into, width: Union[str, float], height: Optional[float]):
    """(figure, axes) to draw into: the axes a panel layout provides, or a new single-panel figure.

    Every figure below takes `into=` so `panel_figure` can lay several of them out side by side without a
    second implementation of each plot."""
    if into is not None:
        return into.figure, into
    fig = _new_figure(width, height)
    return fig, fig.add_subplot(111)


def _thin_legend(leg) -> None:
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)


def panel_label(ax, text: str, style: Optional[PlotStyle] = None) -> None:
    """Bold IEEE-style subfigure caption below the panel, e.g. "a. PSNR" or "(a) PSNR" -> "(a) PSNR".

    Offset in points so it clears the x-axis label regardless of the panel height (lab convention)."""
    import re

    m = re.match(r"^\(?([A-Za-z])[.)]\s*(.*)$", text)
    if m:
        letter, rest = m.group(1).lower(), m.group(2)
        text = f"({letter}) {rest}".strip()
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -32), textcoords="offset points",
                ha="center", va="top", fontsize=plotstyle.resolve(style).panel_label, fontweight="bold", annotation_clip=False)


def set_axis_labels(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None, style: Optional[PlotStyle] = None) -> None:
    """Axis labels with the lab's math-only bump: '$\\lambda$' -> 14 pt, 'PSNR (dB)' -> 11 pt."""
    s = plotstyle.resolve(style)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=s.label_size(xlabel))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=s.label_size(ylabel))


def top_legend(ax, ncol: Optional[int] = None):
    """Bordered legend in one row above the axes (lab's add_top_legend), so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    leg = ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=ncol or len(handles),
                    borderaxespad=0.0, handlelength=2.0, columnspacing=1.2)
    _thin_legend(leg)
    return leg


def _tight_ylim(ax, los: Sequence[float], his: Sequence[float], zero_based: bool = False) -> None:
    """Data-tight y limits with head-room; zero_based anchors the bottom at 0 (only when data is positive)."""
    if not los:
        return
    lo, hi = min(los), max(his)
    span = max(hi - lo, 0.05 * abs(hi) if hi else 1.0)
    if zero_based and lo >= 0:
        ax.set_ylim(0, hi + 0.12 * span)
    else:
        ax.set_ylim(lo - 0.15 * span, hi + 0.25 * span)


# --------------------------------------------------------------------------- sweep

def sweep_figure(
    series_by_group: Mapping[tuple, Sequence[tuple[Any, agg.Stat]]],
    param: str,
    metric: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    log_x: Optional[bool] = None,
    band: bool = True,
    best_by_group: Optional[Mapping[tuple, Any]] = None,
    mark_best: bool = True,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    emphasize: Iterable[Any] = (),
    labels: Optional[Mapping[tuple, str]] = None,
    legend_loc: str = "top",
    caption: Optional[str] = None,
    style: Optional[PlotStyle] = None,
    by: Sequence[str] = (),
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    into: Optional[Any] = None,
) -> Figure:
    """Metric vs swept parameter: mean line with a shaded ± std band (or error bars), best value ringed.

    `legend_loc="top"` puts the bordered legend above the axes (lab convention); any matplotlib loc works too.
    `caption` adds a bold "(a) ..." label under the panel. `by` names the keys the lines are split by, so the
    style can order the legend; a categorical x axis follows the style's order for `param`."""
    st_ = plotstyle.resolve(style)
    with matplotlib.rc_context(ieee_rc(style)):
        fig, ax = _target(into, width, height)
        groups = st_.ordered(" / ".join(by), [g for g, s in series_by_group.items() if s])
        styles = style_map([_label(g, metric) for g in groups], emphasize, style=style)
        xs_all = sorted({x for g in groups for x, _ in series_by_group[g]}, key=lambda x: (isinstance(x, str), x))
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs_all)
        if log_x is None:
            log_x = numeric and len(xs_all) >= 3 and min(xs_all) > 0 and max(xs_all) / min(xs_all) >= 10
        if not numeric:
            xs_all = st_.ordered(param, xs_all)
            _seed_categories(ax, "x", [str(x) for x in xs_all])
        for g in groups:
            name = (labels or {}).get(g) or _label(g, metric)
            st = styles[_label(g, metric)]
            series = list(series_by_group[g])
            xs = [x for x, _ in series] if numeric else [str(x) for x, _ in series]
            ys = [s.mean for _, s in series]
            sd = [s.std for _, s in series]
            mk = dict(marker=st["marker"], markersize=st["markersize"], color=st["color"], linestyle=st["linestyle"],
                      linewidth=st["linewidth"], zorder=st["zorder"], label=name)
            if band and any(v > 0 for v in sd):
                ax.fill_between(xs, [m - s for m, s in zip(ys, sd)], [m + s for m, s in zip(ys, sd)],
                                color=st["color"], alpha=0.15, linewidth=0)
                ax.plot(xs, ys, **mk)
            else:
                ax.errorbar(xs, ys, yerr=sd if any(v > 0 for v in sd) else None, ecolor=st["color"],
                            elinewidth=0.8, capsize=2, capthick=0.8, **mk)
            best = (best_by_group or {}).get(g)
            if mark_best and best is not None and best in dict(series):
                bx = best if numeric else str(best)
                # ring the chosen value: larger hollow marker in the series colour over the filled point
                ax.plot([bx], [dict(series)[best].mean], marker=st["marker"], markersize=st["markersize"] + 4.5,
                        markerfacecolor="none", markeredgecolor=st["color"], markeredgewidth=1.2, linestyle="none", zorder=6)
                if len(groups) == 1:
                    ax.axvline(bx, color=GUIDE_COLOR, linestyle=":", linewidth=1.0, zorder=0)
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(xs_all)
            ax.set_xticklabels([f"{x:g}" for x in xs_all])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        set_axis_labels(ax, xlabel or param, ylabel or metric, style=style)
        _apply_limits(ax, xlim, ylim)
        if len(groups) > 1:
            if legend_loc == "top":
                top_legend(ax)
            else:
                _thin_legend(ax.legend(loc=legend_loc))
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- curves

def curves_figure(
    series_by_group: Mapping[tuple, Any],
    curve: str,
    *,
    xlabel: str = "iteration",
    ylabel: Optional[str] = None,
    band: bool = True,
    log_y: bool = False,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    emphasize: Iterable[Any] = (),
    labels: Optional[Mapping[tuple, str]] = None,
    caption: Optional[str] = None,
    guide: Optional[float] = None,
    style: Optional[PlotStyle] = None,
    by: Sequence[str] = (),
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    into: Optional[Any] = None,
) -> Figure:
    """A per-iteration curve (`curves.CurveStat`) per group: mean line, shaded ± std band, lab style.
    `guide` draws a dotted horizontal reference (1.0 for a ratio such as sigma_hat / sigma)."""
    st_ = plotstyle.resolve(style)
    with matplotlib.rc_context(ieee_rc(style)):
        fig, ax = _target(into, width, height)
        groups = st_.ordered(" / ".join(by), [g for g, cs in series_by_group.items() if cs.mean])
        styles = style_map([_label(g, curve) for g in groups], emphasize, style=style)
        for g in groups:
            cs = series_by_group[g]
            st = styles[_label(g, curve)]
            name = (labels or {}).get(g) or _label(g, curve)
            xs = cs.x
            if band and any(v > 0 for v in cs.std):
                ax.fill_between(xs, [m - s for m, s in zip(cs.mean, cs.std)], [m + s for m, s in zip(cs.mean, cs.std)],
                                color=st["color"], alpha=0.15, linewidth=0)
            ax.plot(xs, cs.mean, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=st["zorder"],
                    marker="o" if len(xs) <= 25 else None, markersize=st["markersize"], label=name)
        if guide is not None:
            ax.axhline(guide, color=GUIDE_COLOR, linestyle=":", linewidth=1.0, zorder=0)
        if log_y:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        set_axis_labels(ax, xlabel, ylabel or curve, style=style)
        _apply_limits(ax, xlim, ylim)
        if len(groups) > 1:
            top_legend(ax, ncol=min(len(groups), 4))
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- trade-off

def tradeoff_figure(
    points_by_series: Mapping[Any, Sequence[Any]],
    x_metric: str,
    y_metric: str,
    *,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    log_x: bool = True,
    hollow: Iterable[Any] = (),
    annotate: bool = True,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    emphasize: Iterable[Any] = (),
    labels: Optional[Mapping[Any, str]] = None,
    caption: Optional[str] = None,
    style: Optional[PlotStyle] = None,
    series_key: str = "method",
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    into: Optional[Any] = None,
) -> Figure:
    """Two metrics against each other (`aggregate.tradeoff_points`): one series per method, its points joined
    along the path key (iteration budget), error bars = std. Series in `hollow` (baselines, reported numbers)
    get open markers and no line: the paper's filled-vs-hollow convention for parameter-free vs tuned."""
    hollow_set = set(hollow)
    st_ = plotstyle.resolve(style)
    with matplotlib.rc_context(ieee_rc(style)):
        fig, ax = _target(into, width, height)
        names = st_.ordered(series_key, list(points_by_series))
        styles = style_map([str(n) for n in names], [str(e) for e in emphasize], style=style)
        for name in names:
            pts = list(points_by_series[name])
            if not pts:
                continue
            st = styles[str(name)]
            xs = [p.x.mean for p in pts]
            ys = [p.y.mean for p in pts]
            open_marker = name in hollow_set
            ax.errorbar(xs, ys, xerr=[p.x.std for p in pts] if any(p.x.std > 0 for p in pts) else None,
                        yerr=[p.y.std for p in pts] if any(p.y.std > 0 for p in pts) else None,
                        color=st["color"], ecolor=st["color"], elinewidth=0.6, capsize=1.5, capthick=0.6,
                        linestyle="none" if (open_marker or len(pts) == 1) else st["linestyle"], linewidth=st["linewidth"],
                        marker="o", markersize=st["markersize"] + 1, markerfacecolor="none" if open_marker else st["color"],
                        markeredgecolor=st["color"], markeredgewidth=1.0, zorder=st["zorder"], label=(labels or {}).get(name, str(name)))
            if annotate and len(pts) > 1:
                for p in pts:
                    ax.annotate(agg.fmt_value(p.label), (p.x.mean, p.y.mean), xytext=(3, 3), textcoords="offset points",
                                fontsize=max(st_.annotation - 2, 1.0), color=st["color"])
        if log_x:
            ax.set_xscale("log")
        set_axis_labels(ax, xlabel or x_metric, ylabel or y_metric, style=style)
        _apply_limits(ax, xlim, ylim)
        if len(names) > 1:
            top_legend(ax, ncol=min(len(names), 4))
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- distribution

def distribution_figure(
    values_by_method: Mapping[Any, Sequence[float]],
    metric: str,
    *,
    ylabel: Optional[str] = None,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    emphasize: Iterable[Any] = (),
    labels: Optional[Mapping[Any, str]] = None,
    show_points: bool = True,
    caption: Optional[str] = None,
    style: Optional[PlotStyle] = None,
    series_key: str = "method",
    ylim: Optional[Sequence[float]] = None,
    into: Optional[Any] = None,
) -> Figure:
    """Box-and-whisker of per-instance values per method (genuine quartiles, not mean ± std), with the points
    jittered alongside so n and outliers are visible."""
    import random

    st_ = plotstyle.resolve(style)
    with matplotlib.rc_context(ieee_rc(style)):
        fig, ax = _target(into, width, height)
        names = st_.ordered(series_key, [m for m, v in values_by_method.items() if len(v)])
        styles = style_map([str(n) for n in names], [str(e) for e in emphasize], style=style)
        data = [list(values_by_method[m]) for m in names]
        bp = ax.boxplot(data, positions=range(1, len(names) + 1), widths=0.55, patch_artist=True, showfliers=not show_points,
                        medianprops=dict(color="black", linewidth=1.0), whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
        rng = random.Random(0)
        for i, (m, box) in enumerate(zip(names, bp["boxes"]), start=1):
            st = styles[str(m)]
            box.set(facecolor=st["color"], alpha=0.35, edgecolor=st["color"], linewidth=st["linewidth"])
            if show_points:
                ax.plot([i + rng.uniform(-0.18, 0.18) for _ in data[i - 1]], data[i - 1], linestyle="none", marker="o",
                        markersize=2.2, color=st["color"], alpha=0.7, zorder=3)
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels([(labels or {}).get(m, str(m)) for m in names])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        set_axis_labels(ax, None, ylabel or metric, style=style)
        _apply_limits(ax, None, ylim)
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- ablation

def ablation_figure(
    rows: Sequence[agg.AblationRow],
    metric: str,
    *,
    higher_is_better: bool = True,
    xlabel: Optional[str] = None,
    fmt: str = ".2f",
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    sort: bool = True,
    labels: Optional[Mapping[str, str]] = None,
    annotate: bool = True,
    caption: Optional[str] = None,
    style: Optional[PlotStyle] = None,
    xlim: Optional[Sequence[float]] = None,
    into: Optional[Any] = None,
) -> Figure:
    """Horizontal bars of (variant − full model). Blue = improves the metric, red = hurts; thin black edges.
    Bars are ranked by effect size unless the style declares an order for `variant`."""
    st_ = plotstyle.resolve(style)
    variants = [r for r in rows if not r.is_base and r.delta.get(metric) is not None]
    if sort:
        variants.sort(key=lambda r: r.delta[metric] * (1 if higher_is_better else -1))
    if st_.order.get(VARIANT_KEY):
        ranked = st_.ordered(VARIANT_KEY, [r.label for r in variants])
        variants.sort(key=lambda r: ranked.index(r.label))
    with matplotlib.rc_context(ieee_rc(style)):
        h = height if height is not None else max(1.2, 0.28 * len(variants) + 0.6)
        fig, ax = _target(into, width, h)
        names = [(labels or {}).get(r.label, r.label) for r in variants]
        deltas = [r.delta[metric] for r in variants]
        errs = [(r.stats[metric].std if r.stats.get(metric) else 0.0) for r in variants]
        improves = [(d > 0) if higher_is_better else (d < 0) for d in deltas]
        colors = [IMPROVES if ok else (HURTS if d != 0 else NEUTRAL) for ok, d in zip(improves, deltas)]
        y = list(range(len(variants)))
        ax.barh(y, deltas, xerr=errs, color=colors, height=0.6, edgecolor="black", linewidth=0.6, alpha=0.9,
                error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, ecolor="black"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.tick_params(axis="y", which="both", length=0)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        set_axis_labels(ax, xlabel or f"$\\Delta$ {metric} vs. full model", style=style)
        if annotate:
            span = max((abs(d) + e for d, e in zip(deltas, errs)), default=1.0) or 1.0
            for yi, d, e in zip(y, deltas, errs):
                off = (e + 0.03 * span) * (1 if d >= 0 else -1)
                ax.annotate(format(d, f"+{fmt}"), (d + off, yi), va="center", ha="left" if d >= 0 else "right",
                            fontsize=st_.annotation)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo - 0.15 * span, hi + 0.15 * span)
        _apply_limits(ax, xlim, None)
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- comparison

def comparison_figure(
    pt: agg.PivotTable,
    metric: str,
    *,
    ylabel: Optional[str] = None,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    row_labels: Optional[Mapping[Any, str]] = None,
    col_labels: Optional[Mapping[Any, str]] = None,
    emphasize: Iterable[Any] = (),
    hatch: bool = False,
    legend_loc: str = "above",
    zero_based: bool = False,
    ylim: Optional[Sequence[float]] = None,
    caption: Optional[str] = None,
    style: Optional[PlotStyle] = None,
    rows_key: str = "method",
    cols_key: Optional[str] = "dataset",
    into: Optional[Any] = None,
) -> Figure:
    """Grouped bars: x = column key (datasets), one bar per row entity (method), error bar = std.

    Method colours with thin black edges (hatch=True adds print-safe hatching). Missing cells are left empty,
    never drawn as 0.
    `legend_loc="above"` puts a framed one-row legend over the axes so it never covers a bar.
    y limits are data-tight by default (PSNR differences of a few dB are invisible from 0); pass
    `zero_based=True` or `ylim` to override. Say which in the caption."""
    st_ = plotstyle.resolve(style)
    with matplotlib.rc_context(ieee_rc(style)):
        fig, ax = _target(into, width, height)
        rows = st_.ordered(rows_key, pt.rows)
        cols = st_.ordered(cols_key, pt.cols)
        styles = style_map(rows, emphasize, style=style)
        n = len(rows)
        group_w = 0.8
        bw = group_w / n
        xs = list(range(len(cols)))
        los: list[float] = []
        his: list[float] = []
        for i, r in enumerate(rows):
            st = styles[r]
            cells = [(x, pt.stat(r, c, metric)) for x, c in zip(xs, cols)]
            cells = [(x, cell) for x, cell in cells if cell is not None]  # missing cells: no bar
            if not cells:
                continue
            offs = [x - group_w / 2 + bw * (i + 0.5) for x, _ in cells]
            means = [cell.mean for _, cell in cells]
            errs = [cell.std for _, cell in cells]
            los += [m - e for m, e in zip(means, errs)]
            his += [m + e for m, e in zip(means, errs)]
            ax.bar(offs, means, width=bw * 0.92, yerr=errs if any(errs) else None,
                   color=st["fill"], hatch=st["hatch"] if hatch else "", edgecolor="black", linewidth=0.6,
                   alpha=0.9, label=(row_labels or {}).get(r, str(r)),
                   error_kw=dict(elinewidth=0.8, capsize=2, capthick=0.8, ecolor="black"))
        ax.set_xticks(xs)
        ax.set_xticklabels([(col_labels or {}).get(c, "" if c is None else str(c)) for c in cols])
        ax.tick_params(axis="x", which="both", length=0)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(-0.5, len(cols) - 0.5)
        set_axis_labels(ax, None, ylabel or metric, style=style)
        if ylim is not None:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        else:
            _tight_ylim(ax, los, his, zero_based)
        if legend_loc in ("above", "top"):
            top_legend(ax, ncol=n if n <= 4 else (n + 1) // 2)
        else:
            _thin_legend(ax.legend(loc=legend_loc, ncol=min(n, 3)))
        if caption:
            panel_label(ax, caption, style=style)
        return fig


# --------------------------------------------------------------------------- panels

def panel_figure(
    panels: Sequence[tuple[Any, Optional[str]]],
    *,
    width: Union[str, float] = "double",
    height: Optional[float] = None,
    ncols: Optional[int] = None,
    style: Optional[PlotStyle] = None,
    letters: bool = True,
) -> Figure:
    """Several plots as one figure, each with its bold `(a)` caption underneath.

    `panels` pairs a drawing callable with an optional caption: `(lambda ax: sweep_figure(..., into=ax), "PSNR vs lambda")`.
    An IEEE figure is usually several panels under one number and one caption, and pasting separately exported
    PDFs together in LaTeX is where panel sizes and font sizes stop matching. Composed here, every panel is drawn
    by the same code, at the same size, in the project's style.
    """
    if not panels:
        raise ValueError("a panel figure needs at least one panel")
    n = len(panels)
    cols = ncols or (n if n <= 3 else (n + 1) // 2)
    rows = math.ceil(n / cols)
    with matplotlib.rc_context(ieee_rc(style)):
        w = width_in(width)
        h = height if height is not None else rows * (w / cols) * 0.62 + 0.35 * rows  # room for the (a) captions
        fig = Figure(figsize=(w, h))
        axes = fig.subplots(rows, cols, squeeze=False)
        for i, (draw, caption) in enumerate(panels):
            ax = axes[i // cols][i % cols]
            draw(ax)
            if letters or caption:
                text = f"{chr(ord('a') + i)}. {caption}" if caption else f"{chr(ord('a') + i)}."
                panel_label(ax, text, style=style)
        for j in range(n, rows * cols):  # an incomplete last row leaves empty axes: drop them
            fig.delaxes(axes[j // cols][j % cols])
        fig.tight_layout(w_pad=2.0, h_pad=3.0)
        return fig


# --------------------------------------------------------------------------- output

# rcParams only apply while the figure is *built* (inside rc_context); saving happens later, so the
# tight bounding box must be passed explicitly or long tick labels get clipped.
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.02)

def save_figure(fig: Figure, path: Union[str, Path], dpi: int = 300, also_png: bool = False) -> list[Path]:
    """Save as the suffix says (.pdf vector by default). Optionally a 300-dpi PNG next to it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, **SAVE_KW)
    out = [p]
    if also_png and p.suffix.lower() != ".png":
        q = p.with_suffix(".png")
        fig.savefig(q, dpi=dpi, **SAVE_KW)
        out.append(q)
    return out


def figure_bytes(fig: Figure, fmt: str = "pdf", dpi: int = 300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, **SAVE_KW)
    return buf.getvalue()


# --------------------------------------------------------------------------- LaTeX glue

def figure_tex(
    graphic: Union[str, Path],
    caption: str = "TODO",
    label: Optional[str] = None,
    width: Union[str, float] = "single",
    position: str = "!t",
) -> str:
    """`figure` (single column, \\columnwidth) or `figure*` (double column, \\textwidth) snippet."""
    single = width in ("single", "ieee-single") or (isinstance(width, (int, float)) and float(width) <= 5.0)
    env = "figure" if single else "figure*"
    w = r"\columnwidth" if single else r"\textwidth"
    name = Path(graphic).with_suffix("").as_posix()
    lines = [f"\\begin{{{env}}}[{position}]", "\\centering", f"\\includegraphics[width={w}]{{{name}}}", f"\\caption{{{caption}}}"]
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\end{{{env}}}")
    return "\n".join(lines) + "\n"


def ieee_preamble() -> str:
    """Packages the generated tables and figures rely on."""
    return "\n".join([
        r"\usepackage{booktabs}   % \toprule, \midrule, \bottomrule, \cmidrule",
        r"\usepackage{amssymb}    % \checkmark in ablation tables",
        r"\usepackage{graphicx}   % \includegraphics",
    ]) + "\n"


def to_grayscale_png(png_bytes: bytes) -> bytes:
    """Convert a PNG to grayscale, for checking that series stay distinguishable in print."""
    from PIL import Image

    im = Image.open(io.BytesIO(png_bytes)).convert("L")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()
