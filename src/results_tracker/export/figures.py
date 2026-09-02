"""Paper figures in the lab's IEEE style (ported from adaptivePnP `ablation_utils.set_paper_style`).

- Times-like serif text: 8 pt base, 11 pt axis labels, 9 pt ticks, 11 pt legend, 10.5 pt bold panel captions.
- Full boxed axes frame (0.8 pt) with inward major + minor ticks on all four sides, no grid.
- Bordered legend (black edge, square corners) placed across the top of the figure.
- Solid tab10-style colours with filled circle markers; the proposed method gets a heavier line and
  larger markers; uncertainty as a shaded band (alpha 0.15), error bars on request.
- Bold "(a) ..." captions below each panel via `panel_label`.
- Figure widths 5.0 in (column) / 10.5 in (page): sized for comfortable review and scaled by LaTeX to
  \columnwidth / \textwidth; pass a number of inches for exact IEEE widths (3.5 / 7.16).
- Deterministic: Figure objects only, no pyplot global state; TrueType fonts embedded.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.ticker
from matplotlib.figure import Figure

from .. import aggregate as agg

# Lab convention (ablation_utils.IEEE_COLUMN_WIDTH / IEEE_PAGE_WIDTH): wider than the literal IEEE
# column so point sizes read as body text once LaTeX scales the figure down.
SINGLE_COL_IN = 5.0
DOUBLE_COL_IN = 10.5
IEEE_SINGLE_COL_IN = 3.5   # literal IEEEtran \columnwidth
IEEE_DOUBLE_COL_IN = 7.16  # literal IEEEtran \textwidth
WIDTHS = {"single": SINGLE_COL_IN, "double": DOUBLE_COL_IN, "ieee-single": IEEE_SINGLE_COL_IN, "ieee-double": IEEE_DOUBLE_COL_IN}

LINE_WIDTH = 1.3
MARKER_SIZE = 3.5
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 11
PANEL_LABEL_SIZE = 10.5
# STIX mathtext renders visibly smaller than serif body text, so a label that is *entirely* math
# (e.g. r"$\lambda$") gets this size instead of AXIS_LABEL_SIZE (lab rule: AXIS_LABEL_SIZE_MATH).
AXIS_LABEL_SIZE_MATH = 14

# Fixed hue order used across the lab's ablation figures (tab10 subset): blue, red, green, purple, orange,
# brown, gray, pink. Assigned in first-seen order, never re-ranked.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f", "#e377c2"]
LINESTYLES = ["-"] * 8          # the lab keeps solid lines; identity is colour + emphasis + marker
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
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


def style_map(names: Sequence[Any], emphasize: Iterable[Any] = ()) -> dict[Any, dict[str, Any]]:
    """Fixed colour/linestyle/marker per entity in first-seen order; emphasised ones are a bit thicker."""
    emph = set(emphasize)
    out: dict[Any, dict[str, Any]] = {}
    for n in names:
        if n in out:
            continue
        i = len(out)
        primary = n in emph
        out[n] = dict(
            color=PALETTE[i % len(PALETTE)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker="o",  # the lab uses filled circles everywhere; identity is carried by colour + emphasis
            fill=BAR_FILLS[i % len(BAR_FILLS)],
            hatch=BAR_HATCHES[i % len(BAR_HATCHES)],
            linewidth=LINE_WIDTH + 0.2 if primary else LINE_WIDTH - 0.1,
            markersize=4.5 if primary else MARKER_SIZE,
            zorder=3 if primary else 2,
        )
    return out


def _label(g: tuple, fallback: str) -> str:
    return " / ".join(map(str, g)) if g else fallback


def _new_figure(width: Union[str, float], height: Optional[float]) -> Figure:
    w = width_in(width)
    h = height if height is not None else w * 0.62
    return Figure(figsize=(w, h))


def _thin_legend(leg) -> None:
    if leg is not None:
        leg.get_frame().set_linewidth(0.8)


def panel_label(ax, text: str) -> None:
    """Bold IEEE-style subfigure caption below the panel, e.g. "a. PSNR" or "(a) PSNR" -> "(a) PSNR".

    Offset in points so it clears the x-axis label regardless of the panel height (lab convention)."""
    import re

    m = re.match(r"^\(?([A-Za-z])[.)]\s*(.*)$", text)
    if m:
        letter, rest = m.group(1).lower(), m.group(2)
        text = f"({letter}) {rest}".strip()
    ax.annotate(text, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -32), textcoords="offset points",
                ha="center", va="top", fontsize=PANEL_LABEL_SIZE, fontweight="bold", annotation_clip=False)


def set_axis_labels(ax, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    """Axis labels with the lab's math-only bump: '$\\lambda$' -> 14 pt, 'PSNR (dB)' -> 11 pt."""

    def size(t: str) -> float:
        t = t.strip()
        return AXIS_LABEL_SIZE_MATH if (t.startswith("$") and t.endswith("$") and t.count("$") == 2) else AXIS_LABEL_SIZE

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=size(xlabel))
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=size(ylabel))


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
) -> Figure:
    """Metric vs swept parameter: mean line with a shaded ± std band (or error bars), best value ringed.

    `legend_loc="top"` puts the bordered legend above the axes (lab convention); any matplotlib loc works too.
    `caption` adds a bold "(a) ..." label under the panel."""
    with matplotlib.rc_context(IEEE_RC):
        fig = _new_figure(width, height)
        ax = fig.add_subplot(111)
        groups = [g for g, s in series_by_group.items() if s]
        styles = style_map([_label(g, metric) for g in groups], emphasize)
        xs_all = sorted({x for g in groups for x, _ in series_by_group[g]}, key=lambda x: (isinstance(x, str), x))
        numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs_all)
        if log_x is None:
            log_x = numeric and len(xs_all) >= 3 and min(xs_all) > 0 and max(xs_all) / min(xs_all) >= 10
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
        set_axis_labels(ax, xlabel or param, ylabel or metric)
        if len(groups) > 1:
            if legend_loc == "top":
                top_legend(ax)
            else:
                _thin_legend(ax.legend(loc=legend_loc))
        if caption:
            panel_label(ax, caption)
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
) -> Figure:
    """Horizontal bars of (variant − full model). Blue = improves the metric, red = hurts; thin black edges."""
    variants = [r for r in rows if not r.is_base and r.delta.get(metric) is not None]
    if sort:
        variants.sort(key=lambda r: r.delta[metric] * (1 if higher_is_better else -1))
    with matplotlib.rc_context(IEEE_RC):
        h = height if height is not None else max(1.2, 0.28 * len(variants) + 0.6)
        fig = _new_figure(width, h)
        ax = fig.add_subplot(111)
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
        set_axis_labels(ax, xlabel or f"$\\Delta$ {metric} vs. full model")
        if annotate:
            span = max((abs(d) + e for d, e in zip(deltas, errs)), default=1.0) or 1.0
            for yi, d, e in zip(y, deltas, errs):
                off = (e + 0.03 * span) * (1 if d >= 0 else -1)
                ax.annotate(format(d, f"+{fmt}"), (d + off, yi), va="center", ha="left" if d >= 0 else "right",
                            fontsize=TICK_LABEL_SIZE)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo - 0.15 * span, hi + 0.15 * span)
        if caption:
            panel_label(ax, caption)
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
    ylim: Optional[tuple[float, float]] = None,
    caption: Optional[str] = None,
) -> Figure:
    """Grouped bars: x = column key (datasets), one bar per row entity (method), error bar = std.

    Method colours with thin black edges (hatch=True adds print-safe hatching). Missing cells are left empty,
    never drawn as 0.
    `legend_loc="above"` puts a framed one-row legend over the axes so it never covers a bar.
    y limits are data-tight by default (PSNR differences of a few dB are invisible from 0); pass
    `zero_based=True` or `ylim` to override. Say which in the caption."""
    with matplotlib.rc_context(IEEE_RC):
        fig = _new_figure(width, height)
        ax = fig.add_subplot(111)
        rows, cols = pt.rows, pt.cols
        styles = style_map(rows, emphasize)
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
        set_axis_labels(ax, None, ylabel or metric)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            _tight_ylim(ax, los, his, zero_based)
        if legend_loc in ("above", "top"):
            top_legend(ax, ncol=n if n <= 4 else (n + 1) // 2)
        else:
            _thin_legend(ax.legend(loc=legend_loc, ncol=min(n, 3)))
        if caption:
            panel_label(ax, caption)
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
