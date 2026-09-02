"""IEEE-sized matplotlib figures (vector PDF by default).

- Widths: single column 3.5 in, double column 7.16 in.
- Serif fonts at 8 pt, TrueType embedded (pdf.fonttype 42).
- Methods keep a fixed colour + line style + marker so they survive grayscale.
- Deterministic: no randomness, no pyplot global state (Figure objects only).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import matplotlib
import matplotlib.ticker
from matplotlib.figure import Figure

from .. import aggregate as agg

SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.16

# Okabe-Ito: colourblind-safe and the de-facto scientific palette. Order: blue, vermillion, green, purple,
# orange, sky blue, black. The GUI keeps its own (brighter) palette; print figures use this one.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Bar fills for print: white -> black ramp with black edges, plus light hatching. Reads in grayscale by construction.
BAR_FILLS = ["#ffffff", "#c8c8c8", "#8c8c8c", "#505050", "#000000", "#e6e6e6"]
BAR_HATCHES = ["", "////", "\\\\\\\\", "xxxx", "....", "++++"]
HATCHES = BAR_HATCHES
IMPROVES, HURTS, NEUTRAL = "#ffffff", "#8c8c8c", "#c8c8c8"

# Classic IEEE look: Times at 8 pt, closed black axes box, inward ticks on all four sides with minor ticks,
# no grid, framed legend with square corners, thin lines, small white-faced markers, error bars with caps.
IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "black",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "lines.linewidth": 0.9,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.8,
    "errorbar.capsize": 1.5,
    "hatch.linewidth": 0.4,
    "legend.frameon": True,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.borderpad": 0.4,
    "legend.handlelength": 2.2,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.0,
    "patch.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 150,
}


def width_in(width: Union[str, float]) -> float:
    if isinstance(width, (int, float)):
        return float(width)
    return {"single": SINGLE_COL_IN, "double": DOUBLE_COL_IN}[width]


def style_map(names: Sequence[Any], emphasize: Iterable[Any] = ()) -> dict[Any, dict[str, Any]]:
    """Fixed colour/linestyle/marker per entity in first-seen order; emphasised ones are a bit thicker."""
    emph = set(emphasize)
    out: dict[Any, dict[str, Any]] = {}
    for n in names:
        if n in out:
            continue
        i = len(out)
        out[n] = dict(
            color=PALETTE[i % len(PALETTE)],
            linestyle=LINESTYLES[i % len(LINESTYLES)],
            marker=MARKERS[i % len(MARKERS)],
            fill=BAR_FILLS[i % len(BAR_FILLS)],
            hatch=BAR_HATCHES[i % len(BAR_HATCHES)],
            linewidth=1.4 if n in emph else 1.0,
            zorder=3 if n in emph else 2,
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
        leg.get_frame().set_linewidth(0.5)


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
    band: bool = False,
    best_by_group: Optional[Mapping[tuple, Any]] = None,
    mark_best: bool = True,
    width: Union[str, float] = "single",
    height: Optional[float] = None,
    emphasize: Iterable[Any] = (),
    labels: Optional[Mapping[tuple, str]] = None,
    legend_loc: str = "best",
) -> Figure:
    """Metric vs swept parameter: mean line with ± std error bars (or a shaded band), best value as a filled marker."""
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
            mk = dict(marker=st["marker"], markerfacecolor="white", markeredgecolor=st["color"])
            if band and any(v > 0 for v in sd):
                ax.fill_between(xs, [m - s for m, s in zip(ys, sd)], [m + s for m, s in zip(ys, sd)],
                                color=st["color"], alpha=0.15, linewidth=0)
                ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                        zorder=st["zorder"], label=name, **mk)
            else:
                ax.errorbar(xs, ys, yerr=sd if any(v > 0 for v in sd) else None, color=st["color"],
                            linestyle=st["linestyle"], linewidth=st["linewidth"], zorder=st["zorder"], label=name,
                            elinewidth=0.5, capthick=0.5, ecolor=st["color"], **mk)
            best = (best_by_group or {}).get(g)
            if mark_best and best is not None and best in dict(series):
                bx = best if numeric else str(best)
                ax.plot([bx], [dict(series)[best].mean], marker=st["marker"], markersize=4.5, markerfacecolor=st["color"],
                        markeredgecolor=st["color"], linestyle="none", zorder=5)
                if len(groups) == 1:
                    ax.axvline(bx, color="black", linestyle=":", linewidth=0.5)
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(xs_all)
            ax.set_xticklabels([f"{x:g}" for x in xs_all])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(xlabel or param)
        ax.set_ylabel(ylabel or metric)
        if len(groups) > 1:
            _thin_legend(ax.legend(loc=legend_loc))
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
) -> Figure:
    """Horizontal bars of (variant − full model). White = improves the metric, gray hatched = hurts; black edges."""
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
        bars = ax.barh(y, deltas, xerr=errs, color=colors, height=0.6, edgecolor="black", linewidth=0.5,
                       error_kw=dict(elinewidth=0.5, capsize=1.5, capthick=0.5, ecolor="black"))
        for b, ok in zip(bars, improves):
            if not ok:
                b.set_hatch("////")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.tick_params(axis="y", which="both", length=0)
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel or f"$\\Delta$ {metric} vs. full model")
        if annotate:
            span = max((abs(d) + e for d, e in zip(deltas, errs)), default=1.0) or 1.0
            for yi, d, e in zip(y, deltas, errs):
                off = (e + 0.03 * span) * (1 if d >= 0 else -1)
                ax.annotate(format(d, f"+{fmt}"), (d + off, yi), va="center", ha="left" if d >= 0 else "right", fontsize=6.5)
            lo, hi = ax.get_xlim()
            ax.set_xlim(lo - 0.15 * span, hi + 0.15 * span)
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
    hatch: bool = True,
    legend_loc: str = "above",
    zero_based: bool = False,
    ylim: Optional[tuple[float, float]] = None,
) -> Figure:
    """Grouped bars: x = column key (datasets), one bar per row entity (method), error bar = std.

    Fills run white -> black with black edges (print-safe). Missing cells are left empty, never drawn as 0.
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
                   color=st["fill"], hatch=st["hatch"] if hatch else "", edgecolor="black", linewidth=0.5,
                   label=(row_labels or {}).get(r, str(r)),
                   error_kw=dict(elinewidth=0.5, capsize=1.5, capthick=0.5, ecolor="black"))
        ax.set_xticks(xs)
        ax.set_xticklabels([(col_labels or {}).get(c, "" if c is None else str(c)) for c in cols])
        ax.tick_params(axis="x", which="both", length=0)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylabel(ylabel or metric)
        if ylim is not None:
            ax.set_ylim(*ylim)
        else:
            _tight_ylim(ax, los, his, zero_based)
        if legend_loc == "above":
            ncol = n if n <= 3 else (n + 1) // 2  # two rows once there are four or more methods
            _thin_legend(ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=ncol, borderaxespad=0.2))
        else:
            _thin_legend(ax.legend(loc=legend_loc, ncol=min(n, 3)))
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
    single = width == "single" or (isinstance(width, (int, float)) and float(width) <= 4.0)
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
