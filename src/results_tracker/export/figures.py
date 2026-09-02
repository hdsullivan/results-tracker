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
from matplotlib.figure import Figure

from .. import aggregate as agg

SINGLE_COL_IN = 3.5
DOUBLE_COL_IN = 7.16

# Same validated categorical order as the GUI (dataviz reference palette).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCHES = ["", "//", "\\\\", "xx", "..", "++"]
IMPROVES, HURTS, NEUTRAL = "#2a78d6", "#e34948", "#9a9a9a"

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
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.0,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
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
            hatch=HATCHES[i % len(HATCHES)],
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
    legend_loc: str = "best",
) -> Figure:
    """Metric vs swept parameter, mean line with ± std band (or error bars), best value marked."""
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
            if band and any(v > 0 for v in sd):
                ax.fill_between(xs, [m - s for m, s in zip(ys, sd)], [m + s for m, s in zip(ys, sd)],
                                color=st["color"], alpha=0.18, linewidth=0)
                ax.plot(xs, ys, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        linewidth=st["linewidth"], zorder=st["zorder"], label=name)
            else:
                ax.errorbar(xs, ys, yerr=sd, color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                            linewidth=st["linewidth"], zorder=st["zorder"], label=name, capsize=1.5, elinewidth=0.6)
            best = (best_by_group or {}).get(g)
            if mark_best and best is not None and best in dict(series):
                bx = best if numeric else str(best)
                ax.plot([bx], [dict(series)[best].mean], marker=st["marker"], markersize=6, markerfacecolor="white",
                        markeredgecolor=st["color"], markeredgewidth=1.0, linestyle="none", zorder=4)
                if len(groups) == 1:
                    ax.axvline(bx, color=st["color"], linestyle=":", linewidth=0.6, alpha=0.7)
        if log_x:
            ax.set_xscale("log")
            ax.set_xticks(xs_all)
            ax.set_xticklabels([f"{x:g}" for x in xs_all])
            ax.minorticks_off()
        ax.set_xlabel(xlabel or param)
        ax.set_ylabel(ylabel or metric)
        if len(groups) > 1:
            ax.legend(loc=legend_loc)
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
    """Horizontal bars of (variant − full model). Blue = improves, red (hatched) = hurts."""
    variants = [r for r in rows if not r.is_base and r.delta.get(metric) is not None]
    if sort:
        variants.sort(key=lambda r: r.delta[metric] * (1 if higher_is_better else -1))
    with matplotlib.rc_context(IEEE_RC):
        h = height if height is not None else max(1.2, 0.28 * len(variants) + 0.6)
        fig = _new_figure(width, h)
        ax = fig.add_subplot(111)
        ax.grid(axis="y", visible=False)
        names = [(labels or {}).get(r.label, r.label) for r in variants]
        deltas = [r.delta[metric] for r in variants]
        errs = [(r.stats[metric].std if r.stats.get(metric) else 0.0) for r in variants]
        improves = [(d > 0) if higher_is_better else (d < 0) for d in deltas]
        colors = [IMPROVES if ok else (HURTS if d != 0 else NEUTRAL) for ok, d in zip(improves, deltas)]
        y = list(range(len(variants)))
        bars = ax.barh(y, deltas, xerr=errs, color=colors, height=0.6, error_kw=dict(elinewidth=0.6, capsize=1.5),
                       edgecolor="none")
        for b, ok in zip(bars, improves):
            if not ok:
                b.set_hatch("//")
                b.set_edgecolor("white")
                b.set_linewidth(0)
        ax.axvline(0, color="#555555", linewidth=0.6)
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
    legend_loc: str = "best",
) -> Figure:
    """Grouped bars: x = column key (datasets), one bar per row entity (method), error bar = std."""
    with matplotlib.rc_context(IEEE_RC):
        fig = _new_figure(width, height)
        ax = fig.add_subplot(111)
        ax.grid(axis="x", visible=False)
        rows, cols = pt.rows, pt.cols
        styles = style_map(rows, emphasize)
        n = len(rows)
        group_w = 0.8
        bw = group_w / n
        xs = list(range(len(cols)))
        for i, r in enumerate(rows):
            means = [pt.stat(r, c, metric).mean if pt.stat(r, c, metric) else 0.0 for c in cols]
            errs = [pt.stat(r, c, metric).std if pt.stat(r, c, metric) else 0.0 for c in cols]
            offs = [x - group_w / 2 + bw * (i + 0.5) for x in xs]
            st = styles[r]
            ax.bar(offs, means, width=bw * 0.92, yerr=errs, color=st["color"], hatch=st["hatch"] if hatch else "",
                   edgecolor="white", linewidth=0.3, label=(row_labels or {}).get(r, str(r)),
                   error_kw=dict(elinewidth=0.6, capsize=1.5, ecolor="#333333"))
        ax.set_xticks(xs)
        ax.set_xticklabels([(col_labels or {}).get(c, "" if c is None else str(c)) for c in cols])
        ax.set_ylabel(ylabel or metric)
        ax.legend(loc=legend_loc, ncol=min(n, 3))
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
