"""Plotly charts for the GUI, styled like the lab's paper figures (see export/figures.py):

white paper background, Times-like serif text, a boxed axes frame with inward mirrored ticks and
minor ticks, no grid, a black-bordered legend in one row above the axes, tab10 colours with filled
circle markers, shaded ± std bands, and the chosen sweep value ringed with a dotted gray guide.
Hover tooltips stay on (the point of the on-screen version); the paper export is matplotlib.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import plotly.graph_objects as go

from ..aggregate import ComparisonTable, Stat

# Same fixed hue order as export/figures.PALETTE (tab10 subset): blue, red, green, purple, orange, brown, gray, pink.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f", "#e377c2"]
SEQUENTIAL = ["#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#08519c", "#08306b"]  # Blues
DIVERGING_POS, DIVERGING_NEG, NEUTRAL = "#1f77b4", "#d62728", "#7f7f7f"  # improves = blue, hurts = red
GUIDE = "#8c8c8c"  # 0.55 gray
FONT = "Times New Roman, Times, STIXGeneral, serif"
# On-screen sizes: the paper style's 8 / 11 / 9 / 11 pt scaled up ~1.35x so they read at browser distance.
SIZE_BASE, SIZE_LABEL, SIZE_TICK, SIZE_LEGEND = 12, 15, 12, 14


def color_for(entities: Sequence[Any]) -> dict[Any, str]:
    """Stable entity -> hue assignment in first-seen order (never re-ranked)."""
    out: dict[Any, str] = {}
    for e in entities:
        if e not in out:
            out[e] = PALETTE[len(out) % len(PALETTE)] if len(out) < len(PALETTE) else "#9a9a9a"
    return out


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _is_math(label: str) -> bool:
    t = label.strip()
    return t.startswith("$") and t.endswith("$") and t.count("$") == 2


def base_layout(fig: go.Figure, title: str = "", ytitle: str = "", xtitle: str = "", legend_top: bool = True) -> go.Figure:
    """Apply the paper look. Axis titles that are pure LaTeX math get a larger size (lab rule)."""
    fig.update_layout(
        template="simple_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family=FONT, size=SIZE_BASE, color="black"),
        margin=dict(l=70, r=25, t=60 if (title or legend_top) else 25, b=60),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
            bordercolor="black", borderwidth=1, bgcolor="white", font=dict(size=SIZE_LEGEND),
            itemsizing="constant", tracegroupgap=0,
        ),
        bargap=0.25,
        bargroupgap=0.06,
        hovermode="closest",
        hoverlabel=dict(font=dict(family=FONT, size=SIZE_BASE)),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=SIZE_LABEL)))
    axis = dict(
        showline=True, linecolor="black", linewidth=1.2, mirror="ticks",
        ticks="inside", ticklen=6, tickwidth=1.1, tickcolor="black",
        showgrid=False, zeroline=False,
        tickfont=dict(size=SIZE_TICK, color="black"),
        minor=dict(ticks="inside", ticklen=3, tickwidth=0.8, tickcolor="black", showgrid=False),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if xtitle:
        fig.update_xaxes(title=dict(text=xtitle, font=dict(size=SIZE_LABEL + (3 if _is_math(xtitle) else 0), color="black")))
    if ytitle:
        fig.update_yaxes(title=dict(text=ytitle, font=dict(size=SIZE_LABEL + (3 if _is_math(ytitle) else 0), color="black")))
    return fig


# --------------------------------------------------------------------------- comparison bars

def comparison_bars(ct: ComparisonTable, metric: str, fmt: str = ".2f", unit: str = "", zero_based: bool = False) -> go.Figure:
    """Mean ± std per row. Method colours with black edges; missing cells get no bar; data-tight y range."""
    fig = go.Figure()
    ylabel = f"{metric} ({unit})" if unit else metric
    los: list[float] = []
    his: list[float] = []

    def cell(row) -> Optional[Stat]:
        return ct.cells[row].get(metric)

    if len(ct.group_by) == 1:
        rows = [r for r in ct.rows if cell(r) is not None]
        colors = color_for([r[0] for r in rows])
        for r in rows:
            c = cell(r)
            los.append(c.mean - c.std)
            his.append(c.mean + c.std)
            fig.add_bar(
                x=[ct.row_label(r)], y=[c.mean], name=ct.row_label(r), showlegend=False,
                error_y=dict(type="data", array=[c.std], visible=c.std > 0, color="black", thickness=1.2, width=4),
                marker=dict(color=colors[r[0]], line=dict(color="black", width=1.2), opacity=0.9),
                customdata=[[c.n]],
                hovertemplate=f"%{{x}}<br>{metric}: %{{y:{fmt}}} ± {format(c.std, fmt)}<br>n=%{{customdata[0]}}<extra></extra>",
            )
        base_layout(fig, ytitle=ylabel, xtitle=ct.group_by[0], legend_top=False)
    else:
        entities = [r[0] for r in ct.rows]
        colors = color_for(entities)
        for ent in colors:
            rows = [r for r in ct.rows if r[0] == ent and cell(r) is not None]
            if not rows:
                continue
            stats = [cell(r) for r in rows]
            los += [c.mean - c.std for c in stats]
            his += [c.mean + c.std for c in stats]
            fig.add_bar(
                name=str(ent),
                x=[" / ".join(map(str, r[1:])) for r in rows],
                y=[c.mean for c in stats],
                error_y=dict(type="data", array=[c.std for c in stats], visible=True, color="black", thickness=1.2, width=4),
                marker=dict(color=colors[ent], line=dict(color="black", width=1.2), opacity=0.9),
                customdata=[[c.n] for c in stats],
                hovertemplate=f"{ent}<br>%{{x}}<br>{metric}: %{{y:{fmt}}} ± %{{error_y.array:{fmt}}}<br>n=%{{customdata[0]}}<extra></extra>",
            )
        base_layout(fig, ytitle=ylabel, xtitle=" / ".join(ct.group_by[1:]))
    if los:
        lo, hi = min(los), max(his)
        span = max(hi - lo, 0.05 * abs(hi) if hi else 1.0)
        fig.update_yaxes(range=[0, hi + 0.12 * span] if (zero_based and lo >= 0) else [lo - 0.15 * span, hi + 0.25 * span])
    fig.update_xaxes(ticks="", minor=dict(ticks=""))  # categorical axis: no tick marks
    return fig


# --------------------------------------------------------------------------- sweeps

def is_log_friendly(xs: Sequence[Any]) -> bool:
    nums = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool)]
    if len(nums) != len(xs) or len(nums) < 3 or min(nums) <= 0:
        return False
    return max(nums) / min(nums) >= 10


def sweep_lines(
    series_by_group: dict[tuple, list[tuple[Any, Any]]],
    param: str,
    metric: str,
    fmt: str = ".2f",
    unit: str = "",
    log_x: bool = False,
    band: bool = True,
    best_by_group: Optional[dict[tuple, Any]] = None,
    emphasize: Sequence[str] = (),
) -> go.Figure:
    """One line per group: filled circle markers, shaded ± std band (or error bars), best value ringed."""
    fig = go.Figure()
    groups = list(series_by_group)
    colors = color_for(groups)
    single = len(groups) == 1
    xs_all = sorted({x for s in series_by_group.values() for x, _ in s}, key=lambda x: (isinstance(x, str), x))
    numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in xs_all)  # else a categorical axis
    for g in groups:
        series = series_by_group[g]
        if not series:
            continue
        name = " / ".join(map(str, g)) if g else metric
        xs = [x if numeric else str(x) for x, _ in series]
        ys = [s.mean for _, s in series]
        sd = [s.std for _, s in series]
        ns = [s.n for _, s in series]
        c = colors[g]
        primary = name in emphasize
        lw, ms = (3.0, 9) if primary else (2.2, 7)
        if band and any(v > 0 for v in sd):
            fig.add_scatter(x=xs, y=[m + s for m, s in zip(ys, sd)], mode="lines", line=dict(width=0),
                            hoverinfo="skip", showlegend=False, legendgroup=name)
            fig.add_scatter(x=xs, y=[m - s for m, s in zip(ys, sd)], mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_rgba(c, 0.15), hoverinfo="skip", showlegend=False, legendgroup=name)
        fig.add_scatter(
            x=xs, y=ys, mode="lines+markers", name=name, legendgroup=name, showlegend=not single,
            line=dict(color=c, width=lw),
            marker=dict(size=ms, color=c, line=dict(color=c, width=0)),
            error_y=None if band else dict(type="data", array=sd, visible=True, color=c, thickness=1.2, width=4),
            customdata=[[s, n] for s, n in zip(sd, ns)],
            hovertemplate=f"{name}<br>{param}=%{{x}}<br>{metric}: %{{y:{fmt}}} ± %{{customdata[0]:{fmt}}}<br>n=%{{customdata[1]}}<extra></extra>",
        )
        best = (best_by_group or {}).get(g)
        if best is not None and best in dict(series):
            bx = best if numeric else str(best)
            fig.add_scatter(
                x=[bx], y=[dict(series)[best].mean], mode="markers", showlegend=False, legendgroup=name,
                hoverinfo="skip", name=f"{name} best",
                marker=dict(symbol="circle-open", size=ms + 9, color=c, line=dict(color=c, width=2.2)),
            )
            if single:
                fig.add_vline(x=bx, line=dict(color=GUIDE, width=1.2, dash="dot"), layer="below")
                # annotations on a log axis take log10(x) in data coordinates
                ax_x = math.log10(best) if (log_x and numeric and best > 0) else bx
                fig.add_annotation(x=ax_x, y=1.0, yref="paper", text=f"best {param} = {best}", showarrow=False,
                                   yanchor="bottom", font=dict(size=SIZE_TICK, color="black"))
    ylabel = f"{metric} ({unit})" if unit else metric
    base_layout(fig, ytitle=ylabel, xtitle=param, legend_top=not single)
    fig.update_layout(hovermode="x unified")
    if not numeric:
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=[str(x) for x in xs_all])
    elif log_x:
        fig.update_xaxes(type="log", tickvals=xs_all, ticktext=[f"{x:g}" for x in xs_all], minor=dict(ticks=""))
    return fig


def best_marker_traces(fig: go.Figure) -> list:
    return [t for t in fig.data if getattr(t.marker, "symbol", None) == "circle-open"]


def sweep_heatmap(xs: Sequence[Any], ys: Sequence[Any], matrix: Sequence[Sequence[Optional[float]]],
                  param_x: str, param_y: str, metric: str, fmt: str = ".2f",
                  higher_is_better: bool = True, best: Optional[tuple[Any, Any]] = None) -> go.Figure:
    """Sequential single-hue (Blues) heatmap; darker = better. Cell values printed; best cell outlined."""
    scale = SEQUENTIAL if higher_is_better else list(reversed(SEQUENTIAL))
    text = [[("" if v is None else format(v, fmt)) for v in row] for row in matrix]
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=[str(x) for x in xs], y=[str(y) for y in ys], text=text, texttemplate="%{text}",
            textfont=dict(family=FONT, size=SIZE_TICK),
            colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
            xgap=2, ygap=2,
            colorbar=dict(title=dict(text=metric, font=dict(size=SIZE_TICK)), thickness=14, outlinecolor="black",
                          outlinewidth=1, ticks="inside", tickfont=dict(size=SIZE_TICK)),
            hovertemplate=f"{param_x}=%{{x}}<br>{param_y}=%{{y}}<br>{metric}: %{{z:{fmt}}}<extra></extra>",
        )
    )
    if best is not None:
        fig.add_scatter(x=[str(best[0])], y=[str(best[1])], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(symbol="square-open", size=40, line=dict(color="black", width=2.2)))
    base_layout(fig, xtitle=param_x, ytitle=param_y, legend_top=False)
    fig.update_xaxes(type="category", ticks="", minor=dict(ticks=""))
    fig.update_yaxes(type="category", ticks="", minor=dict(ticks=""))
    return fig


# --------------------------------------------------------------------------- ablations

def ablation_deltas(labels: Sequence[str], deltas: Sequence[Optional[float]], stds: Sequence[float],
                    metric: str, higher_is_better: bool = True, fmt: str = ".2f", unit: str = "") -> go.Figure:
    """Horizontal bars of (variant − full model). Blue = improves the metric, red = hurts; black edges."""
    fig = go.Figure()
    ys, xs, cols, errs = [], [], [], []
    for lab, d, sd in zip(labels, deltas, stds):
        if d is None:
            continue
        improves = (d > 0) if higher_is_better else (d < 0)
        ys.append(lab)
        xs.append(d)
        cols.append(DIVERGING_POS if improves else (DIVERGING_NEG if d != 0 else NEUTRAL))
        errs.append(sd)
    fig.add_bar(
        y=ys, x=xs, orientation="h", width=0.6,
        marker=dict(color=cols, line=dict(color="black", width=1.2), opacity=0.9),
        error_x=dict(type="data", array=errs, visible=True, color="black", thickness=1.2, width=4),
        hovertemplate=f"%{{y}}<br>Δ {metric}: %{{x:+{fmt}}}<extra></extra>", showlegend=False,
    )
    # value labels sit just beyond the error bar so they never collide with it
    span = max((abs(d) + e for d, e in zip(xs, errs)), default=1.0) or 1.0
    for lab, d, e in zip(ys, xs, errs):
        off = (e + 0.04 * span) * (1 if d >= 0 else -1)
        fig.add_annotation(x=d + off, y=lab, text=format(d, f"+{fmt}"), showarrow=False,
                           xanchor="left" if d >= 0 else "right", font=dict(size=SIZE_TICK, color="black"))
    if xs:
        lo = min(min(d - e for d, e in zip(xs, errs)), 0) - 0.18 * span
        hi = max(max(d + e for d, e in zip(xs, errs)), 0) + 0.18 * span
        fig.update_xaxes(range=[lo, hi])
    fig.add_vline(x=0, line=dict(color="black", width=1.2))
    xlabel = f"Δ {metric} vs full model" + (f" ({unit})" if unit else "")
    base_layout(fig, xtitle=xlabel, legend_top=False)
    fig.update_yaxes(autorange="reversed", ticks="", minor=dict(ticks=""))
    fig.update_layout(height=max(240, 48 * len(ys) + 110), margin=dict(l=20, r=70, t=25, b=60))
    return fig


# --------------------------------------------------------------------------- curves

def curves_lines(series_by_group: dict[tuple, Any], curve: str, ylabel: Optional[str] = None, band: bool = True,
                 log_y: bool = False, guide: Optional[float] = None, members: bool = False) -> go.Figure:
    """One mean line per group with a shaded ± std band (`curves.CurveStat`); `members` overlays the individual
    runs as thin lines; `guide` draws a dotted horizontal reference (1.0 for a ratio)."""
    fig = go.Figure()
    groups = [g for g, cs in series_by_group.items() if cs.mean]
    colors = color_for(groups)
    single = len(groups) == 1
    for g in groups:
        cs = series_by_group[g]
        name = " / ".join(map(str, g)) if g else curve
        c = colors[g]
        xs = cs.x
        if members:
            for run in cs.members:
                fig.add_scatter(x=list(range(len(run))), y=run, mode="lines", line=dict(color=_rgba(c, 0.25), width=0.8),
                                hoverinfo="skip", showlegend=False, legendgroup=name)
        if band and any(v > 0 for v in cs.std):
            fig.add_scatter(x=xs, y=[m + s for m, s in zip(cs.mean, cs.std)], mode="lines", line=dict(width=0),
                            hoverinfo="skip", showlegend=False, legendgroup=name)
            fig.add_scatter(x=xs, y=[m - s for m, s in zip(cs.mean, cs.std)], mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_rgba(c, 0.15), hoverinfo="skip", showlegend=False, legendgroup=name)
        fig.add_scatter(x=xs, y=cs.mean, mode="lines+markers" if len(xs) <= 25 else "lines", name=name, legendgroup=name,
                        showlegend=not single, line=dict(color=c, width=2.2), marker=dict(size=6, color=c),
                        customdata=[[s, n] for s, n in zip(cs.std, cs.n)],
                        hovertemplate=f"{name}<br>iteration %{{x}}<br>{curve}: %{{y:.4g}} ± %{{customdata[0]:.3g}}<br>n=%{{customdata[1]}}<extra></extra>")
    if guide is not None:
        fig.add_hline(y=guide, line=dict(color=GUIDE, width=1.2, dash="dot"), layer="below")
    base_layout(fig, ytitle=ylabel or curve, xtitle="iteration", legend_top=not single)
    fig.update_layout(hovermode="x unified")
    if log_y:
        fig.update_yaxes(type="log")
    return fig


# --------------------------------------------------------------------------- trade-off

def tradeoff_scatter(points_by_series: dict[Any, list], x_metric: str, y_metric: str, x_fmt: str = ".2f", y_fmt: str = ".2f",
                     xlabel: Optional[str] = None, ylabel: Optional[str] = None, log_x: bool = True,
                     hollow: Sequence[Any] = (), labels: Optional[dict] = None) -> go.Figure:
    """Two metrics against each other: one series per method, joined along the path key, std as error bars;
    `hollow` series (baselines, reported) get open markers and no line."""
    fig = go.Figure()
    names = list(points_by_series)
    colors = color_for(names)
    hollow_set = set(hollow)
    for name in names:
        pts = points_by_series[name]
        if not pts:
            continue
        c = colors[name]
        is_open = name in hollow_set
        fig.add_scatter(
            x=[p.x.mean for p in pts], y=[p.y.mean for p in pts], name=(labels or {}).get(name, str(name)),
            mode="markers" if (is_open or len(pts) == 1) else "lines+markers+text",
            text=[str(agg_fmt(p.label)) for p in pts] if len(pts) > 1 else None, textposition="top right",
            textfont=dict(size=SIZE_TICK - 1, color=c),
            line=dict(color=c, width=2.0),
            marker=dict(size=9, color="white" if is_open else c, line=dict(color=c, width=2.0), symbol="circle"),
            error_x=dict(type="data", array=[p.x.std for p in pts], visible=any(p.x.std > 0 for p in pts), color=c, thickness=1.0, width=3),
            error_y=dict(type="data", array=[p.y.std for p in pts], visible=any(p.y.std > 0 for p in pts), color=c, thickness=1.0, width=3),
            customdata=[[str(p.label), p.x.n] for p in pts],
            hovertemplate=f"{name} · %{{customdata[0]}}<br>{x_metric}: %{{x:{x_fmt}}}<br>{y_metric}: %{{y:{y_fmt}}}<br>n=%{{customdata[1]}}<extra></extra>",
        )
    base_layout(fig, ytitle=ylabel or y_metric, xtitle=xlabel or x_metric, legend_top=len(names) > 1)
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def agg_fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- distribution

def distribution_box(values_by_method: dict[Any, Sequence[float]], metric: str, ylabel: Optional[str] = None,
                     labels: Optional[dict] = None, points: bool = True) -> go.Figure:
    """Per-instance values per method as box plots with the points alongside."""
    fig = go.Figure()
    names = [m for m, v in values_by_method.items() if len(v)]
    colors = color_for(names)
    for m in names:
        fig.add_box(y=list(values_by_method[m]), name=(labels or {}).get(m, str(m)), marker=dict(color=colors[m], size=4),
                    line=dict(color=colors[m], width=1.5), fillcolor=_rgba(colors[m], 0.35),
                    boxpoints="all" if points else "outliers", jitter=0.35, pointpos=0, showlegend=False,
                    hovertemplate=f"%{{x}}<br>{metric}: %{{y:.4g}}<extra></extra>")
    base_layout(fig, ytitle=ylabel or metric, xtitle="", legend_top=False)
    fig.update_xaxes(ticks="", minor=dict(ticks=""))
    return fig
