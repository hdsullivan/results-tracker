"""Plotly helpers with a fixed, colorblind-validated categorical palette.

Hues are assigned in fixed order to entities (methods), never cycled or re-ranked.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import plotly.graph_objects as go

from ..aggregate import ComparisonTable, Stat

# Validated categorical order (see dataviz reference palette): blue, orange, aqua, yellow, magenta, green, violet, red
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRID = "rgba(128,128,128,0.18)"


def color_for(entities: Sequence[Any]) -> dict[Any, str]:
    """Stable entity -> hue assignment in first-seen order."""
    out: dict[Any, str] = {}
    for e in entities:
        if e not in out:
            out[e] = PALETTE[len(out) % len(PALETTE)] if len(out) < len(PALETTE) else "#9a9a9a"
    return out


def base_layout(fig: go.Figure, title: str = "", ytitle: str = "", xtitle: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40 if title else 20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        bargap=0.35,
        bargroupgap=0.08,
        hovermode="closest",
    )
    if title:
        fig.update_layout(title=title)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if xtitle:
        fig.update_xaxes(title=xtitle)
    if ytitle:
        fig.update_yaxes(title=ytitle)
    return fig


def comparison_bars(ct: ComparisonTable, metric: str, fmt: str = ".2f", unit: str = "") -> go.Figure:
    """Mean ± std per row. One group key -> single-hue bars; two keys -> x = second key, hue = first."""
    fig = go.Figure()
    ylabel = f"{metric} ({unit})" if unit else metric

    def cell(row) -> Optional[Stat]:
        return ct.cells[row].get(metric)

    if len(ct.group_by) == 1:
        rows = [r for r in ct.rows if cell(r) is not None]
        fig.add_bar(
            x=[ct.row_label(r) for r in rows],
            y=[cell(r).mean for r in rows],
            error_y=dict(type="data", array=[cell(r).std for r in rows], visible=True, thickness=1.2),
            marker_color=PALETTE[0],
            marker_line_width=0,
            width=0.55,
            customdata=[[cell(r).n] for r in rows],
            hovertemplate=f"%{{x}}<br>{metric}: %{{y:{fmt}}} ± %{{error_y.array:{fmt}}}<br>n=%{{customdata[0]}}<extra></extra>",
            showlegend=False,
        )
        return base_layout(fig, ytitle=ylabel, xtitle=ct.group_by[0])

    # two or more keys: hue = first key (entity), x = the rest joined
    entities = [r[0] for r in ct.rows]
    colors = color_for(entities)
    for ent in colors:
        rows = [r for r in ct.rows if r[0] == ent and cell(r) is not None]
        fig.add_bar(
            name=str(ent),
            x=[" / ".join(map(str, r[1:])) for r in rows],
            y=[cell(r).mean for r in rows],
            error_y=dict(type="data", array=[cell(r).std for r in rows], visible=True, thickness=1.2),
            marker_color=colors[ent],
            marker_line_width=0,
            customdata=[[cell(r).n] for r in rows],
            hovertemplate=f"{ent}<br>%{{x}}<br>{metric}: %{{y:{fmt}}} ± %{{error_y.array:{fmt}}}<br>n=%{{customdata[0]}}<extra></extra>",
        )
    return base_layout(fig, ytitle=ylabel, xtitle=" / ".join(ct.group_by[1:]))


# --------------------------------------------------------------------------- sweeps

SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # blue, light -> dark
DIVERGING_POS, DIVERGING_NEG, NEUTRAL = "#2a78d6", "#e34948", "#9a9a9a"  # blue improves, red hurts


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


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
) -> go.Figure:
    """One line per group, mean with a ± std band (or error bars), best x per group ringed."""
    fig = go.Figure()
    groups = list(series_by_group)
    colors = color_for(groups)
    single = len(groups) == 1
    for g in groups:
        series = series_by_group[g]
        if not series:
            continue
        name = " / ".join(map(str, g)) if g else metric
        xs = [x for x, _ in series]
        ys = [s.mean for _, s in series]
        sd = [s.std for _, s in series]
        ns = [s.n for _, s in series]
        c = colors[g]
        if band and any(v > 0 for v in sd):
            fig.add_scatter(x=xs, y=[m + s for m, s in zip(ys, sd)], mode="lines", line=dict(width=0),
                            hoverinfo="skip", showlegend=False, legendgroup=name)
            fig.add_scatter(x=xs, y=[m - s for m, s in zip(ys, sd)], mode="lines", line=dict(width=0),
                            fill="tonexty", fillcolor=_rgba(c, 0.18), hoverinfo="skip", showlegend=False,
                            legendgroup=name)
        best = (best_by_group or {}).get(g)
        sizes = [12 if x == best else 8 for x in xs]
        rings = [2 if x == best else 0 for x in xs]
        fig.add_scatter(
            x=xs, y=ys, mode="lines+markers", name=name, legendgroup=name, showlegend=not single,
            line=dict(color=c, width=2),
            marker=dict(size=sizes, color=c, line=dict(color="white", width=rings)),
            error_y=None if band else dict(type="data", array=sd, visible=True, thickness=1.2),
            customdata=[[s, n] for s, n in zip(sd, ns)],
            hovertemplate=f"{name}<br>{param}=%{{x}}<br>{metric}: %{{y:{fmt}}} ± %{{customdata[0]:{fmt}}}<br>n=%{{customdata[1]}}<extra></extra>",
        )
        if single and best is not None:
            fig.add_vline(x=best, line=dict(color=_rgba(c, 0.5), width=1, dash="dash"))
            fig.add_annotation(x=best, y=1.0, yref="paper", text=f"best {param}={best}", showarrow=False,
                               yanchor="bottom", font=dict(size=11))
    ylabel = f"{metric} ({unit})" if unit else metric
    base_layout(fig, ytitle=ylabel, xtitle=param)
    fig.update_layout(hovermode="x unified")
    if log_x:
        # show only the swept values as ticks; Plotly's default log minor labels (2..9) are noise here
        xs_all = sorted({x for s in series_by_group.values() for x, _ in s})
        fig.update_xaxes(type="log", tickvals=xs_all, ticktext=[f"{x:g}" for x in xs_all])
    return fig


def sweep_heatmap(xs: Sequence[Any], ys: Sequence[Any], matrix: Sequence[Sequence[Optional[float]]],
                  param_x: str, param_y: str, metric: str, fmt: str = ".2f",
                  higher_is_better: bool = True, best: Optional[tuple[Any, Any]] = None) -> go.Figure:
    """Sequential single-hue heatmap; darker = better. Cell values are printed directly."""
    scale = SEQUENTIAL if higher_is_better else list(reversed(SEQUENTIAL))
    text = [[("" if v is None else format(v, fmt)) for v in row] for row in matrix]
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=[str(x) for x in xs], y=[str(y) for y in ys], text=text, texttemplate="%{text}",
            colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)],
            xgap=2, ygap=2, colorbar=dict(title=metric, thickness=12),
            hovertemplate=f"{param_x}=%{{x}}<br>{param_y}=%{{y}}<br>{metric}: %{{z:{fmt}}}<extra></extra>",
        )
    )
    if best is not None:
        fig.add_shape(type="rect", xref="x", yref="y", x0=str(best[0]), x1=str(best[0]), y0=str(best[1]), y1=str(best[1]),
                      line=dict(color="#0b0b0b", width=2), opacity=0.9)
        # rect on a category axis needs half-cell padding: use a marker ring instead of relying on it
        fig.add_scatter(x=[str(best[0])], y=[str(best[1])], mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(symbol="square-open", size=34, line=dict(color="#0b0b0b", width=2)))
    base_layout(fig, xtitle=param_x, ytitle=param_y)
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category", showgrid=False)
    return fig


# --------------------------------------------------------------------------- ablations

def ablation_deltas(labels: Sequence[str], deltas: Sequence[Optional[float]], stds: Sequence[float],
                    metric: str, higher_is_better: bool = True, fmt: str = ".2f", unit: str = "") -> go.Figure:
    """Horizontal bars of (variant - base). Blue = improves the metric, red = hurts it."""
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
        y=ys, x=xs, orientation="h", marker_color=cols, marker_line_width=0, width=0.55,
        error_x=dict(type="data", array=errs, visible=True, thickness=1.2),
        text=[format(v, f"+{fmt}") for v in xs], textposition="outside", cliponaxis=False,
        hovertemplate=f"%{{y}}<br>Δ {metric}: %{{x:+{fmt}}}<extra></extra>", showlegend=False,
    )
    fig.add_vline(x=0, line=dict(color="#9a9a9a", width=1))
    xlabel = f"Δ {metric} vs full model" + (f" ({unit})" if unit else "")
    base_layout(fig, xtitle=xlabel)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_layout(height=max(220, 40 * len(ys) + 80), margin=dict(l=10, r=60, t=20, b=40))
    return fig
