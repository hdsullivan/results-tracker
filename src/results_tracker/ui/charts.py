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
