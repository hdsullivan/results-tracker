"""The GUI's plot-style controls: one sidebar panel for sizes and weights, one expander per chart for
axis ranges, label order and series colours.

The style belongs to the project (`Project.plot_style`), so a choice made on one page holds on every other
page and in every figure the project exports; these widgets are the only place that writes it. They are
*displays* of the stored style -- every edit goes through an `on_change` callback that saves it and lets
Streamlit rerun, so the chart below is drawn from the new style and no page can write a stale one back.

Axis *ranges* are not part of the project style: they describe one view, so they live in `st.session_state`
per (chart, experiment) and are saved into a pinned asset's options (`xlim`, `ylim`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import streamlit as st

from .. import plotstyle
from ..api import set_project
from ..plotstyle import PlotStyle
from .common import KEY_PROJECT, db_path, engine_for, keyed, load_catalog

WIDGET_PREFIX = "stylew_"

#: (field, label, help) of every size the user can set. Sizes are print points; the GUI draws them larger.
SIZE_CONTROLS = (
    ("axis_label", "Axis titles", "The x and y labels, e.g. 'PSNR (dB)'."),
    ("tick", "Tick labels", "The numbers along both axes."),
    ("legend", "Legend", "On screen and in the exported figure."),
    ("annotation", "In-plot text", "Value labels, point labels, heat-map cells, colour-bar ticks."),
    ("base", "Body text", "Hover tooltips on screen; the base size print figures scale from."),
    ("panel_label", "Panel caption", "The bold '(a) …' under an exported figure."),
    ("math_bump", "Math label bump", "Extra size for an axis title that is pure math ($\\lambda$): STIX renders small."),
)
WEIGHT_CONTROLS = (
    ("line_width", "Line width", "Lines and box edges; the emphasised method stays a little heavier."),
    ("marker_size", "Marker size", "Filled circles on lines and scatters."),
)


# --------------------------------------------------------------------------- the project's style

def plot_style(project: Optional[str] = None) -> PlotStyle:
    """The style stored on the project (from the cached catalog), or the lab default."""
    if project is None:
        project = st.session_state.get(KEY_PROJECT)
    for p in load_catalog()["projects"]:
        if p["name"] == project:
            return plotstyle.from_dict(p.get("plot_style"))
    return plotstyle.DEFAULT


def save_plot_style(project: Optional[str], style: PlotStyle) -> None:
    """Write the style to the project. The catalog is cached on the database's mtime, so the next rerun reads
    it back; the record caches are keyed per experiment and are not disturbed."""
    if not project:
        return
    set_project(project, plot_style=style.to_dict(), engine=engine_for(db_path()))


def _synced(widget, label: str, key: str, value: Any, on_change: Callable[[], None], **kw: Any) -> Any:
    """A widget that shows `value` (the stored style) on every run. Safe to re-seed each time because the
    edit is persisted by `on_change` *before* Streamlit reruns the script."""
    st.session_state[key] = value
    return widget(label, key=key, on_change=on_change, **kw)


def _set_field(project: Optional[str], style: PlotStyle, field: str, key: str) -> Callable[[], None]:
    def apply() -> None:
        v = st.session_state.get(key)
        if v is not None:
            save_plot_style(project, style.merge(**{field: float(v)}))
    return apply


# --------------------------------------------------------------------------- sidebar: sizes and weights

def sidebar_plot_style(project: Optional[str] = None) -> PlotStyle:
    """The sizes panel in the sidebar of every page that draws a chart. Returns the style to draw with."""
    if project is None:
        project = st.session_state.get(KEY_PROJECT)
    style = plot_style(project)
    with st.sidebar.expander("Plot style", expanded=False):
        st.caption("Point sizes as the paper figures use them (the GUI draws them ~1.35× larger so they read on "
                   "screen). They apply to every chart of this project and to every figure it exports.")
        for controls, step in ((SIZE_CONTROLS, 0.5), (WEIGHT_CONTROLS, 0.1)):
            cols = st.columns(2)
            for i, (field, label, help_) in enumerate(controls):
                key = f"{WIDGET_PREFIX}{field}"
                lo, hi = plotstyle.limits(field)
                with cols[i % 2]:
                    _synced(st.number_input, label, key, float(getattr(style, field)),
                            _set_field(project, style, field, key), min_value=lo, max_value=hi,
                            step=step, format="%.1f", help=help_)
        c1, c2 = st.columns([2, 1])
        factor = c1.number_input("Scale every size ×", min_value=0.2, max_value=5.0, value=1.0, step=0.1,
                                 key=f"{WIDGET_PREFIX}scale", help="Multiplies all of the above at once.")
        if c2.button("Apply", key=f"{WIDGET_PREFIX}scale_go", disabled=factor == 1.0):
            save_plot_style(project, style.scaled(factor, factor, factor))
            st.session_state[f"{WIDGET_PREFIX}scale"] = 1.0
            st.rerun()
        if st.button("Reset sizes and colours", key=f"{WIDGET_PREFIX}reset",
                     disabled=style.is_default, help="Back to the lab style; the label order is kept."):
            save_plot_style(project, PlotStyle(order=style.order))
            st.rerun()
        if not style.is_default:
            st.caption("This project has a custom style; `results-tracker export paper` renders with the same one.")
    return style


# --------------------------------------------------------------------------- per chart: ranges, order, colours

@dataclass
class ChartControls:
    """What the expander above a chart decided: the style to draw with and the fixed axis ranges."""

    style: PlotStyle
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None

    @property
    def limit_options(self) -> dict[str, Any]:
        """The ranges as a pinned asset stores them (None = fit the data)."""
        return {"xlim": list(self.xlim) if self.xlim else None, "ylim": list(self.ylim) if self.ylim else None}


def _range_input(container, label: str, key: str, help_: str) -> Optional[tuple[float, float]]:
    """`lo,hi` for one axis, kept per view in the session (not on the project)."""
    text = keyed(container.text_input, label, key, "", placeholder="auto", help=help_)
    lim = plotstyle.parse_limits(text)
    if str(text).strip() and lim is None:
        container.caption(":red[give two different numbers, e.g. 28,34]")
    return lim


def chart_controls(
    project: Optional[str],
    style: PlotStyle,
    *,
    key: str,
    series: Sequence[Any] = (),
    series_key: Optional[str] = None,
    series_labels: Optional[Mapping[Any, str]] = None,
    orders: Sequence[tuple[str, Sequence[Any]]] = (),
    colors: bool = True,
    x_name: Optional[str] = "x",
    y_name: Optional[str] = "y",
    log_x: bool = False,
    log_y: bool = False,
) -> ChartControls:
    """Axis ranges, label order and series colours for one chart. Call it just above the chart.

    `series` are the entities that carry a colour (legend entries, bars) and `series_key` the grouping key
    they come from, so their order is stored per key and every chart grouped by it agrees. `orders` adds
    `(key, values)` pairs for the other axes whose label order this chart decides (a categorical x axis, a
    heat map's two parameters). Ranges are per view; order and colours are saved on the project. Pass
    `colors=False` for a chart whose hues are not per series (the ablation bars are coloured by polarity).
    """
    with st.expander("Axes, label order and colours", expanded=False):
        xlim = ylim = None
        wanted = [(name, axis, log) for name, axis, log in ((x_name, "xlim", log_x), (y_name, "ylim", log_y)) if name]
        if wanted:
            hint = "`lo,hi` in data units; blank fits the data."
            cols = st.columns(len(wanted) if len(wanted) > 1 else 2)
            for col, (name, axis, log) in zip(cols, wanted):
                lim = _range_input(col, f"{name} range", f"{key}_{axis}",
                                   hint + (" Give plain values on this log axis (0.01,1)." if log else ""))
                if axis == "xlim":
                    xlim = lim
                else:
                    ylim = lim
        if series_key and len(series) > 1:
            _order_control(project, style, series_key, series, series_labels, label=f"Order of {series_key}", prefix=key)
        for order_key, values in orders:
            if order_key and order_key != series_key and len(values) > 1:
                _order_control(project, style, order_key, values, None, prefix=key,
                               label=f"Order of {order_key} along the axis")
        if series and colors:
            _colour_control(project, style, series, series_labels, prefix=key)
    n = len(list(dict.fromkeys(series)))
    if n > len(plotstyle.PALETTE):
        st.caption(f":orange[{n} series share {len(plotstyle.PALETTE)} colours.] The {len(plotstyle.PALETTE) + 1}th onward "
                   "repeat a hue with a different marker (a dashed line in the exported figure). A plot this crowded is "
                   "usually better split: filter in the sidebar, or group by a coarser key.")
    return ChartControls(style, xlim, ylim)


def _order_control(project: Optional[str], style: PlotStyle, order_key: str, values: Sequence[Any],
                   labels: Optional[Mapping[Any, str]], *, label: str, prefix: str) -> None:
    """A multiselect whose *selection order* is the order `order_key`'s values are drawn in."""
    options = list(dict.fromkeys(str(v) for v in style.ordered(order_key, list(values))))
    key = f"{prefix}_order_{order_key}"

    def apply() -> None:
        picked = [v for v in (st.session_state.get(key) or []) if v in options]
        save_plot_style(project, style.with_order(order_key, picked))

    _synced(st.multiselect, label, key, options, apply, options=options,
            format_func=lambda v: str((labels or {}).get(v, v)),
            help="Click the values in the order you want them drawn; anything left out keeps its place at the "
                 "end. Saved for this key, so every chart, table axis and exported figure grouped by it agrees. "
                 "Palette hues follow this order -- fix a colour below to pin it.")


def _colour_control(project: Optional[str], style: PlotStyle, series: Sequence[Any],
                    labels: Optional[Mapping[Any, str]], *, prefix: str) -> None:
    """One colour picker per series. A colour left at its palette hue is stored as no override, so an
    untouched series keeps following the palette when the order changes."""
    names = list(dict.fromkeys(series))
    current = style.colors_for(names)
    palette = style.palette_colors(names)
    st.caption("Line, bar and box colours. Set one back to its palette hue to unpin it.")
    per_row = 6
    for start in range(0, len(names), per_row):
        chunk = names[start:start + per_row]
        cols = st.columns(per_row)
        for col, name in zip(cols, chunk):
            skey = plotstyle.series_key(name)
            key = f"{prefix}_color_{skey}"

            def apply(name=name, skey=skey, key=key) -> None:
                got = str(st.session_state.get(key) or "").lower()
                save_plot_style(project, style.with_colors({name: None if got == palette[name].lower() else got}))

            with col:
                text = str((labels or {}).get(name, skey)) or "all runs"
                _synced(st.color_picker, text[:22], key, current[name], apply)
