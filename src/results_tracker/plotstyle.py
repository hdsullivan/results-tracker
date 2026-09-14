"""Plot style: font sizes, line and marker weights, series colours and category order.

The lab's IEEE style (`export/figures.IEEE_RC`) is the default. A project may override any of it -- the
GUI writes the overrides to `Project.plot_style` and every renderer reads them, so one style drives both
the on-screen Plotly charts (`ui/charts.py`) and the matplotlib paper figures (`export/figures.py`): what
the GUI shows is what the PDF prints.

Sizes are typographic points as matplotlib means them; the GUI multiplies them by the SCREEN_* scales
(8 pt is unreadable on a monitor) and the print figures use them as they are. Axis *ranges* are not part
of the style: they belong to one view, so they travel as `xlim` / `ylim` arguments and are saved in a
pinned asset's options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

# Fixed hue order across the lab's figures (tab10 subset): blue, red, green, purple, orange, brown, gray, pink.
# Assigned in first-seen order, never re-ranked; `PlotStyle.colors` overrides a single series.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f", "#e377c2"]

# Past the eighth series a hue has to be reused, and colour alone no longer says which line is which (the lab
# draws every line solid with filled circles). The shape -- and in print the dash -- changes on each wrap, so
# the first eight series look exactly as they always have and the ninth is still distinguishable.
MARKERS = ("circle", "square", "diamond", "triangle-up")          # plotly names
MPL_MARKERS = ("o", "s", "D", "^")                                 # the same shapes for matplotlib
MPL_LINESTYLES = ("-", "--", ":", "-.")

# Print points -> on-screen sizes. Fonts read at browser distance; lines and markers are drawn in CSS
# pixels rather than points, so they need their own factors (marker_size 3.5 pt -> 7 px).
SCREEN_FONT_SCALE = 1.35
SCREEN_LINE_SCALE = 1.7
SCREEN_MARKER_SCALE = 2.0

SIZE_FIELDS = ("base", "axis_label", "tick", "legend", "annotation", "panel_label", "math_bump")
WEIGHT_FIELDS = ("line_width", "marker_size")
NUMERIC_FIELDS = SIZE_FIELDS + WEIGHT_FIELDS

SIZE_LIMITS = (1.0, 72.0)
WEIGHT_LIMITS = (0.1, 30.0)
BUMP_LIMITS = (0.0, 24.0)  # math_bump alone may be 0 (no bump at all)


def limits(field: str) -> tuple[float, float]:
    """The range a numeric field is clamped to, for `from_dict` and the GUI's number inputs."""
    if field == "math_bump":
        return BUMP_LIMITS
    return WEIGHT_LIMITS if field in WEIGHT_FIELDS else SIZE_LIMITS


#: `order` is keyed by the grouping key whose values are being ordered (`method`, `dataset`, `config.K`,
#: `derived.kernel_type`), or by one of these pseudo-keys for an axis that is not a record field.
VARIANT_KEY = "variant"      # the ablation chart's bars (one per config variant)
PSEUDO_KEYS = (VARIANT_KEY,)

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_math(label: str) -> bool:
    """A label that is *entirely* LaTeX math (`$\\lambda$`): STIX mathtext renders smaller than serif body
    text, so the lab bumps its size (`math_bump`)."""
    t = str(label).strip()
    return t.startswith("$") and t.endswith("$") and t.count("$") == 2


def series_key(name: Any) -> str:
    """The name a colour override is stored under: a group tuple becomes its legend label (`dpir / Set12`)."""
    if isinstance(name, tuple):
        return " / ".join(map(str, name))
    return "" if name is None else str(name)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


@dataclass(frozen=True)
class PlotStyle:
    """What every chart and figure of a project looks like. All fields optional; defaults are the lab style."""

    base: float = 8.0            # body text (matplotlib font.size): hover labels on screen
    axis_label: float = 11.0     # x / y axis titles
    tick: float = 9.0            # tick labels
    legend: float = 11.0
    annotation: float = 9.0      # in-plot text: value labels, point labels, heatmap cells, colour bar
    panel_label: float = 10.5    # bold "(a) ..." caption under a print panel
    math_bump: float = 3.0       # extra size for an axis label that is pure math ($\lambda$)
    line_width: float = 1.3
    marker_size: float = 3.5
    #: series name (`series_key`) -> "#rrggbb"; anything not listed keeps its palette hue
    colors: Mapping[str, str] = field(default_factory=dict)
    #: grouping key (`method`, `dataset`, `config.K`, `derived.kernel_type`) -> the order its values are
    #: drawn in, as text. Values not listed follow, in their natural order.
    order: Mapping[str, Sequence[str]] = field(default_factory=dict)

    # ----------------------------------------------------------------- sizes

    def scaled(self, fonts: float = 1.0, line: float = 1.0, marker: float = 1.0) -> "PlotStyle":
        return replace(self,
                       **{f: getattr(self, f) * fonts for f in SIZE_FIELDS},
                       line_width=self.line_width * line, marker_size=self.marker_size * marker)

    def screen(self) -> "PlotStyle":
        """The same style at browser distance (`ui/charts.py` works in these units)."""
        return self.scaled(SCREEN_FONT_SCALE, SCREEN_LINE_SCALE, SCREEN_MARKER_SCALE)

    def label_size(self, text: Optional[str]) -> float:
        """Axis-title size with the lab's math-only bump: `$\\lambda$` -> axis_label + math_bump."""
        return self.axis_label + (self.math_bump if text and is_math(text) else 0.0)

    def weights(self, emphasized: bool = False) -> tuple[float, float]:
        """(line width, marker size) for a normal or emphasised (proposed-method) series."""
        if emphasized:
            return self.line_width + 0.2 * (self.line_width / 1.3), self.marker_size + 1.0 * (self.marker_size / 3.5)
        return self.line_width - 0.1 * (self.line_width / 1.3), self.marker_size

    # ----------------------------------------------------------------- colours

    def slots(self, names: Iterable[Any]) -> dict[Any, int]:
        """Palette position per series in first-seen order, before any override.

        `slot % len(PALETTE)` is the hue and `slot // len(PALETTE)` the wrap, which is what tells a renderer
        that two series share a colour and need different shapes.
        """
        out: dict[Any, int] = {}
        for n in names:
            if n not in out:
                out[n] = len(out)
        return out

    def colors_for(self, names: Iterable[Any]) -> dict[Any, str]:
        """Stable name -> hue in first-seen order, with this style's overrides applied. Palette positions are
        counted over every series, so removing an override never reshuffles the others; past the eighth the
        palette wraps (`wrap_of` says which series then share a hue)."""
        slots = self.slots(names)
        return {n: (self.colors.get(series_key(n)) or PALETTE[i % len(PALETTE)]) for n, i in slots.items()}

    def wrap_of(self, names: Iterable[Any]) -> dict[Any, int]:
        """How often the palette has wrapped for each series: 0 for the first eight, 1 for the next eight.
        Renderers use it to pick the marker shape (and the dash in print)."""
        return {n: i // len(PALETTE) for n, i in self.slots(names).items()}

    def palette_colors(self, names: Iterable[Any]) -> dict[Any, str]:
        """The hues these series get with no override applied -- what "reset this colour" restores."""
        return replace(self, colors={}).colors_for(names)

    def with_colors(self, colors: Mapping[Any, Optional[str]]) -> "PlotStyle":
        """This style plus (or minus, for a None value) colour overrides keyed by series name."""
        merged = dict(self.colors)
        for name, hue in colors.items():
            key = series_key(name)
            if hue:
                merged[key] = str(hue).lower()
            else:
                merged.pop(key, None)
        return replace(self, colors=merged)

    # ----------------------------------------------------------------- order

    def ordered(self, key: Optional[str], values: Sequence[Any]) -> list[Any]:
        """`values` in this style's declared order for `key`; values it does not mention keep their place at
        the end (comparison is by text, so 5 and "5" are the same value)."""
        wanted = [str(v) for v in (self.order.get(key or "") or [])]
        if not wanted:
            return list(values)
        rank = {v: i for i, v in enumerate(wanted)}
        return sorted(values, key=lambda v: rank.get(series_key(v), len(rank)))  # sorted() is stable

    def with_order(self, key: str, values: Optional[Sequence[Any]]) -> "PlotStyle":
        """This style with the order of `key` set (or cleared, for an empty list)."""
        merged = {k: list(v) for k, v in self.order.items()}
        if values:
            merged[key] = [series_key(v) for v in values]
        else:
            merged.pop(key, None)
        return replace(self, order=merged)

    # ----------------------------------------------------------------- (de)serialisation

    def to_dict(self) -> dict[str, Any]:
        """Only what differs from the lab default, so a stored style follows later default changes."""
        d = PlotStyle()
        out: dict[str, Any] = {f: getattr(self, f) for f in NUMERIC_FIELDS if getattr(self, f) != getattr(d, f)}
        if self.colors:
            out["colors"] = dict(self.colors)
        if self.order:
            out["order"] = {k: list(v) for k, v in self.order.items()}
        return out

    @property
    def is_default(self) -> bool:
        return not self.to_dict()

    def merge(self, **overrides: Any) -> "PlotStyle":
        """A copy with the given fields replaced; None values are ignored (nothing to override)."""
        return from_dict({**self.to_dict(), **{k: v for k, v in overrides.items() if v is not None}})


DEFAULT = PlotStyle()

_FIELD_NAMES = {f.name for f in fields(PlotStyle)}


def from_dict(data: Optional[Mapping[str, Any]]) -> PlotStyle:
    """A style from stored JSON, ignoring anything unrecognised. Never raises: a hand-edited or older
    `Project.plot_style` must not stop the GUI from drawing."""
    if not data:
        return DEFAULT
    kw: dict[str, Any] = {}
    for name in NUMERIC_FIELDS:
        if name in data:
            try:
                v = float(data[name])
            except (TypeError, ValueError):
                continue
            kw[name] = _clamp(v, *limits(name))
    colors = data.get("colors")
    if isinstance(colors, Mapping):
        kw["colors"] = {str(k): str(v).lower() for k, v in colors.items() if _HEX.match(str(v))}
    order = data.get("order")
    if isinstance(order, Mapping):
        kw["order"] = {str(k): [str(x) for x in v] for k, v in order.items() if isinstance(v, (list, tuple)) and v}
    return PlotStyle(**{k: v for k, v in kw.items() if k in _FIELD_NAMES})


def resolve(style: Optional[PlotStyle]) -> PlotStyle:
    return DEFAULT if style is None else style


def parse_limits(text: Any) -> Optional[tuple[float, float]]:
    """`"28,34"` / `[28, 34]` -> (28.0, 34.0); blank, malformed or non-numeric -> None (axis stays automatic)."""
    if text is None or isinstance(text, bool):
        return None
    if isinstance(text, (list, tuple)):
        parts = [str(v) for v in text]
    else:
        parts = str(text).replace(";", ",").split(",")
    parts = [p.strip() for p in parts if str(p).strip() != ""]
    if len(parts) != 2:
        return None
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return None if lo == hi else (lo, hi)
