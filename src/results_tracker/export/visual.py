"""Qualitative reconstruction comparisons in the lab's IEEE style (adaptivePnP deblur_figures.py).

- Ground truth / measurement block on the left, a narrow spacer, then baselines -> proposed;
- identical zoom box (yellow) magnified into a lower-right inset on every panel, same display range,
  native pixels (nearest-neighbour), same colour map;
- per-panel metric stamp ("31.27 dB / 0.873") with a white backing box;
- error mode: |luminance(x) - luminance(x_ref)| on one pooled 99th-percentile scale, magma, bottom colour bar;
- several rows (iteration budgets, seeds, instances) share the left block and get a row label;
- provenance (source paths, zoom box, display range, error scale) saved next to the figure.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .figures import IEEE_RC, SAVE_KW

BOX_COLOR = "#eda100"  # yellow from the categorical palette: visible on gray and on the magma error map
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
Box = tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------- images

def load_image(path: Union[str, Path]) -> np.ndarray:
    """Float32 array in [0, 1], HxW (gray) or HxWx3 (RGB). 16-bit and float inputs are scaled by dtype range."""
    from PIL import Image

    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I;16L"):
        arr = np.asarray(img, dtype=np.float32) / 65535.0
    elif img.mode in ("I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / arr.max()
    else:
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB") if "RGB" in img.mode or img.mode == "P" else img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def error_map(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|img - ref|, averaged over colour channels; shapes must match."""
    if img.shape != ref.shape:
        raise ValueError(f"shape mismatch {img.shape} vs {ref.shape}")
    e = np.abs(img.astype(np.float32) - ref.astype(np.float32))
    return e.mean(axis=2) if e.ndim == 3 else e


def psnr(img: np.ndarray, ref: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((img.astype(np.float64) - ref.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(data_range**2 / mse)


def list_image_files(dirs: Iterable[Optional[str]]) -> list[str]:
    """Union of image files (relative posix paths) across artifact directories."""
    out: set[str] = set()
    for d in dirs:
        if not d:
            continue
        p = Path(d).expanduser()
        if not p.is_dir():
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXT:
                out.add(f.relative_to(p).as_posix())
    return sorted(out)


ROLE_KEYS = {
    "reconstruction": ("recon", "restored", "output", "estimate", "x_hat", "xhat"),
    "reference": ("ground_truth", "gt", "reference", "clean", "target"),
    "measurement": ("measurement", "input", "degraded", "noisy", "observ", "blurred", "sinogram"),
    "kernel": ("kernel", "psf"),
}


def guess_roles(files: Sequence[str]) -> dict[str, Optional[str]]:
    """Guess which artifact file plays which role from its name (first match wins)."""
    out: dict[str, Optional[str]] = {}
    taken: set[str] = set()
    for role, keys in ROLE_KEYS.items():
        hit = next((f for f in files if f not in taken and any(k in Path(f).stem.lower() for k in keys)), None)
        out[role] = hit
        if hit:
            taken.add(hit)
    if out["reconstruction"] is None:
        rest = [f for f in files if f not in taken]
        out["reconstruction"] = rest[0] if rest else (files[0] if files else None)
    return out


# --------------------------------------------------------------------------- panels

@dataclass
class Panel:
    title: str
    image: np.ndarray
    subtitle: str = ""
    path: Optional[str] = None
    kind: str = "method"  # reference | measurement | method


@dataclass
class VisualSpec:
    """Everything needed to regenerate the figure; saved as a JSON sidecar."""

    experiment: Optional[str] = None
    dataset: Optional[str] = None
    instance: Optional[str] = None
    seed: Optional[int] = None
    image: Optional[str] = None
    reference: Optional[str] = None
    measurement: Optional[str] = None
    crop_box: Optional[Box] = None
    display_range: tuple[float, float] = (0.0, 1.0)
    error_vmax: Optional[float] = None
    mode: str = "image"
    kernel: Optional[str] = None
    rows: list[str] = field(default_factory=list)
    panels: list[dict[str, Any]] = field(default_factory=list)  # title, path, kind[, row]

    def caption_stub(self) -> str:
        parts = []
        where = ", ".join(str(v) for v in [self.dataset, self.instance] if v)
        if where:
            parts.append(f"Sample: {where}" + (f" (seed {self.seed})" if self.seed is not None else "") + ".")
        if self.crop_box:
            x, y, w, h = self.crop_box
            parts.append(f"Yellow box: {w}x{h} px region at ({x}, {y}), magnified in the inset; identical for all panels.")
        parts.append(f"Display range [{self.display_range[0]:g}, {self.display_range[1]:g}] for all panels; native pixels (nearest-neighbour).")
        if self.mode == "error" and self.error_vmax is not None:
            parts.append(f"Error maps show |luminance(x) − luminance(x_ref)| on a shared scale [0, {self.error_vmax:.3g}] "
                         f"(99th percentile of all method panels).")
        if self.rows:
            parts.append("Rows: " + ", ".join(self.rows) + ".")
        names = [p["title"] for p in self.panels if not p.get("row")] + \
                list(dict.fromkeys(p["title"] for p in self.panels if p.get("row")))
        if not self.rows:
            names = [p["title"] for p in self.panels]
        parts.append("Left to right: " + ", ".join(names) + ".")
        if any(p.get("kind") == "method" for p in self.panels):
            parts.append("Numbers on panels are the shown image's own PSNR / SSIM as logged.")
        return " ".join(parts)


def metric_subtitle(metrics: Mapping[str, Any], defs: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> str:
    parts = []
    for m in names:
        v = metrics.get(m)
        if v is None:
            continue
        d = defs.get(m, {})
        s = format(v, d.get("fmt", ".2f"))
        parts.append(f"{s} {d['unit']}" if d.get("unit") else s)
    return " / ".join(parts)


def build_panels(
    records: Sequence[Mapping[str, Any]],
    image: str,
    defs: Mapping[str, Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("psnr",),
    reference: Optional[str] = None,
    measurement: Optional[str] = None,
    kernel: Optional[str] = None,
    titles: Optional[Mapping[str, str]] = None,
) -> tuple[list[Panel], Optional[Panel], list[str]]:
    """One panel per record (in the given order) plus optional reference/measurement panels found
    in the first record directory that has them. The measurement panel is inserted first with
    kind="measurement"; an optional kernel/PSF thumbnail is appended with kind="kernel".
    Returns (panels, reference_panel, problems)."""
    problems: list[str] = []
    panels: list[Panel] = []
    ref_panel: Optional[Panel] = None
    meas_panel: Optional[Panel] = None

    def find(name: Optional[str]) -> Optional[Path]:
        if not name:
            return None
        p = Path(name).expanduser()
        if p.is_absolute() and p.is_file():
            return p
        for r in records:
            if r.get("artifacts_dir"):
                q = Path(r["artifacts_dir"]).expanduser() / name
                if q.is_file():
                    return q
        return None

    rp = find(reference)
    if reference and rp is None:
        problems.append(f"reference image {reference!r} not found")
    elif rp is not None:
        ref_panel = Panel("Reference", load_image(rp), path=str(rp), kind="reference")
    mp = find(measurement)
    if measurement and mp is None:
        problems.append(f"measurement image {measurement!r} not found")
    elif mp is not None:
        meas_panel = Panel("Measurement", load_image(mp), path=str(mp), kind="measurement")

    for r in records:
        title = (titles or {}).get(r.get("method"), r.get("method_label") or str(r.get("method")))
        d = r.get("artifacts_dir")
        p = Path(d).expanduser() / image if d else None
        if p is None or not p.is_file():
            problems.append(f"{title}: {image!r} missing" + (f" in {d}" if d else " (no artifacts_dir)"))
            continue
        panels.append(Panel(title, load_image(p), metric_subtitle(r.get("metrics", {}), defs, metrics), str(p)))
    if meas_panel is not None:
        panels.insert(0, meas_panel)
    shapes = {p.image.shape[:2] for p in panels} | ({ref_panel.image.shape[:2]} if ref_panel else set())
    kp = find(kernel)
    if kernel and kp is None:
        problems.append(f"kernel image {kernel!r} not found")
    elif kp is not None:
        panels.append(Panel("Kernel", load_image(kp), path=str(kp), kind="kernel"))
    if len(shapes) > 1:
        problems.append(f"image sizes differ across panels: {sorted(shapes)}")
    return panels, ref_panel, problems


def build_rows(
    records: Sequence[Mapping[str, Any]],
    row_key: str,
    image: str,
    defs: Mapping[str, Mapping[str, Any]],
    *,
    metrics: Sequence[str] = ("psnr",),
    methods: Optional[Sequence[Any]] = None,
    titles: Optional[Mapping[str, str]] = None,
    reference: Optional[str] = None,
) -> tuple[list[PanelRow], list[str]]:
    """One PanelRow per distinct value of `row_key` ('seed', 'instance' or 'config.<k>'), methods as columns.

    `records` should already be filtered to the dataset (and instance/seed when they are not the row key).
    With `reference`, each row also carries the ground truth found in its own run directories, so error maps
    for rows that show different images (instances) are scored against the right reference."""
    from .. import aggregate as agg

    values = sorted({agg.get_field(r, row_key) for r in records if agg.get_field(r, row_key) is not None},
                    key=lambda v: (isinstance(v, str), v))
    rows: list[PanelRow] = []
    problems: list[str] = []
    name = row_key.split(".")[-1]
    for v in values:
        subset = [r for r in records if agg.get_field(r, row_key) == v]
        chosen = agg.select_runs(subset, methods=methods)
        panels, ref_panel, probs = build_panels(chosen, image, defs, metrics=metrics, titles=titles, reference=reference)
        problems += [f"{name}={v}: {pr}" for pr in probs]
        label = f"${name} = {v:g}$" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{name} = {v}"
        rows.append(PanelRow([p for p in panels if p.kind == "method"], label, reference=ref_panel))
    return rows, problems


# --------------------------------------------------------------------------- figure

# Lab conventions (adaptivePnP deblur_figures.py): IEEE text width, 8 pt serif titles, hidden ticks/spines,
# yellow zoom box + lower-right magnified inset, metric stamp with a white backing box, magma error maps on a
# pooled 99th-percentile scale with a bottom colour bar, GT/Measurement block + spacer column on the left.
IEEE_TEXTWIDTH_IN = 7.16
IEEE_FONT_SIZE = 8
ZOOM_FRACTION = 0.30
ZOOM_CENTER = (0.5, 0.5)
ZOOM_INSET_BOUNDS = (0.5, 0.02, 0.48, 0.48)  # axes fraction, lower-right
ZOOM_EDGE_COLOR = "#ffd400"
KERNEL_INSET_BOUNDS = (0.78, 0.78, 0.2, 0.2)  # axes fraction, upper-right of the Measurement panel
ERROR_CMAP = "magma"
ERROR_VMAX_PERCENTILE = 99.0
SPACER_RATIO = 0.15
VIS_WIDTHS = {"double": IEEE_TEXTWIDTH_IN, "single": 3.5, "ieee-double": IEEE_TEXTWIDTH_IN, "ieee-single": 3.5}


@dataclass
class PanelRow:
    """One row of method panels, e.g. one iteration budget K or one seed; `label` goes on the left.

    `reference` (optional) is the ground truth for *this* row; error maps use it when rows have their own
    reference (e.g. rows = instances). The left block shows the figure-level reference."""

    panels: list[Panel]
    label: str = ""
    reference: Optional[Panel] = None


def luminance(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return (img[..., :3].astype(np.float32) @ w)


def zoom_region(shape: tuple[int, ...], fraction: float = ZOOM_FRACTION, center: tuple[float, float] = ZOOM_CENTER) -> Box:
    """Square zoom box (x, y, side, side) as a fraction of the short side, centred at `center`, clamped inside."""
    h, w = shape[:2]
    side = max(2, int(round(min(h, w) * fraction)))
    cx, cy = int(center[0] * w), int(center[1] * h)
    x0 = min(max(cx - side // 2, 0), w - side)
    y0 = min(max(cy - side // 2, 0), h - side)
    return (x0, y0, side, side)


def _style_panel(ax) -> None:
    """Hide ticks and spines but keep titles/labels usable (unlike ax.axis('off'))."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _show(ax, img: np.ndarray, vmin: float, vmax: float, cmap: str) -> None:
    ax.imshow(img, cmap=cmap if img.ndim == 2 else None, vmin=vmin, vmax=vmax, interpolation="nearest")


def _add_zoom_inset(ax, img: np.ndarray, box: Box, vmin: float, vmax: float, cmap: str) -> None:
    x0, y0, bw, bh = box
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), bw, bh, fill=False, edgecolor=ZOOM_EDGE_COLOR, linewidth=0.9))
    ix, iy, iw, ih = ZOOM_INSET_BOUNDS
    if bh != bw:  # keep the inset's aspect equal to the box's
        ih = min(0.6, iw * bh / bw)
        iy = 0.02
    axins = ax.inset_axes((ix, iy, iw, ih))
    _show(axins, crop(img, box), vmin, vmax, cmap)
    axins.set_xticks([])
    axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor(ZOOM_EDGE_COLOR)
        sp.set_linewidth(0.9)


def _add_kernel_inset(ax, kernel: np.ndarray) -> None:
    axins = ax.inset_axes(KERNEL_INSET_BOUNDS)
    axins.imshow(kernel, cmap="gray" if kernel.ndim == 2 else None, interpolation="nearest")
    axins.set_xticks([])
    axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("white")
        sp.set_linewidth(0.9)


def _stamp(ax, text: str, corner: str) -> None:
    """`31.27 dB / 0.873` in a corner with a legibility backing box (upper-left when a zoom inset is present)."""
    y, va = (0.035, "bottom") if corner == "lower left" else (0.965, "top")
    ax.text(0.035, y, text, transform=ax.transAxes, fontsize=IEEE_FONT_SIZE - 1.5, va=va, ha="left", color="black",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "boxstyle": "round,pad=0.15"})


def reconstruction_figure(
    panels: Union[Sequence[Panel], Sequence[PanelRow]],
    *,
    reference: Optional[Panel] = None,
    measurement: Optional[Panel] = None,
    kernel: Optional[Panel] = None,
    mode: str = "image",
    zoom: bool = False,
    zoom_fraction: float = ZOOM_FRACTION,
    zoom_center: tuple[float, float] = ZOOM_CENTER,
    crop_box: Optional[Box] = None,
    error_vmax: Optional[float] = None,
    display_range: tuple[float, float] = (0.0, 1.0),
    width: Union[str, float] = "double",
    cmap: str = "gray",
    annotate: bool = True,
    show_titles: bool = True,
    error_maps: Optional[bool] = None,
) -> tuple[Figure, VisualSpec]:
    """Lab-style qualitative grid.

    Columns: [Reference | Measurement] (side by side for one row, stacked in one column for several rows),
    a narrow spacer, then one column per method. Rows: one `PanelRow` per iteration budget / seed / instance.

    mode="image": reconstructions, optionally with an identical yellow zoom box + lower-right magnified inset
    on every panel (`zoom`, with `zoom_fraction`/`zoom_center`, or an explicit `crop_box`).
    mode="error": |luminance(x) − luminance(x_ref)| per method panel on one pooled scale (99th percentile,
    or `error_vmax`) with a bottom colour bar; the reference block still shows the images.
    Metric stamps (panel.subtitle) go in the lower-left corner, upper-left when the zoom inset is present.
    """
    if error_maps is not None:  # backwards compatibility with the old flag
        mode = "error" if error_maps else mode
    if mode not in ("image", "error"):
        raise ValueError("mode must be 'image' or 'error'")
    # normalise input into rows; a plain list of Panels (possibly containing a measurement panel) is one row
    if panels and isinstance(panels[0], PanelRow):
        rows: list[PanelRow] = list(panels)  # type: ignore[arg-type]
    else:
        flat = [p for p in panels if p.kind != "measurement"]  # type: ignore[union-attr]
        meas_in = [p for p in panels if p.kind == "measurement"]  # type: ignore[union-attr]
        if meas_in and measurement is None:
            measurement = meas_in[0]
        rows = [PanelRow(list(flat))]
    n_rows = len(rows)
    n_methods = max((len(r.panels) for r in rows), default=0)
    if n_methods == 0 and reference is None and measurement is None:
        raise ValueError("no panels")
    if mode == "error" and reference is None:
        raise ValueError("error mode needs a reference image")

    left = [p for p in (reference, measurement) if p is not None]
    first = (rows[0].panels[0] if rows and rows[0].panels else left[0]).image
    h0, w0 = first.shape[:2]
    if crop_box is not None:
        cx, cy, cw, ch = crop_box
        if cx < 0 or cy < 0 or cx + cw > w0 or cy + ch > h0 or cw <= 0 or ch <= 0:
            raise ValueError(f"crop box {crop_box} outside image {w0}x{h0}")
        box: Optional[Box] = crop_box
    else:
        box = zoom_region(first.shape, zoom_fraction, zoom_center) if zoom else None
    use_zoom = mode == "image" and box is not None

    # column layout
    left_cols = len(left) if n_rows == 1 else (1 if left else 0)
    ratios: list[float] = [1.0] * left_cols + ([SPACER_RATIO] if left_cols and n_methods else []) + [1.0] * n_methods
    n_cols = len(ratios)
    fw = VIS_WIDTHS.get(width, None) if isinstance(width, str) else float(width)
    if fw is None:
        fw = float(width)
    unit = fw / sum(ratios)
    fh = unit * n_rows * (h0 / w0) + 0.3 + (0.35 if mode == "error" else 0.0)
    vmin, vmax = display_range

    # pooled error scale over every method panel
    errs: dict[tuple[int, int], np.ndarray] = {}
    if mode == "error":
        for ri, row in enumerate(rows):
            ref_lum = luminance((row.reference or reference).image)  # a row's own reference wins
            for ci, p in enumerate(row.panels):
                if p.image.shape[:2] != ref_lum.shape:
                    raise ValueError(f"{p.title}: shape {p.image.shape[:2]} differs from reference {ref_lum.shape}")
                errs[(ri, ci)] = np.abs(luminance(p.image) - ref_lum)
        if error_vmax is None:
            pooled = np.concatenate([e.ravel() for e in errs.values()]) if errs else np.zeros(1)
            error_vmax = max(float(np.percentile(pooled, ERROR_VMAX_PERCENTILE)), 1e-6)

    with matplotlib.rc_context({**IEEE_RC, "font.size": IEEE_FONT_SIZE, "axes.titlesize": IEEE_FONT_SIZE,
                                "axes.labelsize": IEEE_FONT_SIZE}):
        fig = Figure(figsize=(fw, fh), dpi=300)
        gs = fig.add_gridspec(n_rows, n_cols, width_ratios=ratios)

        def draw_ref_block(ax, p: Panel) -> None:
            _style_panel(ax)
            _show(ax, p.image, vmin, vmax, cmap)
            if use_zoom:
                _add_zoom_inset(ax, p.image, box, vmin, vmax, cmap)
            if kernel is not None and p.kind == "measurement":
                _add_kernel_inset(ax, kernel.image)
            if show_titles:
                ax.set_title(p.title, fontsize=IEEE_FONT_SIZE)
            if annotate and p.subtitle:
                _stamp(ax, p.subtitle, "upper left" if use_zoom else "lower left")

        if left:
            if n_rows == 1:
                for i, p in enumerate(left):
                    draw_ref_block(fig.add_subplot(gs[0, i]), p)
            else:
                sub = gs[:, 0].subgridspec(len(left), 1, hspace=0.15)
                for i, p in enumerate(left):
                    draw_ref_block(fig.add_subplot(sub[i, 0]), p)

        method_axes = []
        mappable = None
        c0 = left_cols + (1 if left_cols and n_methods else 0)
        for ri, row in enumerate(rows):
            for ci, p in enumerate(row.panels):
                ax = fig.add_subplot(gs[ri, c0 + ci])
                _style_panel(ax)
                method_axes.append(ax)
                if mode == "error":
                    mappable = ax.imshow(errs[(ri, ci)], cmap=ERROR_CMAP, vmin=0.0, vmax=error_vmax, interpolation="nearest")
                else:
                    _show(ax, p.image, vmin, vmax, cmap)
                    if use_zoom:
                        _add_zoom_inset(ax, p.image, box, vmin, vmax, cmap)
                if annotate and p.subtitle:
                    _stamp(ax, p.subtitle, "upper left" if use_zoom else "lower left")
                if ri == 0 and show_titles:
                    ax.set_title(p.title, fontsize=IEEE_FONT_SIZE)
                if ci == 0 and row.label:
                    ax.set_ylabel(row.label, fontsize=IEEE_FONT_SIZE)

        fig.tight_layout(pad=0.4, h_pad=0.6, w_pad=0.3)
        if mode == "error" and mappable is not None:
            cbar = fig.colorbar(mappable, ax=method_axes, location="bottom", shrink=0.6, aspect=40, pad=0.03)
            cbar.set_label("| luminance error |", fontsize=IEEE_FONT_SIZE)
            cbar.ax.tick_params(labelsize=IEEE_FONT_SIZE - 1.5)

    spec = VisualSpec(
        crop_box=box if use_zoom else None, display_range=display_range,
        error_vmax=error_vmax if mode == "error" else None,
        reference=reference.path if reference else None,
        measurement=measurement.path if measurement else None,
        mode=mode, kernel=kernel.path if kernel else None,
        rows=[r.label for r in rows] if n_rows > 1 else [],
        panels=[{"title": p.title, "path": p.path, "kind": p.kind} for p in left]
               + [{"title": p.title, "path": p.path, "kind": p.kind, "row": r.label} for r in rows for p in r.panels],
    )
    return fig, spec


def save_visual(fig: Figure, path: Union[str, Path], spec: VisualSpec, dpi: int = 300, also_png: bool = False) -> list[Path]:
    """Save the figure (PNG recommended for image panels) plus a JSON provenance sidecar."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, **SAVE_KW)
    out = [p]
    if also_png and p.suffix.lower() != ".png":
        q = p.with_suffix(".png")
        fig.savefig(q, dpi=dpi, **SAVE_KW)
        out.append(q)
    side = p.with_suffix(".json")
    side.write_text(json.dumps(asdict(spec), indent=2, default=str))
    out.append(side)
    return out


# --------------------------------------------------------------------------- one-call builder

@dataclass
class VisualResult:
    fig: Figure
    spec: VisualSpec
    problems: list[str]
    omitted: dict[Any, str]


def make_visual(
    records: Sequence[Mapping[str, Any]],
    defs: Mapping[str, Mapping[str, Any]],
    *,
    experiment: Optional[str] = None,
    dataset: Optional[Any] = None,
    seed: Optional[Any] = None,
    instance: Optional[Any] = None,
    image: Optional[str] = None,
    reference: Optional[str] = None,
    measurement: Optional[str] = None,
    kernel: Optional[str] = None,
    methods: Optional[Sequence[Any]] = None,
    metrics: Sequence[str] = ("psnr", "ssim"),
    mode: str = "image",
    zoom: bool = False,
    zoom_fraction: float = ZOOM_FRACTION,
    zoom_center: tuple[float, float] = ZOOM_CENTER,
    crop_box: Optional[Box] = None,
    rows: Optional[str] = None,
    width: Union[str, float] = "double",
    auto_roles: bool = True,
) -> VisualResult:
    """Everything from records to a finished lab-style figure. File roles are guessed from the artifact
    folders when not given (`auto_roles`). Raises ValueError when nothing can be drawn."""
    from .. import aggregate as agg

    pool = agg.completed(records)
    if dataset is not None:
        pool = [r for r in pool if r.get("dataset") == dataset]
    if instance is not None and rows != "instance":
        pool = [r for r in pool if r.get("instance") == instance]
    if seed is not None and rows != "seed":
        pool = [r for r in pool if r.get("seed") == seed]
    with_art = [r for r in pool if r.get("artifacts_dir")]
    if not with_art:
        raise ValueError("no completed runs with an artifacts_dir match")
    if auto_roles and (image is None or (reference is None and measurement is None)):
        roles = guess_roles(list_image_files(r["artifacts_dir"] for r in with_art))
        image = image or roles["reconstruction"]
        reference = reference if reference is not None else roles["reference"]
        measurement = measurement if measurement is not None else roles["measurement"]
        kernel = kernel if kernel is not None else roles["kernel"]
    if image is None:
        raise ValueError("no image files found in the artifact folders")
    if mode == "error" and reference is None:
        raise ValueError("error mode needs a reference (ground truth) image")

    # only runs with artifacts can be drawn; methods without them are reported as omitted, not as errors
    chosen = agg.select_runs(with_art, methods=methods)
    shown = with_art if rows else chosen
    omitted = agg.omitted_methods(records, shown, dataset=dataset)
    if rows:
        aux, ref_panel, problems = build_panels(shown[:1], image, defs, reference=reference, measurement=measurement, kernel=kernel)
        problems = [pr for pr in problems if pr.startswith(("reference", "measurement", "kernel"))]
        row_specs, probs = build_rows(pool, rows, image, defs, metrics=metrics, methods=methods, reference=reference)
        problems += probs
        panel_arg: Any = row_specs
    else:
        aux, ref_panel, problems = build_panels(chosen, image, defs, metrics=metrics, reference=reference,
                                                measurement=measurement, kernel=kernel)
        panel_arg = [p for p in aux if p.kind == "method"]
    meas_panel = next((p for p in aux if p.kind == "measurement"), None)
    ker_panel = next((p for p in aux if p.kind == "kernel"), None)
    if not panel_arg or (rows and not any(r.panels for r in panel_arg)):
        raise ValueError("no method panels could be built: " + "; ".join(problems))
    fig, spec = reconstruction_figure(panel_arg, reference=ref_panel, measurement=meas_panel, kernel=ker_panel, mode=mode,
                                      zoom=zoom, zoom_fraction=zoom_fraction, zoom_center=zoom_center, crop_box=crop_box,
                                      width=width)
    spec.experiment, spec.dataset, spec.instance, spec.seed, spec.image = experiment, dataset, instance, seed, image
    return VisualResult(fig, spec, problems, omitted)
