"""Qualitative reconstruction comparisons: reference | measurement | baselines | proposed.

Conventions (lab's IEEE reconstruction-figure checklist):
- identical crop, display range, interpolation, colour map and error-map scale for every method;
- nearest-neighbour rendering so native pixels are preserved;
- panel order reference -> observation -> baselines -> proposed;
- one shared error scale, shown as a colour bar, error defined as |x - x_ref| (mean over channels);
- provenance (source paths, crop box, display range) saved next to the figure.
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

from .figures import IEEE_RC, SAVE_KW, width_in

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
    panels: list[dict[str, Any]] = field(default_factory=list)  # title, path, run_id

    def caption_stub(self) -> str:
        parts = []
        where = ", ".join(str(v) for v in [self.dataset, self.instance] if v)
        if where:
            parts.append(f"Sample: {where}" + (f" (seed {self.seed})" if self.seed is not None else "") + ".")
        if self.crop_box:
            x, y, w, h = self.crop_box
            parts.append(f"Zoom: {w}x{h} crop at ({x}, {y}), identical for all methods.")
        parts.append(f"Display range [{self.display_range[0]:g}, {self.display_range[1]:g}] for all panels; nearest-neighbour rendering.")
        if self.reference and self.error_vmax is not None:
            parts.append(f"Error maps show |x - x_ref| averaged over channels on a shared scale [0, {self.error_vmax:.3g}].")
        parts.append("Left to right: " + ", ".join(p["title"] for p in self.panels) + ".")
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
    titles: Optional[Mapping[str, str]] = None,
) -> tuple[list[Panel], Optional[Panel], list[str]]:
    """One panel per record (in the given order) plus optional reference/measurement panels found
    in the first record directory that has them. Returns (panels, reference_panel, problems)."""
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
    if len(shapes) > 1:
        problems.append(f"image sizes differ across panels: {sorted(shapes)}")
    return panels, ref_panel, problems


# --------------------------------------------------------------------------- figure

def reconstruction_figure(
    panels: Sequence[Panel],
    *,
    reference: Optional[Panel] = None,
    crop_box: Optional[Box] = None,
    error_maps: bool = False,
    error_vmax: Optional[float] = None,
    display_range: tuple[float, float] = (0.0, 1.0),
    width: Union[str, float] = "double",
    cmap: str = "gray",
    error_cmap: str = "magma",
    show_titles: bool = True,
    show_subtitles: bool = True,
    panel_labels: bool = False,
    box_color: str = BOX_COLOR,
) -> tuple[Figure, VisualSpec]:
    """Grid: columns = [reference] + panels; rows = full view [, zoomed crop] [, error map].

    All panels share `display_range`, `crop_box`, `cmap` and the error scale. Returns the figure and
    the spec (with the resolved error_vmax) for the caption and the provenance sidecar.
    """
    cols: list[Panel] = ([reference] if reference is not None else []) + list(panels)
    if not cols:
        raise ValueError("no panels")
    use_err = error_maps and reference is not None
    rows = 1 + (1 if crop_box else 0) + (1 if use_err else 0)
    n = len(cols)
    h0, w0 = cols[0].image.shape[:2]
    aspect = h0 / w0
    if crop_box:
        cx, cy, cw, ch = crop_box
        if cx < 0 or cy < 0 or cx + cw > w0 or cy + ch > h0 or cw <= 0 or ch <= 0:
            raise ValueError(f"crop box {crop_box} outside image {w0}x{h0}")

    errs: dict[int, np.ndarray] = {}
    if use_err:
        for i, p in enumerate(cols):
            if p.kind == "method" or p.kind == "measurement":
                errs[i] = error_map(p.image, reference.image)
        if error_vmax is None and errs:
            error_vmax = float(max(np.percentile(e, 99.5) for e in errs.values())) or 1.0

    with matplotlib.rc_context(IEEE_RC):
        fw = width_in(width)
        title_h = 0.14 if show_titles else 0.0
        sub_h = 0.14 if show_subtitles else 0.0
        cbar_w = 0.18 if use_err else 0.0
        panel_w = (fw - cbar_w) / n
        row_hs = [panel_w * aspect] + ([panel_w * (ch / cw)] if crop_box else []) + ([panel_w * aspect] if use_err else [])
        fh = sum(row_hs) + title_h + sub_h + 0.02 * (rows - 1) * panel_w
        fig = Figure(figsize=(fw, fh))
        gs = fig.add_gridspec(
            rows, n + (1 if use_err else 0),
            width_ratios=[1] * n + ([cbar_w / panel_w] if use_err else []),
            height_ratios=row_hs, wspace=0.03, hspace=0.03,
            left=0, right=1, top=1 - title_h / fh, bottom=sub_h / fh,
        )
        vmin, vmax = display_range
        last_im = None
        for i, p in enumerate(cols):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(p.image, cmap=cmap if p.image.ndim == 2 else None, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_axis_off()
            if show_titles:
                t = p.title if not panel_labels else f"({chr(97 + i)}) {p.title}"
                ax.set_title(t, fontsize=8, pad=2)
            if crop_box:
                ax.add_patch(Rectangle((cx - 0.5, cy - 0.5), cw, ch, fill=False, edgecolor=box_color, linewidth=0.8))
                axc = fig.add_subplot(gs[1, i])
                axc.imshow(crop(p.image, crop_box), cmap=cmap if p.image.ndim == 2 else None, vmin=vmin, vmax=vmax,
                           interpolation="nearest")
                axc.set_xticks([]); axc.set_yticks([])
                axc.tick_params(which="both", length=0)
                for s in axc.spines.values():
                    s.set_edgecolor(box_color); s.set_linewidth(0.8)
            if use_err:
                axe = fig.add_subplot(gs[rows - 1, i])
                if i in errs:
                    last_im = axe.imshow(errs[i], cmap=error_cmap, vmin=0, vmax=error_vmax, interpolation="nearest")
                else:
                    axe.imshow(np.zeros_like(errs[next(iter(errs))]) if errs else np.zeros((2, 2)), cmap=error_cmap,
                               vmin=0, vmax=error_vmax or 1, interpolation="nearest")
                    axe.text(0.5, 0.5, "reference", ha="center", va="center", fontsize=7, color="white", transform=axe.transAxes)
                axe.set_axis_off()
            bottom_ax = axe if use_err else (axc if crop_box else ax)
            if show_subtitles and p.subtitle:
                bottom_ax.text(0.5, -0.04, p.subtitle, ha="center", va="top", fontsize=7, transform=bottom_ax.transAxes)
        if use_err and last_im is not None:
            cax = fig.add_subplot(gs[rows - 1, n])
            cb = fig.colorbar(last_im, cax=cax)
            cb.set_label("|x − x_ref|", fontsize=7)
            cb.ax.tick_params(labelsize=6, width=0.5, length=2)
            cb.outline.set_linewidth(0.5)
            for j in range(rows - 1):
                fig.add_subplot(gs[j, n]).set_axis_off()

    spec = VisualSpec(
        crop_box=crop_box, display_range=display_range, error_vmax=error_vmax if use_err else None,
        reference=reference.path if reference else None,
        panels=[{"title": p.title, "path": p.path, "kind": p.kind} for p in cols],
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
