import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from PIL import Image  # noqa: E402

from results_tracker import aggregate as agg  # noqa: E402
from results_tracker.export.figures import figure_tex, ieee_preamble, to_grayscale_png, figure_bytes  # noqa: E402
from results_tracker.export.visual import (  # noqa: E402
    IEEE_TEXTWIDTH_IN, Panel, PanelRow, build_panels, build_rows, crop, error_map, list_image_files, load_image,
    guess_roles, luminance, panel_metrics_rows, panel_psnr, psnr, reconstruction_figure, save_visual, zoom_region,
)

DEFS = {"psnr": {"unit": "dB", "higher_is_better": True, "fmt": ".2f"}, "ssim": {"unit": "", "higher_is_better": True, "fmt": ".3f"}}


def _png(path, arr):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8")).save(path)


@pytest.fixture
def art(tmp_path):
    rng = np.random.default_rng(0)
    gt = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6, 48))[None, :] * np.ones((48, 1)), 0, 1)
    recs = []
    for m, sigma, base in [("TV", 0.08, True), ("Ours", 0.02, False)]:
        d = tmp_path / m
        d.mkdir()
        _png(d / "ground_truth.png", gt)
        _png(d / "measurement.png", gt + rng.normal(0, 0.15, gt.shape))
        _png(d / "reconstruction.png", gt + rng.normal(0, sigma, gt.shape))
        (d / "notes.txt").write_text("x")
        recs.append({"run_id": len(recs), "method": m, "method_label": f"{m} [1]" if base else m, "method_is_baseline": base,
                     "dataset": "D", "seed": 0, "instance": None, "config": {}, "metrics": {"psnr": 30.0 if base else 34.0, "ssim": 0.9},
                     "status": "completed", "artifacts_dir": str(d)})
    # a failed run and a run without artifacts must be ignored / reported
    recs.append({**recs[0], "run_id": 9, "method": "PnP", "method_label": "PnP", "status": "completed", "artifacts_dir": None})
    return gt, recs


def test_image_helpers(tmp_path):
    arr = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    _png(tmp_path / "a.png", arr)
    img = load_image(tmp_path / "a.png")
    assert img.shape == (16, 16) and img.dtype == np.float32 and 0 <= img.min() and img.max() <= 1
    assert crop(img, (2, 3, 5, 4)).shape == (4, 5)
    e = error_map(img, np.zeros_like(img))
    assert np.allclose(e, img)
    rgb = np.stack([img] * 3, axis=2)
    assert error_map(rgb, np.zeros_like(rgb)).shape == (16, 16)
    assert psnr(img, img) == float("inf")
    assert 19.9 < psnr(img, img + 0.1) < 20.1
    with pytest.raises(ValueError):
        error_map(img, img[:8])
    assert list_image_files([str(tmp_path), None, "/nonexistent"]) == ["a.png"]


def test_select_runs_orders_baselines_first(art):
    _, recs = art
    sel = agg.select_runs(recs, dataset="D", seed=0)
    assert [r["method"] for r in sel] == ["TV", "PnP", "Ours"]  # baselines (TV, PnP) first, proposed last
    assert [r["method"] for r in agg.select_runs(recs, methods=["Ours", "TV"])] == ["Ours", "TV"]
    assert agg.select_runs(recs, dataset="other") == []


def test_build_panels_reports_problems(art):
    _, recs = art
    panels, ref, problems = build_panels(agg.select_runs(recs), "reconstruction.png", DEFS, metrics=["psnr", "ssim"],
                                         reference="ground_truth.png", measurement="measurement.png")
    assert [p.title for p in panels] == ["Measurement", "TV [1]", "Ours"]
    assert panels[1].subtitle == "30.00 dB / 0.900" and panels[0].subtitle == ""
    assert ref is not None and ref.kind == "reference"
    assert any("PnP" in p and "no artifacts_dir" in p for p in problems)
    _, _, problems = build_panels(recs[:1], "nope.png", DEFS, reference="missing.png")
    assert any("missing" in p for p in problems) and any("reference" in p for p in problems)


def _panels(art):
    _, recs = art
    return build_panels(agg.select_runs(recs), "reconstruction.png", DEFS, reference="ground_truth.png",
                        measurement="measurement.png")


def test_zoom_region_and_luminance():
    assert zoom_region((100, 100), 0.3, (0.5, 0.5)) == (35, 35, 30, 30)
    assert zoom_region((100, 60), 0.5, (0.0, 1.0)) == (0, 70, 30, 30)  # clamped inside
    rgb = np.zeros((4, 4, 3), dtype=np.float32); rgb[..., 1] = 1.0
    assert np.allclose(luminance(rgb), 0.587)
    assert luminance(np.ones((2, 2))).shape == (2, 2)


def test_reconstruction_figure_image_mode_with_zoom(art, tmp_path):
    panels, ref, _ = _panels(art)
    methods = [p for p in panels if p.kind == "method"]
    meas = next(p for p in panels if p.kind == "measurement")
    fig, spec = reconstruction_figure(methods, reference=ref, measurement=meas, zoom=True, width="double")
    # 4 panels (GT, Meas, TV, Ours), each with one zoom inset (inset axes are children, not fig.axes)
    assert len(fig.axes) == 4 and sum(len(a.child_axes) for a in fig.axes) == 4
    assert fig.get_size_inches()[0] == pytest.approx(IEEE_TEXTWIDTH_IN)
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert titles == ["Reference", "Measurement", "TV [1]", "Ours"]
    assert spec.crop_box == zoom_region(ref.image.shape) and spec.mode == "image" and spec.error_vmax is None
    # metric stamps present on method panels (upper-left because of the zoom inset)
    stamps = [t for a in fig.axes for t in a.texts if "dB" in t.get_text()]
    assert len(stamps) == 2 and all(t.get_position()[1] > 0.5 for t in stamps)
    # yellow zoom boxes on every panel
    from matplotlib.patches import Rectangle
    boxes = [pch for a in fig.axes for pch in a.patches if isinstance(pch, Rectangle)]
    assert len(boxes) == 4
    out = save_visual(fig, tmp_path / "vis.png", spec)
    assert [p.suffix for p in out] == [".png", ".json"]
    side = json.loads(out[1].read_text())
    assert side["mode"] == "image" and side["panels"][2]["path"].endswith("TV/reconstruction.png")
    assert "Yellow box" in spec.caption_stub() and "Left to right: Reference, Measurement, TV [1], Ours" in spec.caption_stub()


def test_reconstruction_figure_error_mode(art):
    panels, ref, _ = _panels(art)
    methods = [p for p in panels if p.kind == "method"]
    meas = next(p for p in panels if p.kind == "measurement")
    fig, spec = reconstruction_figure(methods, reference=ref, measurement=meas, mode="error")
    # 4 panels + colour bar
    assert len(fig.axes) == 5
    assert spec.error_vmax is not None and 0 < spec.error_vmax <= 1
    assert "luminance" in spec.caption_stub() and "99th percentile" in spec.caption_stub()
    err_axes = [a for a in fig.axes if a.get_title() in ("TV [1]", "Ours")]
    ims = [a.images[0] for a in err_axes]
    assert all(im.get_clim() == (0.0, spec.error_vmax) for im in ims)  # one shared scale
    assert ims[0].get_array().mean() > ims[1].get_array().mean()  # TV is worse
    with pytest.raises(ValueError):
        reconstruction_figure(methods, mode="error")  # needs a reference


def test_reconstruction_figure_rows_and_kernel(art):
    _, recs = art
    # two "rows": same runs relabelled as K=1 / K=5
    p1, ref, _ = _panels(art)
    methods = [p for p in p1 if p.kind == "method"]
    meas = next(p for p in p1 if p.kind == "measurement")
    kernel = Panel("Kernel", np.eye(5, dtype=np.float32), kind="kernel")
    rows = [PanelRow(methods, "$K = 1$"), PanelRow(methods, "$K = 5$")]
    fig, spec = reconstruction_figure(rows, reference=ref, measurement=meas, kernel=kernel, zoom=True)
    # left block stacked (2 axes) + 4 method panels; insets: 6 zoom + 1 kernel as child axes
    assert len(fig.axes) == 6 and sum(len(a.child_axes) for a in fig.axes) == 7
    assert spec.rows == ["$K = 1$", "$K = 5$"]
    ylabels = [a.get_ylabel() for a in fig.axes if a.get_ylabel()]
    assert ylabels == ["$K = 1$", "$K = 5$"]
    assert "Rows: $K = 1$, $K = 5$" in spec.caption_stub()


def test_build_rows_groups_by_key(art):
    _, recs = art
    more = [{**r, "run_id": 100 + i, "seed": 1, "metrics": {"psnr": 20.0}} for i, r in enumerate(recs[:2])]
    rows, problems = build_rows(recs[:2] + more, "seed", "reconstruction.png", DEFS, metrics=["psnr"])
    assert [r.label for r in rows] == ["$seed = 0$", "$seed = 1$"]
    assert [p.title for p in rows[0].panels] == ["TV [1]", "Ours"] and rows[1].panels[0].subtitle == "20.00 dB"
    assert problems == []


def test_reconstruction_figure_minimal_and_errors(art):
    _, recs = art
    panels, _, _ = build_panels(agg.select_runs(recs), "reconstruction.png", DEFS)
    fig, spec = reconstruction_figure(panels, width="single", show_titles=False)
    assert len(fig.axes) == 2 and spec.error_vmax is None and spec.crop_box is None
    with pytest.raises(ValueError):
        reconstruction_figure(panels, crop_box=(40, 40, 20, 20))
    with pytest.raises(ValueError):
        reconstruction_figure([])
    with pytest.raises(ValueError):
        reconstruction_figure(panels, mode="nope")


def test_latex_glue_and_grayscale():
    t = figure_tex("figures/fig_lambda.pdf", caption="cap", label="fig:l", width="single")
    assert t.startswith("\\begin{figure}[!t]") and "\\includegraphics[width=\\columnwidth]{figures/fig_lambda}" in t and "\\label{fig:l}" in t
    t2 = figure_tex("x.png", width="double")
    assert "figure*" in t2 and "\\textwidth" in t2
    assert "booktabs" in ieee_preamble() and "graphicx" in ieee_preamble()
    fig, _ = reconstruction_figure([Panel("a", np.zeros((8, 8)))], width="single", show_titles=False)
    g = to_grayscale_png(figure_bytes(fig, "png", dpi=50))
    assert Image.open(__import__("io").BytesIO(g)).mode == "L"


def test_rows_use_their_own_reference_for_error_maps(art, tmp_path):
    """Rows showing different images (e.g. instances) must be scored against their own ground truth."""
    p1, ref, _ = _panels(art)
    methods = [p for p in p1 if p.kind == "method"]
    other_ref = Panel("Reference", np.zeros_like(ref.image), kind="reference")  # a wrong reference: all-zero
    rows = [PanelRow(methods, "row A", reference=ref), PanelRow(methods, "row B", reference=other_ref)]
    fig, spec = reconstruction_figure(rows, reference=ref, mode="error")
    err_axes = [a for a in fig.axes if a.images and a.images[0].get_cmap().name == "magma"]
    means = [a.images[0].get_array().mean() for a in err_axes]
    assert len(means) == 4 and max(means[:2]) < min(means[2:])  # row B (zero reference) has far larger error
    # build_rows attaches per-row references when asked
    _, recs = art
    rws, _ = build_rows(recs[:2], "seed", "reconstruction.png", DEFS, reference="ground_truth.png")
    assert rws[0].reference is not None and rws[0].reference.kind == "reference"


def test_guess_roles():
    files = ["error_map.png", "ground_truth.png", "measurement.png", "reconstruction.png", "psf.png"]
    r = guess_roles(files)
    assert r == {"reconstruction": "reconstruction.png", "reference": "ground_truth.png",
                 "measurement": "measurement.png", "kernel": "psf.png"}
    r2 = guess_roles(["out/x_hat.png", "clean.png"])
    assert r2["reconstruction"] == "out/x_hat.png" and r2["reference"] == "clean.png" and r2["measurement"] is None
    assert guess_roles(["a.png"])["reconstruction"] == "a.png" and guess_roles([])["reconstruction"] is None


def test_panel_psnr_and_metrics_rows(art):
    gt, recs = art
    panels, ref, _ = _panels(art)
    methods = [p for p in panels if p.kind == "method"]
    # recomputed PSNR: Ours (sigma 0.02) beats TV (sigma 0.08); identical image -> inf
    assert panel_psnr(methods[1].image, ref.image) > panel_psnr(methods[0].image, ref.image)
    assert panel_psnr(ref.image, ref.image) == float("inf")
    with pytest.raises(ValueError):
        panel_psnr(ref.image, ref.image[:10])
    headers, rows, warns = panel_metrics_rows(recs, panels, ref, DEFS, metrics=["psnr", "ssim"])
    assert headers == ["Panel", "psnr (logged)", "ssim (logged)", "PSNR (from image)", "Δ (dB)"]
    assert [r[0] for r in rows] == ["TV [1]", "Ours"] and rows[0][1] == "30.00" and rows[0][2] == "0.900"
    # the fixture's logged numbers are invented, so they must be flagged against the recomputed ones
    assert len(warns) == 2 and all("recomputed" in w for w in warns)
    # no reference -> no computed columns, no warnings
    h2, r2, w2 = panel_metrics_rows(recs, panels, None, DEFS, metrics=["psnr"])
    assert h2 == ["Panel", "psnr (logged)"] and len(r2[0]) == 2 and w2 == []


def test_make_visual_records_the_seed_and_flags_mixed_seeds(art, tmp_path):
    from results_tracker.export.visual import make_visual
    gt, recs = art
    two = [r for r in recs if r.get("artifacts_dir")]
    # both methods have seed 0 only -> seed recorded on the spec and on every method panel
    vr = make_visual(two, DEFS, dataset="D")
    assert vr.spec.seed == 0 and all(p.get("seed") == 0 for p in vr.spec.panels if p["kind"] == "method")
    assert all(p.get("run_id") is not None for p in vr.spec.panels if p["kind"] == "method")
    assert vr.problems == [] and "seed 0" in vr.spec.caption_stub()
    # give the second method only seed 1 -> no common seed -> flagged
    two[1] = {**two[1], "seed": 1}
    vr2 = make_visual(two, DEFS, dataset="D")
    assert vr2.spec.seed is None and any("different seeds" in p for p in vr2.problems)
