import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
from PIL import Image  # noqa: E402

from results_tracker import aggregate as agg  # noqa: E402
from results_tracker.export.figures import figure_tex, ieee_preamble, to_grayscale_png, figure_bytes  # noqa: E402
from results_tracker.export.visual import (  # noqa: E402
    Panel, build_panels, crop, error_map, list_image_files, load_image, psnr, reconstruction_figure, save_visual,
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


def test_reconstruction_figure_layout_and_sidecar(art, tmp_path):
    gt, recs = art
    panels, ref, _ = build_panels(agg.select_runs(recs), "reconstruction.png", DEFS, reference="ground_truth.png",
                                  measurement="measurement.png")
    fig, spec = reconstruction_figure(panels, reference=ref, crop_box=(10, 10, 16, 16), error_maps=True, width="double")
    # 4 columns (ref + meas + 2 methods) x 3 rows + colour bar + 2 spacer axes
    assert len(fig.axes) == 4 * 3 + 1 + 2
    assert spec.error_vmax is not None and spec.error_vmax > 0
    assert [p["title"] for p in spec.panels] == ["Reference", "Measurement", "TV [1]", "Ours"]
    assert fig.get_size_inches()[0] == pytest.approx(7.16)
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert titles == ["Reference", "Measurement", "TV [1]", "Ours"]
    out = save_visual(fig, tmp_path / "vis.png", spec)
    assert [p.suffix for p in out] == [".png", ".json"]
    side = json.loads(out[1].read_text())
    assert side["crop_box"] == [10, 10, 16, 16] and side["panels"][2]["path"].endswith("TV/reconstruction.png")
    assert "Zoom: 16x16 crop at (10, 10)" in spec.caption_stub() and "shared scale" in spec.caption_stub()
    # error maps: the worse method has the larger mean error
    tv = error_map(panels[1].image, ref.image).mean()
    ours = error_map(panels[2].image, ref.image).mean()
    assert tv > ours


def test_reconstruction_figure_minimal_and_errors(art):
    _, recs = art
    panels, _, _ = build_panels(agg.select_runs(recs), "reconstruction.png", DEFS)
    fig, spec = reconstruction_figure(panels, width="single", panel_labels=True)
    assert len(fig.axes) == 2 and fig.axes[0].get_title().startswith("(a) ")
    assert spec.error_vmax is None
    with pytest.raises(ValueError):
        reconstruction_figure(panels, crop_box=(40, 40, 20, 20))
    with pytest.raises(ValueError):
        reconstruction_figure([])


def test_latex_glue_and_grayscale():
    t = figure_tex("figures/fig_lambda.pdf", caption="cap", label="fig:l", width="single")
    assert t.startswith("\\begin{figure}[!t]") and "\\includegraphics[width=\\columnwidth]{figures/fig_lambda}" in t and "\\label{fig:l}" in t
    t2 = figure_tex("x.png", width="double")
    assert "figure*" in t2 and "\\textwidth" in t2
    assert "booktabs" in ieee_preamble() and "graphicx" in ieee_preamble()
    fig, _ = reconstruction_figure([Panel("a", np.zeros((8, 8)))], width="single", show_titles=False)
    g = to_grayscale_png(figure_bytes(fig, "png", dpi=50))
    assert Image.open(__import__("io").BytesIO(g)).mode == "L"
