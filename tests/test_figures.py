import pytest

pytest.importorskip("matplotlib")

from results_tracker import aggregate as agg  # noqa: E402
from results_tracker.export.figures import (  # noqa: E402
    DOUBLE_COL_IN, SINGLE_COL_IN, ablation_figure, comparison_figure, figure_bytes, save_figure, style_map, sweep_figure,
)


def stat(m, s=0.1, n=3):
    return agg.Stat(mean=m, std=s, n=n, min=m - s, max=m + s, values=[m] * n)


def test_style_map_is_stable_and_distinct():
    sm = style_map(["A", "B", "C"], emphasize=["C"])
    assert sm["A"]["color"] != sm["B"]["color"] and sm["A"]["linestyle"] != sm["B"]["linestyle"]
    assert sm["C"]["linewidth"] > sm["A"]["linewidth"]
    assert style_map(["B", "A"])["B"]["color"] == sm["A"]["color"]  # first-seen order


def test_sweep_figure_log_axis_ticks_and_size(tmp_path):
    series = {(): [(0.01, stat(29.0)), (0.1, stat(31.0)), (1.0, stat(29.5))]}
    fig = sweep_figure(series, "lambda", "psnr", xlabel=r"$\lambda$", ylabel="PSNR (dB)", best_by_group={(): 0.1})
    ax = fig.axes[0]
    assert ax.get_xscale() == "log"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["0.01", "0.1", "1"]
    assert ax.get_ylabel() == "PSNR (dB)"
    assert fig.get_size_inches()[0] == pytest.approx(SINGLE_COL_IN)
    assert ax.get_legend() is None  # single series: title/labels name it
    paths = save_figure(fig, tmp_path / "s.pdf", also_png=True)
    assert [p.suffix for p in paths] == [".pdf", ".png"] and all(p.stat().st_size > 1000 for p in paths)


def test_sweep_figure_multi_group_legend_and_errorbars():
    series = {("A",): [(1, stat(1.0)), (2, stat(2.0))], ("B",): [(1, stat(3.0)), (2, stat(2.5))]}
    fig = sweep_figure(series, "k", "psnr", band=False, width="double", log_x=False)
    ax = fig.axes[0]
    assert ax.get_xscale() == "linear"
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["A", "B"]
    assert fig.get_size_inches()[0] == pytest.approx(DOUBLE_COL_IN)


def test_ablation_figure_bars_and_polarity():
    rows = [
        agg.AblationRow("full model", {}, 3, {"psnr": stat(31.0)}, {"psnr": 0.0}, is_base=True),
        agg.AblationRow("w/o a", {"a": (True, False)}, 3, {"psnr": stat(30.0)}, {"psnr": -1.0}),
        agg.AblationRow("bigger b", {"b": (1, 2)}, 3, {"psnr": stat(31.4)}, {"psnr": 0.4}),
    ]
    fig = ablation_figure(rows, "psnr")
    ax = fig.axes[0]
    bars = [p for p in ax.patches if p.get_height() > 0.5]  # the barh rectangles
    assert len(bars) == 2
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["w/o a", "bigger b"]  # sorted worst first
    hatched = [b for b in bars if b.get_hatch()]
    assert len(hatched) == 1  # the hurting one


def test_comparison_figure_grouped_bars():
    recs = []
    for m, base in [("TV", 27.0), ("Ours", 31.0)]:
        for d, off in [("Set12", 0.5), ("CBSD68", -0.5)]:
            for s in range(2):
                recs.append({"method": m, "dataset": d, "seed": s, "config": {}, "metrics": {"psnr": base + off + 0.1 * s},
                             "status": "completed"})
    pt = agg.pivot_table(recs, "method", "dataset", metrics=["psnr"])
    fig = comparison_figure(pt, "psnr", ylabel="PSNR (dB)", emphasize=["Ours"])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["TV", "Ours"]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["CBSD68", "Set12"]
    assert len([p for p in ax.patches if p.get_width() > 0.1]) == 4
    assert figure_bytes(fig, "png", dpi=72)[:4] == b"\x89PNG"


def test_saved_png_is_not_clipped(tmp_path):
    """Long y labels must be inside the saved image (tight bbox applied at save time)."""
    from PIL import Image

    rows = [
        agg.AblationRow("full model", {}, 3, {"psnr": stat(31.0)}, {"psnr": 0.0}, is_base=True),
        agg.AblationRow("a very long variant description here", {"a": (True, False)}, 3, {"psnr": stat(30.0)}, {"psnr": -1.0}),
    ]
    fig = ablation_figure(rows, "psnr")
    (p,) = save_figure(fig, tmp_path / "a.png", dpi=100)
    img = Image.open(p)
    # tight bbox grows the canvas beyond the nominal 3.5 in x 100 dpi = 350 px when labels stick out
    assert img.width > 350


def test_comparison_figure_skips_missing_cells_and_tight_ylim():
    recs = [{"method": m, "dataset": d, "seed": 0, "config": {}, "metrics": {"psnr": v}, "status": "completed"}
            for m, d, v in [("A", "D1", 30.0), ("A", "D2", 31.0), ("B", "D1", 28.0)]]  # B missing on D2
    pt = agg.pivot_table(recs, "method", "dataset", metrics=["psnr"])
    fig = comparison_figure(pt, "psnr")
    ax = fig.axes[0]
    bars = [p for p in ax.patches if p.get_width() > 0.1]
    assert len(bars) == 3  # not 4: the missing cell has no bar (and no zero-height bar)
    lo, hi = ax.get_ylim()
    assert lo > 20 and hi > 31  # data-tight, not from 0
    fig0 = comparison_figure(pt, "psnr", zero_based=True)
    assert fig0.axes[0].get_ylim()[0] == 0
    fig1 = comparison_figure(pt, "psnr", ylim=(25, 35))
    assert fig1.axes[0].get_ylim() == (25, 35)


def test_ieee_axes_style():
    series = {(): [(1, stat(1.0)), (2, stat(2.0)), (3, stat(1.5))]}
    fig = sweep_figure(series, "k", "psnr", log_x=False)
    ax = fig.axes[0]
    assert all(ax.spines[s].get_visible() for s in ("top", "right", "bottom", "left"))
    assert not any(l.get_visible() for l in ax.get_xgridlines())
    assert ax.yaxis.get_ticks_position() == "default"  # ticks on both sides (left + right) in IEEE style
    assert ax.xaxis.get_ticks_position() == "default"
    data_line = ax.containers[0].lines[0]  # errorbar container: (data line, caplines, barlines)
    assert data_line.get_markerfacecolor() == "white"
