import pytest

pytest.importorskip("plotly")

from results_tracker import aggregate as agg  # noqa: E402
from results_tracker.ui import charts  # noqa: E402


def stat(m, s=0.1, n=3):
    return agg.Stat(mean=m, std=s, n=n, min=m - s, max=m + s, values=[m] * n)


def test_sweep_lines_single_group_has_band_and_best_marker():
    series = {(): [(0.01, stat(29.0)), (0.1, stat(31.0)), (1.0, stat(29.5))]}
    fig = charts.sweep_lines(series, "lambda", "psnr", log_x=True, best_by_group={(): 0.1})
    assert len(fig.data) == 4  # upper band, lower band, line, best ring
    line = fig.data[2]
    assert line.marker.size == 7 and line.line.color == charts.PALETTE[0]
    rings = charts.best_marker_traces(fig)
    assert len(rings) == 1 and list(rings[0].x) == [0.1]
    assert fig.layout.xaxis.type == "log"
    assert any("best lambda = 0.1" in (a.text or "") for a in fig.layout.annotations)
    assert any(s.line.dash == "dot" for s in fig.layout.shapes)  # dotted guide
    # paper look: boxed axes, inward mirrored ticks, no grid, serif font, white paper
    assert fig.layout.xaxis.mirror == "ticks" and fig.layout.xaxis.ticks == "inside" and fig.layout.xaxis.showgrid is False
    assert fig.layout.yaxis.showline is True and fig.layout.paper_bgcolor == "white"
    assert "Times" in fig.layout.font.family


def test_sweep_lines_multi_group_colors_follow_entity():
    series = {("A",): [(1, stat(1.0)), (2, stat(2.0))], ("B",): [(1, stat(3.0))]}
    fig = charts.sweep_lines(series, "k", "psnr", band=False, emphasize=["B"])
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == ["A", "B"]
    assert lines[0].line.color == charts.PALETTE[0] and lines[1].line.color == charts.PALETTE[1]
    assert lines[0].error_y.visible
    assert lines[1].line.width > lines[0].line.width  # emphasised (proposed) method is heavier
    assert fig.layout.legend.bordercolor == "black" and fig.layout.legend.orientation == "h"


def test_is_log_friendly():
    assert charts.is_log_friendly([0.01, 0.1, 1.0])
    assert not charts.is_log_friendly([1, 2, 3])
    assert not charts.is_log_friendly([0, 1, 10])
    assert not charts.is_log_friendly(["a", "b", "c"])


def test_sweep_heatmap_orientation():
    fig = charts.sweep_heatmap([1, 2], [10, 20], [[1.0, 2.0], [3.0, 4.0]], "x", "y", "psnr", best=(2, 20))
    hm = fig.data[0]
    assert list(hm.x) == ["1", "2"] and list(hm.y) == ["10", "20"]
    assert hm.text[1][1] == "4.00"
    assert fig.data[1].marker.symbol == "square-open"


def test_ablation_deltas_polarity_colors():
    fig = charts.ablation_deltas(["w/o a", "w/o b", "w/o c"], [-1.0, 0.5, None], [0.1, 0.1, 0.1], "psnr", higher_is_better=True)
    bar = fig.data[0]
    assert list(bar.y) == ["w/o a", "w/o b"]  # None dropped
    assert list(bar.marker.color) == [charts.DIVERGING_NEG, charts.DIVERGING_POS]
    assert bar.marker.line.color == "black"
    fig = charts.ablation_deltas(["w/o a"], [-1.0], [0.1], "rmse", higher_is_better=False)
    assert list(fig.data[0].marker.color) == [charts.DIVERGING_POS]  # lower rmse = improvement


def test_comparison_bars_skip_missing_and_tight_range():
    recs = [{"method": m, "dataset": d, "seed": 0, "config": {}, "metrics": {"psnr": v}, "status": "completed"}
            for m, d, v in [("A", "D1", 30.0), ("A", "D2", 31.0), ("B", "D1", 28.0)]]
    ct = agg.comparison_table(recs, group_by=["method", "dataset"])
    fig = charts.comparison_bars(ct, "psnr")
    assert [t.name for t in fig.data] == ["A", "B"]
    assert list(fig.data[1].x) == ["D1"]  # B has no D2 bar
    lo, hi = fig.layout.yaxis.range
    assert lo > 20 and hi > 31
    assert fig.data[0].marker.line.color == "black"
    ct1 = agg.comparison_table(recs, group_by=["method"])
    fig1 = charts.comparison_bars(ct1, "psnr", zero_based=True)
    assert fig1.layout.yaxis.range[0] == 0 and len(fig1.data) == 2
