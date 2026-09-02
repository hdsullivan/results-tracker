import pytest

pytest.importorskip("plotly")

from results_tracker import aggregate as agg  # noqa: E402
from results_tracker.ui import charts  # noqa: E402


def stat(m, s=0.1, n=3):
    return agg.Stat(mean=m, std=s, n=n, min=m - s, max=m + s, values=[m] * n)


def test_sweep_lines_single_group_has_band_and_best_marker():
    series = {(): [(0.01, stat(29.0)), (0.1, stat(31.0)), (1.0, stat(29.5))]}
    fig = charts.sweep_lines(series, "lambda", "psnr", log_x=True, best_by_group={(): 0.1})
    assert len(fig.data) == 3  # upper band, lower band, line
    line = fig.data[-1]
    assert list(line.marker.size) == [8, 12, 8]
    assert fig.layout.xaxis.type == "log"
    assert any("best lambda=0.1" in (a.text or "") for a in fig.layout.annotations)


def test_sweep_lines_multi_group_colors_follow_entity():
    series = {("A",): [(1, stat(1.0)), (2, stat(2.0))], ("B",): [(1, stat(3.0))]}
    fig = charts.sweep_lines(series, "k", "psnr", band=False)
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == ["A", "B"]
    assert lines[0].line.color == charts.PALETTE[0] and lines[1].line.color == charts.PALETTE[1]
    assert lines[0].error_y.visible


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
    fig = charts.ablation_deltas(["w/o a"], [-1.0], [0.1], "rmse", higher_is_better=False)
    assert list(fig.data[0].marker.color) == [charts.DIVERGING_POS]  # lower rmse = improvement
