"""The per-project plot style: what it stores, and that one style drives the charts, the print figures and
the pinned assets."""

import pytest

from results_tracker import aggregate as agg
from results_tracker import plotstyle as ps
from results_tracker.api import get_plot_style, log_run, set_project
from results_tracker.plotstyle import VARIANT_KEY, PlotStyle


def stat(m, s=0.1, n=3):
    return agg.Stat(mean=m, std=s, n=n, min=m - s, max=m + s, values=[m] * n)


# --------------------------------------------------------------------------- the style object

def test_only_differences_are_stored_and_read_back():
    style = PlotStyle().merge(tick=20).with_colors({"ours": "#00AA00"}).with_order("method", ["ours", "dpir"])
    assert style.to_dict() == {"tick": 20.0, "colors": {"ours": "#00aa00"}, "order": {"method": ["ours", "dpir"]}}
    assert ps.from_dict(style.to_dict()) == style
    assert PlotStyle().is_default and not style.is_default


def test_a_hand_edited_style_never_breaks_the_gui():
    style = ps.from_dict({"tick": "not a number", "legend": 900, "line_width": -3, "math_bump": 0,
                          "colors": {"a": "red", "b": "#ABCDEF"}, "order": {"method": [], "dataset": ["Set12", 5]},
                          "unknown": 1})
    assert style.tick == PlotStyle().tick                      # unusable value ignored
    assert style.legend == ps.SIZE_LIMITS[1]                   # clamped, not applied literally
    assert style.line_width == ps.WEIGHT_LIMITS[0]
    assert style.math_bump == 0.0                              # the one size that may be zero
    assert style.colors == {"b": "#abcdef"}                    # only #rrggbb survives
    assert style.order == {"dataset": ["Set12", "5"]}          # an empty order is no order; values are text
    assert ps.from_dict(None).is_default and ps.from_dict({}).is_default


def test_screen_sizes_and_emphasis_weights():
    style = PlotStyle()
    assert style.screen().marker_size == 7.0 and style.screen().weights(True)[1] == 9.0
    assert style.weights(True) > style.weights(False)  # the proposed method stays heavier
    assert PlotStyle(marker_size=7.0).screen().marker_size == 14.0
    assert style.label_size("$\\lambda$") == style.axis_label + style.math_bump
    assert style.label_size("PSNR (dB)") == style.axis_label


def test_colours_follow_the_order_unless_pinned():
    style = PlotStyle().with_colors({"b": "#123456"})
    assert style.colors_for(["a", "b", "c"]) == {"a": ps.PALETTE[0], "b": "#123456", "c": ps.PALETTE[2]}
    # pinning b does not shift a and c off their palette slots
    assert style.palette_colors(["a", "b", "c"])["b"] == ps.PALETTE[1]
    assert style.with_colors({"b": None}).colors == {}
    assert style.colors_for([("dpir", "Set12")]) == {("dpir", "Set12"): ps.PALETTE[0]}


def test_past_the_palette_the_hue_repeats_and_the_shape_changes():
    """Screen and print must agree on what a ninth series looks like: the same hue as the first, told apart
    by its marker (and, on paper, by a dashed line) rather than by nothing at all."""
    from results_tracker.export.figures import style_map
    from results_tracker.ui import charts

    names = [f"m{i}" for i in range(10)]
    hues = PlotStyle().colors_for(names)
    assert hues["m8"] == hues["m0"] and hues["m9"] == hues["m1"]  # wrapped, not greyed out
    assert PlotStyle().wrap_of(names)["m7"] == 0 and PlotStyle().wrap_of(names)["m8"] == 1

    printed = style_map(names)
    assert printed["m0"]["marker"] == "o" and printed["m0"]["linestyle"] == "-"   # the usual look is untouched
    assert printed["m8"]["marker"] != printed["m0"]["marker"]
    assert printed["m8"]["linestyle"] != printed["m0"]["linestyle"]
    assert printed["m8"]["color"] == printed["m0"]["color"]

    screen = charts.marker_for(names)
    assert screen["m0"] == "circle" and screen["m8"] != "circle"
    assert charts.color_for(names) == hues  # the same hues as the print figure, not a grey fallback


def test_order_is_by_text_and_keeps_unlisted_values_at_the_end():
    style = PlotStyle().with_order("config.K", [5, 2])
    assert style.ordered("config.K", [2, 5, 10, 20]) == [5, 2, 10, 20]
    assert style.ordered("config.K", ["2", "5"]) == ["5", "2"]  # 5 and "5" are the same value
    assert style.ordered("method", ["b", "a"]) == ["b", "a"]    # no order declared: untouched
    assert style.with_order("config.K", []).order == {}


@pytest.mark.parametrize("text,expected", [
    ("28,34", (28.0, 34.0)), (" 28 ; 34 ", (28.0, 34.0)), ([1, 2], (1.0, 2.0)), ("-1,1", (-1.0, 1.0)),
    ("", None), (None, None), ("28", None), ("a,b", None), ("5,5", None), ("1,2,3", None), (True, None),
])
def test_parse_limits(text, expected):
    assert ps.parse_limits(text) == expected


# --------------------------------------------------------------------------- print figures

def test_style_reaches_every_part_of_a_print_figure():
    from results_tracker.export.figures import ieee_rc, sweep_figure

    style = PlotStyle(tick=20, axis_label=18).with_colors({"A": "#00aa00"}).with_order("method", ["B", "A"])
    rc = ieee_rc(style)
    assert rc["xtick.labelsize"] == 20 and rc["axes.labelsize"] == 18 and rc["font.family"] == "serif"
    series = {("A",): [(1, stat(1.0)), (2, stat(2.0))], ("B",): [(1, stat(3.0)), (2, stat(2.5))]}
    fig = sweep_figure(series, "k", "psnr", style=style, by=["method"], ylim=(0.0, 5.0), xlim=(0.5, 2.5))
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["B", "A"]  # declared order
    assert dict(zip(("B", "A"), (line.get_color() for line in ax.lines)))["A"] == "#00aa00"
    assert ax.get_ylim() == (0.0, 5.0) and ax.get_xlim() == (0.5, 2.5)
    assert ax.get_xticklabels()[0].get_fontsize() == 20


def test_a_categorical_axis_follows_the_declared_order():
    from results_tracker.export.figures import sweep_figure

    series = {(): [("none", stat(27.0)), ("op_norm", stat(28.0)), ("cr_bound", stat(28.5))]}
    plain = sweep_figure(series, "rho_floor", "psnr")
    assert [t.get_text() for t in plain.axes[0].get_xticklabels()] == ["cr_bound", "none", "op_norm"]
    style = PlotStyle().with_order("rho_floor", ["none", "op_norm", "cr_bound"])
    fig = sweep_figure(series, "rho_floor", "psnr", style=style)
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["none", "op_norm", "cr_bound"]


def test_comparison_and_distribution_figures_take_order_colour_and_range():
    from results_tracker.export.figures import comparison_figure, distribution_figure

    recs = [{"method": m, "dataset": d, "seed": 0, "config": {}, "metrics": {"psnr": v}, "status": "completed"}
            for m, d, v in [("A", "D1", 30.0), ("A", "D2", 31.0), ("B", "D1", 28.0), ("B", "D2", 29.0)]]
    pt = agg.pivot_table(recs, "method", "dataset", metrics=["psnr"])
    style = PlotStyle().with_order("dataset", ["D2", "D1"]).with_colors({"A": "#00aa00"})
    fig = comparison_figure(pt, "psnr", style=style, rows_key="method", cols_key="dataset", ylim=(25.0, 35.0))
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["D2", "D1"]
    assert ax.get_ylim() == (25.0, 35.0)
    assert ax.containers[0].patches[0].get_facecolor()[:3] == pytest.approx((0.0, 2 / 3, 0.0), abs=0.01)
    box = distribution_figure({"A": [1.0, 2.0], "B": [2.0, 3.0]}, "psnr",
                              style=PlotStyle().with_order("method", ["B", "A"]), ylim=(0.0, 4.0))
    assert [t.get_text() for t in box.axes[0].get_xticklabels()] == ["B", "A"]
    assert box.axes[0].get_ylim() == (0.0, 4.0)


def test_ablation_bars_can_be_ordered_by_hand():
    from results_tracker.export.figures import ablation_figure

    recs = [{"method": "m", "seed": s, "config": {"a": on_a, "b": on_b}, "status": "completed", "tags": tags,
             "metrics": {"psnr": v}}
            for on_a, on_b, v, tags in [(True, True, 30.0, ["base"]), (False, True, 28.0, []), (True, False, 29.0, [])]
            for s, v in [(0, v), (1, v + 0.1)]]
    rows = agg.ablation_table(recs, metrics=["psnr"])
    ranked = ablation_figure(rows, "psnr")
    assert [t.get_text() for t in ranked.axes[0].get_yticklabels()] == ["w/o a", "w/o b"]  # by effect size: -2.0 then -1.0
    style = PlotStyle().with_order(VARIANT_KEY, ["w/o b", "w/o a"])
    fig = ablation_figure(rows, "psnr", style=style, xlim=(-3.0, 3.0))
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["w/o b", "w/o a"]
    assert fig.axes[0].get_xlim() == (-3.0, 3.0)


# --------------------------------------------------------------------------- on-screen charts

def test_screen_charts_use_the_same_style():
    pytest.importorskip("plotly")
    from results_tracker.ui import charts

    style = PlotStyle(legend=20, tick=16).with_colors({"A": "#00aa00"}).with_order("method", ["B", "A"])
    series = {("A",): [(1, stat(1.0)), (2, stat(2.0))], ("B",): [(1, stat(3.0)), (2, stat(2.5))]}
    fig = charts.sweep_lines(series, "k", "psnr", style=style, by=["method"], ylim=(0.0, 5.0))
    lines = [t for t in fig.data if t.mode == "lines+markers"]
    assert [t.name for t in lines] == ["B", "A"]
    assert dict((t.name, t.line.color) for t in lines)["A"] == "#00aa00"
    assert fig.layout.legend.font.size == pytest.approx(20 * charts.plotstyle.SCREEN_FONT_SCALE)
    assert fig.layout.yaxis.tickfont.size == pytest.approx(16 * charts.plotstyle.SCREEN_FONT_SCALE)
    assert list(fig.layout.yaxis.range) == [0.0, 5.0]


def test_a_range_on_a_log_axis_is_given_in_data_units():
    pytest.importorskip("plotly")
    from results_tracker.ui import charts

    series = {(): [(0.01, stat(29.0)), (0.1, stat(31.0)), (1.0, stat(29.5))]}
    fig = charts.sweep_lines(series, "lambda", "psnr", log_x=True, xlim=(0.01, 1.0))
    assert list(fig.layout.xaxis.range) == [-2.0, 0.0]  # plotly wants log10 bounds
    plain = charts.sweep_lines(series, "lambda", "psnr", log_x=True, xlim=(0.0, 1.0))
    assert plain.layout.xaxis.range is None  # a non-positive bound has no log: leave it automatic


# --------------------------------------------------------------------------- storage and exports

def test_style_round_trips_through_the_project(engine):
    log_run("e", project="p", method="m", seed=0, config={}, metrics={"psnr": 30.0}, engine=engine, git_commit=None)
    assert get_plot_style("p", engine=engine).is_default
    set_project("p", plot_style=PlotStyle().merge(tick=20).to_dict(), engine=engine)
    assert get_plot_style("p", engine=engine).tick == 20.0
    set_project("p", plot_style={}, engine=engine)
    assert get_plot_style("p", engine=engine).is_default
    assert get_plot_style("no-such-project", engine=engine).is_default


def test_an_older_database_gains_the_column(tmp_path):
    """A database written before the style existed must open, not crash (db.add_missing_columns)."""
    import sqlite3

    from results_tracker.db import get_engine

    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE project (id INTEGER PRIMARY KEY, name TEXT, description TEXT, primary_metric TEXT, "
                "studies_dir TEXT, created_at TIMESTAMP)")
    con.execute("INSERT INTO project (name, description, primary_metric, studies_dir) VALUES ('p', '', '', '')")
    con.commit()
    con.close()
    engine = get_engine(path)
    assert get_plot_style("p", engine=engine).is_default
    set_project("p", plot_style={"tick": 20.0}, engine=engine)
    assert get_plot_style("p", engine=engine).tick == 20.0


def test_a_pinned_figure_keeps_its_axis_range_and_the_project_style(engine):
    from results_tracker.api import save_asset
    from results_tracker.export.paper import render_paper

    for k in (2, 5, 10):
        for seed in (0, 1):
            log_run("sweep-k", project="p", experiment_type="sweep", method="ours", dataset="Set12", seed=seed,
                    config={"K": k}, metrics={"psnr": 28 + k * 0.1 + 0.05 * seed}, engine=engine, git_commit=None)
    set_project("p", plot_style=PlotStyle(tick=21).with_colors({"ours": "#00aa00"}).to_dict(), engine=engine)
    save_asset("p", "fig:k", kind="sweep-figure", experiment="sweep-k",
               options={"param": "K", "metric": "psnr", "by": ["method"], "ylim": [28.0, 29.5], "xlim": [1.0, 12.0]},
               engine=engine)
    rendered = render_paper(engine, "p", source="test")
    assert [r.error for r in rendered] == [""]
    assert rendered[0].main_file == "figures/fig-k.pdf" and rendered[0].files[0][1][:4] == b"%PDF"
    # the same spec, rendered by hand, must place the axes and the colour where the style says
    from results_tracker.export.figures import sweep_figure

    series = {("ours",): [(2, stat(28.2)), (5, stat(28.5)), (10, stat(29.0))]}
    fig = sweep_figure(series, "K", "psnr", style=get_plot_style("p", engine=engine), by=["method"],
                       xlim=(1.0, 12.0), ylim=(28.0, 29.5))
    ax = fig.axes[0]
    assert ax.get_xlim() == (1.0, 12.0) and ax.get_ylim() == (28.0, 29.5)
    assert ax.lines[0].get_color() == "#00aa00" and ax.get_yticklabels()[0].get_fontsize() == 21
