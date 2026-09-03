"""Per-iteration curves, trade-off points, instance tables, selection tables, and their figures/asset kinds."""

import json
import math

import pytest

pytest.importorskip("matplotlib")

from results_tracker import aggregate as agg, get_engine, get_metric_defs, get_runs, run_records, save_asset  # noqa: E402
from results_tracker.curves import curve_names, curve_series, normalise, record_curves  # noqa: E402
from results_tracker.demo import PROJECT as DEMO, seed_demo  # noqa: E402
from results_tracker.export import paper  # noqa: E402
from results_tracker.export.figures import curves_figure, distribution_figure, tradeoff_figure  # noqa: E402
from results_tracker.export.latex import selection_latex  # noqa: E402
from results_tracker.recipe import Arm, Study, materialize_selection, run_study  # noqa: E402
from results_tracker.recipe.toy import PROJECT as TOY, toy_studies  # noqa: E402
from results_tracker.ui.charts import curves_lines, distribution_box, tradeoff_scatter  # noqa: E402


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("toy")
    engine = get_engine(tmp / "t.db")
    comparison, sweep, _ = toy_studies(str(tmp / "art"))
    run_study(comparison, engine=engine, log=None)
    run_study(sweep, engine=engine, log=None)
    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt} for k, m in get_metric_defs(engine=engine).items()}
    return engine, defs, lambda e: run_records(get_runs(experiment=e, engine=engine), engine=engine)


def test_curves_from_diagnostics(toy, tmp_path):
    engine, defs, recs = toy
    cmp_ = recs("main-comparison")
    assert curve_names(cmp_) == ["step_sizes"]  # the toy's adaptive method records a bare list; wiener/gd have no curves
    ours = [r for r in cmp_ if r["method"] == "adaptive-gd"]
    assert len(record_curves(ours[0])["step_sizes"]) == 30 and record_curves(next(r for r in cmp_ if r["method"] == "wiener")) == {}
    series = curve_series(cmp_, "step_sizes", group_by=["method"])
    assert list(series) == [("adaptive-gd",)] and series[("adaptive-gd",)].runs == 32 and len(series[("adaptive-gd",)].mean) == 30
    by_noise = curve_series(cmp_, "step_sizes", group_by=["config.noise"])
    assert {g[0] for g in by_noise} == {0.01, 0.05} and all(cs.n[0] == 16 for cs in by_noise.values())
    cs = series[("adaptive-gd",)]
    assert cs.final() is not None and cs.x == list(range(30)) and all(s >= 0 for s in cs.std)
    delta = normalise(cs, "delta")
    assert abs(delta.mean[0]) < 1e-12 and normalise(cs, "value") is cs
    # the adaptivePnP layout: a `curves` dict, unequal lengths and NaNs are pooled per iteration
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "diagnostics.json").write_text(json.dumps({"ok": True, "curves": {"psnr": [20.0, 25.0, None], "rho_k": [1, 2, 3, 4]}}))
    run_dir2 = tmp_path / "run2"
    run_dir2.mkdir()
    (run_dir2 / "diagnostics.json").write_text(json.dumps({"ok": True, "curves": {"psnr": [22.0, 27.0, 28.0, 29.0]}}))
    fake = [{"method": "m", "status": "completed", "metrics": {}, "config": {}, "artifacts_dir": str(run_dir)},
            {"method": "m", "status": "completed", "metrics": {}, "config": {}, "artifacts_dir": str(run_dir2)},
            {"method": "m", "status": "completed", "metrics": {}, "config": {}, "artifacts_dir": None}]
    assert set(curve_names(fake)) == {"psnr", "rho_k"}
    cs = curve_series(fake, "psnr")[()]
    assert cs.runs == 2 and cs.n == [2, 2, 1, 1] and cs.mean[:2] == [21.0, 26.0] and cs.mean[2] == 28.0 and cs.std[0] == pytest.approx(math.sqrt(2))
    assert curve_series(fake, "rho_k")[()].runs == 1
    assert normalise(cs, "ratio").mean[0] == pytest.approx(1.0)
    # figures
    fig = curves_figure(series, "step_sizes", caption="a. step size", guide=1.0)
    assert fig.axes[0].get_ylabel() == "step_sizes" and fig.axes[0].get_lines()
    fig2 = curves_lines(by_noise, "step_sizes", members=True, log_y=True)
    assert fig2.layout.yaxis.type == "log" and len(fig2.data) > 2
    # asset kind
    r = paper.render_asset({"label": "fig:curves", "kind": "curves-figure", "experiment": "main-comparison",
                            "options": {"curve": "step_sizes", "by": ["config.noise"], "normalise": "ratio"}}, cmp_, defs)
    assert not r.error and [f for f, _ in r.files] == ["figures/fig-curves.pdf", "figures/fig-curves.tex"] and "32 runs with curves" in r.note
    r = paper.render_asset({"label": "fig:x", "kind": "curves-figure", "experiment": "x", "options": {"curve": "nope"}}, cmp_, defs)
    assert "no run has" in r.error


def test_tradeoff_points_and_figure(tmp_path):
    db = tmp_path / "d.db"
    seed_demo(db=db)
    engine = get_engine(db)
    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt} for k, m in get_metric_defs(engine=engine).items()}
    recs = run_records(get_runs(experiment="main-comparison", engine=engine), engine=engine)
    pts = agg.tradeoff_points(recs, "runtime_s", "psnr")
    assert set(pts) == {"TV", "PnP-BM3D", "Ours"} and all(len(v) == 1 for v in pts.values())  # DPIR has no runtime
    pts = agg.tradeoff_points(recs, "runtime_s", "psnr", path_key="dataset")
    assert [p.label for p in pts["Ours"]] == ["CBSD68", "Set12"] and pts["Ours"][0].x.n == 3
    fig = tradeoff_figure(pts, "runtime_s", "psnr", hollow=["TV"], caption="a. cost vs quality")
    assert fig.axes[0].get_xscale() == "log" and len(fig.axes[0].containers) == 3
    fig2 = tradeoff_scatter(pts, "runtime_s", "psnr", hollow=["TV"])
    assert fig2.layout.xaxis.type == "log" and [t.name for t in fig2.data] == ["TV", "PnP-BM3D", "Ours"]
    r = paper.render_asset({"label": "fig:tradeoff", "kind": "tradeoff-figure", "experiment": "main-comparison",
                            "options": {"x_metric": "runtime_s", "y_metric": "psnr", "path": "dataset"}}, recs, defs)
    assert not r.error and r.files[0][0] == "figures/fig-tradeoff.pdf"
    r = paper.render_asset({"label": "fig:x", "kind": "tradeoff-figure", "experiment": "x", "options": {"x_metric": "nope"}}, recs, defs)
    assert r.error


def test_instance_table_gains_and_distribution(toy):
    engine, defs, recs = toy
    cmp_ = recs("main-comparison")
    table = agg.instance_table(cmp_, "psnr")
    assert table.methods == ["wiener", "gd", "adaptive-gd"] and len(table.instances) == 4  # 4 phantoms, seeds and conditions pooled
    assert all(table.stat(i, "wiener").n == 8 for i in table.instances)  # 2 blur x 2 noise x 2 seeds
    assert all(table.best_method(i) == "adaptive-gd" for i in table.instances)
    gains = agg.instance_gains(table, "adaptive-gd", "wiener")
    assert len(gains) == 4 and gains[0].gain >= gains[-1].gain > 0
    assert agg.instance_table([], "psnr").methods == []
    fig = distribution_figure({m: table.values(m) for m in table.methods}, "psnr", emphasize=["adaptive-gd"], caption="b. per image")
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["wiener", "gd", "adaptive-gd"]
    fig2 = distribution_box({m: table.values(m) for m in table.methods}, "psnr")
    assert len(fig2.data) == 3
    r = paper.render_asset({"label": "fig:dist", "kind": "distribution-figure", "experiment": "main-comparison", "options": {"metric": "psnr"}}, cmp_, defs)
    assert not r.error and "4 instances" in r.note
    r = paper.render_asset({"label": "fig:x", "kind": "distribution-figure", "experiment": "x", "options": {"metric": "psnr"}},
                           [{**x, "instance": None} for x in cmp_], defs)
    assert "runs need an `instance`" in r.error


def test_selection_table_latex_and_materialize(toy):
    engine, defs, recs = toy
    sweep = recs("reg-sweep")
    sel = agg.selection_table(sweep, "reg", "psnr")
    assert len(sel) == 1 and sel[0].group == () and sel[0].grid == [0.0003, 0.001, 0.003, 0.01, 0.03]
    assert sel[0].best in sel[0].grid and sel[0].runner_up in sel[0].grid and sel[0].margin >= 0
    by_seed = agg.selection_table(sweep, "reg", "psnr", group_by=["seed"])
    assert [s.group for s in by_seed] == [(0,), (1,), (2,)]
    boundary = agg.selection_table([r for r in sweep if r["config"]["reg"] >= 0.003], "reg", "psnr")[0]
    assert boundary.grid == [0.003, 0.01, 0.03] and boundary.at_boundary == (boundary.best in (0.003, 0.03))
    tex = selection_latex(by_seed, "reg", "psnr", ["seed"], defs, label="tab:sel", param_label="$\\lambda$")
    assert "\\label{tab:sel}" in tex and "$\\lambda$" in tex and tex.count("\\\\") == 4 and "\\begin{tabular}{lccl}" in tex  # header + 3 rows
    if boundary.at_boundary:
        assert "dagger" in selection_latex([boundary], "reg", "psnr", [], defs)
    r = paper.render_asset({"label": "tab:sel", "kind": "selection-table", "experiment": "reg-sweep",
                            "options": {"param": "reg", "metric": "psnr", "by": ["seed"]}}, sweep, defs)
    assert not r.error and r.files[0][0] == "tables/tab-sel.tex" and b"\\toprule" in r.files[0][1]
    # write the winner into a comparison spec
    template = Study("cmp", "comparison", "toy-deblur", [Arm("adaptive-gd"), Arm("gd", {"reg": 0.5}), Arm("wiener")], project=TOY)
    out = materialize_selection(template, "reg", {"adaptive-gd": sel[0].best, "gd": 0.1}, note="reg-sweep")
    assert out.methods[0].config == {"reg": sel[0].best} and out.methods[1].config == {"reg": 0.5} and out.methods[2].config == {}
    assert "reg materialized" in out.description and "from reg-sweep" in out.description and template.methods[0].config == {}
