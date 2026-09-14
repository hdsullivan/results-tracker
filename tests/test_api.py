from datetime import datetime

import pytest

from results_tracker import (
    DuplicateRunError,
    DuplicateRunWarning,
    ExperimentType,
    delete_runs,
    RunStatus,
    define_metric,
    get_metric_defs,
    get_runs,
    list_experiments,
    list_projects,
    log_run,
    run_records,
)
from results_tracker.api import guess_higher_is_better


class FakeNumpyScalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def test_log_run_creates_everything(engine):
    r = log_run(
        "exp1", project="paper", method="ours", dataset="Set12", seed=3,
        config={"lambda": 0.1, "solver": {"iters": 50}}, metrics={"psnr": 30.5, "rmse": 0.02},
        experiment_type="sweep", tags=["quick"], git_commit=None, hostname="h", engine=engine,
    )
    assert r.id is not None
    assert [p.name for p in list_projects(engine=engine)] == ["paper"]
    exps = list_experiments("paper", engine=engine)
    assert len(exps) == 1 and exps[0].type == ExperimentType.sweep
    recs = run_records(get_runs(experiment="exp1", engine=engine), engine=engine)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["method"] == "ours" and rec["dataset"] == "Set12" and rec["seed"] == 3
    assert rec["config"] == {"lambda": 0.1, "solver": {"iters": 50}}
    assert rec["metrics"] == {"psnr": 30.5, "rmse": 0.02}
    assert rec["tags"] == ["quick"] and rec["status"] == "completed"


def test_get_or_create_is_idempotent(engine):
    for s in range(3):
        log_run("e", project="p", method="m", dataset="d", seed=s, metrics={"psnr": s}, git_commit=None, engine=engine)
    assert len(list_projects(engine=engine)) == 1
    assert len(list_experiments(engine=engine)) == 1
    recs = run_records(get_runs(engine=engine), engine=engine)
    assert len(recs) == 3
    assert {r["method"] for r in recs} == {"m"}


def test_metric_defs_auto_created_with_direction_heuristic(engine):
    log_run("e", metrics={"psnr": 1, "rmse": 1, "runtime_s": 1, "ssim": 1}, git_commit=None, engine=engine)
    defs = get_metric_defs(engine=engine)
    assert defs["psnr"].higher_is_better and defs["ssim"].higher_is_better
    assert not defs["rmse"].higher_is_better and not defs["runtime_s"].higher_is_better


def test_define_metric_overrides_guess(engine):
    log_run("e", metrics={"score": 1}, git_commit=None, engine=engine)
    define_metric("score", unit="", higher_is_better=False, fmt=".3f", engine=engine)
    m = get_metric_defs(engine=engine)["score"]
    assert not m.higher_is_better and m.fmt == ".3f"


def test_numpy_like_scalars_and_nan(engine):
    log_run("e", metrics={"psnr": FakeNumpyScalar(29.9), "bad": float("nan")}, git_commit=None, engine=engine)
    rec = run_records(get_runs(engine=engine), engine=engine)[0]
    assert rec["metrics"] == {"psnr": 29.9, "bad": None}


def test_non_numeric_metric_rejected(engine):
    with pytest.raises(TypeError):
        log_run("e", metrics={"psnr": "high"}, git_commit=None, engine=engine)


def test_filters(engine):
    log_run("a", project="p1", method="m1", dataset="d1", metrics={"x": 1}, git_commit=None, engine=engine)
    log_run("a", project="p1", method="m2", dataset="d1", metrics={"x": 2}, git_commit=None, engine=engine)
    log_run("b", project="p2", method="m1", dataset="d2", metrics={"x": 3}, status="failed", git_commit=None, engine=engine)
    assert len(get_runs(experiment="a", engine=engine)) == 2
    assert len(get_runs(project="p2", engine=engine)) == 1
    assert len(get_runs(method="m1", engine=engine)) == 2
    assert len(get_runs(dataset="d1", engine=engine)) == 2
    assert len(get_runs(status=RunStatus.failed, engine=engine)) == 1
    assert len(get_runs(status="completed", engine=engine)) == 2


def test_explicit_timestamp_and_no_git(engine):
    ts = datetime(2026, 1, 2, 3, 4, 5)
    r = log_run("e", metrics={"x": 1}, timestamp=ts, git_commit=None, hostname=None, engine=engine)
    assert r.timestamp == ts and r.git_commit is None and r.hostname is None


def test_guess_direction():
    assert guess_higher_is_better("PSNR")
    assert not guess_higher_is_better("val_loss")
    assert not guess_higher_is_better("NRMSE")
    assert not guess_higher_is_better("time_per_iter")


def test_delete_runs(engine):
    ids = [log_run("e", method="m", seed=s, metrics={"x": s}, git_commit=None, engine=engine).id for s in range(3)]
    assert delete_runs([ids[0], 999], engine=engine) == 1  # unknown ids are ignored
    assert [r.id for r in get_runs(engine=engine)] == ids[1:]
    assert delete_runs([], engine=engine) == 0
    assert len(list_experiments(engine=engine)) == 1  # experiment kept


def _log3(engine, **kw):
    return [log_run("e", method="A", dataset="D", seed=s, config={"k": 1}, metrics={"psnr": 30 + s}, git_commit=None,
                    engine=engine, **kw) for s in range(3)]


def test_rerunning_a_script_does_not_inflate_n(engine):
    from results_tracker import aggregate as agg

    first = _log3(engine)
    with pytest.warns(DuplicateRunWarning):
        second = _log3(engine)
    assert [r.id for r in second] == [r.id for r in first]  # existing runs returned
    recs = run_records(get_runs(engine=engine), engine=engine)
    assert agg.comparison_table(recs, ["method"]).cells[("A",)]["psnr"].n == 3


def test_duplicate_policies(engine):
    _log3(engine)
    # a different config is not a duplicate
    log_run("e", method="A", dataset="D", seed=0, config={"k": 2}, metrics={"psnr": 1}, git_commit=None, engine=engine)
    assert len(get_runs(engine=engine)) == 4
    # replace: new values win, count unchanged
    r = log_run("e", method="A", dataset="D", seed=0, config={"k": 1}, metrics={"psnr": 99}, git_commit=None,
                engine=engine, on_duplicate="replace")
    assert len(get_runs(engine=engine)) == 4 and r.metrics == {"psnr": 99}
    # allow: appends
    log_run("e", method="A", dataset="D", seed=0, config={"k": 1}, metrics={"psnr": 5}, git_commit=None, engine=engine, on_duplicate="allow")
    assert len(get_runs(engine=engine)) == 5
    # error
    with pytest.raises(DuplicateRunError):
        log_run("e", method="A", dataset="D", seed=1, config={"k": 1}, metrics={"psnr": 5}, git_commit=None, engine=engine, on_duplicate="error")
    with pytest.raises(ValueError):
        log_run("e", metrics={"x": 1}, git_commit=None, engine=engine, on_duplicate="nope")


def test_failed_duplicate_is_superseded_by_a_rerun(engine):
    log_run("e", method="A", dataset="D", seed=0, config={"k": 1}, metrics={}, status="failed", git_commit=None, engine=engine)
    r = log_run("e", method="A", dataset="D", seed=0, config={"k": 1}, metrics={"psnr": 30}, git_commit=None, engine=engine)
    runs = get_runs(engine=engine)
    assert len(runs) == 1 and runs[0].id == r.id and runs[0].status.value == "completed"


def test_bool_metrics_are_rejected(engine):
    with pytest.raises(TypeError, match="bool"):
        log_run("e", metrics={"converged": True}, git_commit=None, engine=engine)
    r = log_run("e", metrics={"converged": 1}, git_commit=None, engine=engine)  # explicit 0/1 is fine
    assert r.metrics == {"converged": 1}


def test_record_timestamps_are_timezone_aware_utc(engine):
    from datetime import timezone

    log_run("e", metrics={"x": 1}, git_commit=None, engine=engine)
    rec = run_records(get_runs(engine=engine), engine=engine)[0]
    assert rec["timestamp"].tzinfo is not None and rec["timestamp"].utcoffset().total_seconds() == 0
    assert rec["timestamp"].astimezone().tzinfo is not None  # convertible to local time for display


def test_summaries_version_recent_and_notes(engine):
    from datetime import datetime, timezone

    from results_tracker import (add_note, delete_note, experiment_summaries, experiment_version, list_notes, log_run, recent_runs,
                                 set_experiment, set_project)

    v0 = experiment_version("e", project="p", engine=engine)
    assert v0 == (0, 0, "")
    log_run("e", project="p", method="a", dataset="d", seed=0, metrics={"psnr": 1.0}, engine=engine, git_commit=None)
    log_run("e", project="p", method="b", dataset="d", seed=0, metrics={"psnr": 2.0, "ssim": 0.5}, status="failed", engine=engine, git_commit=None)
    log_run("f", project="p", method="a", dataset="d2", seed=0, metrics={"rmse": 0.1}, experiment_type="sweep", engine=engine, git_commit=None)
    v1 = experiment_version("e", project="p", engine=engine)
    assert v1[0] == 2 and v1[1] > 0 and v1[2] and v1 != v0
    assert experiment_version("f", project="p", engine=engine)[0] == 1
    s = {x["experiment"]: x for x in experiment_summaries(engine=engine)}
    assert s["e"]["completed"] == 1 and s["e"]["failed"] == 1 and s["e"]["runs"] == 2 and s["e"]["methods"] == ["a", "b"]
    assert s["e"]["metrics"] == ["psnr", "ssim"] and s["e"]["datasets"] == ["d"] and s["e"]["type"] == "comparison"
    assert isinstance(s["e"]["last"], datetime) and s["e"]["last"].tzinfo == timezone.utc
    assert s["f"]["metrics"] == ["rmse"] and s["f"]["stage"] == ""
    assert [x["experiment"] for x in experiment_summaries("p", engine=engine)] == ["e", "f"] and experiment_summaries("nope", engine=engine) == []
    assert [r.experiment_id for r in recent_runs(2, engine=engine)] and len(recent_runs(2, engine=engine)) == 2
    # stage and studies dir
    set_experiment("e", project="p", stage="superseded", engine=engine)
    assert experiment_summaries("p", engine=engine)[0]["stage"] == "superseded"
    with pytest.raises(ValueError):
        set_experiment("e", project="p", stage="nonsense", engine=engine)
    assert set_project("p", studies_dir="~/studies", engine=engine).studies_dir == "~/studies"
    # notes
    n1 = add_note("p", "beta = 0.5: plateau", experiment="e", engine=engine)
    n2 = add_note("p", "main table on noise 0.01 only", asset_label="tab:main", engine=engine)
    with pytest.raises(ValueError):
        add_note("p", "   ", engine=engine)
    assert [n.id for n in list_notes("p", engine=engine)] == [n2.id, n1.id]
    assert [n.id for n in list_notes("p", asset_label="tab:main", engine=engine)] == [n2.id]
    assert [n.id for n in list_notes("p", experiment="e", engine=engine)] == [n1.id]
    assert delete_note(n1.id, engine=engine) and not delete_note(n1.id, engine=engine) and len(list_notes(engine=engine)) == 1


def test_query_runs_pages_filters_and_counts(engine):
    """The run browser's query: several statuses at once, a text search over what a run carries, newest
    first, one page at a time, and counts over everything that matches (not just the page)."""
    from datetime import timedelta, timezone

    from results_tracker.api import log_run, query_runs

    now = datetime.now(timezone.utc)
    for i in range(7):
        log_run("sweep-a", project="p", method="ours", dataset="D", seed=i, config={"K": i},
                metrics={"psnr": 30.0 + i}, timestamp=now - timedelta(minutes=i), engine=engine, git_commit=None)
    log_run("sweep-a", project="p", method="ours", dataset="D", seed=99, config={"K": 99}, metrics={},
            status="failed", notes="RuntimeError: CUDA out of memory", timestamp=now, engine=engine, git_commit=None)
    log_run("other", project="p", method="tv", dataset="D", seed=0, config={}, metrics={}, status="running",
            timestamp=now, engine=engine, git_commit=None)

    rows, counts = query_runs(project="p", limit=3, engine=engine)
    assert counts == {"completed": 7, "failed": 1, "running": 1, "total": 9}
    assert len(rows) == 3 and rows[0].timestamp >= rows[-1].timestamp  # newest first
    page2, _ = query_runs(project="p", limit=3, offset=3, engine=engine)
    assert not ({r.id for r in rows} & {r.id for r in page2})

    failed, counts = query_runs(project="p", statuses=["failed", "running"], engine=engine)
    assert {r.status.value for r in failed} == {"failed", "running"} and counts["total"] == 2

    hits, counts = query_runs(project="p", search="CUDA", engine=engine)
    assert counts["total"] == 1 and hits[0].notes.startswith("RuntimeError")
    assert query_runs(project="p", search="out of memory", engine=engine)[1]["total"] == 1
    assert query_runs(project="p", experiment="other", engine=engine)[1]["total"] == 1
    assert query_runs(project="p", method="tv", engine=engine)[1]["total"] == 1
    assert query_runs(project="nope", engine=engine)[1]["total"] == 0


def test_rename_experiment_follows_the_name_into_assets_and_notes(engine):
    """An experiment's name is stored by every asset that renders it and every note that mentions it; a rename
    that only touched the experiment row would silently unpin half the paper."""
    from results_tracker.api import add_note, get_asset, list_notes, log_run, rename_experiment, save_asset

    for seed in (0, 1):
        log_run("compare-k2", project="p", method="ours", dataset="D", seed=seed, config={"K": 2},
                metrics={"psnr": 30.0}, engine=engine, git_commit=None)
    log_run("main", project="p", method="ours", dataset="D", seed=0, config={}, metrics={"psnr": 31.0},
            engine=engine, git_commit=None)
    save_asset("p", "tab:main", kind="comparison-table", experiment="main", extra_experiments=["compare-k2"],
               options={}, engine=engine)
    save_asset("p", "fig:k2", kind="sweep-figure", experiment="compare-k2", options={"param": "K", "metric": "psnr"}, engine=engine)
    add_note("p", "K=2 is the paper's setting", experiment="compare-k2", engine=engine)

    touched = rename_experiment("p", "compare-k2", "compare-K2", engine=engine)
    assert touched == {"runs": 2, "assets": 2, "notes": 1, "merged": 0}
    assert get_asset("p", "fig:k2", engine=engine).experiment == "compare-K2"
    assert get_asset("p", "tab:main", engine=engine).extra_experiments == ["compare-K2"]
    assert list_notes("p", engine=engine)[0].experiment == "compare-K2"
    assert {e.name for e in list_experiments("p", engine=engine)} == {"compare-K2", "main"}


def test_renaming_onto_an_existing_experiment_merges_them(engine):
    from results_tracker.api import log_run, rename_experiment

    for name, seed in (("typo-run", 0), ("typo-run", 1), ("real", 2)):
        log_run(name, project="p", method="ours", dataset="D", seed=seed, config={}, metrics={"psnr": 30.0},
                engine=engine, git_commit=None)
    touched = rename_experiment("p", "typo-run", "real", engine=engine)
    assert touched["runs"] == 2 and touched["merged"] == 1
    assert {e.name for e in list_experiments("p", engine=engine)} == {"real"}
    assert len(get_runs(experiment="real", engine=engine)) == 3
    assert rename_experiment("p", "real", "real", engine=engine)["runs"] == 0  # a no-op rename changes nothing
    with pytest.raises(ValueError):
        rename_experiment("p", "gone", "whatever", engine=engine)


def test_delete_experiment_refuses_to_take_runs_by_accident(engine):
    from results_tracker.api import delete_experiment, log_run, save_asset

    log_run("scratch", project="p", method="ours", dataset="D", seed=0, config={}, metrics={"psnr": 30.0},
            engine=engine, git_commit=None)
    save_asset("p", "fig:scratch", kind="sweep-figure", experiment="scratch", options={}, engine=engine)
    with pytest.raises(ValueError, match="still has 1 run"):
        delete_experiment("p", "scratch", engine=engine)
    out = delete_experiment("p", "scratch", delete_runs=True, engine=engine)
    assert out == {"runs": 1, "assets_left_dangling": 1}
    assert not list_experiments("p", engine=engine)


def test_move_runs_into_another_experiment(engine):
    from results_tracker.api import log_run, move_runs

    ids = [log_run("wrong-name", project="p", method="ours", dataset="D", seed=s, config={}, metrics={"psnr": 30.0 + s},
                   engine=engine, git_commit=None).id for s in (0, 1, 2)]
    log_run("right-name", project="p", method="ours", dataset="D", seed=9, config={}, metrics={"psnr": 31.0},
            engine=engine, git_commit=None)
    assert move_runs(ids[:2], "right-name", project="p", engine=engine) == 2
    assert len(get_runs(experiment="right-name", engine=engine)) == 3
    assert len(get_runs(experiment="wrong-name", engine=engine)) == 1
    # a target that does not exist yet is created, keeping the source's type
    assert move_runs(ids[2:], "brand-new", project="p", engine=engine) == 1
    assert len(get_runs(experiment="brand-new", engine=engine)) == 1
    assert move_runs([], "right-name", project="p", engine=engine) == 0
