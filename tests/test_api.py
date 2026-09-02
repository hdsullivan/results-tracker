from datetime import datetime

import pytest

from results_tracker import (
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
