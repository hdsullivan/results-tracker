"""Merging one database into another: runs travel, local curation stays."""

import pytest

from results_tracker import aggregate as agg
from results_tracker.api import (add_note, define_method, get_asset, get_runs, list_notes, log_run, run_records,
                                 save_asset, set_experiment)
from results_tracker.merge import merge_database


def cluster(db, **over):
    """A run as the remote machine logs it."""
    kw = dict(experiment="window", project="paper", method="admm", dataset="D", instance="img0", seed=0,
              config={"floor": "window"}, metrics={"psnr": 30.0}, experiment_type="ablation",
              artifacts_dir="/mnt/cc/artifacts/window/img0", hostname="crimson", db=db)
    kw.update(over)
    return log_run(**kw)


@pytest.fixture
def remote(tmp_path):
    db = tmp_path / "remote.db"
    cluster(db)
    cluster(db, instance="img1", metrics={"psnr": 28.0})
    return db


@pytest.fixture
def local(tmp_path, remote):
    """A local database that has already had one merge and some curation done on top of it."""
    db = tmp_path / "local.db"
    merge_database(remote, db=db)
    return db


def test_merge_brings_runs_across_and_is_idempotent(tmp_path, remote):
    db = tmp_path / "local.db"
    first = merge_database(remote, db=db)
    assert (first.runs_inserted, first.runs_updated, first.runs_unchanged) == (2, 0, 0)
    assert first.created["projects"] == ["paper"] and first.created["experiments"] == ["paper/window"]

    again = merge_database(remote, db=db)
    assert (again.runs_inserted, again.runs_updated, again.runs_unchanged) == (0, 0, 2)
    assert len(get_runs(experiment="window", db=db)) == 2
    assert not again.created.get("projects")


def test_merge_keeps_pins_notes_labels_and_stages(local, remote, tmp_path):
    """The whole point: the things a fetch used to destroy.

    Every one of these is written only where the paper is, and none of them exists in the source.
    """
    save_asset("paper", "tab:main", kind="ablation-table", experiment="window", db=local)
    add_note("paper", "floor=window chosen 2026-09-14", experiment="window", db=local)
    define_method("admm", label=r"AdaptivePnP-ADMM~\cite{us}", position=3, db=local)
    set_experiment("window", project="paper", stage="paper", description="the window ablation", db=local)
    # a run repointed at the local copy of the artifacts, and tagged by hand
    run = get_runs(experiment="window", db=local)[0]
    from results_tracker.db import get_engine
    from sqlmodel import Session
    with Session(get_engine(local)) as s:
        row = s.get(type(run), run.id)
        row.artifacts_dir, row.tags, row.notes = "/Users/me/artifacts/window/img0", ["base"], "kept"
        s.add(row); s.commit()

    cluster(remote, instance="img2", metrics={"psnr": 27.0})  # new work arrives
    report = merge_database(remote, db=local)
    assert (report.runs_inserted, report.runs_updated, report.runs_unchanged) == (1, 0, 2)

    assert get_asset("paper", "tab:main", db=local) is not None
    assert len(list_notes("paper", db=local)) == 1
    recs = {r["instance"]: r for r in run_records(get_runs(experiment="window", db=local), db=local)}
    assert recs["img0"]["method_label"] == r"AdaptivePnP-ADMM~\cite{us}"
    assert recs["img0"]["tags"] == ["base"] and recs["img0"]["notes"] == "kept"
    assert recs["img0"]["artifacts_dir"] == "/Users/me/artifacts/window/img0", "the repoint survives"
    assert recs["img2"]["artifacts_dir"].startswith("/mnt/cc"), "a new run arrives with the source's path"


def test_merge_lands_results_that_changed_since_the_last_merge(local, remote):
    """A run that was `running` at the last merge, and finished since."""
    cluster(remote, instance="img3", status="running", metrics={})
    merge_database(remote, db=local)
    before = next(r for r in get_runs(experiment="window", db=local) if r.instance == "img3")
    assert before.status.value == "running" and before.metrics == {}

    cluster(remote, instance="img3", status="completed", metrics={"psnr": 26.0}, on_duplicate="replace")
    report = merge_database(remote, db=local)
    assert (report.runs_inserted, report.runs_updated) == (0, 1)
    after = next(r for r in get_runs(experiment="window", db=local) if r.instance == "img3")
    assert after.status.value == "completed" and after.metrics == {"psnr": 26.0}


def test_merge_adds_source_tags_without_dropping_local_ones(local, remote):
    """Tags union: the source can teach this database a tag, never un-teach one."""
    from results_tracker.db import get_engine
    from sqlmodel import Session
    run = get_runs(experiment="window", db=local)[0]
    with Session(get_engine(local)) as s:
        row = s.get(type(run), run.id)
        row.tags = ["mine"]
        s.add(row); s.commit()
    cluster(remote, tags=["base", "arm:full model"], on_duplicate="replace")

    merge_database(remote, db=local)
    tags = next(r for r in get_runs(experiment="window", db=local) if r.instance == "img0").tags
    assert tags == ["mine", "base", "arm:full model"]


def test_merge_counts_repeats_rather_than_collapsing_them(tmp_path, remote):
    db = tmp_path / "local.db"
    merge_database(remote, db=db)
    cluster(remote, on_duplicate="allow", metrics={"psnr": 30.5})  # a deliberate second run of one setting
    report = merge_database(remote, db=db)
    assert (report.runs_inserted, report.runs_unchanged) == (1, 2)
    assert len([r for r in get_runs(experiment="window", db=db) if r.instance == "img0"]) == 2


def test_merge_dry_run_and_filters(tmp_path, remote):
    db = tmp_path / "local.db"
    cluster(remote, experiment="other", instance="img9", metrics={"psnr": 20.0})

    report = merge_database(remote, db=db, dry_run=True)
    assert report.runs_inserted == 3 and report.dry_run
    assert get_runs(db=db) == [], "a dry run writes nothing, lookup rows included"

    report = merge_database(remote, db=db, experiments=["window"])
    assert report.runs_inserted == 2
    assert {r.instance for r in get_runs(db=db)} == {"img0", "img1"}
    assert merge_database(remote, db=db, projects=["no-such-paper"]).runs_seen == 0


def test_merge_refuses_to_merge_a_database_into_itself(remote, tmp_path):
    with pytest.raises(ValueError, match="same database"):
        merge_database(remote, db=remote)
    with pytest.raises(FileNotFoundError):
        merge_database(tmp_path / "nope.db", db=remote)


def test_merged_runs_aggregate_exactly_as_the_source_does(tmp_path, remote):
    """The merge must not disturb what the runs mean: same ablation table on both sides."""
    for floor, psnr in (("none", 22.0), ("none", 23.0)):
        cluster(remote, config={"floor": floor}, instance=f"img{psnr}", metrics={"psnr": psnr})
    cluster(remote, instance="img4", metrics={"psnr": 31.0})  # 3 window runs to 2 none: an unambiguous base
    db = tmp_path / "local.db"
    merge_database(remote, db=db)
    here = agg.ablation_table(run_records(get_runs(experiment="window", db=db), db=db), metrics=["psnr"])
    there = agg.ablation_table(run_records(get_runs(experiment="window", db=remote), db=remote), metrics=["psnr"])
    assert [(r.label, r.n, r.stats["psnr"].mean) for r in here] == [(r.label, r.n, r.stats["psnr"].mean) for r in there]
