"""Value maps, derived fields, method order, schema migration."""

import sqlite3

import pytest
from typer.testing import CliRunner

from results_tracker import (
    aggregate as agg, define_method, define_value_map, delete_value_map, get_engine, get_runs, list_methods, list_value_maps,
    log_run, run_records, set_project,
)
from results_tracker.cli import app
from results_tracker.db import add_missing_columns
from results_tracker.demo import PROJECT, seed_demo
from results_tracker.valuemaps import apply_value_maps, derive, format_rules, parse_rules, rule_labels

runner = CliRunner()


def test_rules_grammar_and_derivation():
    rules = parse_rules("isotropic = 0, 1, 2, 3\nanisotropic = 4-7\n# a comment\nmotion = 8-11\n")
    assert rules == [{"label": "isotropic", "values": [0, 1, 2, 3]}, {"label": "anisotropic", "range": [4, 7]}, {"label": "motion", "range": [8, 11]}]
    assert [derive(rules, v) for v in (0, 3, 4.0, 7, 11, 12, None, "x")] == ["isotropic", "isotropic", "anisotropic", "anisotropic", "motion", None, None, None]
    assert format_rules(rules) == "isotropic = 0, 1, 2, 3\nanisotropic = 4-7\nmotion = 8-11"
    assert rule_labels(rules) == ["isotropic", "anisotropic", "motion"]
    assert parse_rules("low = 0.01, 0.03\nhigh = 0.05") == [{"label": "low", "values": [0.01, 0.03]}, {"label": "high", "values": [0.05]}]
    assert parse_rules("neg = -1") == [{"label": "neg", "values": [-1]}]  # a lone negative number is a value, not a range
    assert derive(parse_rules("small = Set12, BSD68"), "Set12") == "small" and derive(parse_rules("five = 5"), 5.0) == "five"
    for bad in ("", "nonsense", "= 1", "x ="):
        with pytest.raises(ValueError):
            parse_rules(bad)
    recs = apply_value_maps([{"config": {"kernel": 5}}, {"config": {"kernel": 9}}, {"config": {}}],
                            [{"name": "kernel_type", "field": "config.kernel", "rules": rules}])
    assert [agg.get_field(r, "derived.kernel_type") for r in recs] == ["anisotropic", "motion", None]
    assert "derived.kernel_type" in agg.grouping_keys(recs) and "config.kernel" in agg.grouping_keys(recs)
    assert agg.grouping_keys(recs, varying_only=True) == ["config.kernel", "derived.kernel_type"]


def test_value_maps_flow_into_records_and_tables(tmp_path):
    db = tmp_path / "d.db"
    seed_demo(db=db)
    engine = get_engine(db)
    vm = define_value_map(PROJECT, "size", field="dataset", rules=parse_rules("small = Set12\nlarge = CBSD68"), description="image set size", engine=engine)
    assert vm.name == "size" and [v.name for v in list_value_maps(PROJECT, engine=engine)] == ["size"]
    with pytest.raises(ValueError):
        define_value_map(PROJECT, "bad", field="dataset", rules=[{"label": "x"}], engine=engine)
    recs = run_records(get_runs(experiment="main-comparison", engine=engine), engine=engine)
    assert {r["derived"]["size"] for r in recs} == {"small", "large"}
    pt = agg.pivot_table(recs, "method", "derived.size", metrics=["psnr"])
    assert set(pt.cols) == {"small", "large"}
    assert "derived.size" in agg.grouping_keys(recs) and "experiment" not in agg.grouping_keys(recs)  # one experiment only
    assert agg.filter_records(recs, {"derived.size": "small"}) and all(r["dataset"] == "Set12" for r in agg.filter_records(recs, {"derived.size": "small"}))
    # method order: positions first, ties in first-seen order
    define_method("Ours", label="Ours", position=-1, engine=engine)
    define_method("TV", label="TV~\\cite{rudin1992}", is_baseline=True, position=5, engine=engine)
    recs = run_records(get_runs(experiment="main-comparison", engine=engine), engine=engine)
    order = agg.method_order(recs)
    assert order[0] == "Ours" and order[-1] == "TV"
    assert [m.name for m in list_methods(engine=engine)][0] == "Ours"
    define_method("TV", label="TV", is_baseline=True, engine=engine)  # position=None keeps 5
    assert next(m for m in list_methods(engine=engine) if m.name == "TV").position == 5
    assert set_project(PROJECT, primary_metric="ssim", engine=engine).primary_metric == "ssim"
    assert delete_value_map(PROJECT, "size", engine=engine) and not delete_value_map(PROJECT, "size", engine=engine)
    assert run_records(get_runs(experiment="main-comparison", engine=engine), engine=engine)[0]["derived"] == {}


def test_cli_method_and_valuemap(tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    db = str(tmp_path / "d.db")
    seed_demo(db=db)
    r = runner.invoke(app, ["valuemap", "set", "size", "-p", PROJECT, "--field", "dataset", "--rule", "small = Set12", "--rule", "large = CBSD68", "--db", db])
    assert r.exit_code == 0 and "derived.size" in r.output
    r = runner.invoke(app, ["valuemap", "list", "-p", PROJECT, "--db", db])
    assert r.exit_code == 0 and "small = Set12" in r.output
    r = runner.invoke(app, ["export", "table", "-e", "main-comparison", "--cols", "derived.size", "--db", db])
    assert r.exit_code == 0 and "\\multicolumn{3}{c}{large}" in r.output and "\\multicolumn{3}{c}{small}" in r.output
    r = runner.invoke(app, ["table", "-e", "main-comparison", "--where", "derived.size=small", "--db", db])
    assert r.exit_code == 0
    r = runner.invoke(app, ["valuemap", "set", "bad", "-p", PROJECT, "--field", "dataset", "--rule", "nonsense", "--db", db])
    assert r.exit_code != 0
    r = runner.invoke(app, ["method", "define", "Ours", "--position", "-1", "--db", db])
    assert r.exit_code == 0
    r = runner.invoke(app, ["method", "list", "--db", db])
    assert r.exit_code == 0 and r.output.index("Ours") < r.output.index("TV")
    r = runner.invoke(app, ["valuemap", "rm", "size", "-p", PROJECT, "--db", db])
    assert r.exit_code == 0 and runner.invoke(app, ["valuemap", "rm", "size", "-p", PROJECT, "--db", db]).exit_code == 1


def test_older_database_gains_new_columns(tmp_path):
    db = tmp_path / "old.db"
    seed_demo(db=db)
    # drop the columns this branch added, as a database written by an older results-tracker would lack them
    con = sqlite3.connect(db)
    con.execute("ALTER TABLE method DROP COLUMN position")
    con.execute("ALTER TABLE project DROP COLUMN primary_metric")
    con.execute("DROP TABLE valuemap")
    con.commit()
    con.close()
    from sqlalchemy import inspect

    engine = get_engine(db)  # create_all makes the missing table, add_missing_columns the missing columns
    cols = {c["name"] for c in inspect(engine).get_columns("method")}
    assert "position" in cols and "valuemap" in inspect(engine).get_table_names()
    assert all(m.position == 0 for m in list_methods(engine=engine))  # the scalar default was applied to old rows
    assert add_missing_columns(engine) == []  # idempotent
    log_run("main-comparison", project=PROJECT, method="Ours", dataset="Set12", seed=9, metrics={"psnr": 1.0}, engine=engine, git_commit=None)
