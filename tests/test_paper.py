"""Paper assets: pin specs, render them into stable files, detect stale exports."""

import json
import zipfile

import pytest

pytest.importorskip("matplotlib")

from typer.testing import CliRunner  # noqa: E402

from results_tracker import (  # noqa: E402
    delete_asset, get_asset, get_engine, get_metric_defs, get_runs, list_assets, log_run, run_records, save_asset, update_asset,
)
from results_tracker.cli import app  # noqa: E402
from results_tracker.demo import PROJECT, seed_demo  # noqa: E402
from results_tracker.export import paper  # noqa: E402

runner = CliRunner()


@pytest.fixture
def demo(tmp_path):
    db = tmp_path / "d.db"
    seed_demo(db=db, artifacts_dir=str(tmp_path / "art"))
    engine = get_engine(db)
    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt} for k, m in get_metric_defs(engine=engine).items()}

    def recs(exp):
        return run_records(get_runs(experiment=exp, engine=engine), engine=engine)

    return db, engine, defs, recs


def test_asset_crud(demo):
    db, engine, _, _ = demo
    a = save_asset(PROJECT, "tab:main", kind="comparison-table", experiment="main-comparison",
                   options={"rows": "method", "cols": "dataset"}, filters={"dataset": ["Set12"]}, caption="Main results", engine=engine)
    b = save_asset(PROJECT, "fig:beta", kind="sweep-figure", experiment="lambda-sweep", options={"param": "lambda", "metric": "psnr"}, engine=engine)
    assert (a.position, b.position) == (0, 1) and a.status.value == "planned" and a.exported_at is None
    assert [x.label for x in list_assets(PROJECT, engine=engine)] == ["tab:main", "fig:beta"]
    update_asset(PROJECT, "fig:beta", engine=engine, position=-1, status="final", notes="Fig. 3")
    assert [x.label for x in list_assets(PROJECT, engine=engine)] == ["fig:beta", "tab:main"]
    assert get_asset(PROJECT, "fig:beta", engine=engine).status.value == "final"
    with pytest.raises(ValueError):
        update_asset(PROJECT, "fig:beta", engine=engine, kind="ablation-table")
    with pytest.raises(LookupError):
        update_asset(PROJECT, "nope", engine=engine, status="draft")
    # re-pinning keeps status/caption/position but replaces the spec and forgets the export
    update_asset(PROJECT, "tab:main", engine=engine, exported_at=a.created_at, fingerprint="abc")
    a2 = save_asset(PROJECT, "tab:main", kind="comparison-table", experiment="main-comparison", options={"rows": "method"}, engine=engine)
    assert a2.caption == "Main results" and a2.position == 0 and a2.exported_at is None and a2.options == {"rows": "method"} and a2.filters == {}
    assert delete_asset(PROJECT, "tab:main", engine=engine) and not delete_asset(PROJECT, "tab:main", engine=engine)
    assert [x.label for x in list_assets(engine=engine)] == ["fig:beta"]


def test_render_every_kind(demo):
    db, engine, defs, recs = demo
    cmp_, sweep, abl = recs("main-comparison"), recs("lambda-sweep"), recs("ablation")
    r = paper.render_asset({"label": "tab:main", "kind": "comparison-table", "experiment": "main-comparison",
                            "filters": {"dataset": ["Set12"]}, "options": {"cols": "none", "std": "small"}, "caption": "Main"}, cmp_, defs, source="d.db")
    assert not r.error and [f for f, _ in r.files] == ["tables/tab-main.tex"]
    tex = r.files[0][1].decode()
    assert "\\label{tab:main}" in tex and "\\caption{Main}" in tex and "Filter: --where 'dataset=Set12'" in tex and "CBSD68" not in tex
    assert r.runs == 10 and r.filters == ["dataset=Set12"] and len(r.fingerprint) == 12
    r = paper.render_asset({"label": "tab:abl", "kind": "ablation-table", "experiment": "ablation", "options": {"metrics": ["psnr"]}}, abl, defs)
    assert not r.error and b"\\checkmark" in r.files[0][1]
    r = paper.render_asset({"label": "tab:sweep", "kind": "sweep-table", "experiment": "lambda-sweep",
                            "options": {"param": "lambda", "metric": "psnr", "param_label": "$\\lambda$"}}, sweep, defs)
    assert not r.error and b"$\\lambda$" in r.files[0][1]
    for kind, exp, opts in [("sweep-figure", sweep, {"param": "lambda", "metric": "psnr", "width": "double", "panel_label": "a. Sweep"}),
                            ("ablation-figure", abl, {"metric": "psnr"}),
                            ("comparison-figure", cmp_, {"metric": "psnr", "emphasize": ["Ours"], "hatch": True}),
                            ("visual-figure", cmp_, {"dataset": "Set12", "seed": 0, "zoom": True, "zoom_center": [0.4, 0.6]})]:
        r = paper.render_asset({"label": f"fig:{kind}", "kind": kind, "experiment": "x", "options": opts, "caption": "Cap"}, exp, defs)
        assert not r.error, (kind, r.error)
        names = [f for f, _ in r.files]
        assert names[:2] == [f"figures/fig-{kind}.pdf", f"figures/fig-{kind}.tex"], names
        assert r.files[0][1][:4] == b"%PDF" and b"\\caption{Cap}" in r.files[1][1] and f"\\label{{fig:{kind}}}".encode() in r.files[1][1]
    assert "figures/fig-visual-figure.json" in names and "not shown: DPIR" in r.note
    r = paper.render_asset({"label": "data:runs", "kind": "runs-csv", "experiment": "main-comparison"}, cmp_, defs)
    assert not r.error and r.files[0][0] == "data/data-runs.csv" and r.runs == 19
    # errors are reported, not raised
    r = paper.render_asset({"label": "fig:x", "kind": "sweep-figure", "experiment": "x", "options": {"param": "nope", "metric": "psnr"}}, sweep, defs)
    assert r.error and not r.files
    r = paper.render_asset({"label": "fig:x", "kind": "comparison-table", "experiment": "x", "filters": {"dataset": "nope"}}, cmp_, defs)
    assert "no completed runs match" in r.error
    with pytest.raises(ValueError):
        paper.render_asset({"label": "x", "kind": "nope", "experiment": "x"}, cmp_, defs)


def test_render_paper_write_zip_and_staleness(demo, tmp_path):
    db, engine, defs, recs = demo
    save_asset(PROJECT, "tab:main", kind="comparison-table", experiment="main-comparison", engine=engine,
               fingerprint=paper.records_fingerprint(recs("main-comparison")))
    save_asset(PROJECT, "fig:beta", kind="sweep-figure", experiment="lambda-sweep", options={"param": "lambda", "metric": "psnr"}, engine=engine)
    save_asset(PROJECT, "fig:old", kind="ablation-figure", experiment="ablation", options={"metric": "psnr"}, status="dropped", engine=engine)
    save_asset(PROJECT, "fig:broken", kind="sweep-figure", experiment="lambda-sweep", options={"param": "nope", "metric": "psnr"}, engine=engine)
    rendered = paper.render_paper(engine, PROJECT, source="d.db")
    assert [r.label for r in rendered] == ["tab:main", "fig:beta", "fig:broken"]  # dropped is skipped, manuscript order kept
    assert rendered[2].error and not rendered[2].files
    out = tmp_path / "paper"
    paths = paper.write_paper(rendered, out, project=PROJECT, source="d.db")
    rel = sorted(p.relative_to(out).as_posix() for p in paths)
    assert rel == ["MANIFEST.json", "README.txt", "figures/fig-beta.pdf", "figures/fig-beta.tex", "preamble.tex", "tables/tab-main.tex"]
    m = json.loads((out / "MANIFEST.json").read_text())
    assert m["project"] == PROJECT and [a["label"] for a in m["assets"]] == ["tab:main", "fig:beta", "fig:broken"]
    assert m["assets"][2]["error"] and m["assets"][0]["files"] == ["tables/tab-main.tex"]
    z = zipfile.ZipFile(__import__("io").BytesIO(paper.zip_paper(rendered, project=PROJECT, source="d.db")))
    assert sorted(z.namelist()) == rel

    # staleness: never exported -> current after mark_exported -> stale once a run is added
    a = get_asset(PROJECT, "tab:main", engine=engine)
    assert paper.staleness(a, recs("main-comparison"))[0] == "never exported"
    paper.mark_exported(engine, PROJECT, rendered)
    a = get_asset(PROJECT, "tab:main", engine=engine)
    assert a.exported_at is not None and paper.staleness(a, recs("main-comparison"))[0] == "current"
    assert get_asset(PROJECT, "fig:broken", engine=engine).exported_at is None  # a failed render is not an export
    log_run("main-comparison", project=PROJECT, method="Ours", dataset="Set12", seed=99, config={"lambda": 0.1, "iters": 50},
            metrics={"psnr": 31.0}, engine=engine, git_commit=None)
    state, detail = paper.staleness(a, recs("main-comparison"))
    assert state == "stale" and "re-export" in detail
    b = get_asset(PROJECT, "fig:broken", engine=engine)
    assert paper.staleness(b, recs("lambda-sweep"))[0] == "never exported"
    assert paper.staleness(a, [])[0] == "no data"
    # a filter that matches nothing
    save_asset(PROJECT, "tab:none", kind="comparison-table", experiment="main-comparison", filters={"dataset": "nope"}, engine=engine)
    assert paper.staleness(get_asset(PROJECT, "tab:none", engine=engine), recs("main-comparison")) == ("no data", "no completed runs match dataset = nope")


def test_cli_paper_and_assets(demo, tmp_path, monkeypatch):
    monkeypatch.setenv("COLUMNS", "250")  # rich would otherwise truncate the table cells
    db, engine, _, _ = demo
    dbs = str(db)
    r = runner.invoke(app, ["asset", "list", "--db", dbs])
    assert r.exit_code == 0 and "no assets pinned" in r.output
    r = runner.invoke(app, ["export", "paper", "-p", PROJECT, "-o", str(tmp_path / "p"), "--db", dbs])
    assert r.exit_code == 1 and "no assets to export" in r.output
    save_asset(PROJECT, "tab:main", kind="comparison-table", experiment="main-comparison", engine=engine)
    save_asset(PROJECT, "fig:abl", kind="ablation-figure", experiment="ablation", options={"metric": "psnr"}, engine=engine)
    r = runner.invoke(app, ["asset", "set", "fig:abl", "-p", PROJECT, "--status", "final", "--position", "-5", "--db", dbs])
    assert r.exit_code == 0 and "status=final" in r.output
    r = runner.invoke(app, ["asset", "set", "fig:abl", "-p", PROJECT, "--db", dbs])
    assert r.exit_code != 0
    r = runner.invoke(app, ["asset", "list", "-p", PROJECT, "--db", dbs])
    assert r.exit_code == 0 and "never exported" in r.output and r.output.index("fig:abl") < r.output.index("tab:main")
    r = runner.invoke(app, ["export", "paper", "-p", PROJECT, "-o", str(tmp_path / "p"), "--dry-run", "--db", dbs])
    assert r.exit_code == 0 and "nothing written" in r.output and not (tmp_path / "p").exists()
    r = runner.invoke(app, ["export", "paper", "-p", PROJECT, "-o", str(tmp_path / "p"), "--db", dbs])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "p" / "tables" / "tab-main.tex").exists() and (tmp_path / "p" / "figures" / "fig-abl.pdf").exists()
    r = runner.invoke(app, ["asset", "list", "-p", PROJECT, "--db", dbs])
    assert r.output.count("current") == 2
    r = runner.invoke(app, ["export", "paper", "-p", PROJECT, "--status", "final", "-o", str(tmp_path / "p.zip"), "--zip", "--db", dbs])
    assert r.exit_code == 0 and set(zipfile.ZipFile(tmp_path / "p.zip").namelist()) == {"figures/fig-abl.pdf", "figures/fig-abl.tex", "preamble.tex", "README.txt", "MANIFEST.json"}
    r = runner.invoke(app, ["asset", "rm", "tab:main", "-p", PROJECT, "--db", dbs])
    assert r.exit_code == 0 and runner.invoke(app, ["asset", "rm", "tab:main", "-p", PROJECT, "--db", dbs]).exit_code == 1
