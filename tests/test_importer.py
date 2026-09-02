import json

from results_tracker import get_runs, run_records
from results_tracker.importer import ImportSpec, coerce, import_path, normalize, read_records


def test_coerce():
    assert coerce("0.1") == 0.1 and coerce("3") == 3 and coerce("true") is True
    assert coerce("") is None and coerce("nan") is None and coerce("abc") == "abc"
    assert coerce(5) == 5


def test_csv_import_with_heuristic_mapping(tmp_path, engine):
    csv = tmp_path / "r.csv"
    csv.write_text(
        "method,dataset,seed,lambda,denoiser,psnr,ssim\n"
        "TV,Set12,0,0.1,none,27.5,0.77\n"
        "Ours,Set12,0,0.1,drunet,31.2,0.88\n"
        "Ours,Set12,1,0.1,drunet,31.4,0.88\n"
    )
    res = import_path(csv, ImportSpec(experiment="cmp", project="p", config_cols=["lambda"]), engine=engine)
    assert res.imported == 3 and res.skipped == 0 and not res.errors
    recs = run_records(get_runs(experiment="cmp", engine=engine), engine=engine)
    ours = [r for r in recs if r["method"] == "Ours"]
    assert len(ours) == 2
    assert ours[0]["metrics"] == {"psnr": 31.2, "ssim": 0.88}
    assert ours[0]["config"] == {"lambda": 0.1, "denoiser": "drunet"}
    assert ours[0]["source"] == "imported" and ours[0]["dataset"] == "Set12"


def test_csv_import_explicit_columns_and_constant_method(tmp_path, engine):
    csv = tmp_path / "r.csv"
    csv.write_text("img,noise,score,elapsed\na.png,25,30.1,1.5\nb.png,25,29.9,1.6\n")
    spec = ImportSpec(experiment="e", method="Ours", dataset="Set12", instance_col="img",
                      metric_cols=["score"], config_cols=["noise"])
    res = import_path(csv, spec, engine=engine)
    assert res.imported == 2
    recs = run_records(get_runs(engine=engine), engine=engine)
    r = recs[0]
    assert r["method"] == "Ours" and r["instance"] == "a.png"
    assert r["metrics"] == {"score": 30.1}  # 'elapsed' not listed as a metric -> config catch-all
    assert r["config"] == {"noise": 25, "elapsed": 1.5}


def test_json_directory_import_nested_and_dedup(tmp_path, engine):
    d = tmp_path / "runs"
    (d / "sub").mkdir(parents=True)
    (d / "a.json").write_text(json.dumps(
        {"method": "Ours", "dataset": "D", "seed": 0, "config": {"lambda": 0.1, "solver": {"iters": 50}},
         "metrics": {"psnr": 30.0}}))
    (d / "sub" / "b.json").write_text(json.dumps(
        [{"method": "Ours", "dataset": "D", "seed": 1, "config": {"lambda": 0.1}, "metrics": {"psnr": 30.5}},
         {"method": "TV", "dataset": "D", "seed": 0, "config": {"lambda": 0.1}, "metrics": {"psnr": 27.0}, "status": "failed"}]))
    assert len(read_records(d)) == 3
    spec = ImportSpec(experiment="e", experiment_type="sweep")
    res = import_path(d, spec, engine=engine)
    assert res.imported == 3
    recs = run_records(get_runs(engine=engine), engine=engine)
    assert recs[0]["config"] == {"lambda": 0.1, "solver": {"iters": 50}}
    assert [r["status"] for r in recs].count("failed") == 1
    assert "a.json" in recs[0]["notes"]
    # re-import: everything is a duplicate
    res2 = import_path(d, spec, engine=engine)
    assert res2.imported == 0 and res2.skipped == 3
    assert len(get_runs(engine=engine)) == 3
    # unless asked to keep them
    res3 = import_path(d, ImportSpec(experiment="e", skip_duplicates=False), engine=engine)
    assert res3.imported == 3 and len(get_runs(engine=engine)) == 6


def test_dry_run_writes_nothing(tmp_path, engine):
    csv = tmp_path / "r.csv"
    csv.write_text("method,psnr\nA,1\nB,2\n")
    res = import_path(csv, ImportSpec(experiment="e"), engine=engine, dry_run=True)
    assert res.imported == 2 and get_runs(engine=engine) == []


def test_normalize_bool_is_config_not_metric():
    kw = normalize({"method": "m", "adaptive": True, "psnr": 30.0, "seed": "3"}, ImportSpec(experiment="e"))
    assert kw["config"] == {"adaptive": True} and kw["metrics"] == {"psnr": 30.0} and kw["seed"] == 3


def test_normalize_heuristic_puts_numeric_in_metrics_unless_mapped():
    raw = {"lambda": 0.1, "psnr": 30.0, "denoiser": "drunet"}
    kw = normalize(raw, ImportSpec(experiment="e"))
    assert kw["metrics"] == {"lambda": 0.1, "psnr": 30.0} and kw["config"] == {"denoiser": "drunet"}
    kw = normalize(raw, ImportSpec(experiment="e", config_cols=["lambda"]))
    assert kw["metrics"] == {"psnr": 30.0} and kw["config"] == {"lambda": 0.1, "denoiser": "drunet"}
    kw = normalize(raw, ImportSpec(experiment="e", metric_cols=["psnr"]))
    assert kw["metrics"] == {"psnr": 30.0} and kw["config"] == {"lambda": 0.1, "denoiser": "drunet"}


def test_row_errors_are_collected(tmp_path, engine):
    csv = tmp_path / "r.csv"
    csv.write_text("method,seed,psnr\nA,notanint,1\nB,0,2\n")
    res = import_path(csv, ImportSpec(experiment="e"), engine=engine)
    assert res.imported == 1 and len(res.errors) == 1 and "row 0" in res.errors[0]
