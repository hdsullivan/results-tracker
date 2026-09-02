import io
import json
import zipfile

import pytest

pytest.importorskip("matplotlib")

from results_tracker import get_engine, get_metric_defs, get_runs, list_experiments, run_records  # noqa: E402
from results_tracker.demo import PROJECT, seed_demo  # noqa: E402
from results_tracker.export.bundle import build_bundle  # noqa: E402


@pytest.fixture
def demo(tmp_path):
    db = tmp_path / "d.db"
    seed_demo(db=db, artifacts_dir=str(tmp_path / "art"))
    engine = get_engine(db)
    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt} for k, m in get_metric_defs(engine=engine).items()}
    exps = {e.name: (e.type.value, run_records(get_runs(experiment=e.name, engine=engine), engine=engine))
            for e in list_experiments(PROJECT, engine=engine)}
    return exps, defs


def test_bundle_contents_and_manifest(demo):
    exps, defs = demo
    data, manifest = build_bundle(exps, defs, project=PROJECT, source="d.db")
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    for expected in ["tables/main-comparison.tex", "tables/ablation.tex", "tables/lambda-sweep_lambda.tex",
                     "figures/main-comparison_psnr.pdf", "figures/ablation_psnr.pdf", "figures/lambda-sweep_lambda.pdf",
                     "figures/main-comparison_Set12_visual.pdf", "figures/main-comparison_Set12_visual.json",
                     "figures/main-comparison_Set12_error.pdf", "figures/main-comparison_CBSD68_visual.pdf",
                     "runs/main-comparison.csv", "runs/lambda-sweep.csv", "runs/ablation.csv",
                     "preamble.tex", "README.txt", "MANIFEST.json"]:
        assert expected in names, expected
    tex = zf.read("tables/main-comparison.tex").decode()
    assert "\\begin{table*}" in tex  # 6 value columns -> wide table chosen automatically
    assert "missing: method=DPIR, dataset=CBSD68" in tex
    m = json.loads(zf.read("MANIFEST.json"))
    assert m["project"] == PROJECT and m["source"] == "d.db"
    kinds = {f["kind"] for f in m["files"]}
    assert {"comparison-table", "comparison-figure", "visual-figure", "visual-error-figure", "sweep-table", "sweep-figure",
            "ablation-table", "ablation-figure", "runs-csv", "figure-tex", "provenance"} <= kinds
    vis = next(f for f in m["files"] if f["kind"] == "visual-figure" and "Set12" in f["file"])
    assert "not shown: DPIR" in vis["note"]
    assert all(zf.getinfo(n).file_size > 0 for n in names)


def test_bundle_without_visuals(demo):
    exps, defs = demo
    data, manifest = build_bundle(exps, defs, project=PROJECT, source="d.db", visual=False, width="double")
    names = zipfile.ZipFile(io.BytesIO(data)).namelist()
    assert not any("visual" in n or "error" in n for n in names)
    assert "figures/main-comparison_psnr.tex" in names
    assert "figure*" in zipfile.ZipFile(io.BytesIO(data)).read("figures/main-comparison_psnr.tex").decode()
