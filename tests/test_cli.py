from typer.testing import CliRunner

from results_tracker.cli import app

runner = CliRunner()


def test_demo_then_tables(tmp_path):
    db = str(tmp_path / "r.db")
    r = runner.invoke(app, ["demo", "--db", db])
    assert r.exit_code == 0, r.output
    assert "demo-paper" in r.output

    r = runner.invoke(app, ["experiments", "--db", db])
    assert r.exit_code == 0 and "main-comparison" in r.output and "lambda-sweep" in r.output

    r = runner.invoke(app, ["table", "-e", "main-comparison", "--by", "method", "--db", db])
    assert r.exit_code == 0, r.output
    for m in ["TV", "PnP-BM3D", "Ours", "psnr", "runtime_s"]:
        assert m in r.output

    r = runner.invoke(app, ["sweep", "-e", "lambda-sweep", "--param", "lambda", "--metric", "psnr", "--db", db])
    assert r.exit_code == 0, r.output
    assert "best lambda=0.1" in r.output

    r = runner.invoke(app, ["ablation", "-e", "ablation", "--db", db])
    assert r.exit_code == 0, r.output
    assert "full model" in r.output and "w/o adaptive" in r.output


def test_log_from_shell(tmp_path):
    db = str(tmp_path / "r.db")
    r = runner.invoke(app, [
        "log", "-e", "exp", "-m", "ours", "-d", "Set12", "--seed", "0",
        "--metric", "psnr=30.1", "--metric", "rmse=0.02", "--param", "lambda=0.1", "--param", "name=abc",
        "--tag", "quick", "--db", db,
    ])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["runs", "--db", db])
    assert r.exit_code == 0 and "ours" in r.output and "30.1" in r.output and "abc" in r.output


def test_ui_stub(tmp_path):
    r = runner.invoke(app, ["ui", "--db", str(tmp_path / "r.db")])
    assert r.exit_code == 1
