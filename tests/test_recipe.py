import json

import pytest
from typer.testing import CliRunner

from results_tracker import aggregate as agg
from results_tracker import get_runs, has_run, run_records
from results_tracker.cli import app
from results_tracker.recipe import (
    Ablation,
    StudyObserver,
    arm_changes,
    default_diagnostics_dir,
    Arm,
    Estimate,
    Instance,
    Knob,
    KnobSpace,
    Method,
    Problem,
    Registry,
    Study,
    Sweep,
    expand,
    run_study,
    validate_study,
)
from results_tracker.recipe.toy import PROJECT, AdaptiveGD, ToyDeblurring, toy_studies

runner = CliRunner()


# --------------------------------------------------------------------------- knobs

def test_knob_validation_and_coercion():
    f = Knob("reg", "float", 0.1, bounds=(0.0, 1.0), log=True)
    assert f.validate(1) == 1.0 and isinstance(f.validate(1), float) and f.validate("0.5") == 0.5
    with pytest.raises(ValueError):
        f.validate(True)
    with pytest.raises(ValueError):
        f.validate(2.0)
    i = Knob("iters", "int", 10, bounds=(1, 100))
    assert i.validate(3.0) == 3 and i.validate("7") == 7
    with pytest.raises(ValueError):
        i.validate(2.5)
    b = Knob("adaptive", "bool", True)
    assert b.validate("false") is False and b.validate(True) is True
    with pytest.raises(ValueError):
        b.validate(1)
    c = Knob("denoiser", "choice", "drunet", choices=("drunet", "dncnn"))
    assert c.validate("dncnn") == "dncnn"
    with pytest.raises(ValueError):
        c.validate("bm3d")
    with pytest.raises(ValueError):
        Knob("x", "choice", None)  # choice without choices
    with pytest.raises(ValueError):
        Knob("x", "float", 5.0, bounds=(0.0, 1.0))  # default outside bounds
    opt = Knob("floor", "float", None, bounds=(0.0, 10.0))
    assert opt.validate(None) is None and opt.validate(2) == 2.0
    assert Knob.from_dict(c.to_dict()) == c and Knob.from_dict(f.to_dict()) == f


def test_knob_space_resolves_defaults_and_rejects_unknown_keys():
    space = KnobSpace([Knob("a", "int", 1), Knob("b", "bool", False)])
    assert space.resolve() == {"a": 1, "b": False}
    assert space.resolve({"b": True}) == {"a": 1, "b": True}
    assert list(space.resolve({"b": True})) == ["a", "b"]  # declaration order
    with pytest.raises(KeyError):
        space.resolve({"c": 3})
    with pytest.raises(ValueError):
        KnobSpace([Knob("a", "int", 1), Knob("a", "int", 2)])
    assert space.diff({"a": 1, "b": False}, {"a": 1, "b": True}) == ["b"]


# --------------------------------------------------------------------------- study specs

def test_study_json_round_trip(tmp_path):
    for study in toy_studies("art"):
        study.save(tmp_path / f"{study.name}.json")
        back = Study.load(tmp_path / f"{study.name}.json")
        assert back == study
    spec = json.loads((tmp_path / "ablation.json").read_text())
    assert spec["ablation"]["arms"] == [{"adaptive": False}, {"warm_start": False}, {"prior": "tikhonov"}]
    assert "sweep" not in spec  # empty fields are not written


def test_expand_and_validation():
    methods = {"adaptive-gd": AdaptiveGD}
    comparison, sweep, ablation = toy_studies()
    jobs = expand(ablation, ToyDeblurring, methods)
    assert len(jobs) == 2 * 2 * 4  # seeds × blur values × (base + 3 arms)
    assert [j.arm for j in jobs[:4]] == ["full model", "adaptive=False", "warm_start=False", "prior=tikhonov"]
    assert jobs[0].tags == ("base",) and all(j.tags == () for j in jobs[1:4])
    assert jobs[0].config["adaptive"] is True and jobs[1].config["adaptive"] is False
    assert jobs[0].condition == {"blur": 1.0, "noise": 0.05}

    jobs = expand(sweep, ToyDeblurring, methods)
    assert [j.config["reg"] for j in jobs[:5]] == [0.0003, 0.001, 0.003, 0.01, 0.03] and len(jobs) == 15

    from results_tracker.recipe.toy import TikhonovGD, Wiener
    jobs = expand(comparison, ToyDeblurring, {"wiener": Wiener, "gd": TikhonovGD, "adaptive-gd": AdaptiveGD})
    assert len(jobs) == 2 * 4 * 3 and jobs[0].config == {"reg": 0.01}

    bad = Study("x", "ablation", "toy-deblur", [Arm("adaptive-gd")], ablation=Ablation(arms=[{"adaptive": False, "iters": 5}]))
    with pytest.raises(ValueError, match="exactly one knob"):
        validate_study(bad, ToyDeblurring, methods)
    bad = Study("x", "ablation", "toy-deblur", [Arm("adaptive-gd")], ablation=Ablation(arms=[{"adaptive": True}]))
    with pytest.raises(ValueError, match="does not differ"):
        validate_study(bad, ToyDeblurring, methods)
    bad = Study("x", "sweep", "toy-deblur", [Arm("adaptive-gd")], sweep=Sweep("lambda", [1, 2]))
    with pytest.raises(ValueError, match="not a knob"):
        validate_study(bad, ToyDeblurring, methods)
    bad = Study("x", "comparison", "toy-deblur", [Arm("adaptive-gd", {"nope": 1})])
    with pytest.raises(KeyError):
        validate_study(bad, ToyDeblurring, methods)
    bad = Study("x", "comparison", "toy-deblur", [Arm("adaptive-gd")], conditions={"dose": [1]})
    with pytest.raises(ValueError, match="not a condition"):
        validate_study(bad, ToyDeblurring, methods)
    bad = Study("x", "comparison", "toy-deblur", [Arm("adaptive-gd")], split="train")
    with pytest.raises(ValueError, match="split"):
        validate_study(bad, ToyDeblurring, methods)


# --------------------------------------------------------------------------- running

def test_run_toy_demo_logs_resumes_and_feeds_the_analysis(tmp_path, engine):
    comparison, sweep, ablation = toy_studies(str(tmp_path / "art"))
    reports = [run_study(s, engine=engine, log=None) for s in (comparison, sweep, ablation)]
    assert [r.logged for r in reports] == [96, 45, 48] and all(r.failed == 0 and r.skipped == 0 for r in reports)

    recs = run_records(get_runs(experiment="main-comparison", project=PROJECT, engine=engine), engine=engine)
    assert {r["method"] for r in recs} == {"wiener", "gd", "adaptive-gd"}
    assert {r["dataset"] for r in recs} == {"Phantoms"} and {r["seed"] for r in recs} == {0, 1}
    r = next(r for r in recs if r["method"] == "adaptive-gd")
    assert set(r["config"]) == {"blur", "noise", "reg", "iters", "prior", "delta", "adaptive", "warm_start"}
    assert {"psnr", "ssim", "runtime_s", "iterations", "final_step"} <= set(r["metrics"])
    assert "step_sizes" not in r["metrics"]  # a curve is not a metric ...
    from pathlib import Path
    run_dir = Path(r["artifacts_dir"])
    assert run_dir.is_relative_to(tmp_path / "art")
    assert {p.name for p in run_dir.iterdir()} == {"reconstruction.png", "ground_truth.png", "measurement.png", "diagnostics.json"}
    assert len(json.loads((run_dir / "diagnostics.json").read_text())["step_sizes"]) == 30  # ... it lives here
    assert r["method_is_baseline"] is False and next(x for x in recs if x["method"] == "wiener")["method_is_baseline"]
    # the proposed method wins the toy comparison, as a demo should
    ct = agg.comparison_table(recs, metrics=["psnr"])
    assert ct.is_best(("adaptive-gd",), "psnr")

    # ablation: base runs tagged, conditions pooled, one row per switched-off component
    recs = run_records(get_runs(experiment="ablation", project=PROJECT, engine=engine), engine=engine)
    ablation_config = recs[0]["config"]
    assert sum("base" in r["tags"] for r in recs) == 12
    assert agg.condition_keys(recs) == ["blur"]
    rows = agg.ablation_table(recs, metrics=["psnr"])
    assert [row.label for row in rows] == ["full model", "prior: huber_tv→tikhonov", "w/o adaptive", "w/o warm_start"]
    assert all(row.n == 12 for row in rows)

    # sweep: five values, three seeds × three instances each
    recs = run_records(get_runs(experiment="reg-sweep", project=PROJECT, engine=engine), engine=engine)
    series = agg.sweep_series(recs, "reg", "psnr")[()]
    assert [x for x, _ in series] == [0.0003, 0.001, 0.003, 0.01, 0.03] and all(st.n == 9 for _, st in series)

    # resume: nothing is recomputed, nothing duplicated
    again = run_study(ablation, engine=engine, log=None)
    assert again.logged == 0 and again.skipped == 48
    assert len(get_runs(experiment="ablation", project=PROJECT, engine=engine)) == 48
    assert has_run("ablation", project=PROJECT, method="adaptive-gd", dataset="Phantoms", instance="phantom_00", seed=0,
                   config=ablation_config, engine=engine)
    assert not has_run("ablation", project=PROJECT, method="adaptive-gd", dataset="Phantoms", instance="phantom_99", seed=0,
                       config=ablation_config, engine=engine)


class Crashy(Method):
    key = "crashy"
    knobs = (Knob("mode", "choice", "raise", choices=("raise", "nan", "huge", "ok")),)
    calls = 0

    def reconstruct(self, instance, config, state):
        Crashy.calls += 1
        if config["mode"] == "raise":
            raise RuntimeError("boom")
        if config["mode"] == "nan":
            return Estimate(instance.measurement * float("nan"))
        if config["mode"] == "huge":  # finite, but so far off that the squared error overflows
            return Estimate(instance.measurement * 0.0 + 1e200)
        return Estimate(instance.measurement)


def test_failures_are_logged_not_raised_and_failed_settings_are_retried(engine):
    reg = Registry()
    reg.problem(ToyDeblurring)
    reg.method(Crashy)
    study = Study("robust", "comparison", "toy-deblur", [Arm("crashy", {"mode": "raise"}), Arm("crashy", {"mode": "nan"}),
                                                          Arm("crashy", {"mode": "ok"})], project="p", n_instances=2)
    report = run_study(study, engine=engine, registry=reg, log=None)
    assert report.logged == 6 and report.failed == 4
    recs = run_records(get_runs(experiment="robust", project="p", engine=engine), engine=engine)
    notes = {r["config"]["mode"]: r["notes"] for r in recs}
    assert notes["raise"].startswith("RuntimeError: boom") and notes["nan"] == "non-finite estimate" and notes["ok"] == ""
    assert {r["status"] for r in recs if r["config"]["mode"] == "ok"} == {"completed"}
    assert all("psnr" in r["metrics"] for r in recs if r["status"] == "completed")
    # a failed setting is not "present": resume recomputes it (and replaces the failed row), the ok one is skipped
    Crashy.calls = 0
    report = run_study(study, engine=engine, registry=reg, log=None)
    assert report.skipped == 2 and report.logged == 4 and Crashy.calls == 4
    assert len(get_runs(experiment="robust", project="p", engine=engine)) == 6


def test_non_finite_metrics_mark_the_run_failed(engine):
    """A finite estimate whose PSNR overflows to -inf is a diverged run, logged as failed with the metric named."""
    reg = Registry()
    reg.problem(ToyDeblurring)
    reg.method(Crashy)
    study = Study("overflow", "comparison", "toy-deblur", [Arm("crashy", {"mode": "huge"})], project="p", n_instances=1)
    report = run_study(study, engine=engine, registry=reg, log=None)
    assert report.logged == 1 and report.failed == 1
    (rec,) = run_records(get_runs(experiment="overflow", project="p", engine=engine), engine=engine)
    assert rec["status"] == "failed" and rec["notes"].startswith("non-finite metric(s): ")
    assert "psnr" in rec["notes"]


def test_registry_resolves_keys_and_import_paths():
    reg = Registry()
    assert reg.resolve_method("results_tracker.recipe.toy:AdaptiveGD") is AdaptiveGD
    assert reg.resolve_problem("results_tracker.recipe.toy.ToyDeblurring") is ToyDeblurring
    with pytest.raises(KeyError):
        reg.resolve_method("adaptive-gd")  # not registered in this fresh registry
    with pytest.raises(KeyError):
        reg.resolve_method("results_tracker.recipe.toy:ToyDeblurring")  # a Problem is not a Method
    assert AdaptiveGD.display_label() == "Ours"

    class Cited(Method):
        key = "c"
        label = "DPIR"
        citation = "zhang2021"

        def reconstruct(self, instance, config, state):
            return Estimate(instance.measurement)

    assert Cited.display_label() == r"DPIR~\cite{zhang2021}"


def test_problem_view_and_instance_defaults():
    import numpy as np

    p = ToyDeblurring()
    assert p.view(np.zeros((4, 4))).shape == (4, 4)
    assert p.view(np.zeros((6, 16, 16))).shape == (16, 16)
    assert p.view(np.zeros((32, 32, 3))).shape == (32, 32, 3)
    assert p.view(np.zeros((6, 16, 16, 16))).shape == (16, 16)
    assert p.view(np.zeros(5)) is None
    inst = next(iter(p.instances({"blur": 1.0, "noise": 0.0}, "test", 1, 0)))
    assert isinstance(inst, Instance) and inst.name == "phantom_00" and inst.measurement.shape == inst.reference.shape
    assert p.dataset_name("test") == "Phantoms" and p.dataset_name("validation") == "Phantoms-val"


# --------------------------------------------------------------------------- CLI

def test_recipe_cli(tmp_path):
    db = str(tmp_path / "r.db")
    r = runner.invoke(app, ["recipe", "demo", "--db", db, "--quiet", "--write-specs", str(tmp_path / "specs"),
                            "--artifacts", str(tmp_path / "art")])
    assert r.exit_code == 0, r.output
    assert "main-comparison: 96 runs logged" in r.output and "ablation: 48 runs logged" in r.output
    specs = sorted(p.name for p in (tmp_path / "specs").iterdir())
    assert specs == ["ablation.json", "main-comparison.json", "reg-sweep.json"]

    r = runner.invoke(app, ["recipe", "validate", str(tmp_path / "specs" / "ablation.json")])
    assert r.exit_code == 0 and "16 jobs" in r.output and "full model" in r.output
    r = runner.invoke(app, ["recipe", "run", str(tmp_path / "specs" / "reg-sweep.json"), "--db", db, "-q"])
    assert r.exit_code == 0 and "0 runs logged" in r.output and "45 already present" in r.output
    r = runner.invoke(app, ["recipe", "knobs", "adaptive-gd", "-i", "results_tracker.recipe.toy"])
    assert r.exit_code == 0 and "warm_start" in r.output and "huber_tv" in r.output
    r = runner.invoke(app, ["recipe", "knobs", "toy-deblur", "-i", "results_tracker.recipe.toy"])
    assert r.exit_code == 0 and "blur" in r.output and "validation" in r.output
    r = runner.invoke(app, ["recipe", "knobs", "nope"])
    assert r.exit_code == 1

    bad = Study.load(tmp_path / "specs" / "ablation.json")
    bad.ablation.arms.append({"iters": 30})  # equals the default: no change
    bad.save(tmp_path / "bad.json")
    r = runner.invoke(app, ["recipe", "validate", str(tmp_path / "bad.json")])
    assert r.exit_code == 1 and "does not differ" in r.output
    r = runner.invoke(app, ["recipe", "run", str(tmp_path / "bad.json"), "--db", db])
    assert r.exit_code == 1

    # the existing exports work on recipe output unchanged
    r = runner.invoke(app, ["export", "table", "-e", "main-comparison", "-p", PROJECT, "--rows", "method",
                            "--cols", "config.blur", "--where", "config.noise=0.05", "--db", db])
    assert r.exit_code == 0, r.output
    assert "Ours" in r.output and r.output.count("\\multicolumn") == 2
    r = runner.invoke(app, ["export", "ablation-table", "-e", "ablation", "-p", PROJECT, "--db", db])
    assert r.exit_code == 0 and "w/o adaptive" in r.output


class Recorder(StudyObserver):
    def __init__(self):
        self.runs, self.done = [], None

    def on_run(self, job, instance, estimate, metrics, run_dir):
        self.runs.append((job.method, job.arm, instance.name, estimate.ok, run_dir))

    def on_study_done(self, report):
        self.done = report


class Configurable(ToyDeblurring):
    key = "toy-configurable"

    def __init__(self, device="cpu", tag="none"):
        self.device, self.tag = device, tag


def test_observers_diagnostics_dir_problem_options_and_provenance(tmp_path, engine):
    reg = Registry()
    reg.problem(Configurable)
    reg.method(AdaptiveGD)
    study = Study("obs", "comparison", "toy-configurable", [Arm("adaptive-gd", {"iters": 3})], project="p", n_instances=2)
    rec = Recorder()
    # no artifacts_dir: curves must still be persisted next to the database
    report = run_study(study, engine=engine, registry=reg, observers=[rec], log=None,
                       problem_options={"device": "cpu", "tag": "x"}, provenance={"oriel": "abc123", "dirty": 0})
    assert report.logged == 2 and rec.done is report
    assert [(m, a, n, ok) for m, a, n, ok, _ in rec.runs] == [("adaptive-gd", "adaptive-gd", "phantom_00", True),
                                                             ("adaptive-gd", "adaptive-gd", "phantom_01", True)]
    run_dir = rec.runs[0][4]
    diag_root = default_diagnostics_dir(engine)
    assert diag_root is not None and run_dir.is_relative_to(diag_root)
    assert sorted(p.name for p in run_dir.iterdir()) == ["diagnostics.json"]  # curves, but no images
    assert len(json.loads((run_dir / "diagnostics.json").read_text())["step_sizes"]) == 3
    recs = run_records(get_runs(experiment="obs", project="p", engine=engine), engine=engine)
    assert all(r["notes"] == "oriel=abc123 dirty=0" for r in recs)
    assert all(r["artifacts_dir"] == str(run_dir) for r in recs if r["instance"] == "phantom_00")
    # a failed run still gets the provenance and the observer call
    reg.method(Crashy)
    study = Study("obs2", "comparison", "toy-configurable", [Arm("crashy", {"mode": "raise"})], project="p")
    rec = Recorder()
    run_study(study, engine=engine, registry=reg, observers=[rec], log=None, provenance={"v": 1})
    assert rec.runs[0][3] is False
    assert run_records(get_runs(experiment="obs2", project="p", engine=engine), engine=engine)[0]["notes"] == "RuntimeError: boom; v=1"


def test_multi_knob_ablation_arms():
    assert arm_changes({"adaptive": False}) == (None, {"adaptive": False})
    assert arm_changes({"label": "no prior", "set": {"prior": "tikhonov", "reg": 0.1}}) == ("no prior", {"prior": "tikhonov", "reg": 0.1})
    for bad in ({}, {"a": 1, "b": 2}, {"set": {}}, {"set": {"a": 1}, "junk": 2}):
        with pytest.raises(ValueError):
            arm_changes(bad)
    methods = {"adaptive-gd": AdaptiveGD}
    study = Study("x", "ablation", "toy-deblur", [Arm("adaptive-gd")],
                  ablation=Ablation(arms=[{"label": "quadratic, weaker", "set": {"prior": "tikhonov", "reg": 0.001}}, {"set": {"iters": 5}}]))
    jobs = expand(study, ToyDeblurring, methods)
    assert [j.arm for j in jobs] == ["full model", "quadratic, weaker", "iters=5"]
    assert jobs[1].config["prior"] == "tikhonov" and jobs[1].config["reg"] == 0.001
    study.ablation.arms.append({"set": {"iters": 30, "reg": 0.003}})  # equals the base on every knob
    with pytest.raises(ValueError, match="does not differ"):
        validate_study(study, ToyDeblurring, methods)
    assert Study.from_dict(study.to_dict()) == study  # the set form round-trips


def test_select_best_is_the_tuning_rule(engine):
    _, sweep, _ = toy_studies()
    run_study(sweep, engine=engine, log=None)
    recs = run_records(get_runs(experiment="reg-sweep", project=PROJECT, engine=engine), engine=engine)
    best = agg.select_best(recs, "reg", "psnr")
    assert list(best) == [()] and best[()] in (0.003, 0.01)
    by_seed = agg.select_best(recs, "reg", "psnr", group_by=["seed"])
    assert set(by_seed) == {(0,), (1,), (2,)}
    assert agg.select_best(recs, "reg", "psnr", higher_is_better=False)[()] == 0.0003


# --------------------------------------------------------------------------- planning layer

def test_pending_subset_narrows_the_grid_and_stays_runnable():
    comparison, sweep, _ = toy_studies()
    from results_tracker.recipe import load_study_classes, pending_subset

    problem_cls, methods = load_study_classes(comparison)
    jobs = expand(comparison, problem_cls, methods)
    pending = [j for j in jobs if j.condition["blur"] == 2.0 and j.seed == 1 and j.method != "gd"]
    sub = pending_subset(comparison, pending, methods)
    assert sub.name == comparison.name and sub.conditions == {"blur": [2.0], "noise": [0.01, 0.05]} and sub.seeds == [1]
    assert [a.method for a in sub.methods] == ["wiener", "adaptive-gd"] and "pending subset" in sub.description
    assert len(expand(sub, problem_cls, methods)) == 4  # 2 noise x 2 arms; the original grid had 24 jobs
    assert "feeds" not in sub.to_dict()  # empty lists stay out of the spec
    problem_cls, methods = load_study_classes(sweep)
    jobs = expand(sweep, problem_cls, methods)
    sub = pending_subset(sweep, [j for j in jobs if j.config["reg"] in (0.001, 0.03)], methods)
    assert sub.sweep.values == [0.001, 0.03] and sub.seeds == [0, 1, 2]
    with pytest.raises(ValueError):
        pending_subset(sweep, [], methods)


def test_feeds_round_trip(tmp_path):
    study, *_ = toy_studies()
    study.feeds = ["tab:main", "fig:visual"]
    study.save(tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text())["feeds"] == ["tab:main", "fig:visual"]
    assert Study.load(tmp_path / "s.json").feeds == ["tab:main", "fig:visual"]


class Peeker(Method):
    """Records how many `running` rows the database holds while it works."""

    key = "peeker"
    knobs = (Knob("k", "int", 1),)
    engine = None
    seen: list = []

    def reconstruct(self, instance, config, state):
        Peeker.seen.append(len(get_runs(status="running", engine=Peeker.engine)))
        return Estimate(instance.measurement)


def test_running_rows_and_swept_knob_are_recorded(engine):
    from results_tracker import list_experiments

    reg = Registry()
    reg.problem(ToyDeblurring)
    reg.method(Peeker)
    Peeker.engine, Peeker.seen = engine, []
    study = Study("peek", "sweep", "toy-deblur", [Arm("peeker")], sweep=Sweep("k", [1, 2]), conditions={"blur": [1.0], "noise": [0.05]},
                  n_instances=2, description="watching the running rows", project=PROJECT)
    report = run_study(study, engine=engine, registry=reg, log=None)
    assert report.logged == 4 and Peeker.seen == [1, 1, 1, 1]  # exactly the current setting was `running` each time
    runs = get_runs(experiment="peek", project=PROJECT, engine=engine)
    assert len(runs) == 4 and {r.status.value for r in runs} == {"completed"}  # every running row was replaced
    exp = next(e for e in list_experiments(PROJECT, engine=engine) if e.name == "peek")
    assert exp.swept_params == ["k"] and exp.description == "watching the running rows" and exp.type.value == "sweep"
    Peeker.seen = []
    run_study(study, engine=engine, registry=reg, log=None, mark_running=False)
    assert Peeker.seen == []  # resumed: nothing ran


def test_declarations_plan_without_the_real_classes(tmp_path):
    from results_tracker.recipe import declared_registry, export_declarations, load_study_classes, registry, save_declarations
    from results_tracker.recipe.declared import load_declarations

    import results_tracker.recipe.toy  # noqa: F401  registers into the default registry

    decl = export_declarations(registry)
    assert {m["key"] for m in decl["methods"]} >= {"wiener", "gd", "adaptive-gd"} and decl["problems"][0]["key"] == "toy-deblur"
    path = save_declarations(registry, tmp_path / "knobs.json")
    reg = load_declarations(path)
    comparison, sweep, ablation = toy_studies()
    for study in (comparison, sweep, ablation):
        real_p, real_m = load_study_classes(study)
        decl_p, decl_m = load_study_classes(study, reg, import_modules=False)
        assert [(j.method, j.config, j.condition, j.seed, j.arm) for j in expand(study, decl_p, decl_m)] == \
               [(j.method, j.config, j.condition, j.seed, j.arm) for j in expand(study, real_p, real_m)]
    assert reg.methods["wiener"].is_baseline and reg.methods["adaptive-gd"].display_label() == AdaptiveGD.display_label()
    assert reg.problems["toy-deblur"].splits == ("test", "validation") and reg.problems["toy-deblur"].metric_definitions["psnr"] == ("dB", True, ".2f")
    with pytest.raises(RuntimeError):
        reg.methods["wiener"]().reconstruct(None, {}, None)
    assert declared_registry({"methods": [], "problems": []}).methods == {}
    # CLI
    r = runner.invoke(app, ["recipe", "export-knobs", "-i", "results_tracker.recipe.toy", "-o", str(tmp_path / "k2.json")])
    assert r.exit_code == 0 and json.loads((tmp_path / "k2.json").read_text())["version"] == 1
    r = runner.invoke(app, ["recipe", "knobs", "adaptive-gd", "-i", "results_tracker.recipe.toy", "--json"])
    assert r.exit_code == 0 and json.loads(r.output)["knobs"][0]["name"]
