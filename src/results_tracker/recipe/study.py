"""Study specs and the generic runner.

A `Study` is data, not code: which problem, which grid of conditions, which methods with which knob
overrides, and either a swept knob (sweep) or a base plus single-knob arms (ablation). `expand` turns it
into `Job`s; `run_study` runs every job on every instance, scores it, saves artifacts, and logs one run
per instance with `log_run`. Re-running a study is a resume: settings already in the database are skipped.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from ..api import _resolve_engine, _to_plain, define_method, define_metric, get_metric_defs, has_run, log_run, set_experiment
from ..models import Method as MethodRow
from .core import Estimate, Instance, Method, Problem, Registry, registry as default_registry

KINDS = ("comparison", "sweep", "ablation")


@dataclass
class Arm:
    """One method in a study with the knobs it is run at (unset knobs take their defaults)."""

    method: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class Sweep:
    knob: str
    values: list[Any]


@dataclass
class Ablation:
    """`base`: knob overrides of the full model (on top of each arm's config); `arms`: what each variant changes.

    An arm is either `{knob: value}` (exactly one knob) or `{"label": "...", "set": {knob: value, ...}}` for a
    variant that only makes sense as a joint change (a floor arm together with its tolerance)."""

    base: dict[str, Any] = field(default_factory=dict)
    arms: list[dict[str, Any]] = field(default_factory=list)


def arm_changes(arm: Mapping[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
    """(label or None, {knob: value}) for either arm form; raises ValueError for anything else."""
    if "set" in arm:
        extra = set(arm) - {"set", "label"}
        if extra or not isinstance(arm["set"], Mapping) or not arm["set"]:
            raise ValueError(f"ablation arm {dict(arm)!r}: use {{'label': ..., 'set': {{knob: value, ...}}}}")
        return arm.get("label"), dict(arm["set"])
    if len(arm) != 1:
        raise ValueError(f"ablation arm {dict(arm)!r} must change exactly one knob (or use the 'set' form)")
    return None, dict(arm)


@dataclass
class Study:
    name: str  # the tracker experiment
    kind: str  # comparison | sweep | ablation
    problem: str  # registry key or module:Class
    methods: list[Arm]
    project: str = "default"
    conditions: dict[str, list[Any]] = field(default_factory=dict)  # condition knob -> values (default: [default])
    split: str = "test"
    n_instances: int = 1
    seeds: list[int] = field(default_factory=lambda: [0])
    sweep: Optional[Sweep] = None
    ablation: Optional[Ablation] = None
    artifacts_dir: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    description: str = ""
    imports: list[str] = field(default_factory=list)  # modules to import first (they register their classes)
    feeds: list[str] = field(default_factory=list)  # paper asset labels this study produces the data for (tab:main, fig:beta)

    # ------------------------------------------------------------------ (de)serialisation

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v not in (None, [], {}, "")} | {"name": self.name, "kind": self.kind}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Study":
        d = dict(d)
        d["methods"] = [Arm(**a) if isinstance(a, Mapping) else Arm(a) for a in d.get("methods", [])]
        if d.get("sweep"):
            d["sweep"] = Sweep(**d["sweep"])
        if d.get("ablation"):
            d["ablation"] = Ablation(**d["ablation"])
        return cls(**d)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Study":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class Job:
    """One (method, full config, condition, seed) cell of a study; runs over every instance."""

    method: str
    config: dict[str, Any]
    condition: dict[str, Any]
    seed: int
    tags: tuple[str, ...] = ()
    arm: str = ""  # human label: "full model", "adaptive=False", "reg=0.1"


@dataclass
class Report:
    experiment: str
    logged: int = 0
    skipped: int = 0
    failed: int = 0

    def __str__(self) -> str:
        return f"{self.experiment}: {self.logged} runs logged ({self.failed} failed), {self.skipped} already present"


# --------------------------------------------------------------------------- validation / expansion

def validate_study(study: Study, problem_cls: type[Problem], methods: Mapping[str, type[Method]]) -> None:
    """Raise ValueError for anything the runner could not execute or the tracker could not display."""
    if study.kind not in KINDS:
        raise ValueError(f"study {study.name!r}: kind must be one of {KINDS}, got {study.kind!r}")
    if not study.methods:
        raise ValueError(f"study {study.name!r}: no methods")
    if study.n_instances < 1 or not study.seeds:
        raise ValueError(f"study {study.name!r}: need n_instances >= 1 and at least one seed")
    if study.split not in problem_cls.splits:
        raise ValueError(f"study {study.name!r}: split {study.split!r} is not one of {list(problem_cls.splits)}")
    cond_space = problem_cls.condition_space()
    for name, values in study.conditions.items():
        if name not in cond_space:
            raise ValueError(f"study {study.name!r}: {name!r} is not a condition of {problem_cls.key!r} ({cond_space.names})")
        if not values:
            raise ValueError(f"study {study.name!r}: condition {name!r} has no values")
        for v in values:
            cond_space[name].validate(v)
    for arm in study.methods:
        cls = methods[arm.method]
        space = cls.space()
        clash = sorted(set(space.names) & set(cond_space.names))
        if clash:
            raise ValueError(f"method {cls.key!r} knob(s) {clash} collide with {problem_cls.key!r} condition names")
        space.resolve(arm.config)  # unknown keys / bad values raise
        if study.sweep is not None:
            if study.sweep.knob not in space:
                raise ValueError(f"sweep knob {study.sweep.knob!r} is not a knob of {cls.key!r} ({space.names})")
            if not study.sweep.values:
                raise ValueError("a sweep needs values")
            for v in study.sweep.values:
                space[study.sweep.knob].validate(v)
        if study.ablation is not None:
            base = space.resolve({**arm.config, **study.ablation.base})
            for changed in study.ablation.arms:
                _, changes = arm_changes(changed)
                for knob in changes:
                    if knob not in space:
                        raise ValueError(f"ablation knob {knob!r} is not a knob of {cls.key!r} ({space.names})")
                if all(space[k].validate(v) == base[k] for k, v in changes.items()):
                    raise ValueError(f"ablation arm {changed!r} does not differ from the base")
    if study.kind == "sweep" and study.sweep is None:
        raise ValueError("a sweep study needs `sweep`")
    if study.kind == "ablation" and study.ablation is None:
        raise ValueError("an ablation study needs `ablation`")


def condition_grid(study: Study, problem_cls: type[Problem]) -> list[dict[str, Any]]:
    """Every condition in the study's grid as a full, validated condition dict (unset knobs at default)."""
    space = problem_cls.condition_space()
    axes = [[space[k.name].validate(v) for v in study.conditions.get(k.name, [k.default])] for k in space]
    return [dict(zip(space.names, values)) for values in product(*axes)] if space.names else [{}]


def _fmt(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def expand(study: Study, problem_cls: type[Problem], methods: Mapping[str, type[Method]]) -> list[Job]:
    """Jobs in run order: seeds, then conditions, then arms and their variants."""
    validate_study(study, problem_cls, methods)
    jobs: list[Job] = []
    for seed in study.seeds:
        for condition in condition_grid(study, problem_cls):
            for arm in study.methods:
                cls = methods[arm.method]
                space = cls.space()
                if study.kind == "sweep" and study.sweep is not None:
                    for v in study.sweep.values:
                        cfg = space.resolve({**arm.config, study.sweep.knob: v})
                        jobs.append(Job(cls.key, cfg, condition, seed, (), f"{study.sweep.knob}={_fmt(cfg[study.sweep.knob])}"))
                elif study.kind == "ablation" and study.ablation is not None:
                    base = space.resolve({**arm.config, **study.ablation.base})
                    jobs.append(Job(cls.key, base, condition, seed, ("base",), "full model"))
                    for changed in study.ablation.arms:
                        label, changes = arm_changes(changed)
                        cfg = space.resolve({**base, **changes})
                        label = label or ", ".join(f"{k}={_fmt(cfg[k])}" for k in changes)
                        jobs.append(Job(cls.key, cfg, condition, seed, (), label))
                else:
                    jobs.append(Job(cls.key, space.resolve(arm.config), condition, seed, (), cls.key))
    return jobs


# --------------------------------------------------------------------------- running

def _slug(d: Mapping[str, Any]) -> str:
    text = "_".join(f"{k}={_fmt(v)}" for k, v in d.items()) or "default"
    return re.sub(r"[^A-Za-z0-9_.=+-]+", "-", text)


def _config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha1(json.dumps(_to_plain(dict(config)), sort_keys=True, default=str).encode()).hexdigest()[:8]


def _is_finite(x: Any) -> bool:
    """False if `x` is array-like with a non-finite entry; True when it cannot tell."""
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    try:
        import numpy as np

        return bool(np.isfinite(np.asarray(x, dtype=float)).all())
    except Exception:
        return True


def _scalar_metrics(diagnostics: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    scalars: dict[str, float] = {}
    rest: dict[str, Any] = {}
    for k, v in diagnostics.items():
        plain = _to_plain(v)
        if isinstance(plain, (int, float)) and not isinstance(plain, bool):
            scalars[k] = plain
        else:
            rest[k] = plain
    return scalars, rest


def _define(problem: Problem, method_classes: Iterable[type[Method]], engine) -> None:
    """Seed metric and method definitions once; never overwrite what is already in the database."""
    defined = set(get_metric_defs(engine=engine))
    definitions = {"runtime_s": ("s", False, ".3f"), **problem.metric_definitions}
    for name, (unit, hib, fmt) in definitions.items():
        if name not in defined:
            define_metric(name, unit=unit, higher_is_better=hib, fmt=fmt, engine=engine)
    from sqlmodel import Session, select

    with Session(engine) as s:
        known = {m.name for m in s.exec(select(MethodRow)).all()}
    for cls in method_classes:
        if cls.key not in known:
            define_method(cls.key, label=cls.display_label(), is_baseline=cls.is_baseline, engine=engine)


def pending_subset(study: Study, pending: Sequence[Job], methods: Mapping[str, type[Method]]) -> Study:
    """A copy of `study` narrowed to the part of its grid that still holds work: only the condition values,
    seeds, sweep values and method arms that occur in `pending` jobs. A grid is a product, so this is a superset
    of the pending cells; running it resumes, so nothing already logged is recomputed. Use it to hand the rest
    of a study to another machine or to run one slice at a time."""
    if not pending:
        raise ValueError(f"study {study.name!r}: nothing is pending")
    keyed = lambda v: (isinstance(v, str), v)  # noqa: E731
    conditions = {k: sorted({j.condition[k] for j in pending}, key=keyed) for k in dict.fromkeys(k for j in pending for k in j.condition)}
    seeds = sorted({j.seed for j in pending})
    pending_keys = {j.method for j in pending}
    arms = [arm for arm in study.methods if methods[arm.method].key in pending_keys]
    sweep = None
    if study.sweep is not None:
        sweep = Sweep(study.sweep.knob, sorted({j.config[study.sweep.knob] for j in pending}, key=keyed))
    note = f"pending subset of {study.name} written {datetime.now(timezone.utc):%Y-%m-%d}: {len(pending)} of the jobs"
    return replace(study, conditions=conditions, seeds=seeds, methods=arms, sweep=sweep,
                   description=f"{study.description} · {note}" if study.description else note)


def load_study_classes(
    study: Study, registry: Optional[Registry] = None, *, import_modules: bool = True
) -> tuple[type[Problem], dict[str, type[Method]]]:
    """Import the study's plugin modules and resolve its problem and method classes.

    With `import_modules=False` the study's `imports` are skipped and `registry` must already hold the classes
    (a planning-only registry built from a knob declaration file, see `recipe.declared`)."""
    reg = registry or default_registry
    if import_modules:
        for module in study.imports:
            importlib.import_module(module)
    problem_cls = reg.resolve_problem(study.problem)
    methods = {arm.method: reg.resolve_method(arm.method) for arm in study.methods}
    return problem_cls, methods


class StudyObserver:
    """Callbacks a caller can attach to `run_study`; override what you need.

    `on_run` fires after each instance is logged (whether it completed or failed); `on_study_done` once
    at the end. This is how a repo streams its own on-disk result files while the study runs."""

    def on_run(self, job: Job, instance: Instance, estimate: Estimate, metrics: Mapping[str, Any],
               run_dir: Optional[Path]) -> None:
        pass

    def on_study_done(self, report: Report) -> None:
        pass


def default_diagnostics_dir(engine) -> Optional[Path]:
    """`<database>.diagnostics/` next to a file database; None for an in-memory one."""
    database = getattr(getattr(engine, "url", None), "database", None)
    if not database or database == ":memory:":
        return None
    return Path(database).with_suffix(".diagnostics")


def run_study(
    study: Study,
    *,
    db=None,
    engine=None,
    registry: Optional[Registry] = None,
    resume: bool = True,
    artifacts_dir: Optional[str] = None,
    diagnostics_dir: Optional[str] = None,
    problem_options: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any]] = None,
    observers: Sequence[StudyObserver] = (),
    log: Optional[Callable[[str], None]] = print,
    mark_running: bool = True,
) -> Report:
    """Run every job of `study` on every instance and log one run per instance.

    A method that raises, returns a non-finite estimate, or whose metrics cannot be computed yields a
    `failed` run with the message in `notes`; the grid keeps going. With `resume` (default) a setting
    already logged as completed is skipped without being recomputed, so an interrupted study is simply
    started again.

    `artifacts_dir` overrides the study's own and receives images plus `diagnostics.json` per run. Without
    one, non-scalar diagnostics (curves) still survive: they go to `diagnostics_dir`, by default
    `<database>.diagnostics/`, so a run's trajectory is never lost for want of an artifacts folder.
    `problem_options` are passed to the problem's constructor (device, data root); they describe the
    machine, not the experiment, so they stay out of the run config. `provenance` (e.g. a dependency's
    commit) is appended to every run's notes. `observers` receive each logged run. With `mark_running` (default)
    a `running` row is logged when a setting starts and replaced by the result, so the GUI shows work in flight
    and a hard crash leaves a visible `running` row instead of nothing (the resume recomputes it)."""
    problem_cls, method_classes = load_study_classes(study, registry)
    problem = problem_cls(**dict(problem_options or {}))
    jobs = expand(study, problem_cls, method_classes)
    engine = _resolve_engine(engine, db)
    _define(problem, method_classes.values(), engine)
    set_experiment(study.name, project=study.project, experiment_type=study.kind, description=study.description or None,
                   swept_params=[study.sweep.knob] if study.sweep is not None else None, engine=engine)
    provenance_note = " ".join(f"{k}={v}" for k, v in (provenance or {}).items())

    methods: dict[str, Method] = {cls.key: cls() for cls in method_classes.values()}
    for m in methods.values():
        if not m.supports(problem):
            raise ValueError(f"method {m.key!r} does not support problem {problem.key!r}")
    dataset = problem.dataset_name(study.split)
    root = Path(artifacts_dir or study.artifacts_dir) if (artifacts_dir or study.artifacts_dir) else None
    with_images = root is not None
    if root is None:
        root = Path(diagnostics_dir) if diagnostics_dir else default_diagnostics_dir(engine)
    states: dict[tuple[str, Any], Any] = {}
    instances: dict[tuple[str, int], list[Instance]] = {}
    report = Report(study.name)
    say = log or (lambda _s: None)

    for job in jobs:
        key = (json.dumps(job.condition, sort_keys=True, default=str), job.seed)
        if key not in instances:
            instances[key] = list(problem.instances(job.condition, study.split, study.n_instances, job.seed))
        method = methods[job.method]
        full_config = {**job.condition, **job.config}
        for inst in instances[key]:
            if resume and has_run(study.name, project=study.project, method=job.method, dataset=dataset,
                                  instance=inst.name, seed=job.seed, config=full_config, engine=engine):
                report.skipped += 1
                continue
            skey = (job.method, method.setup_key(job.config))
            if skey not in states:
                states[skey] = method.setup(problem, job.config)
            if mark_running:
                log_run(study.name, project=study.project, experiment_type=study.kind, method=job.method, dataset=dataset,
                        instance=inst.name, seed=job.seed, config=full_config, metrics={}, status="running",
                        notes=f"started {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC", tags=[*study.tags, *job.tags],
                        engine=engine, on_duplicate="replace")
            t0 = perf_counter()
            try:
                est = method.reconstruct(inst, job.config, states[skey])
                if est.ok and not _is_finite(est.x):
                    est = Estimate(est.x, est.diagnostics, False, "non-finite estimate")
            except Exception as e:  # a crash is a result too
                est = Estimate.failed(f"{type(e).__name__}: {e}")
            runtime = perf_counter() - t0

            metrics: dict[str, Any] = {}
            if est.ok and inst.reference is not None:
                try:
                    metrics.update(problem.metrics(est, inst))
                except Exception as e:
                    est = Estimate(est.x, est.diagnostics, False, f"metrics failed: {type(e).__name__}: {e}")
            scalars, rest = _scalar_metrics(est.diagnostics)
            metrics.update({k: v for k, v in scalars.items() if k not in metrics})
            metrics.setdefault("runtime_s", runtime)

            artifacts = None
            if root is not None:
                arm_dir = f"{re.sub(r'[^A-Za-z0-9_.=+-]+', '-', job.arm)}-{_config_hash(job.config)}"
                run_dir = root / study.name / job.method / arm_dir / _slug(job.condition) / f"seed{job.seed}" / inst.name
                run_dir.mkdir(parents=True, exist_ok=True)
                if est.ok and with_images:
                    problem.save_artifacts(run_dir, inst, est)
                (run_dir / "diagnostics.json").write_text(json.dumps(
                    {"method": job.method, "arm": job.arm, "config": _to_plain(full_config), "ok": est.ok,
                     "message": est.message,
                     **({"metric_convention": dict(problem.metric_convention)} if problem.metric_convention else {}),
                     **rest}, indent=2, default=str))
                artifacts = str(run_dir)

            notes = "; ".join(part for part in (est.message[:400], provenance_note) if part)
            log_run(study.name, project=study.project, experiment_type=study.kind, method=job.method, dataset=dataset,
                    instance=inst.name, seed=job.seed, config=full_config, metrics=metrics,
                    status="completed" if est.ok else "failed", notes=notes,
                    artifacts_dir=artifacts, tags=[*study.tags, *job.tags], engine=engine)
            report.logged += 1
            report.failed += not est.ok
            for observer in observers:
                observer.on_run(job, inst, est, metrics, Path(artifacts) if artifacts else None)
            verdict = "FAILED " + est.message if not est.ok else " ".join(
                f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float) and k != "runtime_s")
            say(f"  {study.name} {job.method} [{job.arm}] {_slug(job.condition)} seed={job.seed} {inst.name}: {verdict}")
    for observer in observers:
        observer.on_study_done(report)
    say(str(report))
    return report
