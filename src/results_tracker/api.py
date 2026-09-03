"""Public logging and query API.

    from results_tracker import log_run
    log_run("lambda-sweep", project="adaptive-pnp", method="ours",
            dataset="Set12", seed=0, config=cfg, metrics={"psnr": 31.2})
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import warnings
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar, Union

from sqlmodel import Session, SQLModel, select

from .db import get_engine, session_scope
from .models import Asset, AssetStatus, Dataset, Experiment, ExperimentType, Method, Metric, Project, Run, RunStatus

T = TypeVar("T", bound=SQLModel)

# Metric names that are minimised unless the user says otherwise.
_LOWER_IS_BETTER_HINTS = (
    "loss", "error", "err", "mse", "rmse", "nrmse", "mae", "lpips", "fid",
    "time", "runtime", "seconds", "iters", "iterations", "cost", "residual",
)


class AUTO:  # sentinel: "capture this automatically"
    pass


class DuplicateRunWarning(UserWarning):
    """A run with the same experiment/method/dataset/instance/seed/config already exists."""


class DuplicateRunError(ValueError):
    """Raised by log_run(on_duplicate="error")."""


ON_DUPLICATE = ("skip", "replace", "allow", "error")


def config_fingerprint(config: Optional[dict]) -> str:
    return json.dumps(_to_plain(config or {}), sort_keys=True, default=str)


def _find_duplicates(session: Session, exp: Experiment, method_id, dataset_id, instance, seed, config) -> list[Run]:
    """Existing runs of the same setting: same experiment, method, dataset, instance, seed and config."""
    stmt = select(Run).where(Run.experiment_id == exp.id, Run.method_id == method_id, Run.dataset_id == dataset_id)
    stmt = stmt.where(Run.instance == instance) if instance is not None else stmt.where(Run.instance.is_(None))  # type: ignore[union-attr]
    stmt = stmt.where(Run.seed == seed) if seed is not None else stmt.where(Run.seed.is_(None))  # type: ignore[union-attr]
    fp = config_fingerprint(config)
    return [r for r in session.exec(stmt).all() if config_fingerprint(r.config) == fp]


# --------------------------------------------------------------------------- helpers

def _resolve_engine(engine=None, db=None):
    return engine if engine is not None else get_engine(db)


def get_or_create(session: Session, model: Type[T], name: str, **defaults: Any) -> T:
    obj = session.exec(select(model).where(model.name == name)).first()  # type: ignore[attr-defined]
    if obj is None:
        obj = model(name=name, **defaults)
        session.add(obj)
        session.flush()
    return obj


def _get_or_create_experiment(
    session: Session, project: Project, name: str, exp_type: ExperimentType, description: str = ""
) -> Experiment:
    exp = session.exec(
        select(Experiment).where(Experiment.project_id == project.id, Experiment.name == name)
    ).first()
    if exp is None:
        exp = Experiment(project_id=project.id, name=name, type=exp_type, description=description)
        session.add(exp)
        session.flush()
    return exp


def guess_higher_is_better(metric_name: str) -> bool:
    n = metric_name.lower()
    return not any(h in n for h in _LOWER_IS_BETTER_HINTS)


def _ensure_metric_defs(session: Session, names: Iterable[str]) -> None:
    for name in names:
        if session.exec(select(Metric).where(Metric.name == name)).first() is None:
            session.add(Metric(name=name, higher_is_better=guess_higher_is_better(name)))
    session.flush()


def _to_plain(value: Any) -> Any:
    """Make numpy / torch scalars JSON-serialisable; NaN -> None."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "item") and callable(value.item):  # numpy / torch 0-d
        try:
            value = value.item()
        except Exception:  # pragma: no cover
            pass
    if hasattr(value, "tolist") and callable(value.tolist):  # arrays
        return value.tolist()
    if isinstance(value, Number):
        if isinstance(value, float) and math.isnan(value):
            return None
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain(v) for v in value]
    return str(value)


def _clean_metrics(metrics: Optional[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        v = _to_plain(v)
        if isinstance(v, bool):
            raise TypeError(f"metric {k!r} is a bool; log flags in `config`/`tags`, or an explicit 0/1 (int) if you mean a rate")
        if v is not None and not isinstance(v, Number):
            raise TypeError(f"metric {k!r} must be numeric or None, got {type(v).__name__}")
        out[str(k)] = v
    return out


def current_git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- write API

def log_run(
    experiment: str,
    *,
    project: str = "default",
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    instance: Optional[str] = None,
    seed: Optional[int] = None,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
    experiment_type: Union[str, ExperimentType] = ExperimentType.comparison,
    status: Union[str, RunStatus] = RunStatus.completed,
    source: str = "logged",
    artifacts_dir: Optional[str] = None,
    notes: str = "",
    tags: Optional[Sequence[str]] = None,
    timestamp: Optional[datetime] = None,
    git_commit: Union[str, None, Type[AUTO]] = AUTO,
    hostname: Union[str, None, Type[AUTO]] = AUTO,
    on_duplicate: str = "skip",
    db=None,
    engine=None,
) -> Run:
    """Record one run. Creates the project / experiment / method / dataset on first use.

    `experiment_type` only matters the first time an experiment is seen.

    Duplicate protection (`on_duplicate`): a run with the same experiment, method, dataset, instance, seed
    and config already in the database is a duplicate *setting*. Re-running a script must not inflate n, so:

    - "skip" (default): keep the existing completed run and return it, with a DuplicateRunWarning. If the
      existing duplicates are all failed/running, they are replaced by the new run (a re-run after a crash).
    - "replace": delete the existing duplicates and log the new run.
    - "allow": append regardless (deliberate repeats without a seed; the importer uses this).
    - "error": raise DuplicateRunError.
    """
    if on_duplicate not in ON_DUPLICATE:
        raise ValueError(f"on_duplicate must be one of {ON_DUPLICATE}")
    engine = _resolve_engine(engine, db)
    exp_type = ExperimentType(experiment_type)
    run_status = RunStatus(status)
    clean_metrics = _clean_metrics(metrics)
    clean_config = _to_plain(config or {})

    with session_scope(engine) as s:
        proj = get_or_create(s, Project, project)
        exp = _get_or_create_experiment(s, proj, experiment, exp_type)
        meth = get_or_create(s, Method, method) if method else None
        ds = get_or_create(s, Dataset, dataset) if dataset else None
        _ensure_metric_defs(s, clean_metrics.keys())

        if on_duplicate != "allow":
            dups = _find_duplicates(s, exp, meth.id if meth else None, ds.id if ds else None, instance, seed, clean_config)
            if dups:
                setting = f"{experiment}/{method}/{dataset}" + (f"/{instance}" if instance else "") + (f"/seed {seed}" if seed is not None else "")
                completed = [d for d in dups if d.status == RunStatus.completed]
                if on_duplicate == "error":
                    raise DuplicateRunError(f"run already logged for {setting} (ids {[d.id for d in dups]})")
                if on_duplicate == "skip" and completed:
                    warnings.warn(f"duplicate run skipped for {setting}: existing run id {completed[-1].id} kept "
                                  f"(pass on_duplicate='replace' to overwrite, 'allow' to append)", DuplicateRunWarning, stacklevel=2)
                    return completed[-1]
                # "replace", or "skip" with only failed/running duplicates: the new run supersedes them
                for d in dups:
                    s.delete(d)
                s.flush()

        run = Run(
            experiment_id=exp.id,
            method_id=meth.id if meth else None,
            dataset_id=ds.id if ds else None,
            instance=instance,
            seed=seed,
            config=clean_config,
            metrics=clean_metrics,
            status=run_status,
            source=source,
            git_commit=current_git_commit() if git_commit is AUTO else git_commit,
            hostname=socket.gethostname() if hostname is AUTO else hostname,
            artifacts_dir=str(artifacts_dir) if artifacts_dir else None,
            notes=notes,
            tags=list(tags or []),
        )
        if timestamp is not None:
            run.timestamp = timestamp
        s.add(run)
        s.flush()
        return run


def has_run(
    experiment: str,
    *,
    project: str = "default",
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    instance: Optional[str] = None,
    seed: Optional[int] = None,
    config: Optional[dict] = None,
    completed_only: bool = True,
    db=None,
    engine=None,
) -> bool:
    """Is this exact setting (experiment, method, dataset, instance, seed, config) already logged?

    The question a resumable runner asks before recomputing anything. With `completed_only` (default)
    a failed or running duplicate does not count, so a crashed setting is re-run."""
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        proj = s.exec(select(Project).where(Project.name == project)).first()
        if proj is None:
            return False
        exp = s.exec(select(Experiment).where(Experiment.project_id == proj.id, Experiment.name == experiment)).first()
        if exp is None:
            return False
        meth = s.exec(select(Method).where(Method.name == method)).first() if method else None
        ds = s.exec(select(Dataset).where(Dataset.name == dataset)).first() if dataset else None
        if (method and meth is None) or (dataset and ds is None):
            return False
        dups = _find_duplicates(s, exp, meth.id if meth else None, ds.id if ds else None, instance, seed, _to_plain(config or {}))
        if completed_only:
            dups = [d for d in dups if d.status == RunStatus.completed]
        return bool(dups)


def define_metric(
    name: str, *, unit: str = "", higher_is_better: bool = True, fmt: str = ".2f", db=None, engine=None
) -> Metric:
    engine = _resolve_engine(engine, db)
    with session_scope(engine) as s:
        m = s.exec(select(Metric).where(Metric.name == name)).first()
        if m is None:
            m = Metric(name=name)
            s.add(m)
        m.unit, m.higher_is_better, m.fmt = unit, higher_is_better, fmt
        s.flush()
        return m


def define_method(name: str, *, label: str = "", is_baseline: bool = False, db=None, engine=None) -> Method:
    engine = _resolve_engine(engine, db)
    with session_scope(engine) as s:
        m = get_or_create(s, Method, name)
        m.label, m.is_baseline = label, is_baseline
        s.flush()
        return m


def delete_runs(run_ids: Iterable[int], db=None, engine=None) -> int:
    """Delete runs by id. Returns how many rows were removed. Projects, experiments, methods and
    metric definitions are left in place (a re-run of the same setting will reuse them)."""
    engine = _resolve_engine(engine, db)
    ids = [int(i) for i in run_ids]
    if not ids:
        return 0
    with session_scope(engine) as s:
        rows = list(s.exec(select(Run).where(Run.id.in_(ids))).all())  # type: ignore[attr-defined]
        for r in rows:
            s.delete(r)
        return len(rows)


def set_experiment(
    name: str,
    *,
    project: str = "default",
    experiment_type: Union[str, ExperimentType, None] = None,
    description: Optional[str] = None,
    swept_params: Optional[Sequence[str]] = None,
    base_run_id: Optional[int] = None,
    db=None,
    engine=None,
) -> Experiment:
    """Create or annotate an experiment: its description, the knob(s) a sweep varies (`swept_params`, what the
    Sweep page and the Overview headline default to) and the ablation base run. `experiment_type` only matters
    when the experiment does not exist yet."""
    engine = _resolve_engine(engine, db)
    with session_scope(engine) as s:
        proj = get_or_create(s, Project, project)
        exp = _get_or_create_experiment(s, proj, name, ExperimentType(experiment_type or "comparison"))
        if description is not None:
            exp.description = description
        if swept_params is not None:
            exp.swept_params = list(swept_params)
        if base_run_id is not None:
            exp.base_run_id = base_run_id
        s.flush()
        return exp


# --------------------------------------------------------------------------- paper assets

def save_asset(
    project: str,
    label: str,
    *,
    kind: str,
    experiment: str,
    options: Optional[dict] = None,
    filters: Optional[dict] = None,
    caption: Optional[str] = None,
    status: Union[str, AssetStatus, None] = None,
    notes: Optional[str] = None,
    position: Optional[int] = None,
    fingerprint: Optional[str] = None,
    db=None,
    engine=None,
) -> Asset:
    """Pin (or re-pin) a paper asset: the table or figure `label` of `project` is rendered from `experiment`
    with `filters` and `options` (see export.paper.KINDS). Re-pinning replaces kind, experiment, filters and
    options and clears `exported_at` (the exported file no longer matches); status, caption, notes and
    position are kept unless given. New assets go to the end of the manuscript order."""
    engine = _resolve_engine(engine, db)
    with session_scope(engine) as s:
        proj = get_or_create(s, Project, project)
        a = s.exec(select(Asset).where(Asset.project_id == proj.id, Asset.label == label)).first()
        if a is None:
            last = max((x.position for x in s.exec(select(Asset).where(Asset.project_id == proj.id)).all()), default=-1)
            a = Asset(project_id=proj.id, label=label, kind=kind, experiment=experiment, position=last + 1)
            s.add(a)
        a.kind, a.experiment = kind, experiment
        a.options = _to_plain(options or {})
        a.filters = _to_plain(filters or {})
        a.exported_at = None
        if caption is not None:
            a.caption = caption
        if status is not None:
            a.status = AssetStatus(status)
        if notes is not None:
            a.notes = notes
        if position is not None:
            a.position = int(position)
        if fingerprint is not None:
            a.fingerprint = fingerprint
        a.updated_at = datetime.now(timezone.utc)
        s.flush()
        return a


def list_assets(project: Optional[str] = None, db=None, engine=None) -> list[Asset]:
    """Assets in manuscript order (position, then label)."""
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        stmt = select(Asset)
        if project:
            stmt = stmt.join(Project, Asset.project_id == Project.id).where(Project.name == project)
        return sorted(s.exec(stmt).all(), key=lambda a: (a.position, a.label))


def get_asset(project: str, label: str, db=None, engine=None) -> Optional[Asset]:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        return s.exec(select(Asset).join(Project, Asset.project_id == Project.id)
                      .where(Project.name == project, Asset.label == label)).first()


def update_asset(project: str, label: str, db=None, engine=None, **fields: Any) -> Asset:
    """Change bookkeeping fields (status, position, caption, notes, exported_at, fingerprint, label) in place."""
    engine = _resolve_engine(engine, db)
    allowed = {"status", "position", "caption", "notes", "exported_at", "fingerprint", "label"}
    bad = set(fields) - allowed
    if bad:
        raise ValueError(f"cannot update {sorted(bad)}; re-pin the asset to change kind, experiment, filters or options")
    with session_scope(engine) as s:
        a = s.exec(select(Asset).join(Project, Asset.project_id == Project.id)
                   .where(Project.name == project, Asset.label == label)).first()
        if a is None:
            raise LookupError(f"no asset {label!r} in project {project!r}")
        for k, v in fields.items():
            setattr(a, k, AssetStatus(v) if k == "status" else v)
        a.updated_at = datetime.now(timezone.utc)
        s.flush()
        return a


def delete_asset(project: str, label: str, db=None, engine=None) -> bool:
    engine = _resolve_engine(engine, db)
    with session_scope(engine) as s:
        a = s.exec(select(Asset).join(Project, Asset.project_id == Project.id)
                   .where(Project.name == project, Asset.label == label)).first()
        if a is None:
            return False
        s.delete(a)
        return True


# --------------------------------------------------------------------------- read API

def list_projects(db=None, engine=None) -> list[Project]:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        return list(s.exec(select(Project).order_by(Project.name)).all())


def list_experiments(project: Optional[str] = None, db=None, engine=None) -> list[Experiment]:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        stmt = select(Experiment)
        if project:
            stmt = stmt.join(Project).where(Project.name == project)
        return list(s.exec(stmt.order_by(Experiment.name)).all())


def get_metric_defs(db=None, engine=None) -> dict[str, Metric]:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        return {m.name: m for m in s.exec(select(Metric)).all()}


def get_runs(
    experiment: Optional[str] = None,
    project: Optional[str] = None,
    method: Optional[str] = None,
    dataset: Optional[str] = None,
    status: Optional[Union[str, RunStatus]] = None,
    db=None,
    engine=None,
) -> list[Run]:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        stmt = select(Run).join(Experiment, Run.experiment_id == Experiment.id)
        if experiment:
            stmt = stmt.where(Experiment.name == experiment)
        if project:
            stmt = stmt.join(Project, Experiment.project_id == Project.id).where(Project.name == project)
        if method:
            stmt = stmt.join(Method, Run.method_id == Method.id).where(Method.name == method)
        if dataset:
            stmt = stmt.join(Dataset, Run.dataset_id == Dataset.id).where(Dataset.name == dataset)
        if status:
            stmt = stmt.where(Run.status == RunStatus(status))
        return list(s.exec(stmt.order_by(Run.timestamp)).all())


def run_records(runs: Iterable[Run], db=None, engine=None) -> list[dict[str, Any]]:
    """Flatten Run rows to plain dicts with names instead of ids.

    This is the format the aggregation functions consume, so the analysis layer
    never touches the ORM.
    """
    engine = _resolve_engine(engine, db)
    runs = list(runs)
    with Session(engine) as s:
        methods = {m.id: m for m in s.exec(select(Method)).all()}
        datasets = {d.id: d for d in s.exec(select(Dataset)).all()}
        experiments = {e.id: e for e in s.exec(select(Experiment)).all()}
    out = []
    for r in runs:
        m = methods.get(r.method_id)
        d = datasets.get(r.dataset_id)
        e = experiments.get(r.experiment_id)
        ts = r.timestamp
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)  # stored in UTC; SQLite drops the zone
        out.append(
            {
                "run_id": r.id,
                "experiment": e.name if e else None,
                "experiment_type": e.type.value if e else None,
                "method": m.name if m else None,
                "method_label": (m.label or m.name) if m else None,
                "method_is_baseline": bool(m.is_baseline) if m else False,
                "dataset": d.name if d else None,
                "instance": r.instance,
                "seed": r.seed,
                "config": dict(r.config or {}),
                "metrics": dict(r.metrics or {}),
                "status": r.status.value if isinstance(r.status, RunStatus) else r.status,
                "source": r.source,
                "tags": list(r.tags or []),
                "timestamp": ts,
                "git_commit": r.git_commit,
                "artifacts_dir": r.artifacts_dir,
                "notes": r.notes,
            }
        )
    return out
