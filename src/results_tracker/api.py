"""Public logging and query API.

    from results_tracker import log_run
    log_run("lambda-sweep", project="adaptive-pnp", method="ours",
            dataset="Set12", seed=0, config=cfg, metrics={"psnr": 31.2})
"""

from __future__ import annotations

import math
import socket
import subprocess
from datetime import datetime
from numbers import Number
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar, Union

from sqlmodel import Session, SQLModel, select

from .db import get_engine, session_scope
from .models import Dataset, Experiment, ExperimentType, Method, Metric, Project, Run, RunStatus

T = TypeVar("T", bound=SQLModel)

# Metric names that are minimised unless the user says otherwise.
_LOWER_IS_BETTER_HINTS = (
    "loss", "error", "err", "mse", "rmse", "nrmse", "mae", "lpips", "fid",
    "time", "runtime", "seconds", "iters", "iterations", "cost", "residual",
)


class AUTO:  # sentinel: "capture this automatically"
    pass


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
    db=None,
    engine=None,
) -> Run:
    """Record one run. Creates the project / experiment / method / dataset on first use.

    `experiment_type` only matters the first time an experiment is seen.
    """
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
                "timestamp": r.timestamp,
                "git_commit": r.git_commit,
                "artifacts_dir": r.artifacts_dir,
                "notes": r.notes,
            }
        )
    return out
