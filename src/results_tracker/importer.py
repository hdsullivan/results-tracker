"""Bulk import of existing results from CSV files or JSON run files/directories.

    spec = ImportSpec(experiment="main-comparison", project="paper",
                      metric_cols=["psnr", "ssim"], config_cols=["lambda"])
    import_path("old_results/", spec, engine=engine)

Column mapping is done once in the spec. Anything not mapped falls back to a
heuristic: numeric columns become metrics (unless `metric_cols` is given, in which
case only those are metrics), everything else becomes config. A swept numeric
parameter such as `lambda` must therefore be listed in `config_cols`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from numbers import Number
from pathlib import Path
from typing import Any, Iterable, Optional, Union

from sqlmodel import Session, select

from .api import _resolve_engine, log_run
from .models import Experiment, Project, Run

PathLike = Union[str, Path]

RESERVED = ("method", "dataset", "seed", "instance", "status", "experiment", "project", "tags", "notes")


@dataclass
class ImportSpec:
    experiment: str
    project: str = "default"
    experiment_type: str = "comparison"
    # constants applied to every row (override columns)
    method: Optional[str] = None
    dataset: Optional[str] = None
    # column / key names in the source
    method_col: str = "method"
    dataset_col: str = "dataset"
    seed_col: str = "seed"
    instance_col: str = "instance"
    status_col: str = "status"
    metric_cols: list[str] = field(default_factory=list)
    config_cols: list[str] = field(default_factory=list)
    source: str = "imported"
    tags: list[str] = field(default_factory=list)
    skip_duplicates: bool = True


@dataclass
class ImportResult:
    imported: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)
    run_ids: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        s = f"imported {self.imported}, skipped {self.skipped} duplicate(s)"
        if self.errors:
            s += f", {len(self.errors)} error(s)"
        return s


# --------------------------------------------------------------------------- readers

def coerce(value: Any) -> Any:
    """CSV cells are strings; turn '0.1' -> 0.1, 'true' -> True, '' -> None."""
    if not isinstance(value, str):
        return value
    v = value.strip()
    if v == "" or v.lower() in ("nan", "none", "null"):
        return None
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def records_from_csv(path: PathLike) -> list[dict[str, Any]]:
    with open(path, newline="") as f:
        return [{k: coerce(v) for k, v in row.items() if k is not None} for row in csv.DictReader(f)]


def records_from_json(path: PathLike) -> list[dict[str, Any]]:
    """One JSON file: a run dict, or a list of run dicts. A directory: every *.json inside, recursively."""
    p = Path(path)
    if p.is_dir():
        out: list[dict[str, Any]] = []
        for f in sorted(p.rglob("*.json")):
            for r in records_from_json(f):
                r.setdefault("_file", str(f))
                out.append(r)
        return out
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise ValueError(f"{p}: expected a JSON object or a list of objects")
    return data


def read_records(path: PathLike) -> list[dict[str, Any]]:
    p = Path(path)
    if p.is_dir() or p.suffix.lower() == ".json":
        return records_from_json(p)
    if p.suffix.lower() in (".csv", ".tsv"):
        return records_from_csv(p)
    raise ValueError(f"don't know how to read {p} (use .csv, .json, or a directory of .json)")


# --------------------------------------------------------------------------- mapping

def normalize(raw: dict[str, Any], spec: ImportSpec) -> dict[str, Any]:
    """Turn one source row into keyword arguments for `log_run`."""
    row = dict(raw)
    row.pop("_file", None)
    nested_metrics = row.pop("metrics", None) if isinstance(row.get("metrics"), dict) else None
    nested_config = row.pop("config", None) if isinstance(row.get("config"), dict) else None

    method = spec.method if spec.method is not None else row.pop(spec.method_col, None)
    dataset = spec.dataset if spec.dataset is not None else row.pop(spec.dataset_col, None)
    seed = row.pop(spec.seed_col, None)
    instance = row.pop(spec.instance_col, None)
    status = row.pop(spec.status_col, None) or "completed"
    for k in RESERVED:
        row.pop(k, None)

    metrics: dict[str, Any] = dict(nested_metrics or {})
    config: dict[str, Any] = dict(nested_config or {})
    if spec.metric_cols:
        metrics.update({c: row.pop(c) for c in spec.metric_cols if c in row})
    if spec.config_cols:
        config.update({c: row.pop(c) for c in spec.config_cols if c in row})
    # Whatever is left: numeric columns become metrics unless the user listed metrics
    # explicitly (or the source has a nested "metrics" dict); everything else is config.
    for k, v in list(row.items()):
        if nested_metrics is None and not spec.metric_cols and isinstance(v, Number) and not isinstance(v, bool):
            metrics[k] = v
        else:
            config[k] = v

    return dict(
        method=None if method is None else str(method),
        dataset=None if dataset is None else str(dataset),
        seed=None if seed is None else int(seed),
        instance=None if instance is None else str(instance),
        status=str(status),
        metrics=metrics,
        config=config,
    )


def fingerprint(experiment: str, kw: dict[str, Any]) -> str:
    key = {
        "experiment": experiment,
        "method": kw.get("method"),
        "dataset": kw.get("dataset"),
        "seed": kw.get("seed"),
        "instance": kw.get("instance"),
        "config": kw.get("config") or {},
        "metrics": kw.get("metrics") or {},
    }
    return json.dumps(key, sort_keys=True, default=str)


def _existing_fingerprints(engine, project: str, experiment: str) -> set[str]:
    from .api import run_records  # local import: avoids a cycle at module load

    with Session(engine) as s:
        exp = s.exec(
            select(Experiment).join(Project).where(Project.name == project, Experiment.name == experiment)
        ).first()
        if exp is None:
            return set()
        runs = list(s.exec(select(Run).where(Run.experiment_id == exp.id)).all())
    out = set()
    for r in run_records(runs, engine=engine):
        out.add(fingerprint(experiment, r))
    return out


# --------------------------------------------------------------------------- import

def import_records(
    raws: Iterable[dict[str, Any]], spec: ImportSpec, *, db=None, engine=None, dry_run: bool = False
) -> ImportResult:
    engine = _resolve_engine(engine, db)
    result = ImportResult()
    seen = _existing_fingerprints(engine, spec.project, spec.experiment) if spec.skip_duplicates else set()
    for i, raw in enumerate(raws):
        try:
            kw = normalize(raw, spec)
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"row {i}: {e}")
            continue
        fp = fingerprint(spec.experiment, kw)
        if spec.skip_duplicates and fp in seen:
            result.skipped += 1
            continue
        seen.add(fp)
        if dry_run:
            result.imported += 1
            continue
        try:
            run = log_run(
                spec.experiment, project=spec.project, experiment_type=spec.experiment_type,
                source=spec.source, tags=spec.tags, git_commit=None, on_duplicate="allow",
                notes=f"imported from {raw.get('_file', '')}".strip(), engine=engine, **kw,
            )
        except Exception as e:  # noqa: BLE001
            result.errors.append(f"row {i}: {e}")
            continue
        result.imported += 1
        result.run_ids.append(run.id)
    return result


def import_path(path: PathLike, spec: ImportSpec, *, db=None, engine=None, dry_run: bool = False) -> ImportResult:
    return import_records(read_records(path), spec, db=db, engine=engine, dry_run=dry_run)
