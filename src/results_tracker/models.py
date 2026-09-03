"""SQLModel schema.

Everything is a Run. Experiment type (comparison / sweep / ablation) only
changes how runs are grouped and displayed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ExperimentType(str, Enum):
    comparison = "comparison"
    sweep = "sweep"
    ablation = "ablation"


class RunStatus(str, Enum):
    completed = "completed"
    failed = "failed"
    running = "running"


class AssetStatus(str, Enum):
    """Where a paper asset stands: planned (data may not exist yet), draft, final, or dropped from the paper."""

    planned = "planned"
    draft = "draft"
    final = "final"
    dropped = "dropped"


class Project(SQLModel, table=True):
    """One per paper."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    primary_metric: str = ""  # the metric headlines and default figures use ("" = guess psnr/ssim/first)
    created_at: datetime = Field(default_factory=utcnow)


class Metric(SQLModel, table=True):
    """How to display and rank a metric (PSNR up, RMSE down, ...)."""

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    unit: str = ""
    higher_is_better: bool = True
    fmt: str = ".2f"


class Dataset(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    description: str = ""
    instances: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Method(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    label: str = ""  # display label for tables; falls back to name
    is_baseline: bool = False
    position: int = 0  # display order in tables and legends (ties: first seen)


class Experiment(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str = Field(index=True)
    type: ExperimentType = Field(default=ExperimentType.comparison)
    description: str = ""
    swept_params: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    # Ablation reference run. Plain int (not FK) to avoid a circular FK with Run.
    base_run_id: Optional[int] = None
    created_at: datetime = Field(default_factory=utcnow)


class Run(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    experiment_id: int = Field(foreign_key="experiment.id", index=True)
    method_id: Optional[int] = Field(default=None, foreign_key="method.id", index=True)
    dataset_id: Optional[int] = Field(default=None, foreign_key="dataset.id", index=True)
    instance: Optional[str] = None  # e.g. a single test image within the dataset
    seed: Optional[int] = None
    config: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: RunStatus = Field(default=RunStatus.completed)
    source: str = "logged"  # logged | imported | reported (copied from a paper)
    git_commit: Optional[str] = None
    hostname: Optional[str] = None
    timestamp: datetime = Field(default_factory=utcnow)
    artifacts_dir: Optional[str] = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Asset(SQLModel, table=True):
    """One table or figure of the manuscript, pinned from a GUI view: which experiment, which filter, which
    rendering options. `results-tracker export paper` regenerates exactly these. `fingerprint` hashes the runs
    the asset was last rendered from, so the Paper page can say when the data moved under it."""

    __table_args__ = (UniqueConstraint("project_id", "label"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    label: str = Field(index=True)  # the LaTeX label: tab:main, fig:beta
    kind: str  # one of export.paper.KINDS
    experiment: str
    extra_experiments: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # pooled with `experiment`
    filters: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # the `where` filter
    options: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))  # rendering options for `kind`
    caption: str = ""
    status: AssetStatus = Field(default=AssetStatus.planned)
    position: int = 0  # manuscript order
    notes: str = ""
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    exported_at: Optional[datetime] = None
    fingerprint: Optional[str] = None  # records_fingerprint of the runs at the last pin or export


class ValueMap(SQLModel, table=True):
    """A derived, labelled grouping of a field's values for one project: `config.kernel` 0-3 -> "isotropic",
    4-7 -> "anisotropic", 8-11 -> "motion" becomes the field `derived.kernel_type`, usable wherever a config key
    is (row/column keys, filters, facets, table headers). Rules: `{"label": ..., "values": [...]}` or
    `{"label": ..., "range": [lo, hi]}` (inclusive); the first matching rule wins, unmatched values stay None."""

    __table_args__ = (UniqueConstraint("project_id", "name"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: int = Field(foreign_key="project.id", index=True)
    name: str = Field(index=True)  # the derived field name: derived.<name>
    field: str  # the source field: config.<key>, method, dataset, ...
    rules: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    description: str = ""
