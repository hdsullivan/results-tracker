"""results-tracker: log experiment runs, aggregate them, export paper-ready tables."""

from .api import (
    DuplicateRunError,
    DuplicateRunWarning,
    define_method,
    delete_runs,
    define_metric,
    get_metric_defs,
    get_runs,
    list_experiments,
    list_projects,
    log_run,
    run_records,
)
from .db import get_engine, resolve_db_path
from .models import Dataset, Experiment, ExperimentType, Method, Metric, Project, Run, RunStatus

__all__ = [
    "log_run",
    "delete_runs",
    "DuplicateRunWarning",
    "DuplicateRunError",
    "define_metric",
    "define_method",
    "get_metric_defs",
    "get_runs",
    "run_records",
    "list_experiments",
    "list_projects",
    "get_engine",
    "resolve_db_path",
    "Project",
    "Metric",
    "Dataset",
    "Method",
    "Experiment",
    "ExperimentType",
    "Run",
    "RunStatus",
]
