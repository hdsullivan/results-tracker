"""Shared Streamlit helpers: database selection, cached loading, page selectors."""

from __future__ import annotations

import os
from typing import Any, Optional

import streamlit as st

from ..api import get_metric_defs, get_runs, list_experiments, list_projects, run_records
from ..db import DEFAULT_DB, ENV_VAR, get_engine, resolve_db_path

Record = dict[str, Any]


def db_path() -> str:
    if "db" not in st.session_state:
        st.session_state["db"] = resolve_db_path(os.environ.get(ENV_VAR) or DEFAULT_DB)
    return st.session_state["db"]


def _mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


@st.cache_resource(show_spinner=False)
def engine_for(path: str):
    return get_engine(path)


@st.cache_data(show_spinner=False)
def _load_records(path: str, mtime: float, project: Optional[str], experiment: Optional[str]) -> list[Record]:
    engine = engine_for(path)
    runs = get_runs(experiment=experiment, project=project, engine=engine)
    return run_records(runs, engine=engine)


@st.cache_data(show_spinner=False)
def _load_metric_defs(path: str, mtime: float) -> dict[str, dict[str, Any]]:
    return {
        k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt}
        for k, m in get_metric_defs(engine=engine_for(path)).items()
    }


@st.cache_data(show_spinner=False)
def _load_catalog(path: str, mtime: float) -> dict[str, list[dict[str, Any]]]:
    engine = engine_for(path)
    projs = {p.id: p.name for p in list_projects(engine=engine)}
    exps = [
        {"project": projs[e.project_id], "experiment": e.name, "type": e.type.value, "description": e.description}
        for e in list_experiments(engine=engine)
    ]
    return {"projects": [{"name": n} for n in projs.values()], "experiments": exps}


def load_records(project: Optional[str] = None, experiment: Optional[str] = None) -> list[Record]:
    p = db_path()
    return _load_records(p, _mtime(p), project, experiment)


def load_metric_defs() -> dict[str, dict[str, Any]]:
    p = db_path()
    return _load_metric_defs(p, _mtime(p))


def load_catalog() -> dict[str, list[dict[str, Any]]]:
    p = db_path()
    return _load_catalog(p, _mtime(p))


def sidebar_db() -> str:
    """Database picker in the sidebar. Returns the active path."""
    with st.sidebar:
        new = st.text_input("Database", value=db_path(), help="SQLite file. Set $RESULTS_TRACKER_DB to change the default.")
        if new != st.session_state["db"]:
            st.session_state["db"] = resolve_db_path(new)
            st.rerun()
        if st.button("Refresh", help="Re-read the database"):
            st.cache_data.clear()
            st.rerun()
    return db_path()


def select_project_experiment(
    types: Optional[tuple[str, ...]] = None, prefer: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Sidebar selectors. Returns (project, experiment); either may be None if nothing exists.

    `types` restricts the list to those experiment types; `prefer` lists experiments
    of that type first so the page opens on a sensible default.
    """
    cat = load_catalog()
    projects = [p["name"] for p in cat["projects"]]
    with st.sidebar:
        if not projects:
            st.info("No projects yet. Run `results-tracker demo` or log a run.")
            return None, None
        project = st.selectbox("Project", projects, key="project")
        exps = [e for e in cat["experiments"] if e["project"] == project and (types is None or e["type"] in types)]
        if prefer:
            exps.sort(key=lambda e: e["type"] != prefer)
        if not exps:
            st.info("No experiments of this type in the project.")
            return project, None
        labels = {f"{e['experiment']}  ({e['type']})": e["experiment"] for e in exps}
        chosen = st.selectbox("Experiment", list(labels), key=f"experiment_{'_'.join(types or ())}_{prefer or ''}")
        return project, labels[chosen]


def fmt_for(defs: dict[str, dict[str, Any]], metric: str) -> str:
    return defs.get(metric, {}).get("fmt", ".2f")


def hib_map(defs: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {k: v["higher_is_better"] for k, v in defs.items()}
