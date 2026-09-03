"""Shared Streamlit helpers: database selection, cached loading, page selectors, the `where` filter.

The project, the experiment and the filter are one selection for the whole GUI: every page reads and
writes the same `st.session_state` entries, and they are mirrored into the URL query string
(`?db=…&project=…&experiment=…&where=config.K=5`) so a view can be bookmarked or pasted into notes.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

import streamlit as st

from .. import aggregate as agg
from ..api import get_metric_defs, get_runs, list_experiments, list_projects, run_records
from ..db import DEFAULT_DB, ENV_VAR, get_engine, resolve_db_path

Record = dict[str, Any]
Where = dict[str, list[Any]]  # field -> accepted values (any of)

# Selection keys in st.session_state. Plain values, not widget keys: pages render their own widgets
# from these and write the choice back, so a page can restrict the option list without losing the
# selection made elsewhere.
KEY_DB, KEY_PROJECT, KEY_EXPERIMENT, KEY_WHERE = "db", "project_name", "experiment_name", "where"

FILTER_BASE_FIELDS = ("method", "dataset", "instance", "seed", "status", "source")


def _default_db() -> str:
    return resolve_db_path(os.environ.get(ENV_VAR) or DEFAULT_DB)


def db_path() -> str:
    if KEY_DB not in st.session_state:
        st.session_state[KEY_DB] = resolve_db_path(st.query_params.get("db") or _default_db())
    return st.session_state[KEY_DB]


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


# --------------------------------------------------------------------------- URL <-> session state

def _take_query_params() -> None:
    """Seed the selection from the URL once per session (the URL is the bookmark, the session the truth)."""
    if st.session_state.get("_query_taken"):
        return
    st.session_state["_query_taken"] = True
    qp = st.query_params
    if qp.get("project"):
        st.session_state[KEY_PROJECT] = qp.get("project")
    if qp.get("experiment"):
        st.session_state[KEY_EXPERIMENT] = qp.get("experiment")
    items = qp.get_all("where")
    if items:
        try:
            st.session_state[KEY_WHERE] = {k: (list(v) if isinstance(v, list) else [v]) for k, v in agg.parse_where(items).items()}
        except ValueError:
            st.session_state[KEY_WHERE] = {}


def _sync_query_params() -> None:
    """Mirror the selection into the URL; the database only when it is not the default one."""
    qp = st.query_params
    db = st.session_state.get(KEY_DB)
    wanted: dict[str, Any] = {
        "db": db if db and db != _default_db() else None,
        "project": st.session_state.get(KEY_PROJECT),
        "experiment": st.session_state.get(KEY_EXPERIMENT),
        "where": where_items(st.session_state.get(KEY_WHERE, {})) or None,
    }
    for k, v in wanted.items():
        current = qp.get_all(k)
        if v is None:
            if current:
                del qp[k]
        elif current != (v if isinstance(v, list) else [v]):
            qp[k] = v


# --------------------------------------------------------------------------- sidebar: database, selection

def sidebar_db() -> str:
    """Database picker in the sidebar. Returns the active path."""
    with st.sidebar:
        new = st.text_input("Database", value=db_path(), help="SQLite file. Set $RESULTS_TRACKER_DB to change the default.")
        if new != st.session_state[KEY_DB]:
            st.session_state[KEY_DB] = resolve_db_path(new)
            st.rerun()
        if st.button("Refresh", help="Re-read the database"):
            st.cache_data.clear()
            st.rerun()
    return db_path()


def select_project_experiment(
    types: Optional[tuple[str, ...]] = None, prefer: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Sidebar selectors. Returns (project, experiment); either may be None if nothing exists.

    `types` restricts the list to those experiment types; `prefer` lists experiments of that type first so
    the page opens on a sensible default. The choice is shared by every page and mirrored into the URL.
    """
    cat = load_catalog()
    projects = [p["name"] for p in cat["projects"]]
    with st.sidebar:
        if not projects:
            st.info("No projects yet. Run `results-tracker demo` or log a run.")
            return None, None
        _take_query_params()
        current = st.session_state.get(KEY_PROJECT)
        project = st.selectbox("Project", projects, index=projects.index(current) if current in projects else 0)
        st.session_state[KEY_PROJECT] = project
        exps = [e for e in cat["experiments"] if e["project"] == project and (types is None or e["type"] in types)]
        if prefer:
            exps.sort(key=lambda e: e["type"] != prefer)
        if not exps:
            st.info("No experiments of this type in the project.")
            _sync_query_params()
            return project, None
        names = [e["experiment"] for e in exps]
        labels = [f"{e['experiment']}  ({e['type']})" for e in exps]
        current = st.session_state.get(KEY_EXPERIMENT)
        chosen = st.selectbox("Experiment", labels, index=names.index(current) if current in names else 0)
        experiment = names[labels.index(chosen)]
        st.session_state[KEY_EXPERIMENT] = experiment
        _sync_query_params()
        return project, experiment


# --------------------------------------------------------------------------- sidebar: where filter

def _fmt_value(v: Any) -> str:
    if v is None:
        return "(none)"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _distinct(records: Iterable[Record], field: str) -> list[Any]:
    seen: list[Any] = []
    for r in records:
        v = agg.get_field(r, field)
        if not any(agg.same_value(v, s) for s in seen):
            seen.append(v)
    return sorted(seen, key=lambda x: (x is None, isinstance(x, str), x if x is not None else 0))


def filter_fields(records: list[Record]) -> list[str]:
    """Fields worth filtering on: base fields and config keys that take more than one value here."""
    fields = [f for f in FILTER_BASE_FIELDS if len(_distinct(records, f)) > 1]
    config_keys = sorted({k for r in records for k in agg.flatten(r.get("config", {}))})
    fields += [f"config.{k}" for k in config_keys if len(_distinct(records, f"config.{k}")) > 1]
    return fields


def _keyed_multiselect(label: str, options: list[str], key: str, seed: list[str], help: Optional[str] = None) -> list[str]:
    """A multiselect whose state lives under `key`, seeded from `seed` the first time it appears and pruned to
    `options` afterwards. Keyed widgets keep their identity across reruns, so clearing a value never resets
    the widget; Streamlit drops the key when a page without the widget is shown, and the seed brings it back."""
    ss = st.session_state
    ss[key] = [o for o in (ss[key] if key in ss else seed) if o in options]
    return st.multiselect(label, options, key=key, help=help)


def sidebar_filter(records: list[Record]) -> list[Record]:
    """The shared `where` filter in the sidebar; returns the records that match.

    Mirrors the CLI's `--where field=value`: every page applies the same filter to the same experiment,
    so a table, its sweep and its export agree on which runs they describe. Fields whose values do not
    vary here are not offered; a stored filter on a field this experiment lacks is dropped.
    """
    if not records:
        return []
    fields = filter_fields(records)
    stored: Where = st.session_state.get(KEY_WHERE, {})
    with st.sidebar:
        st.markdown("**Filter**")
        selected = _keyed_multiselect("Filter on", fields, "where_fields", list(stored),
                                      help="Keep only runs whose field has one of the chosen values; the same filter is applied on "
                                           "every page and mirrored in the URL. Same as `--where field=value` on the command line.")
        where: Where = {}
        for field in selected:
            values = _distinct(records, field)
            labels = [_fmt_value(v) for v in values]
            seed = [labels[i] for i, v in enumerate(values) if any(agg.same_value(v, w) for w in stored.get(field, []))]
            picked = _keyed_multiselect(field, labels, f"where_value_{field}", seed,
                                        help=f"Values of {field} in this experiment; none = no constraint")
            if picked:
                where[field] = [values[labels.index(lbl)] for lbl in picked]
        out = agg.filter_records(records, where) if where else list(records)
        if where:
            st.caption(f"{len(out)} of {len(records)} runs match · {where_text(where)}")
            if not out:
                st.warning("No runs match the filter.")
    st.session_state[KEY_WHERE] = where
    _sync_query_params()
    return out


def active_where() -> Where:
    return dict(st.session_state.get(KEY_WHERE, {}))


def where_text(where: Optional[Where] = None) -> str:
    """Human form of the filter: `dataset = Set12 · config.K ∈ {2, 5}`."""
    where = active_where() if where is None else where
    parts = []
    for k, vs in where.items():
        vs = list(vs) if isinstance(vs, (list, tuple, set)) else [vs]
        parts.append(f"{k} = {_fmt_value(vs[0])}" if len(vs) == 1 else f"{k} ∈ {{{', '.join(_fmt_value(v) for v in vs)}}}")
    return " · ".join(parts)


def _cli_value(v: Any) -> str:
    if isinstance(v, str):
        try:
            json.loads(v)
        except ValueError:
            return v  # plain word: parse_where keeps it as a string
        return json.dumps(v)  # looks like a number/bool: quote it so it stays a string
    return json.dumps(v, separators=(",", ":"))


def where_items(where: Optional[Where] = None) -> list[str]:
    """The filter as `field=value` items for `--where` and the URL (`agg.parse_where` inverts this)."""
    where = active_where() if where is None else where
    items = []
    for k, vs in where.items():
        vs = list(vs) if isinstance(vs, (list, tuple, set)) else [vs]
        items.append(f"{k}={_cli_value(vs[0]) if len(vs) == 1 else _cli_value(vs)}")
    return items


def where_cli() -> str:
    """`--where 'a=1' --where 'b=[2,3]'` for the active filter, or an empty string."""
    return " ".join(f"--where '{it}'" for it in where_items())


def fmt_for(defs: dict[str, dict[str, Any]], metric: str) -> str:
    return defs.get(metric, {}).get("fmt", ".2f")


def hib_map(defs: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {k: v["higher_is_better"] for k, v in defs.items()}
