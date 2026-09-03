"""Shared Streamlit helpers: database selection, cached loading, page selectors, the `where` filter.

The project, the experiment and the filter are one selection for the whole GUI: every page reads and
writes the same `st.session_state` entries, and they are mirrored into the URL query string
(`?db=…&project=…&experiment=…&where=config.K=5`) so a view can be bookmarked or pasted into notes.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Optional, Sequence

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
    if qp.get("asset"):
        open_asset(qp.get("asset"), project=qp.get("project"))
        del qp["asset"]  # consumed: the session now carries the selection, and edits must not be undone on rerun


def open_asset(label: str, project: Optional[str] = None) -> bool:
    """Make a pinned asset the current selection: its project, experiment and filter, plus a prefill of the
    Export page's widgets with its options (see export.prefill_from_asset). Returns False if unknown."""
    from ..api import get_asset, list_assets

    engine = engine_for(db_path())
    a = get_asset(project, label, engine=engine) if project else next((x for x in list_assets(engine=engine) if x.label == label), None)
    if a is None:
        return False
    if not project:
        from ..api import list_projects

        project = {p.id: p.name for p in list_projects(engine=engine)}[a.project_id]
    from .export import prefill_from_asset

    ss = st.session_state
    ss[KEY_PROJECT], ss[KEY_EXPERIMENT] = project, a.experiment
    ss[KEY_WHERE] = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in (a.filters or {}).items()}
    for k in [k for k in ss if k == "where_fields" or str(k).startswith("where_value_")]:
        del ss[k]  # the filter widgets reseed from KEY_WHERE
    ss["_prefill"] = prefill_from_asset(a)
    ss["opened_asset"] = (a.label, a.experiment)
    return True


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


def page_url(page: str, **params: Any) -> str:
    """A relative link to another page of this GUI carrying the selection: `export?db=…&project=…&asset=…`.
    Following a link starts a new Streamlit session, so everything the target needs must be in the URL; the
    database is included only when it is not the default one."""
    from urllib.parse import urlencode

    db = st.session_state.get(KEY_DB)
    query = {"db": db} if db and db != _default_db() else {}
    query.update({k: v for k, v in params.items() if v is not None})
    return f"{page}?{urlencode(query)}" if query else page


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


def select_project() -> Optional[str]:
    """Sidebar project selector, shared by every page and mirrored into the URL. None when the database is empty."""
    cat = load_catalog()
    projects = [p["name"] for p in cat["projects"]]
    with st.sidebar:
        if not projects:
            st.info("No projects yet. Run `results-tracker demo` or log a run.")
            return None
        _take_query_params()
        current = st.session_state.get(KEY_PROJECT)
        project = st.selectbox("Project", projects, index=projects.index(current) if current in projects else 0)
        st.session_state[KEY_PROJECT] = project
        _sync_query_params()
        return project


def select_project_experiment(
    types: Optional[tuple[str, ...]] = None, prefer: Optional[str] = None
) -> tuple[Optional[str], Optional[str]]:
    """Sidebar selectors. Returns (project, experiment); either may be None if nothing exists.

    `types` restricts the list to those experiment types; `prefer` lists experiments of that type first so
    the page opens on a sensible default. The choice is shared by every page and mirrored into the URL.
    """
    project = select_project()
    if project is None:
        return None, None
    cat = load_catalog()
    with st.sidebar:
        exps = [e for e in cat["experiments"] if e["project"] == project and (types is None or e["type"] in types)]
        if prefer:
            exps.sort(key=lambda e: e["type"] != prefer)
        if not exps:
            st.info("No experiments of this type in the project.")
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

_fmt_value = agg.fmt_value


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


def _take_seed(key: str, seed: Any) -> tuple[Any, bool]:
    """(value, forced): a pending prefill for `key` (from an opened asset) wins over the stored widget state."""
    pre = st.session_state.get("_prefill") or {}
    if key in pre:
        return pre.pop(key), True
    return seed, False


def keyed_multiselect(label: str, options: list[Any], key: str, seed: Sequence[Any] = (), **kw) -> list[Any]:
    """A multiselect whose state lives under `key`, seeded from `seed` the first time it appears and pruned to
    `options` afterwards. Keyed widgets keep their identity across reruns, so clearing a value never resets
    the widget; Streamlit drops the key when a page without the widget is shown, and the seed brings it back."""
    ss = st.session_state
    val, forced = _take_seed(key, seed)
    current = list(val or ()) if (forced or key not in ss) else list(ss[key])
    ss[key] = [o for o in current if o in options]
    return st.multiselect(label, options, key=key, **kw)


def keyed_selectbox(label: str, options: list[Any], key: str, seed: Any = None, **kw) -> Any:
    """A selectbox with stable identity under `key`; an invalid stored value falls back to the first option."""
    ss = st.session_state
    val, forced = _take_seed(key, seed)
    current = val if (forced or key not in ss) else ss[key]
    if options:
        ss[key] = current if current in options else options[0]
    else:
        ss.pop(key, None)
    return st.selectbox(label, options, key=key, **kw)


def keyed_radio(label: str, options: list[Any], key: str, seed: Any = None, **kw) -> Any:
    ss = st.session_state
    val, forced = _take_seed(key, seed)
    current = val if (forced or key not in ss) else ss[key]
    ss[key] = current if current in options else options[0]
    return st.radio(label, options, key=key, **kw)


def keyed(widget, label: str, key: str, seed: Any, **kw) -> Any:
    """Any other keyed widget (text_input, text_area, checkbox, number_input, slider): `widget(label, key=key, **kw)`
    with its state seeded on first appearance or from a pending prefill."""
    ss = st.session_state
    val, forced = _take_seed(key, seed)
    if forced or key not in ss:
        ss[key] = val
    return widget(label, key=key, **kw)


_keyed_multiselect = keyed_multiselect


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
    return agg.where_text(active_where() if where is None else where)


def where_items(where: Optional[Where] = None) -> list[str]:
    """The filter as `field=value` items for `--where` and the URL."""
    return agg.where_items(active_where() if where is None else where)


def where_cli() -> str:
    """`--where 'a=1' --where 'b=[2,3]'` for the active filter, or an empty string."""
    return agg.where_cli(active_where())


# --------------------------------------------------------------------------- pin to paper

def pin_to_paper(choices: dict[str, dict[str, Any]], *, records: list[Record], key: str,
                 suggested_label: Optional[str] = None, caption: Optional[str] = None) -> None:
    """An expander that saves the current view as a paper asset (models.Asset) of the selected project.

    `choices` maps asset kind -> rendering options for that kind (one entry, or several to pick from). The
    asset is pinned to the current experiment and filter; `records` are the runs behind the view (for the
    fingerprint). `caption` fixes the manuscript caption; otherwise a text area asks for one.
    """
    from ..api import get_asset, save_asset
    from ..export.paper import KIND_TITLES, default_label, records_fingerprint

    ss = st.session_state
    project, experiment = ss.get(KEY_PROJECT), ss.get(KEY_EXPERIMENT)
    if not project or not experiment or not choices:
        return
    opened = ss.get("opened_asset")
    opened_label = opened[0] if opened and opened[1] == experiment else None
    kinds = list(choices)
    with st.expander("📌 Pin to paper" + (f" — {KIND_TITLES[kinds[0]]}" if len(kinds) == 1 else ""), expanded=False):
        kind = kinds[0] if len(kinds) == 1 else keyed_selectbox("As", kinds, f"{key}_kind", kinds[0], format_func=KIND_TITLES.get)
        engine = engine_for(db_path())
        c1, c2 = st.columns([2, 1])
        with c1:
            label = keyed(st.text_input, "LaTeX label", f"{key}_label", opened_label or suggested_label or default_label(kind, experiment),
                          help="tab:… for tables, fig:… for figures; also the file name under tables/ or figures/").strip()
        existing = get_asset(project, label, engine=engine) if label else None
        with c2:
            status = keyed_selectbox("Status", ["planned", "draft", "final"], f"{key}_status",
                                     existing.status.value if existing and existing.status.value != "dropped" else "planned")
        if caption is None:
            caption = keyed(st.text_area, "Caption in the manuscript (blank = auto-generated)", f"{key}_caption",
                            existing.caption if existing else "", height=70)
        where = active_where()
        st.caption(f"Renders **{experiment}**" + (f" with filter {where_text(where)}" if where else "") + f" as {KIND_TITLES[kind]}"
                   + (f". `{label}` exists ({existing.kind} of {existing.experiment}); pinning replaces what it renders." if existing else "."))
        with st.expander("Options that will be saved"):
            st.json(choices[kind])
        if st.button("Pin", key=f"{key}_button", disabled=not label, type="primary"):
            save_asset(project, label, kind=kind, experiment=experiment, options=choices[kind], filters=where,
                       caption=caption if caption is not None else None, status=status,
                       fingerprint=records_fingerprint(records), engine=engine)
            ss["opened_asset"] = (label, experiment)
            st.success(f"Pinned `{label}`. It is listed on the Paper page; `results-tracker export paper -p {project}` renders it.")


def fmt_for(defs: dict[str, dict[str, Any]], metric: str) -> str:
    return defs.get(metric, {}).get("fmt", ".2f")


def hib_map(defs: dict[str, dict[str, Any]]) -> dict[str, bool]:
    return {k: v["higher_is_better"] for k, v in defs.items()}
