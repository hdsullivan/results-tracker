"""Settings page: what the tables and figures take as given — metric direction, unit and format; method labels,
baseline flags and display order; the project's primary metric; and value maps that derive labelled groupings
from raw config values (kernel index -> kernel type). Everything here was CLI-only before (`metric define`,
`method define`, `valuemap set`); the page writes through the same API.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import streamlit as st

from .. import aggregate as agg
from ..api import (
    define_metric, define_method, define_value_map, delete_value_map, get_metric_defs, list_methods, list_value_maps, set_project,
)
from ..valuemaps import derive, format_rules, parse_rules
from .common import db_path, engine_for, keyed, keyed_selectbox, load_catalog, load_records, select_project, sidebar_db
from .tables import generic_html

NEW = "— new value map —"


def _saved(msg: str) -> None:
    st.cache_data.clear()
    st.session_state["settings_flash"] = msg
    st.rerun()


def render() -> None:
    st.title("Settings")
    sidebar_db()
    project = select_project()
    if project is None:
        return
    engine = engine_for(db_path())
    if "settings_flash" in st.session_state:
        st.success(st.session_state.pop("settings_flash"))
    tab_metrics, tab_methods, tab_maps, tab_project = st.tabs(["Metrics", "Methods", "Value maps", "Project"])
    with tab_metrics:
        _metrics(engine)
    with tab_methods:
        _methods(engine)
    with tab_maps:
        _value_maps(project, engine)
    with tab_project:
        _project(project, engine)


def _metrics(engine) -> None:
    st.caption("Direction decides what is bold; unit and format appear in every table header and cell. Guessed from the name "
               "on first log; fix it here once (`results-tracker metric define` does the same).")
    defs = get_metric_defs(engine=engine)
    if not defs:
        st.info("No metrics logged yet.")
        return
    edits: dict[str, tuple[str, bool, str]] = {}
    h = st.columns([2, 2, 2, 2])
    for col, title in zip(h, ("Metric", "Unit", "Direction", "Format")):
        col.markdown(f"**{title}**")
    for name, m in defs.items():
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        c1.markdown(f"`{name}`")
        unit = c2.text_input("unit", value=m.unit, key=f"set_metric_{name}_unit", label_visibility="collapsed")
        direction = c3.selectbox("direction", ["higher is better", "lower is better"], index=0 if m.higher_is_better else 1,
                                 key=f"set_metric_{name}_dir", label_visibility="collapsed")
        fmt = c4.text_input("format", value=m.fmt, key=f"set_metric_{name}_fmt", label_visibility="collapsed",
                            help="Python format spec, e.g. .2f or .3f")
        edits[name] = (unit, direction.startswith("higher"), fmt)
    if st.button("Save metrics", key="set_metrics_save", type="primary"):
        changed = 0
        for name, (unit, hib, fmt) in edits.items():
            try:
                format(1.0, fmt)
            except ValueError:
                st.error(f"`{name}`: {fmt!r} is not a format spec")
                return
            m = defs[name]
            if (unit, hib, fmt) != (m.unit, m.higher_is_better, m.fmt):
                define_metric(name, unit=unit, higher_is_better=hib, fmt=fmt, engine=engine)
                changed += 1
        _saved(f"{changed} metric definition(s) updated.")


def _methods(engine) -> None:
    st.caption("Labels are what tables and legends print (a `~\\cite{key}` suffix is kept in LaTeX, stripped on screen). "
               "Position fixes the row order (lower first; ties keep first-seen order); baselines lead visual figures.")
    methods = list_methods(engine=engine)
    if not methods:
        st.info("No methods logged yet.")
        return
    edits: dict[str, tuple[str, bool, int]] = {}
    h = st.columns([2, 3, 1, 1])
    for col, title in zip(h, ("Method", "Label", "Baseline", "Position")):
        col.markdown(f"**{title}**")
    for m in methods:
        c1, c2, c3, c4 = st.columns([2, 3, 1, 1])
        c1.markdown(f"`{m.name}`")
        label = c2.text_input("label", value=m.label, key=f"set_method_{m.name}_label", label_visibility="collapsed", placeholder=m.name)
        base = c3.checkbox("baseline", value=bool(m.is_baseline), key=f"set_method_{m.name}_base", label_visibility="collapsed")
        pos = int(c4.number_input("position", value=int(m.position or 0), step=1, key=f"set_method_{m.name}_pos", label_visibility="collapsed"))
        edits[m.name] = (label, base, pos)
    if st.button("Save methods", key="set_methods_save", type="primary"):
        changed = 0
        for m in methods:
            label, base, pos = edits[m.name]
            if (label, base, pos) != (m.label, bool(m.is_baseline), int(m.position or 0)):
                define_method(m.name, label=label, is_baseline=base, position=pos, engine=engine)
                changed += 1
        _saved(f"{changed} method(s) updated.")


def _value_maps(project: str, engine) -> None:
    st.caption("A value map derives a labelled field from a raw one: `config.kernel` 0-3 → isotropic, 4-7 → anisotropic, "
               "8-11 → motion becomes `derived.kernel_type`, usable as a row, column or filter key on every page and in "
               "`--rows` / `--cols` / `--where`. Columns follow the rule order. One line per rule: `label = v1, v2` or `label = lo-hi`.")
    maps = {vm.name: vm for vm in list_value_maps(project, engine=engine)}
    records = load_records(project)
    fields = [k for k in agg.grouping_keys(records, base=("method", "dataset", "instance", "seed", "experiment")) if not k.startswith("derived.")]
    pick = keyed_selectbox("Value map", [NEW] + list(maps), "set_vm_pick", NEW)
    current = maps.get(pick)
    c1, c2 = st.columns(2)
    with c1:
        name = keyed(st.text_input, "Derived field name", "set_vm_name", current.name if current else "", placeholder="kernel_type",
                     help="Used as derived.<name>")
    with c2:
        field = keyed_selectbox("Source field", fields, "set_vm_field", current.field if current else (fields[0] if fields else None))
    rules_text = keyed(st.text_area, "Rules (one per line)", "set_vm_rules", format_rules(current.rules) if current else "", height=120,
                       placeholder="isotropic = 0-3\nanisotropic = 4-7\nmotion = 8-11")
    description = keyed(st.text_input, "Description", "set_vm_desc", current.description if current else "")
    rules: list[dict[str, Any]] = []
    if str(rules_text).strip():
        try:
            rules = parse_rules(str(rules_text))
        except ValueError as e:
            st.error(str(e))
    if rules and field:
        counts = Counter((agg.fmt_value(agg.get_field(r, field)), derive(rules, agg.get_field(r, field))) for r in records)
        rows = [[raw, label or "— unmatched —", n] for (raw, label), n in sorted(counts.items(), key=lambda kv: str(kv[0][0]))]
        st.markdown(generic_html(["value", "→ label", "runs"], rows, left_cols=2,
                                 caption=f"How the rules map the {len(rows)} distinct values of {field} in this project's runs."),
                    unsafe_allow_html=True)
        unmatched = [r for r in rows if r[1] == "— unmatched —"]
        if unmatched:
            st.caption(f"{len(unmatched)} value(s) match no rule and will be grouped as (none).")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save value map", key="set_vm_save", type="primary", disabled=not (str(name).strip() and rules and field)):
        define_value_map(project, str(name).strip(), field=field, rules=rules, description=str(description), engine=engine)
        if current and current.name != str(name).strip():
            delete_value_map(project, current.name, engine=engine)
        st.session_state.setdefault("_prefill", {})["set_vm_pick"] = str(name).strip()  # a drawn widget's state is set on the next run
        _saved(f"Saved derived.{str(name).strip()} ({len(rules)} rules); it is now offered as a grouping and filter key.")
    if current and b2.button("Delete", key="set_vm_delete"):
        delete_value_map(project, current.name, engine=engine)
        for k in ("set_vm_pick", "set_vm_name", "set_vm_field", "set_vm_rules", "set_vm_desc"):
            st.session_state.pop(k, None)
        _saved(f"Deleted derived.{current.name}.")


def _project(project: str, engine) -> None:
    cat = load_catalog()
    entry = next((p for p in cat["projects"] if p["name"] == project), {})
    metrics = sorted(get_metric_defs(engine=engine))
    options = ["(guess: psnr, ssim, first)"] + metrics
    current = entry.get("primary_metric") or ""
    primary = st.selectbox("Primary metric", options, index=options.index(current) if current in options else 0, key="set_primary",
                           help="What the Overview headline and default figures use for this project.")
    if st.button("Save project", key="set_project_save", type="primary"):
        set_project(project, primary_metric="" if primary.startswith("(") else primary, engine=engine)
        _saved(f"Primary metric of {project}: {primary}.")
