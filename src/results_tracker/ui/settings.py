"""Settings page: what the tables and figures take as given — metric direction, unit and format; method labels,
baseline flags and display order; how the project's plots look (sizes, colours, label order); the project's
primary metric; and value maps that derive labelled groupings from raw config values (kernel index -> kernel
type). Everything here was CLI-only before (`metric define`, `method define`, `valuemap set`); the page writes
through the same API.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Optional

import streamlit as st

from .. import aggregate as agg
from ..api import (
    define_metric, define_method, define_value_map, delete_value_map, get_metric_defs, list_assets, list_methods, list_value_maps,
    set_project,
)
from ..valuemaps import derive, format_rules, parse_rules
from .common import db_path, engine_for, invalidate_records, keyed, keyed_selectbox, load_catalog, load_records, select_project, sidebar_db
from .tables import generic_html

NEW = "— new value map —"


def _saved(msg: str, *, records: bool = False) -> None:
    """Flash `msg` and rerun. `records=True` for an edit that changes what a record carries (a method's label
    or position, a value map's rules); everything else is cached on the database's mtime and reloads itself."""
    if records:
        invalidate_records()
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
    tab_metrics, tab_methods, tab_exps, tab_maps, tab_plots, tab_project = st.tabs(
        ["Metrics", "Methods", "Experiments", "Value maps", "Plots", "Project"])
    with tab_metrics:
        _metrics(engine)
    with tab_methods:
        _methods(engine)
    with tab_exps:
        _experiments(project, engine)
    with tab_maps:
        _value_maps(project, engine)
    with tab_plots:
        _plots(project)
    with tab_project:
        _project(project, engine)


def _experiments(project: str, engine) -> None:
    from ..api import set_experiment
    from ..models import EXPERIMENT_STAGES

    st.caption("Stage sorts the experiments of a paper: **paper** feeds the manuscript, **exploratory** is scratch work, **superseded** "
               "is kept for the record but hidden from every selector (a sidebar checkbox shows it). The description is what the "
               "Overview and the study specs print.")
    exps = [e for e in load_catalog()["experiments"] if e["project"] == project]
    if not exps:
        st.info("No experiments in this project.")
        return
    edits = {}
    h = st.columns([2, 1, 1, 4])
    for col, title in zip(h, ("Experiment", "Type", "Stage", "Description")):
        col.markdown(f"**{title}**")
    for e in exps:
        c1, c2, c3, c4 = st.columns([2, 1, 1, 4])
        c1.markdown(f"`{e['experiment']}`")
        c2.markdown(e["type"])
        stage = c3.selectbox("stage", list(EXPERIMENT_STAGES), index=list(EXPERIMENT_STAGES).index(e.get("stage") or ""),
                             key=f"set_exp_{e['experiment']}_stage", label_visibility="collapsed", format_func=lambda v: v or "(unsorted)")
        desc = c4.text_input("description", value=e.get("description") or "", key=f"set_exp_{e['experiment']}_desc", label_visibility="collapsed")
        edits[e["experiment"]] = (stage, desc)
    if st.button("Save experiments", key="set_exps_save", type="primary"):
        changed = 0
        for e in exps:
            stage, desc = edits[e["experiment"]]
            if (stage, desc) != (e.get("stage") or "", e.get("description") or ""):
                set_experiment(e["experiment"], project=project, experiment_type=e["type"], stage=stage, description=desc, engine=engine)
                changed += 1
        _saved(f"{changed} experiment(s) updated.")
    _rename_or_delete(project, exps, engine)


def _rename_or_delete(project: str, exps: list[dict], engine) -> None:
    """Fix a mistyped experiment name, merge two that should be one, or drop a botched grid.

    A name typed into a script was permanent until now: `experiment set` only edits the stage and the
    description, so the only cure was SQL. The name is stored by the assets that render the experiment and by
    the notes that mention it, which `rename_experiment` follows; study specs on disk are named here because
    nothing can rewrite them for you.
    """
    from ..api import delete_experiment, rename_experiment

    st.divider()
    st.markdown("**Rename, merge or delete**")
    names = [e["experiment"] for e in exps]
    chosen = keyed_selectbox("Experiment", names, "set_exp_admin", names[0])
    entry = next(e for e in exps if e["experiment"] == chosen)
    runs, completed = entry.get("runs", 0), entry.get("completed", 0)
    assets = [a for a in list_assets(project, engine=engine) if a.experiment == chosen or chosen in (a.extra_experiments or [])]
    specs = _specs_naming(project, chosen)
    st.caption(f"`{chosen}` · {runs} run(s), {completed} completed · {len(assets)} pinned asset(s) name it"
               + (f" ({', '.join(f'`{a.label}`' for a in assets[:4])}{'…' if len(assets) > 4 else ''})" if assets else ""))

    c1, c2 = st.columns([3, 1])
    new_name = c1.text_input("New name", value=chosen, key="set_exp_newname",
                             help="An existing name merges the two: the runs move into it and this one goes.").strip()
    merging = new_name in names and new_name != chosen
    if c2.button("Rename" if not merging else "Merge", key="set_exp_rename", type="primary",
                 disabled=not new_name or new_name == chosen):
        touched = rename_experiment(project, chosen, new_name, engine=engine)
        st.session_state.setdefault("_prefill", {})["set_exp_admin"] = new_name
        _saved(f"{'Merged' if touched['merged'] else 'Renamed'} `{chosen}` into `{new_name}`: {touched['runs']} run(s) moved, "
               f"{touched['assets']} asset(s) and {touched['notes']} note(s) followed.", records=True)
    if merging:
        st.warning(f"`{new_name}` exists: its runs and `{chosen}`'s will become one experiment, keeping `{new_name}`'s type and stage.")
    if specs:
        st.caption(f"Study spec(s) naming `{chosen}`: {', '.join(f'`{s}`' for s in specs)} — rename the experiment there too, or the "
                   "Studies page will plan it as a new one.")

    with st.expander(f"Delete `{chosen}`"):
        st.warning(f"Deletes the experiment and its {runs} run(s) from the database. Artifact folders on disk are kept, and so are "
                   "pinned assets naming it — they will report no data rather than rendering something else. This cannot be undone.")
        armed = st.checkbox(f"Yes, delete `{chosen}` and its {runs} run(s)", key="set_exp_delete_confirm")
        if st.button("Delete experiment", key="set_exp_delete", type="primary", disabled=not armed):
            out = delete_experiment(project, chosen, delete_runs=True, engine=engine)
            for k in ("set_exp_admin", "set_exp_newname", "set_exp_delete_confirm"):
                st.session_state.pop(k, None)
            _saved(f"Deleted `{chosen}` with {out['runs']} run(s)."
                   + (f" {out['assets_left_dangling']} pinned asset(s) now have no data." if out["assets_left_dangling"] else ""),
                   records=True)


def _specs_naming(project: str, experiment: str) -> list[str]:
    """Study specs in the project's studies directory whose `name` is this experiment (they are files on disk;
    a rename in the database cannot reach them)."""
    import json as _json

    from .studies import default_studies_dir

    folder = default_studies_dir(db_path(), project)
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.rglob("*.json")):
        try:
            d = _json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("name") == experiment:
            out.append(path.name)
    return out


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
        _saved(f"{changed} method(s) updated.", records=changed > 0)


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
        _saved(f"Saved derived.{str(name).strip()} ({len(rules)} rules); it is now offered as a grouping and filter key.", records=True)
    if current and b2.button("Delete", key="set_vm_delete"):
        delete_value_map(project, current.name, engine=engine)
        for k in ("set_vm_pick", "set_vm_name", "set_vm_field", "set_vm_rules", "set_vm_desc"):
            st.session_state.pop(k, None)
        _saved(f"Deleted derived.{current.name}.", records=True)


def _plots(project: str) -> None:
    """Everything about this project's plots in one place: the sizes (also in every page's sidebar), the colour of
    each method, and the order its values are drawn in. The same style renders the on-screen charts and every
    exported figure, so nothing here is screen-only."""
    from .. import plotstyle
    from .styling import SIZE_CONTROLS, WEIGHT_CONTROLS, plot_style, save_plot_style

    style = plot_style(project)
    st.caption("Sizes are the points the paper figures use; the GUI draws them ~1.35× larger so they read on screen. "
               "They apply to every chart of this project and to every figure `results-tracker export paper` writes.")
    edits: dict[str, float] = {}
    for controls, step in ((SIZE_CONTROLS, 0.5), (WEIGHT_CONTROLS, 0.1)):
        cols = st.columns(len(controls))
        for col, (field, label, help_) in zip(cols, controls):
            lo, hi = plotstyle.limits(field)
            edits[field] = col.number_input(label, value=float(getattr(style, field)), min_value=lo, max_value=hi,
                                            step=step, format="%.1f", help=help_, key=f"set_style_{field}")
    new = style.merge(**edits)

    records = load_records(project)
    known = [m.name for m in list_methods(engine=engine_for(db_path()))]
    ordered = agg.method_order(records)  # the project's display order (Method.position, then first seen)
    names = ordered + [m for m in known if m not in ordered]
    labels = agg.method_labels(records)
    if names:
        st.markdown("**Method colours**")
        st.caption("The hue each method gets wherever it appears, shown in this project's display order (Settings → Methods "
                   "sets it). A method left at its palette colour is not pinned: it takes the slot its position in the chart "
                   "gives it (first drawn blue, then red, green, purple, orange, brown, gray, pink), which changes when a "
                   "chart draws a different subset or a different order. Pin a colour to fix it everywhere.")
        palette = style.palette_colors(names)
        current = style.colors_for(names)
        picked: dict[str, Optional[str]] = {}
        per_row = 6
        for start in range(0, len(names), per_row):
            chunk = names[start:start + per_row]
            cols = st.columns(per_row)
            for col, name in zip(cols, chunk):
                got = col.color_picker(str(labels.get(name, name))[:22], value=current[name], key=f"set_style_color_{name}")
                picked[name] = None if str(got).lower() == palette[name].lower() else got
        new = new.with_colors(picked)

    if style.order:
        st.markdown("**Label order**")
        st.caption("Set on a chart's *Axes, label order and colours* expander; this is what is stored.")
        rows = [[key, ", ".join(values)] for key, values in sorted(style.order.items())]
        st.markdown(generic_html(["grouping key", "order its values are drawn in"], rows, left_cols=2,
                                 caption="Values not listed follow at the end, in their natural order."),
                    unsafe_allow_html=True)
        drop = st.multiselect("Forget the order of", sorted(style.order), key="set_style_drop")
        for key in drop:
            new = new.with_order(key, None)

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save plot style", key="set_style_save", type="primary", disabled=new.to_dict() == style.to_dict()):
        save_plot_style(project, new)
        _saved("Plot style saved; every chart and exported figure of this project uses it.")
    if b2.button("Reset to the lab style", key="set_style_reset", disabled=style.is_default):
        save_plot_style(project, plotstyle.PlotStyle())
        for k in [k for k in st.session_state if str(k).startswith("set_style_")]:
            del st.session_state[k]
        _saved("Plot style reset to the lab's IEEE style.")

    sources = [p["name"] for p in load_catalog()["projects"] if p["name"] != project and p.get("plot_style")]
    if sources:
        st.markdown("**Start from another paper**")
        st.caption("Copies its sizes, colours and label order over this project's. Method names that this project does "
                   "not use are carried along harmlessly; nothing else about the project changes.")
        c1, c2 = st.columns([2, 1])
        source = c1.selectbox("Copy the plot style of", sources, key="set_style_source")
        if c2.button("Copy style", key="set_style_copy"):
            save_plot_style(project, plot_style(source))
            for k in [k for k in st.session_state if str(k).startswith("set_style_")]:
                del st.session_state[k]
            _saved(f"Copied the plot style of {source}.")


def _project(project: str, engine) -> None:
    cat = load_catalog()
    entry = next((p for p in cat["projects"] if p["name"] == project), {})
    metrics = sorted(get_metric_defs(engine=engine))
    options = ["(guess: psnr, ssim, first)"] + metrics
    current = entry.get("primary_metric") or ""
    primary = st.selectbox("Primary metric", options, index=options.index(current) if current in options else 0, key="set_primary",
                           help="What the Overview headline and default figures use for this project.")
    studies_dir = st.text_input("Studies directory", value=entry.get("studies_dir") or "", key="set_studies_dir",
                                placeholder="~/Documents/Research/adaptivePnP/studies",
                                help="Where this project's study specs live; the Studies and Paper pages read it. Blank = "
                                     "$RESULTS_TRACKER_STUDIES or studies/ beside the database.")
    if st.button("Save project", key="set_project_save", type="primary"):
        set_project(project, primary_metric="" if primary.startswith("(") else primary, studies_dir=studies_dir.strip(), engine=engine)
        _saved(f"Project {project}: primary metric {primary}" + (f", studies in {studies_dir.strip()}" if studies_dir.strip() else "") + ".")
