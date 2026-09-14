"""Import page: bring existing results in from a CSV or JSON file, with the mapping shown before anything is written.

`results-tracker import` has done this from the command line all along; what it cannot do is show you how your
columns were read *before* you commit. A results file is usually a row per (method, image, seed) with a few
numbers and a few settings, and the one thing that goes wrong is a swept parameter landing in `metrics` -- the
preview below names what became a metric, what became config, and what the heuristic is unsure about.

Nothing is written until the button at the end; the dry run above it counts what would be imported and what
would be skipped as a duplicate.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ..importer import ImportSpec, import_records, normalize, parameter_like_metrics, read_records
from .common import db_path, engine_for, invalidate_records, keyed, keyed_multiselect, keyed_selectbox, load_catalog, page_url, sidebar_db

NEW = "— new —"
TYPES = ("comparison", "sweep", "ablation")
SOURCES = {"imported": "imported — results this repository produced",
           "reported": "reported — numbers copied from a paper (never re-run, shown hollow in figures)"}


def _read_upload(upload) -> list[dict[str, Any]]:
    """The uploaded bytes as raw records. Written to a temp file so the importer's own readers do the parsing --
    the GUI must not grow a second, subtly different CSV reader."""
    suffix = Path(upload.name).suffix.lower() or ".csv"
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / f"upload{suffix}"
        path.write_bytes(upload.getvalue())
        return read_records(path)


def _columns(raws: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(k for r in raws for k in r if k != "_file"))


def render() -> None:
    st.title("Import runs")
    sidebar_db()
    st.caption("A CSV (one row per run) or a JSON file — an object, or a list of them. Nothing is written until you "
               "press Import; the same thing from a terminal is `results-tracker import`.")

    upload = st.file_uploader("Results file", type=["csv", "tsv", "json"], key="imp_file")
    if upload is None:
        st.info("Pick a file to see how its columns will be read.")
        with st.expander("What the importer expects"):
            st.markdown(
                "- One row (or JSON object) per run. Columns named `method`, `dataset`, `seed`, `instance`, `status` are "
                "recognised as such; `metrics` and `config` may also be nested objects in JSON.\n"
                "- Everything else: numeric columns become **metrics**, the rest become **config** — override below.\n"
                "- A run already in the database (same experiment, method, dataset, instance, seed and config) is skipped, "
                "so re-importing a file that grew only adds the new rows.")
        return

    try:
        raws = _read_upload(upload)
    except Exception as e:  # noqa: BLE001 - a malformed file is the user's problem to see, not a traceback
        st.error(f"Could not read `{upload.name}`: {type(e).__name__}: {e}")
        return
    if not raws:
        st.warning(f"`{upload.name}` has no rows.")
        return
    columns = _columns(raws)
    st.success(f"`{upload.name}`: {len(raws)} row(s), {len(columns)} column(s).")
    with st.expander("First rows as read", expanded=False):
        st.dataframe(pd.DataFrame(raws[:20]).astype(str), width="stretch", hide_index=True)

    cat = load_catalog()
    projects = [p["name"] for p in cat["projects"]]
    c1, c2, c3 = st.columns(3)
    with c1:
        project = keyed_selectbox("Project", projects + [NEW], "imp_project", projects[0] if projects else NEW)
        if project == NEW:
            project = keyed(st.text_input, "New project", "imp_project_new", "").strip()
    with c2:
        experiments = [e["experiment"] for e in cat["experiments"] if e["project"] == project]
        experiment = keyed_selectbox("Experiment", experiments + [NEW], "imp_experiment", experiments[0] if experiments else NEW)
        if experiment == NEW:
            experiment = keyed(st.text_input, "New experiment", "imp_experiment_new", Path(upload.name).stem).strip()
    with c3:
        etype = keyed_selectbox("Type", list(TYPES), "imp_type", "comparison",
                                help="Only used when the experiment does not exist yet.")

    st.markdown("**Columns**")
    st.caption("Leave the metric list empty to let the importer decide (numeric → metric). Naming metrics explicitly is "
               "what stops a swept parameter such as `K` from becoming one.")
    c1, c2 = st.columns(2)
    with c1:
        metric_cols = keyed_multiselect("Metric columns", columns, "imp_metrics", [],
                                        help="Empty = every numeric column that is not recognised as a field.")
    with c2:
        config_cols = keyed_multiselect("Config columns", columns, "imp_config", [],
                                        help="Empty = everything that is not a metric or a recognised field.")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = keyed(st.text_input, "Method for every row", "imp_method", "",
                       placeholder="blank = the `method` column").strip()
    with c2:
        dataset = keyed(st.text_input, "Dataset for every row", "imp_dataset", "",
                        placeholder="blank = the `dataset` column").strip()
    with c3:
        source = keyed_selectbox("Source", list(SOURCES), "imp_source", "imported", format_func=SOURCES.get)
    c1, c2 = st.columns(2)
    with c1:
        tags = [t.strip() for t in keyed(st.text_input, "Tags (comma-separated)", "imp_tags", "").split(",") if t.strip()]
    with c2:
        skip_duplicates = keyed(st.checkbox, "Skip rows already in the database", "imp_skip", True)

    spec = ImportSpec(experiment=experiment, project=project, experiment_type=etype,
                      method=method or None, dataset=dataset or None,
                      metric_cols=list(metric_cols), config_cols=list(config_cols),
                      source=source, tags=tags, skip_duplicates=skip_duplicates)
    if not project or not experiment:
        st.warning("Name the project and the experiment to import into.")
        return

    st.markdown("**How the first row will be read**")
    try:
        first = normalize(raws[0], spec)
    except Exception as e:  # noqa: BLE001
        st.error(f"Row 1 cannot be mapped: {type(e).__name__}: {e}")
        return
    c1, c2 = st.columns(2)
    c1.json({k: v for k, v in first.items() if k not in ("metrics", "config")})
    c2.json({"metrics": first.get("metrics"), "config": first.get("config")})
    if not first.get("metrics"):
        st.warning("No metric was recognised in this row — the runs would carry no numbers. Name the metric columns above.")

    engine = engine_for(db_path())
    dry = import_records(raws, spec, engine=engine, dry_run=True)
    suspicious = parameter_like_metrics([normalize(r, spec) for r in raws], explicit=spec.metric_cols)
    st.caption(f"Dry run: **{dry.imported}** row(s) would be imported"
               + (f", **{dry.skipped}** skipped as already present" if dry.skipped else "")
               + (f", **{len(dry.errors)}** could not be mapped" if dry.errors else "") + ".")
    for e in dry.errors[:5]:
        st.caption(f":red[{e}]")
    if suspicious:
        st.warning("These look like swept parameters, not metrics: " + ", ".join(f"`{s}`" for s in suspicious)
                   + ". Add them to the config columns, or they will be averaged as results.")

    if st.button(f"Import {dry.imported} run(s) into {experiment}", type="primary", key="imp_go", disabled=not dry.imported):
        result = import_records(raws, spec, engine=engine)
        invalidate_records()
        st.session_state["imp_done"] = {"imported": result.imported, "skipped": result.skipped,
                                        "errors": result.errors[:5], "experiment": experiment, "project": project}
        st.rerun()

    done = st.session_state.get("imp_done")
    if done and done["experiment"] == experiment:
        st.success(f"Imported {done['imported']} run(s) into **{done['experiment']}**"
                   + (f", skipped {done['skipped']} already present" if done["skipped"] else "") + ".")
        href = page_url("comparison", project=done["project"], experiment=done["experiment"])
        st.markdown(f'<a href="{href}" target="_self">Open {done["experiment"]} on the Comparison page</a>', unsafe_allow_html=True)
        for e in done["errors"]:
            st.caption(f":red[{e}]")
    with st.expander("The same import from a terminal"):
        bits = [f"results-tracker import {upload.name} -e {experiment} -p {project} --type {etype}"]
        bits += [f"--method {method}"] if method else []
        bits += [f"--dataset {dataset}"] if dataset else []
        bits += [f"--metric-col {c}" for c in metric_cols] + [f"--config-col {c}" for c in config_cols]
        bits += [f"--tag {t}" for t in tags] + ([f"--source {source}"] if source != "imported" else [])
        bits += [] if skip_duplicates else ["--keep-duplicates"]
        bits += [f"--db {db_path()}"]
        st.code(" ".join(bits), language="bash")
