"""Runs page: every run of a project in one filterable list -- what failed and why, what is still running,
and how to find a run without knowing which experiment it is in.

The analysis pages aggregate completed runs; this is where the ones they drop are accounted for. A failed run
carries the runner's exception message in its notes (`recipe.study.run_study` writes it there), shown inline
here instead of only on Run detail after guessing which run to open. A job that was killed leaves a `running`
row that nothing will ever complete: the cleanup below removes the ones older than a threshold, which is the
only way to do it outside `results-tracker delete --status running`.

Everything is SQL with one page at a time (`api.query_runs`), so a 50k-run database is never loaded to show
50 rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd
import streamlit as st

from ..api import delete_runs, list_methods, query_runs, run_records
from .common import (db_path, engine_for, fmt_timestamp, keyed, keyed_multiselect, keyed_selectbox, load_catalog,
                     page_url, select_project, sidebar_db)

ALL = "— all —"
STATUSES = ("completed", "failed", "running")
PAGE_SIZES = [50, 100, 200, 500]
STALE_LIMIT = 1000  # running rows examined by the cleanup; more than this is a broken pipeline, not a GUI job


def _age_hours(ts: Optional[datetime]) -> Optional[float]:
    if ts is None:
        return None
    ts = ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds() / 3600


def fmt_age(hours: Optional[float]) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return f"{hours * 60:.0f} min"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"


def _frame(recs: list[dict[str, Any]], project: Optional[str]) -> tuple[pd.DataFrame, dict]:
    """The page's rows as a frame, plus its column config.

    Triage columns only -- who, when, what state, and what the runner said. Metric values are deliberately
    absent: a list spanning experiments would carry a mostly empty column per metric of every other
    experiment. The numbers are one click away on Run detail, and complete in the CSV download.
    """
    rows = []
    for r in recs:
        row = {
            "run": page_url("run", project=project, experiment=r["experiment"], run=r["run_id"]),
            "id": r["run_id"],
            "logged": fmt_timestamp(r["timestamp"]),
            "experiment": r["experiment"],
            "method": r["method"] or "—",
            "dataset": r["dataset"] or "—",
            "instance": r["instance"] or "—",
            # text, not a number: a column mixing ints with the "—" placeholder cannot be serialised to
            # Arrow, and st.dataframe raises rather than drawing the table
            "seed": "—" if r["seed"] is None else str(r["seed"]),
            "status": r["status"],
            "metrics": ", ".join(f"{k}={v:g}" for k, v in sorted(r["metrics"].items()) if isinstance(v, (int, float)))[:60] or "—",
            "message": (r.get("notes") or "").split(";")[0][:300],
        }
        rows.append(row)
    config = {
        "run": st.column_config.LinkColumn("open", display_text="open", width="small",
                                           help="Opens this run on the Run detail page"),
        "metrics": st.column_config.TextColumn("metrics", width="medium", help="As logged; the CSV below has every value"),
        "message": st.column_config.TextColumn("message", width="large",
                                               help="What the runner recorded: a failed run's exception, a running row's start time"),
    }
    return pd.DataFrame(rows), config


def render() -> None:
    st.title("Runs")
    sidebar_db()
    project = select_project()
    if project is None:
        return
    engine = engine_for(db_path())
    experiments = [e["experiment"] for e in load_catalog()["experiments"] if e["project"] == project]

    with st.sidebar:
        st.markdown("**Runs**")
        experiment = keyed_selectbox("Experiment", [ALL] + experiments, "runs_experiment", ALL)
        statuses = keyed_multiselect("Status", list(STATUSES), "runs_status", [],
                                     help="Empty = every status. The analysis pages only ever show completed runs.")
        methods = [m.name for m in list_methods(engine=engine)]
        method = keyed_selectbox("Method", [ALL] + methods, "runs_method", ALL)
        search = keyed(st.text_input, "Search", "runs_search", "", placeholder="message, config value, host, commit",
                       help="Substring of what a run carries as text: its notes (where a crash message lands), instance, "
                            "config, tags, hostname or commit.")
        per_page = keyed_selectbox("Rows per page", PAGE_SIZES, "runs_per_page", 100)

    scope = dict(project=project, experiment=None if experiment == ALL else experiment,
                 method=None if method == ALL else method, search=search.strip())
    signature = (project, experiment, tuple(statuses), method, search.strip(), per_page)
    if st.session_state.get("_runs_signature") != signature:  # a new question starts at the first page
        st.session_state["_runs_signature"] = signature
        st.session_state["runs_offset"] = 0
    offset = int(st.session_state.get("runs_offset", 0))

    rows, counts = query_runs(statuses=statuses, limit=per_page, offset=offset, engine=engine, **scope)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matching runs", counts["total"])
    c2.metric("Completed", counts["completed"])
    c3.metric("Failed", counts["failed"], help="Their message is in the table below.")
    c4.metric("Running", counts["running"], help="A row a job logged at its start; a killed job leaves it behind.")
    if not counts["total"]:
        st.info("No run matches these filters. Clear the search or the status filter in the sidebar.")
        _stale_running(project, scope["experiment"], engine)
        return

    recs = run_records(rows, engine=engine)
    frame, column_config = _frame(recs, project)
    shown = f"{offset + 1}–{min(offset + len(rows), counts['total'])} of {counts['total']}"
    st.caption(f"Newest first · showing {shown} · sort a column by clicking its header · "
               "tick rows on the left to move or delete them")
    event = st.dataframe(frame, width="stretch", hide_index=True, column_config=column_config,
                         on_select="rerun", selection_mode="multi-row", key="runs_table")
    picked = [recs[i] for i in getattr(event.selection, "rows", []) if i < len(recs)]
    if picked:
        _selected_actions(picked, project, experiments, engine)

    b1, b2, b3, _ = st.columns([1, 1, 2, 4])
    if b1.button("← Newer", disabled=offset == 0, key="runs_prev"):
        st.session_state["runs_offset"] = max(0, offset - per_page)
        st.rerun()
    if b2.button("Older →", disabled=offset + per_page >= counts["total"], key="runs_next"):
        st.session_state["runs_offset"] = offset + per_page
        st.rerun()
    from ..export.csv import runs_csv

    b3.download_button("Download this page (CSV)", runs_csv(recs), file_name=f"{project}-runs.csv", mime="text/csv",
                       key="runs_csv", help="The rows above. `results-tracker runs` exports any selection from the command line.")

    if counts["failed"]:
        st.caption(f"{counts['failed']} failed run(s) match. Filter **Status → failed** to see only those; the message "
                   "column holds what the runner recorded, and the full notes are on Run detail.")
    _stale_running(project, scope["experiment"], engine)


def _selected_actions(picked: list[dict[str, Any]], project: str, experiments: list[str], engine) -> None:
    """Move or delete the ticked runs.

    Runs logged under a mistyped experiment name, or a handful that crashed and should not sit in the table,
    were a `results-tracker delete` away before; here they are the rows you are already looking at.
    """
    from ..api import move_runs

    ids = [r["run_id"] for r in picked]
    sources = sorted({r["experiment"] for r in picked})
    st.markdown(f"**{len(ids)} run(s) selected** — #{', #'.join(str(i) for i in ids[:8])}"
                + (f" and {len(ids) - 8} more" if len(ids) > 8 else "") + f" · from {', '.join(sources)}")
    c1, c2 = st.columns(2)
    with c1:
        target = st.text_input("Move to experiment", value="", key="runs_move_to",
                               placeholder="an existing name, or a new one",
                               help="The runs keep everything else. Moving two runs of the same setting together "
                                    "leaves both: the Comparison page's coverage audit is what shows that.").strip()
        if st.button(f"Move {len(ids)} run(s)", key="runs_move", disabled=not target or list(sources) == [target]):
            n = move_runs(ids, target, project=project, engine=engine)
            st.session_state.pop("runs_move_to", None)
            st.success(f"Moved {n} run(s) into `{target}`.")
            st.rerun()
    with c2:
        st.caption("Deleting removes the rows from the database only; artifact folders on disk are kept. "
                   "Tables and figures that used them will change. This cannot be undone.")
        armed = st.checkbox(f"Yes, delete {len(ids)} run(s)", key="runs_delete_confirm")
        if st.button(f"Delete {len(ids)} run(s)", key="runs_delete", type="primary", disabled=not armed):
            n = delete_runs(ids, engine=engine)
            st.session_state.pop("runs_delete_confirm", None)
            st.success(f"Deleted {n} run(s).")
            st.rerun()


def _stale_running(project: str, experiment: Optional[str], engine) -> None:
    """Running rows nothing will finish. The runner logs a `running` row when a job starts and replaces it when
    the job ends, so a row that is hours old means the job was killed (a walltime limit, an OOM, a lost node)."""
    rows, counts = query_runs(project=project, experiment=experiment, statuses=["running"], limit=STALE_LIMIT, engine=engine)
    if not rows:
        return
    aged = sorted(((r, _age_hours(r.timestamp)) for r in rows), key=lambda p: (p[1] is None, -(p[1] or 0)))
    with st.expander(f"Still running ({counts['running']})", expanded=False):
        hours = float(keyed(st.number_input, "Treat as stale after (hours)", "runs_stale_hours", 12.0,
                            min_value=0.0, step=1.0, help="A job that really is still running should be younger than this."))
        stale = [(r, age) for r, age in aged if age is not None and age >= hours]
        recs = run_records([r for r, _ in aged], engine=engine)
        by_id = {r["run_id"]: r for r in recs}
        st.markdown("\n".join(
            f"- **#{r.id}** · {by_id[r.id]['experiment']} · {by_id[r.id]['method'] or '—'} · started {fmt_age(age)} ago"
            + ("  ← stale" if age is not None and age >= hours else "")
            for r, age in aged[:20]))
        if len(aged) > 20:
            st.caption(f"{len(aged) - 20} more not listed.")
        if not stale:
            st.caption(f"None older than {hours:g} h. A row that never clears means the job died without logging a result.")
            return
        st.warning(f"{len(stale)} run(s) have been running for more than {hours:g} h. Deleting them removes the rows only; "
                   "re-running the study logs them again (it resumes, so finished work is not repeated).")
        armed = st.checkbox(f"Yes, delete {len(stale)} stale running row(s)", key="runs_stale_confirm")
        if st.button("Delete stale rows", type="primary", disabled=not armed, key="runs_stale_delete"):
            n = delete_runs([r.id for r, _ in stale], engine=engine)
            st.session_state.pop("runs_stale_confirm", None)
            st.success(f"Deleted {n} stale running row(s).")
            st.rerun()
