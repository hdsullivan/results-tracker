"""Overview page: what is in the database, in the paper look.

Three IEEE-style tables: the experiments (with run counts, methods, datasets, metrics), the results at a
glance (each experiment's headline read off the data: best method, best swept value, largest ablation drop),
and the most recent runs. The sortable data grids stay available in expanders.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from ..export.latex import display_metric_name
from typing import Optional

from .common import db_path, engine_for, hib_map, load_catalog, load_metric_defs, load_records, sidebar_db
from .tables import fmt_stat, generic_html


def _fmt_ts(ts: Any, with_time: bool = True) -> str:
    """Local wall-clock time (records carry UTC)."""
    try:
        local = ts.astimezone() if ts.tzinfo is not None else ts
        return local.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return "—"


def _val(name: str, st_html: str, unit: str = "", suffix: str = "") -> str:
    """Value-column HTML: escaped text around the (trusted) fmt_stat markup."""
    return f"{html.escape(name)} {st_html}" + (f" {html.escape(unit)}" if unit else "") + (f" {html.escape(suffix)}" if suffix else "")


def experiments_rows(summaries: list[dict], project: Optional[str] = None) -> list[list[Any]]:
    """From the SQL summaries (api.experiment_summaries): no run is loaded for this table."""
    rows = []
    for e in summaries:
        runs = f"{e['completed']}" + (f" (+{e['failed']} failed)" if e["failed"] else "") + (f" ({e['running']} running)" if e["running"] else "")
        metrics = [display_metric_name(m) for m in e["metrics"]]
        shown = metrics if len(metrics) <= 6 else metrics[:6] + [f"+{len(metrics) - 6} more"]
        rows.append([
            e["experiment"], e["type"], e.get("stage") or "—", runs,
            ", ".join(map(str, e["methods"])) or "—", ", ".join(map(str, e["datasets"])) or "—",
            ", ".join(shown) or "—", _fmt_ts(e["last"], with_time=False) if e["last"] else "—",
        ])
    return rows


def glance_rows(cat: dict, recs: list[dict], defs: dict, records_for=None) -> list[list[Any]]:
    """One line per experiment: what a reader would want to know first. `records_for(project, experiment)` loads
    one experiment's records (cached per experiment); without it `recs` holds every run."""
    hib = hib_map(defs)
    primary_by_project = {p["name"]: p.get("primary_metric") for p in cat["projects"]}
    out = []
    for e in cat["experiments"]:
        if e.get("stage") == "superseded":
            out.append([e["experiment"], e["type"], "superseded", ""])
            continue
        rs = agg.completed(records_for(e["project"], e["experiment"]) if records_for else [r for r in recs if r["experiment"] == e["experiment"]])
        if not rs:
            out.append([e["experiment"], e["type"], "no completed runs", ""])
            continue
        metrics = agg.metric_names(rs)
        declared = primary_by_project.get(e["project"])
        primary = declared if declared in metrics else next((m for m in ("psnr", "ssim") if m in metrics), metrics[0] if metrics else None)
        if primary is None:
            out.append([e["experiment"], e["type"], "no metrics", ""])
            continue
        fmt = defs.get(primary, {}).get("fmt", ".2f")
        unit = defs.get(primary, {}).get("unit", "")
        name = display_metric_name(primary)
        if e["type"] == "sweep":
            from .common import swept_params

            keys = agg.varying_config_keys(rs) or sorted({k for r in rs for k in agg.flatten(r["config"])})
            declared = [k for k in swept_params(e["project"], e["experiment"]) if k in keys]  # recorded, else from the spec
            keys = declared + [k for k in keys if k not in declared]  # the study's own swept knob first
            if keys:
                series = agg.sweep_series(rs, keys[0], primary)
                s = series.get((), [])
                best = agg.best_sweep_value(s, hib.get(primary, True))
                st_ = dict(s).get(best)
                out.append([e["experiment"], "sweep", f"best {keys[0]} = {best:g}" if isinstance(best, (int, float)) else f"best {keys[0]} = {best}",
                            _val(name, fmt_stat(st_, fmt), unit)])
                continue
        if e["type"] == "ablation":
            try:
                rows = agg.ablation_table(rs, metrics=[primary])
            except agg.AmbiguousBaseError:
                out.append([e["experiment"], "ablation", "base ambiguous", "tag the full model's runs with 'base'"])
                continue
            variants = [r for r in rows if not r.is_base and r.delta.get(primary) is not None]
            base = next((r for r in rows if r.is_base), None)
            if variants and base:
                sign = 1 if hib.get(primary, True) else -1
                worst = min(variants, key=lambda r: r.delta[primary] * sign)
                delta = format(worst.delta[primary], "+" + fmt).replace("-", "−")
                out.append([e["experiment"], "ablation", f"largest drop: {worst.label}",
                            _val(f"{name} {delta}", "", unit).rstrip() + " vs full model " + fmt_stat(base.stats.get(primary), fmt)])
                continue
        labels = agg.method_labels(rs)
        datasets = list(dict.fromkeys(r["dataset"] for r in rs if r.get("dataset") is not None))
        col_key = "dataset" if len(datasets) > 1 else None  # never pool across datasets a method was not run on
        pt = agg.pivot_table(rs, "method", col_key, metrics=[primary], higher_is_better=hib)
        heads, vals = [], []
        for c in pt.cols:
            best = next((r for r in pt.rows if pt.is_best(r, c, primary)), None)
            if best is None:
                continue
            heads.append(f"{labels.get(best, best)}" + (f" on {c}" if c is not None else ""))
            vals.append(fmt_stat(pt.stat(best, c, primary), fmt))
        if not heads:
            out.append([e["experiment"], e["type"], "—", ""])
            continue
        out.append([e["experiment"], e["type"], "best method: " + "; ".join(heads),
                    " · ".join(_val(name, v, unit) for v in vals) + html.escape(f" · {len(pt.rows)} methods")])
    return out


def recent_rows(recs: list[dict], defs: dict, n: int = 15, link=None) -> tuple[list[str], list[list[Any]]]:
    """`recs` are the newest runs already (api.recent_runs); `link(record) -> href` turns the id into a link to Run detail."""
    recent = sorted(recs, key=lambda r: r["timestamp"] or 0, reverse=True)[:n]
    metrics = agg.metric_names(recent)[:4]
    headers = ["#", "logged (local time)", "experiment", "method", "dataset", "seed", "status"] + [display_metric_name(m) for m in metrics]
    rows = []
    for r in recent:
        vals = []
        for m in metrics:
            v = r["metrics"].get(m)
            vals.append("—" if v is None else format(v, defs.get(m, {}).get("fmt", ".2f")).replace("-", "−"))
        rid = f'<a href="{link(r)}" target="_self">#{r["run_id"]}</a>' if link else r["run_id"]
        rows.append([rid, _fmt_ts(r["timestamp"]), r["experiment"], r["method"] or "—", r["dataset"] or "—",
                     "—" if r["seed"] is None else r["seed"], r["status"], *vals])
    return headers, rows


def _paper_line(records_for) -> None:
    """One line on the pinned paper assets: how many, how many final, how many need a re-export."""
    from collections import Counter

    from ..api import list_assets, list_projects
    from ..export.paper import asset_experiments, staleness
    from .common import db_path, engine_for

    engine = engine_for(db_path())
    assets = list_assets(engine=engine)
    if not assets:
        return
    projs = {p.id: p.name for p in list_projects(engine=engine)}
    states = Counter(staleness(a, [r for e in asset_experiments(a) for r in records_for(projs.get(a.project_id), e)])[0]
                     for a in assets if a.status.value != "dropped")
    n_final = sum(a.status.value == "final" for a in assets)
    st.caption(f"**Paper:** {sum(states.values())} pinned assets · {n_final} final · {states.get('stale', 0)} stale · "
               f"{states.get('never exported', 0)} never exported — details on the Paper page.")


def render() -> None:
    from ..api import add_note, delete_note, experiment_summaries, list_notes, list_projects, recent_runs, run_records
    from .common import db_path, engine_for, page_url

    st.title("Results Tracker")
    sidebar_db()
    cat = load_catalog()
    defs = load_metric_defs()
    engine = engine_for(db_path())
    summaries = experiment_summaries(engine=engine)
    n_runs = sum(e["runs"] for e in summaries)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", len(cat["projects"]))
    c2.metric("Experiments", len(cat["experiments"]))
    c3.metric("Runs", n_runs)
    c4.metric("Failed / running", f"{sum(e['failed'] for e in summaries)} / {sum(e['running'] for e in summaries)}")

    if not n_runs:
        st.info("Empty database. Seed a demo with `results-tracker demo`, or log runs with `results_tracker.log_run`.")
        return
    recent = run_records(recent_runs(15, engine=engine), engine=engine)
    _paper_line(records_for=load_records)

    st.subheader("Experiments")
    projects = ", ".join(p["name"] for p in cat["projects"])
    st.markdown(generic_html(
        ["Experiment", "Type", "Stage", "Runs", "Methods", "Datasets", "Metrics", "Last run"], experiments_rows(summaries),
        caption=f"Experiments in {projects}. Run counts are completed runs; failed and running runs in parentheses. Stage (paper / "
                "exploratory / superseded) is set on the Settings page; superseded experiments are hidden from the selectors.",
        number=1, left_cols=3,
    ), unsafe_allow_html=True)

    st.subheader("Results at a glance")
    st.markdown(generic_html(
        ["Experiment", "Type", "Headline", "Value"], glance_rows(cat, [], defs, records_for=load_records),
        caption="One line per experiment, read off the completed runs: best method (mean ± std over datasets and seeds), "
                "best swept value, or the setting whose removal costs the most. Details on the respective pages.",
        number=2, left_cols=4, raw_html_cols=[3],
    ), unsafe_allow_html=True)

    st.subheader("Recent runs")
    exp_project = {e["experiment"]: e["project"] for e in cat["experiments"]}
    headers, rows = recent_rows(recent, defs, link=lambda r: page_url("run", project=exp_project.get(r["experiment"]),
                                                                        experiment=r["experiment"], run=r["run_id"]))
    st.markdown(generic_html(headers, rows, caption="Most recent runs, newest first; metrics as logged. The id opens the run.", number=3,
                             left_cols=5, raw_html_cols=[0]), unsafe_allow_html=True)

    with st.expander("Sortable grids"):
        st.dataframe(pd.DataFrame([{k: v for k, v in e.items() if k not in ("methods", "datasets", "metrics", "swept_params")} for e in summaries]),
                     width="stretch", hide_index=True)
        st.dataframe(pd.DataFrame([
            {"id": r["run_id"], "time": r["timestamp"], "experiment": r["experiment"], "method": r["method"],
             "dataset": r["dataset"], "seed": r["seed"], "status": r["status"], **r["metrics"]} for r in recent
        ]), width="stretch", hide_index=True)

    st.subheader("Notes")
    notes = list_notes(engine=engine)
    proj_names = [p["name"] for p in cat["projects"]]
    proj_by_id = {p.id: p.name for p in list_projects(engine=engine)}
    with st.form("note_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1, 3])
        note_project = c1.selectbox("Project", proj_names, key="note_project")
        note_exp = c2.selectbox("Experiment (optional)", ["—"] + [e["experiment"] for e in cat["experiments"]], key="note_experiment")
        text = c3.text_input("Decision or observation", key="note_text", placeholder="beta = 0.5 chosen: plateau in deblurring-ema-beta")
        if st.form_submit_button("Add note") and text.strip():
            add_note(note_project, text, experiment=None if note_exp == "—" else note_exp, engine=engine)
            st.rerun()
    if not notes:
        st.caption("No notes yet. A dated line per decision keeps the reasoning next to the results; notes attached to a paper asset "
                   "are shown on the Paper page.")
    for n in notes[:20]:
        c1, c2 = st.columns([11, 1])
        tags = " · ".join(t for t in (proj_by_id.get(n.project_id, ""), n.experiment or "", f"`{n.asset_label}`" if n.asset_label else "") if t)
        c1.markdown(f"**{_fmt_ts(n.created_at, with_time=False)}** · {tags} — {n.text}")
        if c2.button("🗑", key=f"note_del_{n.id}", help="Delete this note"):
            delete_note(n.id, engine=engine)
            st.rerun()
    if len(notes) > 20:
        st.caption(f"{len(notes) - 20} older notes not shown (`results-tracker note list`).")
