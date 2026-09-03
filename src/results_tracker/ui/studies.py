"""Studies page: planned experiments next to what the database says is done, and a form to plan new ones.

A *planned* experiment is a study spec (JSON) in the studies directory; `expand` turns it into jobs (arm x
condition x seed), each of which should have `n_instances` completed runs. *Completed* is derived, never
stored: for every job the page counts completed, failed and running runs in the database whose config matches
the job's exactly (the same fingerprint the runner's resume uses), so this page and `results-tracker recipe
run` can never disagree about what is left. From the completed runs' `runtime_s` it estimates the compute
still needed. A study names the paper assets it `feeds`; a pending-only spec can be downloaded to hand the rest
of a grid to another machine. Below the list, a "New study" form builds a spec from the knobs the registered
methods and problems declare, or edits / clones an existing one; when a spec's imports cannot be loaded here,
declarations from `<studies dir>/knobs.json` (`results-tracker recipe export-knobs`) stand in.
"""

from __future__ import annotations

import importlib
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from ..api import config_fingerprint, list_assets
from ..recipe import Ablation, Arm, Job, Knob, Registry, Study, Sweep, arm_changes, expand, load_declarations, load_study_classes, pending_subset
from ..recipe import registry as default_registry
from .common import KEY_PROJECT, db_path, engine_for, keyed, keyed_multiselect, keyed_selectbox, load_records, select_project, sidebar_db
from .tables import generic_html

STUDIES_ENV = "RESULTS_TRACKER_STUDIES"
DECLARATIONS = "knobs.json"
STATE_COLOURS = {"done": "#d5eed8", "running": "#cfe3f7", "partial": "#fde9b6", "failed": "#f6c9c9", "missing": "#ececec", "unknown": "#ffffff"}


# --------------------------------------------------------------------------- derivation

def default_studies_dir(db: str, project: Optional[str] = None) -> Path:
    """The project's own studies directory (Settings → Project), else $RESULTS_TRACKER_STUDIES, else `studies/` next to
    the database if it exists, else ./studies."""
    if project:
        from .common import project_studies_dir

        declared = project_studies_dir(project)
        if declared:
            return Path(declared).expanduser()
    if os.environ.get(STUDIES_ENV):
        return Path(os.environ[STUDIES_ENV]).expanduser()
    beside = Path(db).expanduser().resolve().parent / "studies" if db != ":memory:" else None
    if beside is not None and beside.is_dir():
        return beside
    return Path("studies")


@dataclass
class JobStatus:
    job: Job
    expected: int
    completed: int
    failed: int
    running: int = 0

    @property
    def state(self) -> str:
        if self.completed >= self.expected:
            return "done"
        if self.running:
            return "running"
        if self.failed:
            return "failed"
        if self.completed:
            return "partial"
        return "missing"


def job_statuses(study: Study, jobs: list[Job], records: list[dict]) -> list[JobStatus]:
    """Completed/failed/running run counts per job, matched on (method, seed, exact config) like the runner's resume."""
    counts: dict[tuple, Counter] = {}
    for r in records:
        key = (r.get("method"), r.get("seed"), config_fingerprint(r.get("config") or {}))
        counts.setdefault(key, Counter())[r.get("status")] += 1
    out = []
    for job in jobs:
        c = counts.get((job.method, job.seed, config_fingerprint({**job.condition, **job.config})), Counter())
        out.append(JobStatus(job, study.n_instances, min(c["completed"], study.n_instances), c["failed"], c["running"]))
    return out


def median_runtime(records: list[dict]) -> Optional[float]:
    """Median `runtime_s` of the completed runs (the runner logs it for every run), None without any."""
    times = [r["metrics"]["runtime_s"] for r in records
             if r.get("status") == "completed" and isinstance(r.get("metrics", {}).get("runtime_s"), (int, float))]
    return statistics.median(times) if times else None


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    if seconds < 90:
        return f"{seconds:.0f} s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"


@dataclass
class Planned:
    path: Path
    study: Optional[Study]
    error: str = ""
    jobs: list[Job] | None = None
    statuses: list[JobStatus] | None = None
    methods: dict | None = None  # method key/ref -> class, as resolved for `expand`
    declared: bool = False  # classes came from knobs.json, not from the spec's imports
    runtime: Optional[float] = None  # median seconds per completed run

    @property
    def totals(self) -> tuple[int, int, int, int]:
        """(expected, completed, failed, running) runs over every job."""
        if not self.statuses:
            return 0, 0, 0, 0
        return (sum(s.expected for s in self.statuses), sum(s.completed for s in self.statuses),
                sum(s.failed for s in self.statuses), sum(s.running for s in self.statuses))

    @property
    def pending_jobs(self) -> list[Job]:
        return [s.job for s in (self.statuses or []) if s.state != "done"]

    @property
    def seconds_left(self) -> Optional[float]:
        """Pending runs times the median runtime of a completed run; None until one run has completed."""
        if self.runtime is None or not self.statuses:
            return None
        pending_runs = sum(s.expected - s.completed for s in self.statuses)
        return pending_runs * self.runtime


def resolve_classes(study: Study, studies_dir: Path) -> tuple[Any, dict, bool]:
    """(problem_cls, methods, declared): the study's classes from its imports, else from `<studies dir>/knobs.json`."""
    try:
        problem_cls, methods = load_study_classes(study)
        return problem_cls, methods, False
    except Exception as e:  # noqa: BLE001 - the declaration file is the fallback
        decl = studies_dir / DECLARATIONS
        if not decl.is_file():
            raise
        try:
            problem_cls, methods = load_study_classes(study, load_declarations(decl), import_modules=False)
        except Exception:  # noqa: BLE001
            raise e from None
        return problem_cls, methods, True


def load_planned(studies_dir: Path) -> list[Planned]:
    out: list[Planned] = []
    for path in sorted(studies_dir.rglob("*.json")):
        if path.name == DECLARATIONS:
            continue
        try:
            study = Study.load(path)
        except Exception as e:  # noqa: BLE001 - a bad file is reported in the table, not raised
            out.append(Planned(path, None, f"not a study spec: {type(e).__name__}: {e}"))
            continue
        planned = Planned(path, study)
        try:
            problem_cls, methods, declared = resolve_classes(study, studies_dir)
            planned.jobs = expand(study, problem_cls, methods)
            planned.methods, planned.declared = methods, declared
        except Exception as e:  # noqa: BLE001
            planned.error = f"{type(e).__name__}: {e}"
        if planned.jobs is not None:
            records = load_records(study.project, study.name)
            planned.statuses = job_statuses(study, planned.jobs, records)
            planned.runtime = median_runtime(records)
        out.append(planned)
    return out


def studies_feeding(planned: list[Planned], project: str, asset_label: str, experiment: str) -> list[Planned]:
    """Studies that produce an asset's data: those naming it in `feeds`, else the one producing its experiment."""
    named = [p for p in planned if p.study and p.study.project == project and asset_label in p.study.feeds]
    if named:
        return named
    return [p for p in planned if p.study and p.study.project == project and p.study.name == experiment]


def _slug(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()) or "—"


def grid_html(statuses: list[JobStatus]) -> str:
    """Arms down, conditions across; each cell coloured by state with `completed/expected` (`+failed`, `▶running`)."""
    arms = list(dict.fromkeys((s.job.method, s.job.arm) for s in statuses))
    conditions = list(dict.fromkeys(json.dumps(s.job.condition, sort_keys=True, default=str) for s in statuses))
    seeds = sorted({s.job.seed for s in statuses})
    by_cell: dict[tuple, list[JobStatus]] = {}
    for s in statuses:
        by_cell.setdefault(((s.job.method, s.job.arm), json.dumps(s.job.condition, sort_keys=True, default=str)), []).append(s)
    head = "<th>arm</th>" + "".join(f"<th>{_slug(json.loads(c))}</th>" for c in conditions)
    rows = []
    for arm in arms:
        cells = [f"<td style='text-align:left'>{arm[1] if arm[1] != arm[0] else arm[0]}</td>"]
        for c in conditions:
            group = by_cell.get((arm, c), [])
            done = sum(s.completed for s in group)
            exp = sum(s.expected for s in group)
            failed = sum(s.failed for s in group)
            running = sum(s.running for s in group)
            state = ("done" if group and done >= exp else "running" if running else "failed" if failed else "partial" if done else "missing")
            text = f"{done}/{exp}" + (f" +{failed}✗" if failed else "") + (f" ▶{running}" if running else "")
            cells.append(f"<td style='background:{STATE_COLOURS[state]};text-align:center'>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    seed_note = f" · seeds {', '.join(map(str, seeds))} pooled per cell" if len(seeds) > 1 else ""
    legend = " ".join(f"<span style='background:{STATE_COLOURS[k]};padding:0 .5em'>{k}</span>" for k in ("done", "running", "partial", "failed", "missing"))
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:0.9em'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p style='font-size:0.85em'>{legend} — completed instances / expected per job{seed_note}</p>")


def run_commands(path: Path, db: str) -> str:
    """The shell lines that validate and (resumably) run a spec against this database."""
    return (f"results-tracker recipe validate {path}\n"
            f"results-tracker recipe run {path} --db {db}   # resumes: settings already logged are skipped")


# --------------------------------------------------------------------------- page

def render() -> None:
    st.title("Studies")
    db = sidebar_db()
    project = select_project()
    with st.sidebar:
        st.markdown("**Plans**")
        if st.session_state.get("_studies_dir_for") != project:  # a new project: its own default directory
            st.session_state["_studies_dir_for"] = project
            st.session_state.pop("studies_dir", None)
        studies_dir = Path(keyed(st.text_input, "Studies directory", "studies_dir", str(default_studies_dir(db, project)),
                                 help="Every *.json study spec below this folder is a planned experiment. Set it once per project on "
                                      f"the Settings page, or ${STUDIES_ENV} for every project.")).expanduser()
    if not studies_dir.is_dir():
        st.info(f"`{studies_dir}` does not exist. Point the sidebar at a folder of study specs, or plan one below.")
        planned: list[Planned] = []
    else:
        planned = load_planned(studies_dir)

    if planned:
        _overview(planned, studies_dir)
        choices = {str(p.path.relative_to(studies_dir)): p for p in planned if p.study is not None}
        if choices:
            chosen = choices[keyed_selectbox("Study", list(choices), "studies_pick", list(choices)[0])]
            _study_detail(chosen, db, project or chosen.study.project, studies_dir)

    st.divider()
    _new_study_form(studies_dir, planned, project)


def _overview(planned: list[Planned], studies_dir: Path) -> None:
    rows = []
    for p in planned:
        exp, done, failed, running = p.totals
        study = p.study
        rows.append({
            "spec": str(p.path.relative_to(studies_dir)),
            "experiment": study.name if study else "—",
            "kind": study.kind if study else "—",
            "problem": study.problem if study else "—",
            "arms": len(study.methods) if study else 0,
            "jobs": len(p.jobs) if p.jobs is not None else None,
            "runs done": done, "expected": exp, "failed": failed, "running": running,
            "progress": (done / exp) if exp else 0.0,
            "time left": fmt_duration(p.seconds_left) if p.jobs is not None and done < exp else ("" if done >= exp and exp else "—"),
            "feeds": ", ".join(study.feeds) if study and study.feeds else "",
            "status": ("done" if exp and done >= exp else "not runnable: " + p.error if p.error
                       else "running" if running else "in progress" if done else "planned"),
        })
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"progress": st.column_config.ProgressColumn("progress", min_value=0.0, max_value=1.0, format="%.0f%%")})
    timed = [p for p in planned if p.seconds_left is not None and p.totals[1] < p.totals[0]]
    untimed = [p for p in planned if p.jobs is not None and p.seconds_left is None and p.totals[1] < p.totals[0]]
    if timed or untimed:
        total = sum(p.seconds_left or 0 for p in timed)
        st.caption(f"**Compute left: ~{fmt_duration(total)}** for {len(timed)} unfinished stud{'y' if len(timed) == 1 else 'ies'} "
                   "(pending runs × the median `runtime_s` of that study's completed runs, so setup and data loading are not counted)"
                   + (f"; {len(untimed)} unfinished stud{'y has' if len(untimed) == 1 else 'ies have'} no completed run to time yet." if untimed else "."))
    if any(p.declared for p in planned):
        st.caption(f"Some specs could not import their modules here; their knobs come from `{studies_dir / DECLARATIONS}` "
                   "(refresh it with `results-tracker recipe export-knobs` in the repo's environment).")
    broken = [p for p in planned if p.error]
    if broken:
        st.warning(f"{len(broken)} spec(s) could not be expanded. If the error is an import error, either install the repo that "
                   "declares the methods into the GUI's environment (`pip install -e /path/to/repo --no-deps`), or write its "
                   f"declarations next to the specs: `results-tracker recipe export-knobs -i <module> -o {studies_dir / DECLARATIONS}`.")
        for p in broken:
            st.caption(f"`{p.path.name}`: {p.error}")


def _study_detail(p: Planned, db: str, project: str, studies_dir: Path) -> None:
    study = p.study
    assert study is not None
    st.subheader(study.name)
    if study.description:
        st.caption(study.description)
    exp, done, failed, running = p.totals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("jobs", len(p.jobs) if p.jobs is not None else "—")
    c2.metric("runs completed", f"{done} / {exp}")
    c3.metric("failed / running", f"{failed} / {running}")
    c4.metric("time left", fmt_duration(p.seconds_left) if exp and done < exp else ("done" if exp else "—"),
              help="pending runs × median runtime_s of this study's completed runs")
    if study.feeds:
        assets = {a.label: a for a in list_assets(study.project, engine=engine_for(db_path()))}
        parts = [f"`{lbl}` ({assets[lbl].status.value})" if lbl in assets else f"`{lbl}` (not pinned yet)" for lbl in study.feeds]
        st.caption("Feeds paper assets: " + ", ".join(parts) + " — readiness is shown on the Paper page.")
    if p.statuses:
        st.markdown(grid_html(p.statuses), unsafe_allow_html=True)
        records = load_records(study.project, study.name)
        # the runner lays runs out as <root>/<study>/<method>/<arm>/<condition>/<seed>/<instance>
        roots = sorted({str(Path(a).parents[5]) if len(Path(a).parts) > 6 else a
                        for a in (r.get("artifacts_dir") for r in records) if a})
        if roots:
            st.caption("Results on disk: " + ", ".join(f"`{r}`" for r in roots[:3]) + (" …" if len(roots) > 3 else ""))
        rows = [[s.job.method, s.job.arm, _slug(s.job.condition), s.job.seed, s.state, f"{s.completed}/{s.expected}", s.failed, s.running]
                for s in p.statuses if s.state != "done"]
        if rows:
            with st.expander(f"Pending, running or failed jobs ({len(rows)})"):
                st.markdown(generic_html(["method", "arm", "condition", "seed", "state", "done", "failed", "running"], rows, left_cols=3),
                            unsafe_allow_html=True)
    elif p.error:
        st.error(p.error)
    st.code(run_commands(p.path, db), language="bash")
    pending = p.pending_jobs
    if p.jobs is not None and pending and len(pending) < len(p.jobs) and p.methods is not None:
        sub = pending_subset(study, pending, p.methods)
        c1, c2 = st.columns([1, 2])
        c1.download_button("Download pending-only spec", json.dumps(sub.to_dict(), indent=2, default=str) + "\n",
                           file_name=f"{study.name}-pending.json", mime="application/json", key="studies_pending_dl")
        c2.caption(f"The grid narrowed to the {len(pending)} unfinished jobs ({len(sub.seeds)} seed(s), "
                   f"{' × '.join(f'{len(v)} {k}' for k, v in sub.conditions.items())}, {len(sub.methods)} arm(s)"
                   + (f", {len(sub.sweep.values)} sweep values" if sub.sweep else "") + "): run it on another machine against the "
                   "same database; it resumes, so nothing done here is recomputed.")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Edit in form below", key="studies_edit", disabled=p.error != "" and p.methods is None):
        _prefill_form(p, clone=False)
        st.rerun()
    if b2.button("Clone in form below", key="studies_clone", disabled=p.error != "" and p.methods is None):
        _prefill_form(p, clone=True)
        st.rerun()
    with st.expander("Spec"):
        st.json(study.to_dict())


# --------------------------------------------------------------------------- authoring

def _fmt(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _knob_widget(knob: Knob, key: str):
    """A keyed widget for one knob (seeded with its default, or a pending prefill). Returns the value."""
    label = f"{knob.name}" + (f" — {knob.doc}" if knob.doc else "")
    if knob.kind == "bool":
        return keyed(st.checkbox, label, key, bool(knob.default))
    if knob.kind == "choice":
        return keyed_selectbox(label, list(knob.choices or ()), key, knob.default)
    if knob.kind in ("float", "int"):
        text = keyed(st.text_input, label, key, "" if knob.default is None else _fmt(knob.default),
                     help=f"{knob.kind}" + (f", bounds {list(knob.bounds)}" if knob.bounds else "") + ("; empty = none" if knob.default is None else ""))
        return None if str(text).strip() == "" else knob.validate(str(text).strip())
    return keyed(st.text_input, label, key, "" if knob.default is None else str(knob.default)) or None


def _values(knob: Knob, text: str) -> list[Any]:
    return [knob.validate(t.strip()) for t in str(text).split(",") if t.strip()]


def _prefill_form(p: Planned, clone: bool) -> None:
    """Queue the widget states that reproduce `p.study` in the New-study form (see common.keyed)."""
    study = p.study
    assert study is not None and p.methods is not None
    pre: dict[str, Any] = {
        "new_imports": ", ".join(study.imports), "new_name": f"{study.name}-copy" if clone else study.name, "new_kind": study.kind,
        "new_project": study.project, "new_problem": study.problem, "new_split": study.split, "new_n": int(study.n_instances),
        "new_seeds": ", ".join(map(str, study.seeds)), "new_desc": study.description, "new_feeds": ", ".join(study.feeds),
        "new_arms": len(study.methods), "new_filename": "" if clone else p.path.name, "new_overwrite": not clone,
    }
    for k, values in study.conditions.items():
        pre[f"new_cond_{k}"] = list(values) if any(x.name == k and x.kind == "choice" for x in _problem_conditions(p)) else ", ".join(map(_fmt, values))
    for i, arm in enumerate(study.methods):
        pre[f"new_arm_{i}_method"] = arm.method
        space = p.methods[arm.method].space()
        for knob, value in arm.config.items():
            if knob in space:
                k = space[knob]
                pre[f"new_arm_{i}_{knob}"] = value if k.kind in ("bool", "choice") else ("" if value is None else _fmt(value))
    if study.sweep is not None:
        pre["new_sweep_knob"] = study.sweep.knob
        pre["new_sweep_values"] = ", ".join(map(_fmt, study.sweep.values))
    if study.ablation is not None:
        pre["new_abl_n"] = len(study.ablation.arms)
        space = p.methods[study.methods[0].method].space()
        for i, arm in enumerate(study.ablation.arms):
            label, changes = arm_changes(arm)
            pre[f"new_abl_{i}_knobs"] = list(changes)
            pre[f"new_abl_{i}_label"] = label or ""
            for knob, value in changes.items():
                if knob in space:
                    k = space[knob]
                    pre[f"new_abl_{i}_{knob}"] = value if k.kind in ("bool", "choice") else ("" if value is None else _fmt(value))
    st.session_state.setdefault("_prefill", {}).update(pre)


def _problem_conditions(p: Planned) -> list[Knob]:
    try:
        problem_cls, _, _ = resolve_classes(p.study, p.path.parent)  # type: ignore[arg-type]
        return list(problem_cls.condition_space())
    except Exception:  # noqa: BLE001
        return []


def _registry_for_form(imports: list[str], studies_dir: Path) -> tuple[Registry, list[str], bool]:
    """Import the modules into the default registry; on failure fall back to `<studies dir>/knobs.json`."""
    errors = []
    for m in imports:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            errors.append(f"cannot import `{m}`: {type(e).__name__}: {e}")
    if default_registry.problems and default_registry.methods:
        return default_registry, errors, False
    decl = studies_dir / DECLARATIONS
    if decl.is_file():
        try:
            return load_declarations(decl), errors, True
        except Exception as e:  # noqa: BLE001
            errors.append(f"cannot read {decl}: {type(e).__name__}: {e}")
    return default_registry, errors, False


def _new_study_form(studies_dir: Path, planned: list[Planned], project: Optional[str]) -> None:
    st.header("New study")
    st.caption("Build a spec from the knobs the registered methods and problems declare (or edit / clone one from above); save it into "
               "the studies directory and run it from a terminal with `results-tracker recipe run`.")
    known_imports = sorted({m for p in planned if p.study for m in p.study.imports})
    imports_text = keyed(st.text_input, "Import modules (comma-separated) that register methods and problems", "new_imports",
                         ", ".join(known_imports), help="e.g. adaptivepnp.recipes. If they cannot be imported in the GUI's environment, "
                                                        f"declarations from `{studies_dir / DECLARATIONS}` are used instead.")
    imports = [m.strip() for m in str(imports_text).split(",") if m.strip()]
    reg, errors, declared = _registry_for_form(imports, studies_dir)
    for msg in errors:
        (st.caption if declared else st.error)(msg)
    if declared:
        st.caption(f"Planning from the declarations in `{studies_dir / DECLARATIONS}`.")
    if not reg.problems or not reg.methods:
        st.info("No problems or methods registered yet: add the module that declares them above, or write "
                f"`{studies_dir / DECLARATIONS}` with `results-tracker recipe export-knobs`.")
        return

    projects = sorted({p.study.project for p in planned if p.study}) or [project or "default"]
    c1, c2, c3 = st.columns(3)
    with c1:
        name = keyed(st.text_input, "Experiment name", "new_name", "", placeholder="deblurring-beta-sweep")
    with c2:
        kind = keyed_selectbox("Kind", ["comparison", "sweep", "ablation"], "new_kind", "comparison")
    with c3:
        study_project = keyed(st.text_input, "Project", "new_project", project if project in projects else projects[0])
    problem_key = keyed_selectbox("Problem", sorted(reg.problems), "new_problem", sorted(reg.problems)[0])
    problem_cls = reg.problems[problem_key]
    c1, c2, c3 = st.columns(3)
    with c1:
        split = keyed_selectbox("Split", list(problem_cls.splits), "new_split", problem_cls.splits[0])
    with c2:
        n_instances = int(keyed(st.number_input, "Instances per job", "new_n", 1, min_value=1, step=1))
    with c3:
        seeds = [int(s) for s in str(keyed(st.text_input, "Seeds (comma-separated)", "new_seeds", "0")).split(",") if s.strip()]
    description = keyed(st.text_input, "Description", "new_desc", "")
    asset_labels = [a.label for a in list_assets(study_project or None, engine=engine_for(db_path()))] if study_project else []
    feeds_text = keyed(st.text_input, "Feeds paper assets (comma-separated labels)", "new_feeds", "",
                       placeholder="tab:main, fig:beta", help="Assets of this project: " + (", ".join(asset_labels) or "none pinned yet")
                       + ". The Paper page shows each asset's readiness from the studies that feed it.")
    feeds = [f.strip() for f in str(feeds_text).split(",") if f.strip()]

    st.markdown("**Conditions** (one value or a comma-separated list per knob; the grid is their product)")
    conditions: dict[str, list] = {}
    cond_knobs = list(problem_cls.condition_space())
    cols = st.columns(max(1, min(4, len(cond_knobs))))
    for i, knob in enumerate(cond_knobs):
        with cols[i % len(cols)]:
            if knob.kind == "choice":
                vals = keyed_multiselect(knob.name, list(knob.choices or ()), f"new_cond_{knob.name}", [knob.default])
            else:
                text = keyed(st.text_input, knob.name, f"new_cond_{knob.name}", _fmt(knob.default), help=knob.doc or knob.kind)
                try:
                    vals = _values(knob, text)
                except ValueError as e:
                    st.error(str(e))
                    vals = []
            if vals:
                conditions[knob.name] = vals

    st.markdown("**Method arms** (only knobs set away from their defaults are written into the spec)")
    n_arms = int(keyed(st.number_input, "Arms", "new_arms", 1, min_value=1, max_value=8, step=1))
    arms: list[Arm] = []
    for i in range(n_arms):
        method_key = keyed_selectbox(f"Arm {i + 1} method", sorted(reg.methods), f"new_arm_{i}_method", sorted(reg.methods)[0])
        space = reg.methods[method_key].space()
        config: dict[str, Any] = {}
        with st.expander(f"Arm {i + 1} knobs ({method_key})", expanded=False):
            for knob in space:
                try:
                    v = _knob_widget(knob, key=f"new_arm_{i}_{knob.name}")
                except ValueError as e:
                    st.error(str(e))
                    continue
                if v != knob.default:
                    config[knob.name] = v
        arms.append(Arm(method_key, config))

    sweep = ablation = None
    space0 = reg.methods[arms[0].method].space()
    if kind == "sweep":
        c1, c2 = st.columns([1, 2])
        with c1:
            knob_name = keyed_selectbox("Swept knob", space0.names, "new_sweep_knob", space0.names[0])
        with c2:
            text = keyed(st.text_input, "Values (comma-separated)", "new_sweep_values", "")
        try:
            sweep = Sweep(knob_name, _values(space0[knob_name], text))
        except ValueError as e:
            st.error(str(e))
    elif kind == "ablation":
        st.markdown("**Ablation arms** (the arm configs above are the full model; each arm changes one knob, or several "
                    "at once as a labelled joint change)")
        n_abl = int(keyed(st.number_input, "Ablation arms", "new_abl_n", 1, min_value=1, max_value=12, step=1))
        arm_dicts: list[dict[str, Any]] = []
        for i in range(n_abl):
            with st.container(border=True):
                changed = keyed_multiselect(f"Arm {i + 1}: knob(s) to change", space0.names, f"new_abl_{i}_knobs", [])
                changes: dict[str, Any] = {}
                if changed:
                    cols = st.columns(min(3, len(changed)))
                    for j, knob_name in enumerate(changed):
                        with cols[j % len(cols)]:
                            try:
                                changes[knob_name] = _knob_widget(space0[knob_name], key=f"new_abl_{i}_{knob_name}")
                            except ValueError as e:
                                st.error(str(e))
                if len(changes) > 1:
                    label = keyed(st.text_input, "Label for this joint change", f"new_abl_{i}_label", "", placeholder="w/o floor and tolerance")
                    arm_dicts.append({"label": label or ", ".join(f"{k}={_fmt(v)}" for k, v in changes.items()), "set": changes})
                elif changes:
                    arm_dicts.append(dict(changes))
        ablation = Ablation(base={}, arms=arm_dicts)

    study = Study(name=name or "unnamed", kind=kind, problem=problem_key, methods=arms, project=study_project or "default",
                  conditions=conditions, split=split, n_instances=n_instances, seeds=seeds or [0], sweep=sweep,
                  ablation=ablation, description=description, imports=imports, feeds=feeds)
    try:
        problem_cls_, methods = load_study_classes(study, reg, import_modules=False)
        jobs = expand(study, problem_cls_, methods)
    except Exception as e:  # noqa: BLE001
        st.error(f"Not valid yet: {type(e).__name__}: {e}")
        jobs = None
    if jobs is not None:
        st.success(f"{len(jobs)} jobs × {n_instances} instances = {len(jobs) * n_instances} runs")
        with st.expander("Spec preview"):
            st.json(study.to_dict())
        c1, c2 = st.columns([2, 1])
        with c1:
            typed = keyed(st.text_input, "File name", "new_filename", "", placeholder=f"{name or '<name>'}.json",
                          help="Defaults to <experiment name>.json inside the studies directory.")
        filename = str(typed).strip() or (f"{name}.json" if name else "")
        with c2:
            overwrite = keyed(st.checkbox, "Overwrite if it exists", "new_overwrite", False)
        if st.button("Save spec", key="new_save", disabled=not filename):
            target = studies_dir / filename
            if target.exists() and not overwrite:
                st.error(f"`{target}` exists; tick overwrite to replace it.")
            else:
                study.save(target)
                st.success(f"Saved `{target}`. Run it with: `results-tracker recipe run {target} --db {db_path()}`")
                st.cache_data.clear()
    # a prefill entry for a widget that was not rendered (a knob of another method) must not fire later
    pre = st.session_state.get("_prefill") or {}
    for k in [k for k in pre if str(k).startswith("new_")]:
        pre.pop(k)
