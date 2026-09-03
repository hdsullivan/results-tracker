"""Studies page: planned experiments next to what the database says is done, and a form to plan new ones.

A *planned* experiment is a study spec (JSON) in the studies directory; `expand` turns it into jobs (arm x
condition x seed), each of which should have `n_instances` completed runs. *Completed* is derived, never
stored: for every job the page counts completed and failed runs in the database whose config matches the
job's exactly (the same fingerprint the runner's resume uses), so this page and `results-tracker recipe
run` can never disagree about what is left. Below the list, a "New study" form builds a spec from the
knobs the registered methods and problems declare.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import streamlit as st

from ..api import config_fingerprint
from ..recipe import Ablation, Arm, Job, Knob, Study, Sweep, expand, load_study_classes, registry
from .common import db_path, load_records, sidebar_db
from .tables import generic_html

STUDIES_ENV = "RESULTS_TRACKER_STUDIES"
STATE_COLOURS = {"done": "#d5eed8", "partial": "#fde9b6", "failed": "#f6c9c9", "missing": "#ececec", "unknown": "#ffffff"}


# --------------------------------------------------------------------------- derivation

def default_studies_dir(db: str) -> Path:
    """$RESULTS_TRACKER_STUDIES, else `studies/` next to the database if it exists, else ./studies."""
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

    @property
    def state(self) -> str:
        if self.completed >= self.expected:
            return "done"
        if self.failed:
            return "failed"
        if self.completed:
            return "partial"
        return "missing"


def job_statuses(study: Study, jobs: list[Job], records: list[dict]) -> list[JobStatus]:
    """Completed/failed run counts per job, matched on (method, seed, exact config) like the runner's resume."""
    counts: dict[tuple, Counter] = {}
    for r in records:
        key = (r.get("method"), r.get("seed"), config_fingerprint(r.get("config") or {}))
        counts.setdefault(key, Counter())[r.get("status")] += 1
    out = []
    for job in jobs:
        c = counts.get((job.method, job.seed, config_fingerprint({**job.condition, **job.config})), Counter())
        out.append(JobStatus(job, study.n_instances, min(c["completed"], study.n_instances), c["failed"]))
    return out


@dataclass
class Planned:
    path: Path
    study: Optional[Study]
    error: str = ""
    jobs: list[Job] | None = None
    statuses: list[JobStatus] | None = None

    @property
    def totals(self) -> tuple[int, int, int]:
        """(expected, completed, failed) over every job."""
        if not self.statuses:
            return 0, 0, 0
        return (sum(s.expected for s in self.statuses), sum(s.completed for s in self.statuses),
                sum(s.failed for s in self.statuses))


def load_planned(studies_dir: Path) -> list[Planned]:
    out: list[Planned] = []
    for path in sorted(studies_dir.rglob("*.json")):
        try:
            study = Study.load(path)
        except Exception as e:  # noqa: BLE001 - a bad file is reported in the table, not raised
            out.append(Planned(path, None, f"not a study spec: {type(e).__name__}: {e}"))
            continue
        planned = Planned(path, study)
        try:
            problem_cls, methods = load_study_classes(study)
            planned.jobs = expand(study, problem_cls, methods)
        except Exception as e:  # noqa: BLE001
            planned.error = f"{type(e).__name__}: {e}"
        if planned.jobs is not None:
            planned.statuses = job_statuses(study, planned.jobs, load_records(study.project, study.name))
        out.append(planned)
    return out


def _slug(d: dict[str, Any]) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items()) or "—"


def grid_html(statuses: list[JobStatus]) -> str:
    """Arms down, conditions across; each cell coloured by state with `completed/expected` (`+failed`)."""
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
            state = "done" if group and done >= exp else "failed" if failed else "partial" if done else "missing"
            text = f"{done}/{exp}" + (f" +{failed}✗" if failed else "")
            cells.append(f"<td style='background:{STATE_COLOURS[state]};text-align:center'>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    seed_note = f" · seeds {', '.join(map(str, seeds))} pooled per cell" if len(seeds) > 1 else ""
    legend = " ".join(f"<span style='background:{STATE_COLOURS[k]};padding:0 .5em'>{k}</span>" for k in ("done", "partial", "failed", "missing"))
    return (f"<div style='overflow-x:auto'><table style='border-collapse:collapse;font-size:0.9em'>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
            f"<p style='font-size:0.85em'>{legend} — completed instances / expected per job{seed_note}</p>")


# --------------------------------------------------------------------------- page

def render() -> None:
    st.title("Studies")
    db = sidebar_db()
    with st.sidebar:
        st.markdown("**Plans**")
        studies_dir = Path(st.text_input("Studies directory", value=str(default_studies_dir(db)),
                                         help="Every *.json study spec below this folder is a planned experiment. "
                                              f"Set ${STUDIES_ENV} to change the default.")).expanduser()
    if not studies_dir.is_dir():
        st.info(f"`{studies_dir}` does not exist. Point the sidebar at a folder of study specs, or plan one below.")
        planned: list[Planned] = []
    else:
        planned = load_planned(studies_dir)

    if planned:
        rows = []
        for p in planned:
            exp, done, failed = p.totals
            study = p.study
            rows.append({
                "spec": str(p.path.relative_to(studies_dir)),
                "experiment": study.name if study else "—",
                "kind": study.kind if study else "—",
                "problem": study.problem if study else "—",
                "arms": len(study.methods) if study else 0,
                "jobs": len(p.jobs) if p.jobs is not None else None,
                "runs done": done, "expected": exp, "failed": failed,
                "progress": (done / exp) if exp else 0.0,
                "status": "done" if exp and done >= exp else ("not runnable: " + p.error if p.error else ("in progress" if done else "planned")),
            })
        st.dataframe(rows, hide_index=True, width="stretch",
                     column_config={"progress": st.column_config.ProgressColumn("progress", min_value=0.0, max_value=1.0, format="%.0f%%")})
        broken = [p for p in planned if p.error]
        if broken:
            st.warning(f"{len(broken)} spec(s) could not be expanded. If the error is an import error, install the repo that "
                       "declares the methods into the GUI's environment, e.g. `pip install -e /path/to/repo --no-deps`; the "
                       "declarations need no compute libraries.")
            for p in broken:
                st.caption(f"`{p.path.name}`: {p.error}")

        choices = {str(p.path.relative_to(studies_dir)): p for p in planned if p.study is not None}
        chosen = choices[st.selectbox("Study", list(choices))] if choices else None
        if chosen is not None:
            _study_detail(chosen, db)

    st.divider()
    _new_study_form(studies_dir, planned)


def _study_detail(p: Planned, db: str) -> None:
    study = p.study
    assert study is not None
    st.subheader(study.name)
    if study.description:
        st.caption(study.description)
    exp, done, failed = p.totals
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("jobs", len(p.jobs) if p.jobs is not None else "—")
    c2.metric("runs completed", f"{done} / {exp}")
    c3.metric("failed", failed)
    c4.metric("kind", study.kind)
    if p.statuses:
        st.markdown(grid_html(p.statuses), unsafe_allow_html=True)
        records = load_records(study.project, study.name)
        # the runner lays runs out as <root>/<study>/<method>/<arm>/<condition>/<seed>/<instance>
        roots = sorted({str(Path(a).parents[5]) if len(Path(a).parts) > 6 else a
                        for a in (r.get("artifacts_dir") for r in records) if a})
        if roots:
            st.caption("Results on disk: " + ", ".join(f"`{r}`" for r in roots[:3]) + (" …" if len(roots) > 3 else ""))
        rows = [[s.job.method, s.job.arm, _slug(s.job.condition), s.job.seed, s.state, f"{s.completed}/{s.expected}", s.failed]
                for s in p.statuses if s.state != "done"]
        if rows:
            with st.expander(f"Pending or failed jobs ({len(rows)})"):
                st.markdown(generic_html(["method", "arm", "condition", "seed", "state", "done", "failed"], rows, left_cols=3),
                            unsafe_allow_html=True)
    elif p.error:
        st.error(p.error)
    st.code(f"results-tracker recipe validate {p.path}\nresults-tracker recipe run {p.path} --db {db}", language="bash")
    with st.expander("Spec"):
        st.json(study.to_dict())


# --------------------------------------------------------------------------- authoring

def _knob_widget(knob: Knob, key: str, value: Any = None):
    """A widget for one knob, pre-filled with `value` (default: the knob's default). Returns the value."""
    v = knob.default if value is None else value
    label = f"{knob.name}" + (f" — {knob.doc}" if knob.doc else "")
    if knob.kind == "bool":
        return st.checkbox(label, value=bool(v), key=key)
    if knob.kind == "choice":
        opts = list(knob.choices or ())
        return st.selectbox(label, opts, index=opts.index(v) if v in opts else 0, key=key)
    if knob.kind in ("float", "int"):
        text = st.text_input(label, value="" if v is None else (f"{v:g}" if isinstance(v, float) else str(v)), key=key,
                             help=f"{knob.kind}" + (f", bounds {list(knob.bounds)}" if knob.bounds else "") + ("; empty = none" if knob.default is None else ""))
        return None if text.strip() == "" else knob.validate(text.strip())
    return st.text_input(label, value="" if v is None else str(v), key=key) or None


def _values(knob: Knob, text: str) -> list[Any]:
    return [knob.validate(t.strip()) for t in text.split(",") if t.strip()]


def _new_study_form(studies_dir: Path, planned: list[Planned]) -> None:
    st.header("New study")
    st.caption("Build a spec from the knobs the registered methods and problems declare; save it into the studies "
               "directory and run it from a terminal with `results-tracker recipe run`.")
    known_imports = sorted({m for p in planned if p.study for m in p.study.imports})
    imports = st.text_input("Import modules (comma-separated) that register methods and problems",
                            value=", ".join(known_imports), key="new_imports",
                            help="e.g. adaptivepnp.recipes. The modules must be importable in the GUI's environment.")
    import importlib

    for m in [m.strip() for m in imports.split(",") if m.strip()]:
        try:
            importlib.import_module(m)
        except Exception as e:  # noqa: BLE001
            st.error(f"cannot import `{m}`: {type(e).__name__}: {e}")
    if not registry.problems or not registry.methods:
        st.info("No problems or methods registered yet: add the module that declares them above.")
        return

    projects = sorted({p.study.project for p in planned if p.study}) or ["default"]
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Experiment name", value="", key="new_name", placeholder="deblurring-beta-sweep")
    kind = c2.selectbox("Kind", ["comparison", "sweep", "ablation"], key="new_kind")
    project = c3.text_input("Project", value=projects[0], key="new_project")
    problem_key = st.selectbox("Problem", sorted(registry.problems), key="new_problem")
    problem_cls = registry.problems[problem_key]
    c1, c2, c3 = st.columns(3)
    split = c1.selectbox("Split", list(problem_cls.splits), key="new_split")
    n_instances = int(c2.number_input("Instances per job", min_value=1, value=1, step=1, key="new_n"))
    seeds = [int(s) for s in c3.text_input("Seeds (comma-separated)", value="0", key="new_seeds").split(",") if s.strip()]
    description = st.text_input("Description", value="", key="new_desc")

    st.markdown("**Conditions** (one value or a comma-separated list per knob; the grid is their product)")
    conditions: dict[str, list] = {}
    cols = st.columns(max(1, min(4, len(problem_cls.conditions))))
    for i, knob in enumerate(problem_cls.condition_space()):
        with cols[i % len(cols)]:
            if knob.kind == "choice":
                vals = st.multiselect(knob.name, list(knob.choices or ()), default=[knob.default], key=f"new_cond_{knob.name}")
            else:
                text = st.text_input(knob.name, value=f"{knob.default:g}" if isinstance(knob.default, float) else str(knob.default),
                                     key=f"new_cond_{knob.name}", help=knob.doc or knob.kind)
                try:
                    vals = _values(knob, text)
                except ValueError as e:
                    st.error(str(e))
                    vals = []
            if vals and vals != [knob.default]:
                conditions[knob.name] = vals
            elif vals:
                conditions[knob.name] = vals

    st.markdown("**Method arms** (only knobs set away from their defaults are written into the spec)")
    n_arms = int(st.number_input("Arms", min_value=1, max_value=8, value=1, step=1, key="new_arms"))
    arms: list[Arm] = []
    for i in range(n_arms):
        method_key = st.selectbox(f"Arm {i + 1} method", sorted(registry.methods), key=f"new_arm_{i}_method")
        space = registry.methods[method_key].space()
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
    space0 = registry.methods[arms[0].method].space()
    if kind == "sweep":
        c1, c2 = st.columns([1, 2])
        knob_name = c1.selectbox("Swept knob", space0.names, key="new_sweep_knob")
        text = c2.text_input("Values (comma-separated)", value="", key="new_sweep_values")
        try:
            sweep = Sweep(knob_name, _values(space0[knob_name], text))
        except ValueError as e:
            st.error(str(e))
    elif kind == "ablation":
        text = st.text_area("Arms, one per line as knob=value (the arm configs above are the full model)",
                            value="", key="new_ablation_arms", height=100)
        arm_dicts = []
        for line in text.splitlines():
            if "=" in line:
                k, _, v = line.partition("=")
                arm_dicts.append({k.strip(): v.strip()})
        ablation = Ablation(base={}, arms=arm_dicts)

    study = Study(name=name or "unnamed", kind=kind, problem=problem_key, methods=arms, project=project or "default",
                  conditions=conditions, split=split, n_instances=n_instances, seeds=seeds or [0], sweep=sweep,
                  ablation=ablation, description=description,
                  imports=[m.strip() for m in imports.split(",") if m.strip()])
    try:
        problem_cls_, methods = load_study_classes(study)
        jobs = expand(study, problem_cls_, methods)
    except Exception as e:  # noqa: BLE001
        st.error(f"Not valid yet: {type(e).__name__}: {e}")
        jobs = None
    if jobs is not None:
        st.success(f"{len(jobs)} jobs × {n_instances} instances = {len(jobs) * n_instances} runs")
        with st.expander("Spec preview"):
            st.json(study.to_dict())
        c1, c2 = st.columns([2, 1])
        typed = c1.text_input("File name", value="", key="new_filename", placeholder=f"{name or '<name>'}.json",
                              help="Defaults to <experiment name>.json inside the studies directory.")
        filename = typed.strip() or (f"{name}.json" if name else "")
        overwrite = c2.checkbox("Overwrite if it exists", value=False, key="new_overwrite")
        if st.button("Save spec", key="new_save", disabled=not filename):
            target = studies_dir / filename
            if target.exists() and not overwrite:
                st.error(f"`{target}` exists; tick overwrite to replace it.")
            else:
                study.save(target)
                st.success(f"Saved `{target}`. Run it with: `results-tracker recipe run {target} --db {db_path()}`")
                st.cache_data.clear()
