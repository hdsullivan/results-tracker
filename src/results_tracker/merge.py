"""Merge the runs of one results database into another, leaving local curation alone.

A results database holds two kinds of thing. The runs are produced somewhere else -- a cluster, a
colleague's machine -- and that machine owns them. Everything around them is curation done where the
paper is written: pinned assets, notes, value maps, the project's plot style, metric formats, method
labels and display order, experiment stages, and the local `artifacts_dir` each run was repointed to
after the run folders were fetched. Copying the database file over the top takes the first and
destroys the second, which is why `results-tracker merge` copies runs instead of files.

    results-tracker merge incoming.db --db paper.db

Runs are matched across the two files by *setting*, not by `id`: `id` is a per-file autoincrement and
means nothing outside its own database. The setting is the same one `log_run` uses for duplicate
protection -- project, experiment, method, dataset, instance, seed and config -- so a run merged twice
is recognised the second time and a study can be merged while it is still being written.

Who wins, per field:

- a setting not in the destination is inserted whole, the source's tags, notes and artifacts_dir included
- a setting already there keeps its local `tags` (the source's are added, never removed -- a run tagged
  `base` locally stays tagged), its local `notes` and its local `artifacts_dir`, and takes the source's
  `metrics`, `status`, `timestamp`, `git_commit` and `hostname`, so a run that was `running` at the last
  merge lands as `completed` at the next one
- `project`, `experiment`, `method`, `dataset` and `metric` rows are created when the destination has no
  row of that name, and are never modified: that is where labels, formats, positions, stages and the
  plot style live
- `asset`, `note` and `valuemap` are never read or written

Repeats are counted, not collapsed: a setting the source has three times and the destination once
updates the one and inserts two.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from sqlalchemy import bindparam, insert, select, update

from .api import _resolve_engine, config_fingerprint
from .db import get_engine
from .models import Dataset, Experiment, Method, Metric, Project, Run

PathLike = Union[str, Path]

#: run columns the source owns; a run that finished since the last merge changes these
RESULT_FIELDS = ("metrics", "status", "timestamp", "git_commit", "hostname")
#: run columns the destination owns; the source never overwrites a value that is already set
LOCAL_FIELDS = ("tags", "notes", "artifacts_dir")
#: everything needed to insert a run, minus the foreign keys, which are remapped
RUN_FIELDS = ("instance", "seed", "config", "metrics", "status", "source", "git_commit", "hostname",
              "timestamp", "artifacts_dir", "notes", "tags")

#: lookup tables copied by name, never updated: (model, the fields carried when creating a new row)
LOOKUPS = (
    (Method, ("name", "label", "is_baseline", "position")),
    (Dataset, ("name", "description", "instances")),
    (Metric, ("name", "unit", "higher_is_better", "fmt")),
)


@dataclass
class MergeReport:
    runs_inserted: int = 0
    runs_updated: int = 0
    runs_unchanged: int = 0
    #: table name -> names created in the destination
    created: dict[str, list[str]] = field(default_factory=dict)
    dry_run: bool = False

    @property
    def runs_seen(self) -> int:
        return self.runs_inserted + self.runs_updated + self.runs_unchanged

    def summary(self) -> str:
        head = "would merge" if self.dry_run else "merged"
        parts = [f"{head} {self.runs_seen} runs: {self.runs_inserted} new, {self.runs_updated} updated, "
                 f"{self.runs_unchanged} unchanged"]
        for table, names in self.created.items():
            if names:
                shown = ", ".join(names[:6]) + (f" (+{len(names) - 6} more)" if len(names) > 6 else "")
                parts.append(f"{len(names)} new {table}: {shown}")
        return " · ".join(parts)


def _rows(conn, model, fields: Sequence[str], extra: Sequence[str] = ("id",)) -> list[dict[str, Any]]:
    cols = [getattr(model, c) for c in (*extra, *fields)]
    return [dict(zip((*extra, *fields), r)) for r in conn.execute(select(*cols)).all()]


def _run_key(project: str, experiment: str, method: Optional[str], dataset: Optional[str],
             instance: Optional[str], seed: Optional[int], config: Any) -> tuple:
    """The setting a run is: what `log_run` treats as a duplicate, plus the project the experiment is in.

    Experiment names are unique per project, not globally, so two papers may both have a `main-comparison`.
    """
    return (project, experiment, method, dataset, instance, seed, config_fingerprint(config))


def merge_database(
    source: PathLike,
    *,
    db: Optional[PathLike] = None,
    engine=None,
    projects: Optional[Sequence[str]] = None,
    experiments: Optional[Sequence[str]] = None,
    dry_run: bool = False,
) -> MergeReport:
    """Merge the runs of `source` into the database named by `db` / `engine`. See the module docstring.

    `projects` and `experiments` restrict what is taken from the source (by name); everything else in it is
    left behind. `dry_run` reports what would happen and writes nothing.
    """
    src_path = Path(source).expanduser()
    if not src_path.is_file():
        raise FileNotFoundError(f"no database at {src_path}")
    dst_engine = _resolve_engine(engine, db)
    src_engine = get_engine(src_path)
    dst_file = dst_engine.url.database
    if dst_file and Path(dst_file).resolve() == src_path.resolve():
        raise ValueError(f"source and destination are the same database ({src_path})")
    report = MergeReport(dry_run=dry_run)
    want_projects, want_experiments = (set(projects) if projects else None), (set(experiments) if experiments else None)

    # "commit as you go": nothing is written until the commit at the end, so a dry run simply never reaches it
    with src_engine.connect() as src, dst_engine.connect() as dst:
        # ---------------------------------------------------------------- lookups, by name
        src_projects = {r["id"]: r for r in _rows(src, Project, ("name", "description", "primary_metric",
                                                                 "studies_dir", "plot_style"))}
        src_experiments = {r["id"]: r for r in _rows(src, Experiment, ("project_id", "name", "type", "description",
                                                                      "swept_params", "stage"))}
        src_methods = {r["id"]: r["name"] for r in _rows(src, Method, ("name",))}
        src_datasets = {r["id"]: r["name"] for r in _rows(src, Dataset, ("name",))}

        def keep(exp: dict) -> bool:
            project = src_projects[exp["project_id"]]["name"]
            return ((want_projects is None or project in want_projects)
                    and (want_experiments is None or exp["name"] in want_experiments))

        live = {eid for eid, e in src_experiments.items() if keep(e)}
        live_projects = {src_experiments[eid]["project_id"] for eid in live}

        project_ids = {r["name"]: r["id"] for r in _rows(dst, Project, ("name",))}
        for pid in live_projects:
            row = src_projects[pid]
            if row["name"] not in project_ids:
                fields = {k: row[k] for k in ("name", "description", "primary_metric", "studies_dir", "plot_style")}
                project_ids[row["name"]] = dst.execute(insert(Project).values(**fields)).inserted_primary_key[0]
                report.created.setdefault("projects", []).append(row["name"])

        name_ids: dict[Any, dict[str, int]] = {}
        for model, fields in LOOKUPS:
            have = {r["name"]: r["id"] for r in _rows(dst, model, ("name",))}
            for row in _rows(src, model, fields):
                if row["name"] not in have:
                    have[row["name"]] = dst.execute(
                        insert(model).values(**{k: row[k] for k in fields})).inserted_primary_key[0]
                    report.created.setdefault(model.__tablename__ + "s", []).append(row["name"])
            name_ids[model] = have

        # experiments are unique per project, not globally, so they key on (project name, experiment name)
        dst_projects = {r["id"]: r["name"] for r in _rows(dst, Project, ("name",))}
        exp_ids = {(dst_projects.get(r["project_id"]), r["name"]): r["id"]
                   for r in _rows(dst, Experiment, ("project_id", "name"))}
        src_to_dst_exp: dict[int, int] = {}
        for eid in live:
            row = src_experiments[eid]
            project = src_projects[row["project_id"]]["name"]
            key = (project, row["name"])
            if key not in exp_ids:
                fields = {k: row[k] for k in ("name", "type", "description", "swept_params", "stage")}
                exp_ids[key] = dst.execute(
                    insert(Experiment).values(project_id=project_ids[project], **fields)).inserted_primary_key[0]
                report.created.setdefault("experiments", []).append(f"{project}/{row['name']}")
            src_to_dst_exp[eid] = exp_ids[key]

        # ---------------------------------------------------------------- runs, by setting
        dst_projects = {r["id"]: r["name"] for r in _rows(dst, Project, ("name",))}
        dst_experiments = {r["id"]: (dst_projects.get(r["project_id"]), r["name"])
                           for r in _rows(dst, Experiment, ("project_id", "name"))}
        dst_methods = {r["id"]: r["name"] for r in _rows(dst, Method, ("name",))}
        dst_datasets = {r["id"]: r["name"] for r in _rows(dst, Dataset, ("name",))}

        # The config goes into the key and is not kept: on a paper-sized database these rows are most of
        # the memory the merge uses, and nothing downstream needs the destination's copy of it.
        index: dict[tuple, list[dict[str, Any]]] = {}
        for r in _rows(dst, Run, ("experiment_id", "method_id", "dataset_id", "instance", "seed", "config",
                                  *RESULT_FIELDS, *LOCAL_FIELDS)):
            project, experiment = dst_experiments.get(r["experiment_id"], (None, None))
            if experiment is None:
                continue  # an orphaned run: no experiment row to name it
            key = _run_key(project, experiment, dst_methods.get(r["method_id"]), dst_datasets.get(r["dataset_id"]),
                           r["instance"], r["seed"], r.pop("config"))
            index.setdefault(key, []).append(r)

        inserts: list[dict[str, Any]] = []
        updates: list[dict[str, Any]] = []
        src_runs = _rows(src, Run, ("experiment_id", "method_id", "dataset_id", *RUN_FIELDS))
        src_runs.sort(key=lambda r: (r["timestamp"], r["id"]))
        for r in src_runs:
            if r["experiment_id"] not in src_to_dst_exp:
                continue
            exp = src_experiments[r["experiment_id"]]
            key = _run_key(src_projects[exp["project_id"]]["name"], exp["name"], src_methods.get(r["method_id"]),
                           src_datasets.get(r["dataset_id"]), r["instance"], r["seed"], r["config"])
            waiting = index.get(key)
            if not waiting:
                inserts.append({
                    "experiment_id": src_to_dst_exp[r["experiment_id"]],
                    "method_id": name_ids[Method].get(src_methods.get(r["method_id"])),
                    "dataset_id": name_ids[Dataset].get(src_datasets.get(r["dataset_id"])),
                    **{k: r[k] for k in RUN_FIELDS},
                })
                report.runs_inserted += 1
                continue
            mine = waiting.pop(0)
            change = {k: r[k] for k in RESULT_FIELDS if r[k] != mine[k]}
            # the source may know tags this database does not; it never knows which ones to drop
            tags = list(mine["tags"] or [])
            new_tags = [t for t in (r["tags"] or []) if t not in tags]
            if new_tags:
                change["tags"] = tags + new_tags
            for local in ("notes", "artifacts_dir"):
                if not mine[local] and r[local]:
                    change[local] = r[local]
            if change:
                updates.append({"_id": mine["id"], **change})
                report.runs_updated += 1
            else:
                report.runs_unchanged += 1

        if dry_run:
            return report  # closing the connection without a commit discards the lookup rows too
        if inserts:
            dst.execute(insert(Run), inserts)
        # one statement per shape of change, since executemany needs every row to bind the same columns
        for fields in {tuple(sorted(u.keys())) for u in updates}:
            batch = [u for u in updates if tuple(sorted(u.keys())) == fields]
            stmt = update(Run).where(Run.id == bindparam("_id")).values(
                **{f: bindparam(f) for f in fields if f != "_id"})
            dst.execute(stmt, batch)
        dst.commit()
    return report
