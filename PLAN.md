# Results Tracker GUI — Build Plan

Goal: a local GUI for tracking paper results across three experiment types
(method comparisons, parameter sweeps, ablation studies), with one-click export
to paper-ready LaTeX tables and IEEE-style figures.

## 1. Assumptions (correct me if wrong)

- Lab code is Python (numpy / torch / jax). Results currently live in ad-hoc
  folders of .npy/.json/.csv files plus notebook plots.
- Single user or small lab, runs on a laptop or lab workstation. No cloud.
- The competition rewards a working demo plus something the lab would actually
  adopt, so speed to a polished vertical slice matters more than breadth.

## 2. Stack (recommended)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Matches lab code; results can be logged from the same scripts |
| Storage | SQLite via SQLModel | Zero setup, single file, easy to back up or commit |
| GUI | Streamlit (multipage) | Fastest path to a usable GUI; Plotly for interactive plots |
| Export | matplotlib + Jinja2 | Static IEEE figures and LaTeX tables, reusing the lab's plot/table conventions |
| Packaging | `pip install -e .` + `results-tracker` CLI | One command to launch the app or log a run |

Alternative if the judges want something more "app-like": NiceGUI (same Python
stack, more layout control). Avoid a React frontend unless someone on the team
already lives in JS; it doubles the surface area for a one-week build.

## 3. Data model

Core idea: everything is a **Run**. Experiment type is just how you group and
view runs.

- **Project** — one per paper. name, description.
- **Metric** — name, unit, higher_is_better, display format (e.g. `.2f`).
- **Dataset** — name, optional per-instance list (e.g. 10 test images).
- **Method** — name, is_baseline flag, display label for tables.
- **Experiment** — belongs to a Project. type ∈ {comparison, sweep, ablation},
  name, swept parameter names (for sweeps), base config id (for ablations).
- **Run** — belongs to an Experiment. method_id, dataset_id, seed,
  `config` (JSON), `metrics` (JSON: metric_name → value), git_commit, timestamp,
  status, artifacts_dir (path to images/logs), notes, tags.

Metrics stored as JSON keeps the schema stable as you add PSNR, SSIM, runtime,
iterations, etc. Config stored as JSON lets ablations be computed as diffs
against the base config rather than hand-labeled.

## 4. Ingestion (three doors)

1. **Python logger** — `from results_tracker import log_run;
   log_run(experiment="sweep-lambda", config=cfg, metrics=m, artifacts=dir)`.
   Auto-captures git commit, timestamp, hostname.
2. **Bulk import** — point at a folder of JSON/CSV run outputs, map columns
   to fields once, import. Handles the backlog of existing results.
3. **Manual form** — for numbers copied from a competitor's paper (baselines
   you didn't rerun). Flag these as `source=reported`.

## 5. Views (pages)

1. **Overview** — projects, run counts, recent runs, failed runs.
2. **Comparison** — pick experiment, rows = methods, columns = metrics
   (optionally × datasets). Aggregates over seeds as mean ± std. Best value
   bold, second-best underlined. Toggle: per-instance table vs aggregate.
3. **Sweep** — pick x parameter, y metric, group-by (method or dataset).
   Line plot with error bands. 2-parameter sweeps render as a heatmap. Mark
   the chosen default value.
4. **Ablation** — base config vs variants. Table of "what was removed/changed"
   (computed from config diff) with metric deltas vs full model. Waterfall or
   bar chart of deltas.
5. **Run detail** — full config, metrics, config diff against any other run,
   image gallery from artifacts_dir (reconstructions, error maps), log tail.
6. **Export** — for the current view: LaTeX table (booktabs, IEEE column
   width), PDF/PNG figure (IEEE single/double column sizes), CSV. Copy button
   for LaTeX.

## 6. Build phases (about one working week)

| Phase | Deliverable | Est. | Status |
|---|---|---|---|
| 0 | Confirm scope, pick stack, write 3 demo experiments as fixtures | 0.5 day | done |
| 1 | Schema, SQLite setup, `log_run` API, CLI, tests for aggregation | 1 day | done |
| 2 | Bulk import + Comparison page + Run detail page | 1 day | done |
| 3 | Sweep page + Ablation page (config-diff logic) | 1 day | next |
| 4 | Export: LaTeX tables, IEEE figures, CSV | 1 day | |
| 5 | Demo dataset, README with GIF, polish, 5-minute pitch | 0.5 day | |

Stop-early rule: if time runs short, ship phases 0–2 plus LaTeX table export.
A comparison table that goes straight into a paper is the strongest single
demo.

## 7. Demo story for the competition

Walk through one fake paper end to end:
1. Log 3 methods × 2 datasets × 3 seeds from a script (30 seconds).
2. Open Comparison, show mean ± std with best bolded.
3. Open Sweep for regularization strength, show the knee, pick the default.
4. Open Ablation, show which component matters.
5. Click Export, paste the LaTeX into a slide.

## 8. Risks and mitigations

- **Heterogeneous result formats** → JSON metrics field + one-time column
  mapper in the importer.
- **Large image artifacts** → store paths, not blobs. Thumbnails on demand.
- **Streamlit reruns feel slow with many runs** → cache queries with
  `st.cache_data`; SQLite handles tens of thousands of runs fine.
- **Scope creep** → the six pages above are the whole app. No auth, no
  multi-user, no cloud.

## 9. Open questions

1. Competition deadline and what the judges score (usefulness, polish, novelty)?
2. Where do results live today, and in what format? (Decides the importer.)
3. Team size and whether anyone prefers a JS frontend?
4. Any must-have beyond the three experiment types (e.g. runtime vs quality
   Pareto plots, convergence curves)?
