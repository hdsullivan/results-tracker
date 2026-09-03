# GUI review and roadmap (September 2026)

Scope: the Streamlit GUI as a day-to-day dashboard for the adaptivePnP paper: track results, plan
experiments, and mark what goes into the manuscript. Reviewed on branch `recipe-layer` (d125682),
136 tests passing, against `demo.db`, `toy.db` and the adaptivePnP study specs in `studies/`.

## Where it stands

- Eight pages (Overview, Comparison, Sweep, Ablation, Visual, Run detail, Studies, Export) cover the
  three experiment kinds well; tables and figures already match the paper look, and every page has a
  LaTeX / figure expander. Audits (coverage, ambiguous base, panel-metric mismatch) are a real strength.
- The recipe layer makes *planned* work first-class: the Studies page derives progress per job from the
  database, and the New-study form builds specs from declared knobs.
- What is missing is the layer above experiments: nothing records **which view is a paper result**,
  and the planning side stops at "spec saved" (no cost, no link to the paper, no launch, no edit).
- Several adaptivePnP figures have no GUI counterpart yet: convergence curves, runtime-vs-quality,
  per-image distributions, kernel-type grouping, the global-parameter selection table.

## 1. Paper layer (highest value)

**Goal:** any table or figure you look at can be pinned as a planned paper asset, and the paper is
regenerated from exactly those pins.

- **`Asset` table** (project, kind: table / figure / visual / csv, `label` like `tab:main`, source
  experiment(s), the page's option dict as JSON, caption, status: planned / draft / final / dropped,
  `exported_at`, a fingerprint of the records it was rendered from). One row per thing in the manuscript.
- **"Pin to paper" button** on Comparison, Sweep, Ablation, Visual and Export: saves the current
  selections (grouping, metrics, filter, std style, width, zoom box, instance ...) as an Asset. Re-opening
  an Asset restores the page exactly, so a figure is reproducible from the GUI, not from memory.
- **Paper page** (new): the asset list in manuscript order with status, source experiments, data
  completeness (from the linked study's progress, see section 2) and **staleness**: the record
  fingerprint at last export vs now, so "Table II changed since you last exported it" is visible.
- **`export paper`**: the bundle regenerates the pinned assets only, with stable file names
  (`tables/tab-main.tex`, `figures/fig-beta.pdf`) and a manifest; the Makefile in `docs/INTEGRATION.md`
  collapses to one target. The current all-experiments bundle stays as "everything".
- **Paper tags on experiments**: a `paper` / `exploratory` / `superseded` state on `Experiment` so the
  Overview separates the manuscript's experiments from scratch work, and dropped experiments stop
  cluttering selectors (hidden by default, toggle to show). *Still open*: it needs a column on an existing
  table, i.e. a small migration helper in `db.py` (`create_all` only adds new tables).

Done 2026-09-03: `Asset` table + API, `export/paper.py`, Pin-to-paper on Comparison / Sweep / Ablation / Visual /
Export, the Paper page (status, order, caption, notes, staleness, export to directory or zip, *Open in Export*
restores a view via `?asset=`), `results-tracker export paper` and `asset list|set|rm`. Not done: the experiment
stage column above; restoring an asset on the analysis pages themselves (only Export restores widgets).

## 2. Planning layer

- **Study → Asset link**: a study spec names the assets it feeds (`"feeds": ["tab:main", "fig:beta"]`).
  The Paper page then shows per-asset readiness (jobs done / expected), the Studies page shows what a
  study is for.
- **Cost estimate per study**: median wall time per completed job (from a runtime metric or
  `time_iteration_total`) times pending jobs, shown next to progress; a sum across pending studies gives
  "compute left before submission".
- **Pending-job commands**: the study detail already prints `recipe run`; add a "pending only" filter
  export (a JSON spec restricted to the missing / failed cells) and a copyable shell block per machine
  (`/mnt/cc` paths from `haley-research-env`). Direct launch from the GUI stays local-only and last.
- **Edit and clone specs**: load an existing spec into the New-study form (today it only creates);
  "clone with changes" for the K2 / K5 / K10 families. Ablation arms via knob widgets (with the `set`
  joint form) instead of free text.
- **In-flight visibility**: `run_study` never writes `status=running`; log a running row at job start
  (replaced on completion) so the Overview's "running" count and the Studies grid show live work and
  crashes that never logged a failure.
- **Knob spaces without importing the repo**: the form imports `adaptivepnp.recipes` into the GUI's
  `.venv`, which does not have torch. Let `results-tracker recipe knobs --json` dump the declared spaces
  next to the specs and have the form read that file when the import fails.
- **Use the unused schema fields**: `Experiment.swept_params` and `base_run_id` exist but nothing sets or
  reads them. `run_study` should record the swept knob and base config; the Sweep page and the Overview
  headline then default to the real swept knob instead of the alphabetically first varying key (with
  conditions in the config, that is often `kernel`, not `beta`).

Done 2026-09-03 (step 3): `feeds` on the spec and readiness on the Paper page; compute-left from median
`runtime_s`; pending-only spec download; edit / clone into the form; ablation arms from knob widgets (single or
joint `set`); `running` rows from `run_study`; `swept_params` recorded and used as the default swept knob;
`recipe export-knobs` + `knobs.json` fallback on the Studies page. Not done: direct launch from the GUI (by design,
last), `base_run_id` (the `base` tag covers ablations), per-machine shell blocks beyond the run/validate lines.

## 3. Views adaptivePnP needs

- **Condition filter and facets**: ~~a shared sidebar `where` filter (`config.noise=0.01`, `config.K=5`) on
  every page, mirroring the CLI `--where`~~ (done; `--where config.K=[2,5]` now means any of); plus a per-project **value map** (`kernel 0-3 → isotropic`,
  `4-7 → anisotropic`, `8-11 → motion`; paper labels G/A/M) used for grouping, facet columns and table
  headers. This replaces the kernel-taxonomy code in `plot_cost_vs_quality.py` and `plot_boxplots.py`.
- **Cross-experiment tables**: the paper's main table has K across columns but `compare-K2/5/10` are
  separate experiments. Let Comparison and Export take several experiments (union, audited for
  overlap), or a project-wide "paper table" mode grouped by `config.K`.
- **Curves page** (new): per-iteration curves from `diagnostics.json` (PSNR / SSIM vs iteration,
  `rho_k`, `sigma_hat_over_true`), mean ± std over instances, one line per arm, faceted by condition.
  Ports `plot_cbsd68_convergence.py` and `plot_convergence_by_family.py`.
- **Two-metric scatter**: runtime vs PSNR with a path over K per method, hollow markers for
  `source=reported` or `is_baseline`. Ports `plot_cost_vs_quality.py`.
- **Per-instance distribution**: box / strip plot from the instance rows already in the database, plus a
  **per-instance table** behind every comparison cell (click a cell → the runs behind it). Also an
  instance picker "where ours gains most / least vs baseline X" to choose the visual-figure image
  (today hand-picked in `make_paper_figures.sh`).
- **Selection table for tuning**: `aggregate.select_best` per (method, K[, scale]) as a table with a
  boundary flag when the winner sits at the grid edge; a "materialize into spec" button that calls
  `recipes/tuned.py::materialize`. Ports `make_global_param_table.py`.

## 4. UX and plumbing

- ~~**One selection across pages**~~ (done): project, experiment and filter live in `session_state` and in the
  URL (`?db=…&project=…&experiment=…&where=…`). Still to do: `asset=` once the Paper layer exists, and a run id
  in the URL for Run detail.
- **Settings page** (new, small): edit metric direction / unit / format, method label / baseline /
  **display order** (paper tables want a fixed order: baselines, then ours), condition value maps, primary
  metric per project (the Overview currently guesses `psnr` / `ssim`). All CLI-only today.
- **Scale**: the Overview loads every record of every experiment into dicts; adaptivePnP studies are
  12 kernels × 3 noise × 20 images × arms. Compute Overview counts and headlines with SQL / grouped
  queries and load full records only for the selected experiment. Cache is keyed on file mtime, so any
  logged run invalidates everything; a per-experiment cache key would help during long runs.
- **Notes**: editable `Experiment.description` in the GUI and a dated free-text decision log per project
  ("beta = 0.5 chosen: plateau in `deblurring-ema-beta`"), surfaced on the Paper page next to the asset it
  justifies.
- **Small things**: database picker as a dropdown of recent files; studies directory stored per project
  rather than retyped; "Refresh" also re-imports registered modules; Run detail reachable by run id in the
  URL.

## Suggested order

| Step | Delivers | Size |
|---|---|---|
| 1 | Shared `where` filter + shared selection / URL params (§3, §4) | done 2026-09-03 |
| 2 | `Asset` table, Pin-to-paper on all pages, Paper page, `export paper` (§1) | done 2026-09-03 |
| 3 | Study ↔ asset link, cost estimate, pending-only spec, edit / clone form, running status (§2) | done 2026-09-03 |
| 4 | Value maps + facets, cross-experiment tables, Settings page (§3, §4) | 2 days |
| 5 | Curves page, two-metric scatter, distribution / instance picker, selection table (§3) | 3 days |
| 6 | Overview via SQL, notes, small UX items (§4) | 1 day |

Steps 1–3 make the GUI the place where the paper is planned and tracked; steps 4–5 retire the
remaining `adaptivepnp/paper/*.py` scripts (M6 of the refactor); step 6 is polish.
