# results-tracker

Track paper results (method comparisons, parameter sweeps, ablations) in SQLite, view
them in a Streamlit GUI, export booktabs LaTeX tables and IEEE-sized matplotlib figures.

## Commands

```bash
source .venv/bin/activate            # or use .venv/bin/<tool> directly
pytest -q                            # 115 tests, ~9 s
results-tracker demo --db demo.db --reset --artifacts demo_artifacts
results-tracker recipe demo --db toy.db --reset --artifacts toy_artifacts --write-specs specs   # recipe-layer demo
results-tracker ui --db demo.db      # GUI on http://localhost:8501 (falls back to a free port)
results-tracker export table -e main-comparison --db demo.db
results-tracker export bundle -p demo-paper -o /tmp/bundle.zip --db demo.db   # a default table/figure per experiment
results-tracker export paper -p demo-paper -o paper_assets --db demo.db        # the pinned paper assets (Paper page / asset list)
results-tracker recipe export-knobs -i results_tracker.recipe.toy -o specs/knobs.json  # declarations so the GUI can plan without the repo
python scripts/screenshot.py http://localhost:8501 docs/screenshots   # needs Chrome + websocket-client
```

## Docs

- `docs/USER_GUIDE.html` — the printable user guide (connect a repo, declare, plan, run, look, publish); open in a browser and
  print to PDF. `docs/INTEGRATION.md` — how a lab repo adopts the tracker (instrumentation, import, Makefile, conventions).
- `docs/PITCH.md` — the five-minute demo script. `PLAN.md` and `docs/GUI_ROADMAP.md` are internal
  planning notes, kept locally and gitignored: the repo is public, and they are written to the author.
- `docs/RECIPES.md` — the recipe layer: declare a `Method` and a `Problem`, run comparison/sweep/ablation studies from JSON specs.

## Layout

- `src/results_tracker/models.py` — SQLModel schema. Everything is a `Run` (config JSON + metrics JSON).
- `api.py` — `log_run` and query helpers; `run_records` flattens ORM rows to dicts and applies the project's value maps
  (`derived.<name>` fields, see `valuemaps.py`). `db.add_missing_columns` migrates older databases when a model gains a column.
- `aggregate.py` — pure-Python stats over record dicts: comparison/pivot tables, sweeps, ablations
  (config diff vs base), grid audit, selection tables (tuning winners with a boundary flag), trade-off points,
  instance tables and gains. No ORM, no pandas. Ranking always uses unrounded means.
  `curves.py` — per-iteration curves read from each run's `diagnostics.json`, pooled per group (NaN-aware, no numpy).
- `importer.py` — CSV / JSON bulk import with a one-time column mapping and duplicate skipping.
- `export/latex.py`, `export/figures.py`, `export/visual.py`, `export/csv.py`, `export/bundle.py` — paper exports
  (tables, quantitative figures, qualitative image grids, CSV). `figures.py` also holds `figure_tex` / `ieee_preamble`.
  `export/paper.py` — the paper layer: `Asset` rows (models.py) are specs `{kind, experiment, filters, options}` pinned
  from GUI views; `render_asset` renders one from records, `render_paper`/`write_paper` regenerate a project's assets into
  stable file names, `records_fingerprint`/`staleness` tell when the data moved under an export.
  `visual.py` ports `adaptivePnP/.../utils/deblur_figures.py` (zoom inset, metric stamp, error colour bar, GT block).
- `ui/` — Streamlit pages; `ui/common.py` holds cached loaders, sidebar selectors (project, experiment, pooled experiments,
  filter), the keyed-widget helpers and the `pin_to_paper` expander; `ui/paper.py` lists and exports the pinned assets;
  `ui/settings.py` edits metrics, methods, value maps and the primary metric; `ui/curves.py` and `ui/tradeoff.py` are the
  per-iteration and cost-vs-quality views; `export/paper.KIND_PAGE` says which page configures (and restores) an asset kind;
  `ui/charts.py` (Plotly)
  and `ui/tables.py` (HTML in the IEEEtran/booktabs look) mirror the export styling on screen.
- `recipe/` — `knobs.py` (declared parameter spaces), `core.py` (`Method`, `Problem`, `Instance`, `Estimate`, registry),
  `study.py` (`Study` spec with `feeds`, `expand`, `pending_subset`, resumable `run_study` that logs `running` rows and
  calls `log_run`), `declared.py` (knob declarations as JSON → planning-only stand-in classes), `toy.py` (numpy phantom
  deblurring, the recipe demo). Core is array-library agnostic; only `toy.py` imports numpy.
- `cli.py` — Typer app; `export`, `metric` and `recipe` are sub-apps.
- `demo.py` — deterministic synthetic paper used by tests and the pitch.

## Conventions

- Aggregation and export functions take plain record dicts so they are testable without a DB. Grouping/filter keys are
  `method`, `dataset`, ..., `config.<key>` and `derived.<name>`; offer them via `agg.grouping_keys`, order rows with
  `agg.method_order` and derived columns with `agg.value_order`.
- Recipe runs put the condition (kernel, noise, ...) in `config` next to the method knobs; `ablation_table` treats
  keys that vary among `base`-tagged runs as conditions and pools over them. Re-running a study is a resume (`has_run`).
- log_run deduplicates on (experiment, method, dataset, instance, seed, config); the importer passes on_duplicate='allow'.
- Never rescale an image by its own max (shared display range); never pool rows over different dataset sets silently.
- Metric direction lives in the `Metric` table; guessed from the name on first log, override with
  `results-tracker metric define`.
- Streamlit widgets: use `width="stretch"` (not the deprecated `use_container_width`).
- Experiment pages load records in a fixed order: `select_project_experiment` → `load_records` → `sidebar_filter`
  → `agg.completed`. Project, experiment and the `where` filter live in `st.session_state` under the `KEY_*` names
  in `ui/common.py` and are mirrored into `st.query_params`. Selectors are rendered from those values (`index=`),
  filter multiselects through `_keyed_multiselect` (one key shared by every page, pruned to the current options
  before rendering); never give a shared selector a per-page widget key, or the choice is lost on page switch.
- Colours and chart style: both the GUI (`ui/charts.py`, Plotly) and the print exports
  (`export/figures.py`, matplotlib) follow the lab's paper style ported from
  `adaptivePnP-paper-workspace/.../utils/ablation_utils.py::set_paper_style` (tab10 colours, filled
  circles, boxed axes, inward ticks, top bordered legend, bold panel captions, 5.0/10.5 in widths).
  Keep the two in sync if the lab style changes. Assign hues in fixed first-seen order; never cycle by rank.
- UI tests use `AppTest.from_string("from results_tracker.ui import X; X.render()")`.
- A long-running `streamlit run` does not reload edited *imported* modules reliably; restart it.
- The Overview never loads all runs: `api.experiment_summaries` (SQL) for the tables, `api.recent_runs` for the recent list;
  `load_records(project, experiment)` is cached on `api.experiment_version` (count, max id, max timestamp), so pages only
  reload the experiment that changed. Keep new summary needs in SQL rather than in Python over records.
- Experiments carry a `stage` (paper / exploratory / superseded); `select_project_experiment` hides superseded ones unless the
  sidebar checkbox is on. `Note` rows are the dated decision log (Overview, Paper page per asset, `note` CLI).
