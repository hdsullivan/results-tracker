# results-tracker

Track paper results (method comparisons, parameter sweeps, ablations) in SQLite, view
them in a Streamlit GUI, export booktabs LaTeX tables and IEEE-sized matplotlib figures.

## Commands

```bash
source .venv/bin/activate            # or use .venv/bin/<tool> directly
pytest -q                            # 93 tests, ~8 s
results-tracker demo --db demo.db --reset --artifacts demo_artifacts
results-tracker ui --db demo.db      # GUI on http://localhost:8501 (falls back to a free port)
results-tracker export table -e main-comparison --db demo.db
results-tracker export bundle -p demo-paper -o /tmp/bundle.zip --db demo.db   # all paper assets at once
python scripts/screenshot.py http://localhost:8501 docs/screenshots   # needs Chrome + websocket-client
```

## Layout

- `src/results_tracker/models.py` — SQLModel schema. Everything is a `Run` (config JSON + metrics JSON).
- `api.py` — `log_run` and query helpers; `run_records` flattens ORM rows to dicts.
- `aggregate.py` — pure-Python stats over record dicts: comparison/pivot tables, sweeps, ablations
  (config diff vs base), grid audit. No ORM, no pandas. Ranking always uses unrounded means.
- `importer.py` — CSV / JSON bulk import with a one-time column mapping and duplicate skipping.
- `export/latex.py`, `export/figures.py`, `export/visual.py`, `export/csv.py`, `export/bundle.py` — paper exports
  (tables, quantitative figures, qualitative image grids, CSV). `figures.py` also holds `figure_tex` / `ieee_preamble`.
  `visual.py` ports `adaptivePnP/.../utils/deblur_figures.py` (zoom inset, metric stamp, error colour bar, GT block).
- `ui/` — Streamlit pages; `ui/common.py` holds cached loaders and sidebar selectors; `ui/charts.py` (Plotly)
  and `ui/tables.py` (HTML in the IEEEtran/booktabs look) mirror the export styling on screen.
- `cli.py` — Typer app; `export` and `metric` are sub-apps.
- `demo.py` — deterministic synthetic paper used by tests and the pitch.

## Conventions

- Aggregation and export functions take plain record dicts so they are testable without a DB.
- Metric direction lives in the `Metric` table; guessed from the name on first log, override with
  `results-tracker metric define`.
- Streamlit widgets: use `width="stretch"` (not the deprecated `use_container_width`).
- Colours and chart style: both the GUI (`ui/charts.py`, Plotly) and the print exports
  (`export/figures.py`, matplotlib) follow the lab's paper style ported from
  `adaptivePnP-paper-workspace/.../utils/ablation_utils.py::set_paper_style` (tab10 colours, filled
  circles, boxed axes, inward ticks, top bordered legend, bold panel captions, 5.0/10.5 in widths).
  Keep the two in sync if the lab style changes. Assign hues in fixed first-seen order; never cycle by rank.
- UI tests use `AppTest.from_string("from results_tracker.ui import X; X.render()")`.
- A long-running `streamlit run` does not reload edited *imported* modules reliably; restart it.
