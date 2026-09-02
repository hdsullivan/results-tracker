# results-tracker

Track paper results — method comparisons, parameter sweeps, ablation studies —
and export them as paper-ready tables and figures.

**Status:** phase 3 — schema, logging API, CLI, aggregation, bulk import, and all
five GUI pages (Overview, Comparison, Sweep, Ablation, Run detail). Paper exports
(LaTeX tables, IEEE figures) come in phase 4. See [PLAN.md](PLAN.md).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui]"
```

## Quick start

```bash
results-tracker demo --db results.db          # synthetic paper: 3 experiments
results-tracker experiments --db results.db
results-tracker table  -e main-comparison --by method --by dataset --db results.db
results-tracker sweep  -e lambda-sweep --param lambda --metric psnr --db results.db
results-tracker ablation -e ablation --db results.db
```

Set `RESULTS_TRACKER_DB=/path/to/results.db` to avoid passing `--db` every time.

## GUI

```bash
results-tracker ui --db results.db        # opens http://localhost:8501
```

Pages:

- **Overview** — what is in the database.
- **Comparison** — methods × metrics, mean ± std, best in bold, second
  underlined, bar chart, CSV download. Rows can be grouped by method, dataset,
  instance, seed, or any config key.
- **Sweep** — metric vs one swept config key (lines with ± std band, best value
  marked, log axis when the values span a decade) or vs two keys (heatmap).
  One line per method or dataset if you ask for it.
- **Ablation** — every config variant vs the full model: which settings changed
  (✓ / ✗ / value), mean ± std with Δ (absolute or %), and a bar chart of the
  deltas. The base is the run tagged `base`, else the most common config, or
  any run you pick.
- **Run detail** — config, metrics, config and metric diff against any other
  run, image gallery and log tail from the run's `artifacts_dir`.

## Importing existing results

```bash
# CSV: one row per run. Numeric columns become metrics unless listed as config.
results-tracker import old_results.csv -e main-comparison -p my-paper \
    --config-col lambda --config-col iters

# Directory of JSON run files ({"method":..., "config": {...}, "metrics": {...}})
results-tracker import runs/ -e lambda-sweep --type sweep

# Numbers copied from another paper
results-tracker import baselines.csv -e main-comparison --source reported

results-tracker import old_results.csv -e main-comparison --dry-run   # show the mapping first
```

Re-importing the same file is safe: identical runs are skipped.

## Logging from your own code

```python
from results_tracker import log_run

log_run(
    "lambda-sweep",                 # experiment name (created on first use)
    project="adaptive-pnp",         # one per paper
    method="ours", dataset="Set12", seed=0,
    config={"lambda": 0.1, "denoiser": "drunet", "adaptive": True},
    metrics={"psnr": 31.2, "ssim": 0.91, "runtime_s": 6.4},
    experiment_type="sweep",        # comparison | sweep | ablation (first call only)
    artifacts_dir="runs/2026-09-02/lambda0.1",  # images/logs live on disk, not in the DB
)
```

Git commit and hostname are captured automatically. Numpy/torch scalars are
converted; NaN becomes null. Metric direction (higher/lower is better) is
guessed from the name and can be overridden:

```bash
results-tracker metric define nrmse --lower --fmt .4f
```

## Data model

Everything is a **Run** (config JSON + metrics JSON) inside an **Experiment**
inside a **Project**. Comparisons group runs by method; sweeps group by a
config key; ablations group by the config diff against a base run (tag one
run `base`, or the most common config is used).

## Tests

```bash
pytest
```
