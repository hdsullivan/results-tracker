# results-tracker

Track paper results — method comparisons, parameter sweeps, ablation studies —
and export them as paper-ready tables and figures.

**Status:** phase 1 (schema, logging API, CLI, aggregation). GUI comes in phase 2.
See [PLAN.md](PLAN.md).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
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
