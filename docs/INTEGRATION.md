# Using results-tracker with an existing repo

This guide takes a research repo that already produces numbers and images and
wires it to results-tracker so that every table and figure in the paper is
generated from a database of logged runs. It assumes Python 3.10+ and a repo
with some experiment script (`run_deblur.py`, `compare.py`, `train.py`, …).

Time budget: about an hour for the instrumentation, then a one-off import of
whatever results you already have.

---

## 1. The mental model (two minutes)

Everything is a **run**: one execution of one method on one dataset (optionally
one image and one seed), with

- a `config` dict — every knob, including the ones you did not vary,
- a `metrics` dict — the numbers you would put in a table,
- an optional `artifacts_dir` — a folder on disk with the reconstruction and
  friends.

Runs live in **experiments**, experiments in a **project** (one per paper). The
experiment *type* only decides how runs are grouped on screen:

| Type | Groups runs by | Page shows |
|---|---|---|
| `comparison` | method × dataset | mean ± std table, best bold, bar chart, qualitative grid |
| `sweep` | one config key | metric vs parameter, plateau / sensitivity |
| `ablation` | config diff against the full model | ✓/✗ matrix, deltas, effect sizes |

Nothing else needs to be declared up front. Methods, datasets and metrics are
created the first time they are logged.

---

## 2. Install into the repo's environment

Inside the environment your experiments run in:

```bash
pip install -e /path/to/results-tracker[ui]
```

(`[ui]` pulls Streamlit, Plotly, pandas and matplotlib; the logging API alone
needs only SQLModel.) Or add it to the repo's `requirements.txt` /
`pyproject.toml` as a path or git dependency.

Decide where the database lives. One SQLite file per paper, next to the
manuscript, works well:

```bash
export RESULTS_TRACKER_DB=~/research/my-paper/results.db
```

Put that line in the repo's `.envrc`, `activate` script or Makefile so every
command and every script sees the same file. The database is small (tens of
thousands of runs are a few MB) and safe to commit alongside the paper.

Sanity check:

```bash
results-tracker init
results-tracker --version
```

---

## 3. Declare metrics and methods once

Metric direction and display precision are stored in the database. The name
heuristic (`psnr` ↑, `rmse` ↓, `time` ↓ …) is usually right; fix anything it
gets wrong and set the units that should appear in table headers:

```bash
results-tracker metric define psnr --unit dB --fmt .2f
results-tracker metric define ssim --fmt .3f
results-tracker metric define nrmse --lower --fmt .4f
results-tracker metric define runtime_s --lower --unit s --fmt .1f
```

Method labels are what appears in tables and legends. Give baselines a citation
and mark them as baselines so qualitative figures order them before yours; a
`position` fixes the row order of every table (or set both on the Settings page):

```python
from results_tracker import define_method
define_method("pnp_admm", label=r"PnP-ADMM~\cite{venkatakrishnan2013}", is_baseline=True)
define_method("dpir",     label=r"DPIR~\cite{zhang2021}",              is_baseline=True)
define_method("adaptive_pnp", label="AdaptivePnP", position=9)   # ours: not a baseline, last row
```

Labels containing a backslash are passed to LaTeX untouched; plain labels are
escaped.

---

## 4. Instrument the experiment script

The only call you need is `log_run`. Put it where your script already has the
config, the metrics and the output folder. A typical inverse-problems loop:

```python
from pathlib import Path
from results_tracker import log_run

PROJECT = "adaptive-pnp"

for image_idx, (x_true, y) in enumerate(dataset):
    out_dir = Path(args.out) / f"kernel{args.kernel}_noise{args.noise}" / method / f"img{image_idx:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        x_hat, info = reconstruct(y, method=method, K=args.K, lam=args.lam)
        save_png(out_dir / "reconstruction.png", x_hat)
        save_png(out_dir / "ground_truth.png",   x_true)
        save_png(out_dir / "measurement.png",    y)
        save_png(out_dir / "kernel.png",         kernel)          # optional PSF thumbnail
        log_run(
            "deblur-main",                       # experiment (created on first use)
            project=PROJECT,
            experiment_type="comparison",
            method=method,
            dataset="CBSD68",
            instance=f"img{image_idx:02d}",     # per-image results; tables aggregate over them
            seed=args.seed,
            config={"K": args.K, "lambda": args.lam, "kernel": args.kernel,
                    "noise": args.noise, "denoiser": args.denoiser},
            metrics={"psnr": psnr(x_hat, x_true), "ssim": ssim(x_hat, x_true),
                     "runtime_s": info["runtime"]},
            artifacts_dir=str(out_dir),
        )
    except Exception as e:                         # a crash is a result too
        log_run("deblur-main", project=PROJECT, method=method, dataset="CBSD68",
                instance=f"img{image_idx:02d}", seed=args.seed,
                config={"K": args.K, "lambda": args.lam, "kernel": args.kernel,
                        "noise": args.noise, "denoiser": args.denoiser},
                metrics={}, status="failed", notes=str(e)[:500])
        raise
```

Rules of thumb:

- **Log the whole config**, not just the varied knob. Ablations are computed as
  config diffs, and a knob you did not log cannot be diffed.
- **Metrics are numbers.** numpy / torch scalars are fine, NaN becomes null,
  strings are rejected.
- **Same file names in every run folder.** The Visual page finds images by name
  across methods (`reconstruction.png`, `ground_truth.png`, `measurement.png`,
  `kernel.png` are recognised automatically; anything else can be picked by
  hand). Images must be the same pixel size across methods.
- **Use `instance` for per-image rows.** Comparison tables average over
  instances and seeds unless you ask them not to; the Visual page lets you pick
  one instance.
- Git commit and hostname are captured automatically. Pass `git_commit=None`
  to opt out (e.g. in tests).
- **Re-running a script is safe.** A run with the same experiment, method,
  dataset, instance, seed and config is a duplicate *setting*: by default
  `log_run` keeps the existing completed run (with a `DuplicateRunWarning`) so
  n never inflates, and replaces a failed one. Use `on_duplicate="replace"` to
  overwrite, `"allow"` for deliberate repeats without a seed, `"error"` to be
  strict. Booleans are not metrics; log flags in `config` or as explicit 0/1.

### Hydra / OmegaConf

```python
from omegaconf import OmegaConf
log_run(cfg.experiment, project=cfg.project, method=cfg.method.name, dataset=cfg.data.name,
        config=OmegaConf.to_container(cfg, resolve=True), metrics=metrics, artifacts_dir=out_dir)
```

Nested configs are fine; keys are flattened as `data.name`, `solver.iters` when
displayed or diffed.

### Ablations

Log the full model with `tags=["base"]` and each variant with **exactly one**
config key changed. Everything else (labels like "w/o warm_start", the ✓/✗
matrix, the deltas, the effect sizes) is derived from the diff:

```python
for variant in [{}, {"adaptive": False}, {"warm_start": False}, {"denoiser": "dncnn"}]:
    cfg = {**BASE, **variant}
    log_run("ablation", project=PROJECT, experiment_type="ablation", method="adaptive_pnp",
            dataset="Set12", seed=seed, config=cfg, metrics=run(cfg),
            tags=["base"] if not variant else [])
```

### Sweeps

Sweep one numeric config key with repeated seeds; the page finds the varying
key itself, switches to a log axis when the values span a decade, and reports
the plateau:

```python
for lam in [0.01, 0.03, 0.1, 0.3, 1.0]:
    for seed in range(3):
        log_run("lambda-sweep", project=PROJECT, experiment_type="sweep", method="adaptive_pnp",
                dataset="Set12", seed=seed, config={**BASE, "lambda": lam}, metrics=run(lam, seed))
```

---

## 5. Import the results you already have

Most repos already have a `metrics.csv` or a folder of JSON files. Import them
once instead of re-running. Always dry-run first to see how columns map:

```bash
results-tracker import paper_results/metrics.csv -e deblur-main -p adaptive-pnp \
    --method-col method --dataset-col dataset --instance-col image_idx --seed-col seed \
    --config-col K --config-col kernel_idx --config-col noise_level \
    --dry-run
```

How columns are treated:

- `--method-col`, `--dataset-col`, `--instance-col`, `--seed-col`, `--status-col`
  name the identity columns (defaults: `method`, `dataset`, `instance`, `seed`,
  `status`).
- `--config-col` marks numeric columns that are *parameters*, not metrics
  (`K`, `noise_level`, `lambda`). Unlisted numeric columns become metrics;
  everything else becomes config.
- `--metric-col` does the opposite: list metrics explicitly and every other
  column becomes config.
- JSON files with nested `{"config": {...}, "metrics": {...}}` need no mapping.
  A directory is imported recursively.

Re-importing the same file is safe: identical runs are skipped.

**Method strings that encode a parameter** (e.g. `adaptive_pnp_K5`) should be split
before import so `K` lands in the config, where the sweep and comparison pages
can use it. A five-line pandas script does it:

```python
import pandas as pd, re
df = pd.read_csv("metrics.csv")
m = df["method"].str.extract(r"^(?P<method>.+?)(?:_K(?P<K>\d+))?$")
df["method"], df["K"] = m["method"], m["K"].astype("float")
df.to_csv("metrics_split.csv", index=False)
```

**Baselines copied from a paper** go in as reported numbers so tables and
audits can tell them apart from your runs:

```bash
results-tracker import reported_baselines.csv -e deblur-main -p adaptive-pnp --source reported
```

They will appear with `n = 1`, no std, and the comparison audit will list the
datasets they do not cover.

---

## 6. Day-to-day

```bash
results-tracker ui                         # uses $RESULTS_TRACKER_DB
```

- **Overview** — is everything there? Failed runs are counted, not hidden. The
  "results at a glance" table is the paper's story in three lines. Counts come
  from SQL and records are cached per experiment, so it stays quick as the
  database grows; run ids link to Run detail; a dated **notes** log at the
  bottom keeps decisions next to the results (`results-tracker note add`).
- **Stages.** On Settings → Experiments, mark an experiment *paper*,
  *exploratory* or *superseded* (or `results-tracker experiment set NAME --stage
  superseded`). Superseded experiments disappear from every selector unless the
  sidebar checkbox shows them, so old attempts stop cluttering the menus without
  being deleted.
- **One selection for every page.** Project, experiment and the *Filter* in the
  sidebar are shared: pick `deblur-main` with `config.noise = 0.01` on the
  Comparison page and the Sweep, Visual, Run detail and Export pages show the
  same runs. The selection is mirrored in the URL
  (`?experiment=deblur-main&where=config.noise%3D0.01`), so a view can be
  bookmarked or pasted into notes and slides. The filter has the CLI's `--where`
  grammar (`config.K=[2,5]` means any of); the Export page prints the equivalent
  flags and records the filter in every table's provenance comment.
- **Comparison** — check the audit before trusting a table: `7/8 cells
  present, 1 missing` means a method was never run on a dataset, and "rows
  pooled over different datasets" means a pooled mean compares methods on
  different data. With several datasets the page keeps dataset as a row key by
  default for that reason.
- **Sweep** — the sensitivity table tells you whether the chosen default sits
  on a plateau or on a knife edge.
- **Ablation** — the effect-size verdict tells you which deltas the paper can
  claim; "within noise" means run more seeds or drop the claim.
- **Visual** — the panel-metrics table recomputes PSNR and SSIM from the
  images on disk and flags any panel that disagrees with its logged number by
  more than 0.05 dB or 0.005. A mismatch usually means a figure was regenerated
  from a different checkpoint than the table. Panels always show the same seed
  and instance for every method; if no common one exists the page says so.
  Float images are never rescaled per image: pass a data range shared by all
  panels.

Every page has a LaTeX expander and a paper-figure expander; the Export page has
the full set of options (captions, labels, std style, widths).

- **Studies** — every spec under the studies folder against the database: which
  jobs are done, running, failed or missing, an estimate of the compute left
  (pending runs × the median `runtime_s` of the study's completed runs), the
  paper assets a study `feeds`, a downloadable pending-only spec for another
  machine, and *Edit* / *Clone* into the form below. If the GUI's environment
  cannot import the repo, `results-tracker recipe export-knobs -i <module> -o
  studies/knobs.json` gives it the declarations it needs to plan.
- **Curves** — per-iteration diagnostics from each run's `diagnostics.json`
  (PSNR, step size, noise-level estimate ...): mean ± std over runs, one line
  per method arm or condition, individual runs on request, normalised to the
  first iteration if wanted. The recipe runner writes the file; a `log_run`
  user writes `{"curves": {"psnr": [...], ...}}` into the run's `artifacts_dir`.
- **Trade-off** — two metrics against each other (runtime vs PSNR), one series
  per method joined along a path key such as the iteration budget `K`;
  baselines and reported numbers get hollow markers.
- **Comparison → Per instance** — the spread behind every mean: box plots per
  method, the instance × method table, and instances ranked by the gain of your
  method over a baseline, to pick the image for a qualitative figure honestly.
- **Sweep → Selection** — the tuning rule as a table: the winning value per
  method (or per K), the grid it was chosen from, a flag when the winner sits at
  a grid edge, and *Write the selection into a comparison spec* to put the tuned
  value into the arms of a committed spec.
- **Settings** — metric direction / unit / format, method labels, baseline
  flags and **display order**, the project's primary metric, and **value maps**:
  derived, labelled groupings of a raw field (`config.kernel` 0-3 → isotropic,
  4-7 → anisotropic, 8-11 → motion becomes `derived.kernel_type`). Derived
  fields work wherever a config key does: row and column keys, filters, facets,
  `--rows derived.kernel_type`, `--where derived.kernel_type=motion`; columns
  follow the rule order.
- **Pooling experiments.** *Also include experiments* on the Comparison and
  Export pages unions several experiments of the project (the paper's main table
  with K across columns from `compare-K2`, `compare-K5`, `compare-K10`); group by
  `experiment` or the config key that differs. Pinned assets remember the pool.
- **Pin to paper.** Under every table and figure (Comparison, Sweep, Ablation,
  Visual, Export) a *Pin to paper* expander saves the current view — experiment,
  filter and rendering options — as a paper **asset** with a LaTeX label
  (`tab:main`, `fig:beta`) and a status (planned / draft / final). The **Paper**
  page lists the assets in manuscript order, says whether each export is still
  *current* or *stale* (runs were added or replaced since), lets you edit status,
  order, caption and notes, and exports them all. *Open in Export* restores an
  asset's view so you can change it and pin again.

---

## 7. Wire the paper to the database

Generate the paper's assets into the manuscript tree and `\input` them, so a
re-run of the experiments followed by one command refreshes the whole results
section.

The short way: pin every table and figure of the manuscript from the GUI (see
*Pin to paper* above), then

```bash
results-tracker export paper -p adaptive-pnp -o paper/assets
```

writes `tables/tab-main.tex`, `figures/fig-beta.pdf` + `.tex` snippet, `data/*.csv`,
`preamble.tex` and `MANIFEST.json` under `paper/assets/`, with stable names taken
from the labels. `results-tracker asset list` shows which assets are stale;
`asset set fig:beta --status final` and `asset rm` do the bookkeeping. The
Makefile then has one target:

```make
assets:
	results-tracker export paper -p $(P) -o assets
```

The long way, one export command per asset, still works and is what the pinned
assets run under the hood. `paper/Makefile`:

```make
RT = results-tracker
P  = adaptive-pnp
export RESULTS_TRACKER_DB = $(CURDIR)/results.db

assets: tables/main.tex tables/ablation.tex figures/lambda.pdf figures/ablation.pdf figures/visual.pdf

tables/main.tex:
	$(RT) export table -e deblur-main -p $(P) --label tab:main --std small -o $@

tables/ablation.tex:
	$(RT) export ablation-table -e ablation -p $(P) --label tab:ablation -o $@

figures/lambda.pdf:
	$(RT) export sweep-fig -e lambda-sweep -p $(P) --param lambda --metric psnr \
	      --xlabel '$$\lambda$$' --ylabel 'PSNR (dB)' --panel-label 'a. Regularization sweep' -o $@

figures/ablation.pdf:
	$(RT) export ablation-fig -e ablation -p $(P) --metric psnr -o $@

figures/visual.pdf:
	$(RT) export visual -e deblur-main -p $(P) -d CBSD68 --instance img18 --zoom -o $@ --tex

bundle:
	$(RT) export bundle -p $(P) -o build/paper_bundle.zip

.PHONY: assets bundle
```

In the manuscript:

```latex
\usepackage{booktabs,amssymb,graphicx}   % results-tracker export preamble
...
\input{tables/main.tex}
\begin{figure}[!t]\centering\includegraphics[width=\columnwidth]{figures/lambda}
\caption{...}\label{fig:lambda}\end{figure}
```

Every generated `.tex` starts with a provenance comment (database, experiment,
run count, timestamp). Never edit the numbers in these files; edit the caption
text in the export options or in the `\caption` you write around
`\includegraphics`.

`results-tracker export bundle` does all of the above in one shot and adds
`MANIFEST.json`, which records what was generated from which experiment and how
many runs — useful as a supplementary artifact.

---

## 8. Conventions worth agreeing on in the lab

- One project per paper; experiment names that read well in a caption
  (`deblur-main`, `lambda-sweep`, `ablation-init`).
- Method keys are stable identifiers (`pnp_admm`); display labels carry the
  citation and can change without touching runs.
- Seeds are logged as `seed`; per-image results as `instance`. Do not encode
  either in the method name.
- The full model of every ablation is tagged `base`.
- Reported numbers are imported with `--source reported`, never typed into a
  table.
- `results.db` lives with the paper and is committed with it.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| A metric is ranked the wrong way (best is bold on the worst value) | `results-tracker metric define <name> --lower` (or without `--lower`) |
| A swept parameter shows up as a metric column | The importer warns about integer columns with few distinct values; re-import with `--config-col <name>`, or log it inside `config` |
| "ambiguous base" on the Ablation page | No run is tagged `base` and two configs are equally common; tag the full model's runs or pick the base in the sidebar |
| Two runs of one method share a title in the Visual page | They differ only by seed/instance; pick one in the sidebar or use *Rows* |
| "Not shown: X — no artifacts_dir" on the Visual page | X was imported without images (e.g. reported); expected |
| Panel metrics warn about a PSNR gap | The image on disk is not the one the number was computed on; regenerate one of them |
| The comparison table overflows the column | Follow the width hint: `--std small`, `--font footnotesize`, drop a metric, or `--env 'table*'` |
| Wrong run logged | `results-tracker delete 42` (or `delete -e exp --status failed`), or the *Delete this run* expander on the Run detail page; both show what goes and ask first. Note the deletion in the commit message |
| GUI shows stale data | Click *Refresh* in the sidebar (results are cached per database modification time) |
| A page shows fewer runs than expected | Check the sidebar *Filter*: it applies on every page until cleared, and the caption says `n of m runs match` |
| GUI shows a different database than the one you launched | Another GUI was already running; the launch says so and prints its own port. The browser tab title names the database file, and the sidebar *Database* box switches it |
| Port 8501 busy | `results-tracker ui` picks the next free port and prints it |

---

## 10. Checklist before submission

- [ ] `results-tracker export table` audit shows every cell present, or the
      missing cells are explained in the text.
- [ ] Ablation verdicts: every claimed component is *clear*; nothing *within
      noise* is claimed.
- [ ] Sweep plateau matches the default you chose.
- [ ] Visual page panel metrics agree with the table for every figure in the
      paper.
- [ ] `results-tracker export bundle` regenerates all assets without manual
      edits; `MANIFEST.json` is in the supplementary material.
- [ ] `results.db` is committed with the manuscript at the submitted revision.
