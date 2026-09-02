# Results Tracker — 5-minute pitch

**One line.** Every number in the paper traces back to a logged run, and the
table in the PDF was generated, not typed.

## The problem (30 s)

- Results live in notebooks, `.npy` files and Slack screenshots. By submission
  time nobody knows which run produced Table 2.
- Re-running a baseline means re-typing a table by hand, and typos in a
  results table are the most embarrassing kind of erratum.
- Sweeps and ablations are the same data as the main comparison, but they get
  plotted with a different script every time.

## The idea (30 s)

Everything is a **run**: a config dict, a metrics dict, and where the artifacts
live. Comparisons, sweeps and ablations are just three ways of *grouping* runs.
One SQLite file per lab. One line of Python to log. A GUI to look. One command
to export the paper table.

## Live demo (3 min)

Set up beforehand: `results-tracker demo --db demo.db --reset --artifacts demo_artifacts`
and `results-tracker ui --db demo.db` already open on the Overview page.

1. **Log from code** (20 s). Show the snippet:
   ```python
   log_run("lambda-sweep", project="adaptive-pnp", method="ours", dataset="Set12",
           seed=0, config=cfg, metrics={"psnr": 31.2, "ssim": 0.91})
   ```
   Git commit and hostname are captured for free. Numpy scalars are fine.

2. **Comparison** (40 s). Open the Comparison page. Three methods, two datasets,
   three seeds: mean ± std, best bold, second underlined. Switch rows to
   `method` + `dataset`. Point at the reported-only baseline: it exists on one
   dataset, and the tool knows.

3. **Sweep** (30 s). Open Sweep. λ on a log axis, shaded std band, best value
   ringed and named. One run diverged; its cell shows n = 2 instead of hiding it.

4. **Ablation** (30 s). Open Ablation. The tool diffs each config against the
   full model, so the ✓/✗ matrix and the deltas are computed, not labelled.
   "Removing the adaptive step costs 0.9 dB" is read straight off the chart.

5. **Export** (50 s). Open Export. The audit says 7/8 cells present, 1 missing.
   Copy the LaTeX. Switch to Ablation figure, download the PDF. Say: *this table
   compiled under IEEEtran two minutes after the last run finished, with a
   provenance comment naming the database and run count.*

6. **Visual** (30 s). Open Visual. Reference, measurement, three methods; the
   same 32×32 crop on every panel; error maps on one shared scale. The
   reported-only baseline is listed as "not shown" rather than quietly missing.
   Download the PNG and the provenance JSON that names every source file.

7. **Run detail** (20 s). Click one run: config, metrics, the reconstruction and
   error map from disk, and a config diff against any other run.

## Why it will get used (30 s)

- Zero infrastructure: one file, `pip install -e .`, works offline.
- Import the backlog: `results-tracker import old_results.csv -e main-comparison`.
- Exports follow the lab's IEEE table and plot conventions, so the output goes
  into the paper unchanged.
- 74 tests; the export tables compile under pdflatex.

## Likely questions

- *Why not W&B / MLflow?* They track training. This tracks *paper results*:
  the unit is a method × dataset × seed cell and the output is a booktabs table.
  Nothing to host, nothing to log in to, and the database is a file you can
  commit next to the manuscript.
- *Multi-user?* SQLite handles a lab sharing one file on a network drive for
  reads; for concurrent heavy logging, point each person at their own file and
  import. A Postgres backend is a one-line engine URL change if ever needed.
- *Images?* Stored as paths, never blobs. The gallery reads them from disk.
- *What if my metric direction is wrong?* `results-tracker metric define nrmse --lower`.
- *Custom LaTeX?* Every exporter takes captions, labels, method labels with
  `\cite`, std style, font size, and `table*`. Or use `--env none` and wrap
  it yourself.
