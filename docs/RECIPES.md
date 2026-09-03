# Recipes: methods, problems, and studies

`results_tracker.recipe` is the layer that lets a paper repo describe *what it can run* once, so
that comparisons, parameter sweeps and ablations become specs (JSON) instead of hand-written loops.
The tracker's tables, figures and GUI then work on the output unchanged.

```
Method   -- reconstructs: measurement -> estimate, behaviour fully described by declared knobs
Problem  -- poses: conditions (kernel, noise, dose ...) -> instances (measurement, reference, forward)
Study    -- a spec: problem × condition grid × method arms × (swept knob | base + single-knob arms)
run_study(study)  -- runs every job on every instance, scores it, saves artifacts, logs one run each
```

The core has no array-library dependency. Arrays are opaque to it: numpy, torch, and CuPy methods
coexist, and only the problem's own metric code reads them.

Try it:

```bash
results-tracker recipe demo --db toy.db --reset --artifacts toy_artifacts --write-specs specs
results-tracker ui --db toy.db
results-tracker recipe validate specs/ablation.json
results-tracker recipe knobs adaptive-gd -i results_tracker.recipe.toy
results-tracker export table -e main-comparison -p toy-paper --cols config.blur --where config.noise=0.05 --db toy.db
```

---

## 1. Implementing a method

```python
from results_tracker.recipe import Method, Knob, Estimate

class SnapPnP(Method):
    key = "snap_pnp"                 # stable id used in runs; never changes
    label = "SNAP-PnP"               # display label, editable in the database later
    citation = ""                    # BibTeX key -> rendered label~\cite{key} in tables
    is_baseline = False
    knobs = (
        Knob("algorithm", "choice", "pgm", choices=("pgm", "admm")),
        Knob("K", "int", 20, bounds=(1, 500), doc="iteration budget"),
        Knob("beta", "float", 0.5, bounds=(0.0, 1.0), doc="EMA weight on the step-size estimate"),
        Knob("reg_strength", "float", 1.0, bounds=(0.1, 10.0), log=True),
        Knob("freeze_sigma", "bool", False),
        Knob("denoiser", "choice", "drunet", choices=("drunet", "dncnn_blind")),
    )

    def setup_key(self, config):          # one denoiser per distinct value, shared across configs
        return config["denoiser"]

    def setup(self, problem, config):     # expensive, once per setup_key per study
        return load_denoiser(config["denoiser"])

    def reconstruct(self, instance, config, state):
        x_hat, info = pnp_recon(instance.measurement, instance.forward, state, **config)
        return Estimate(x_hat, {"iterations": info["iters"], "sigma_hat": info["sigma"], "curve": info["curve"]})
```

Rules:

- Everything that changes a result is a knob. A knob that is set but not declared is an error
  (`KnobSpace.resolve` raises), so a run's config always describes the run.
- `reconstruct` returns an `Estimate`; a crash or a non-finite estimate becomes a `failed` run
  with the message in `notes`, and the grid carries on.
- Numeric scalars in `diagnostics` become metrics of the run (`iterations`, `sigma_hat`). Anything
  else (curves, arrays) is written to the run's `diagnostics.json` and never parsed.
- `supports(problem)` lets a method decline a problem (a 2-D denoiser given a volume).

## 2. Implementing a problem

```python
from results_tracker.recipe import Problem, Knob, Instance

class Deblurring(Problem):
    key = "deblurring"
    conditions = (
        Knob("kernel", "int", 0, bounds=(0, 11), doc="DPIR benchmark kernel index"),
        Knob("noise", "float", 0.03, bounds=(0.0, 0.2)),
    )
    splits = ("test", "validation")
    metric_definitions = {"psnr": ("dB", True, ".2f"), "ssim": ("", True, ".3f"), "lpips": ("", False, ".3f")}
    display_range = (0.0, 1.0)

    def dataset_name(self, split):        # the tracker `dataset`: the image set, not the degradation
        return "CBSD68" if split == "test" else "DIV2K"

    def instances(self, condition, split, n, seed):
        operator = Convolution(kernels[condition["kernel"]])
        for i, x in enumerate(load_images(split, n)):
            y = operator(x) + condition["noise"] * randn(seed, i)
            yield Instance(name=f"image_{i:03d}", measurement=y, reference=x, forward=operator, condition=condition)

    def metrics(self, estimate, instance):
        return {"psnr": ..., "ssim": ..., "lpips": ...}
```

The condition goes into every run's `config` next to the method's knobs (names must not collide),
so tables can group by `config.kernel` and ablations pool over it. `view` / `save_artifacts`
default to writing `reconstruction.png`, `ground_truth.png`, `measurement.png` for the Visual
page (middle slice for volumes, shared display range); override `save_artifacts` when the
measurement lives on another grid (a sinogram, k-space).

Register both so specs can name them by key:

```python
from results_tracker.recipe import registry
registry.method(SnapPnP); registry.problem(Deblurring)   # or use them as decorators
```

## 3. Writing a study

```json
{
  "name": "deblurring-ema-beta",  "kind": "sweep",  "project": "adaptive-pnp",
  "problem": "deblurring",  "imports": ["adaptivepnp.recipes"],
  "conditions": {"kernel": [0, 1, 2, 3], "noise": [0.03, 0.05]},
  "methods": [{"method": "snap_pnp", "config": {"algorithm": "admm"}}],
  "sweep": {"knob": "beta", "values": [0.0, 0.25, 0.5, 0.75, 1.0]},
  "split": "test",  "n_instances": 20,  "seeds": [0, 1],
  "artifacts_dir": "results/tracker_artifacts"
}
```

| kind | extra field | what `expand` produces |
|---|---|---|
| `comparison` | — | one job per method arm |
| `sweep` | `sweep: {knob, values}` | one job per arm × value |
| `ablation` | `ablation: {base, arms}` | the full model (tagged `base`) plus one job per arm; an arm is `{knob: value}` (exactly one knob) or `{"label": ..., "set": {knob: value, ...}}` for a joint change, and must differ from the base |

Every job runs on every instance of every condition and seed; unset knobs and conditions take their
defaults. `results-tracker recipe validate spec.json` lists the jobs without running anything.
Tuning a baseline is a `sweep` on the validation split; the selection is a query afterwards
(`results-tracker sweep -e ... --param rho --metric psnr`).

Run it, in the repo's own environment:

```bash
results-tracker recipe run studies/deblurring-ema-beta.json --db paper/results.db
```

`run_study` also takes `problem_options` (passed to the problem's constructor: device, data root — machine
facts that stay out of the run config), `provenance` (a dict appended to every run's notes, e.g. a
dependency's commit), and `observers` (`StudyObserver` subclasses whose `on_run` fires after every logged
run, the hook a repo uses to stream its own result files). Without an `artifacts_dir`, curves and other
non-scalar diagnostics still land in `<database>.diagnostics/<study>/...` so no trajectory is lost.
`aggregate.select_best(records, "rho", "psnr", group_by=["config.scale"])` is the tuning rule: the swept
value with the best mean metric, per group.

Runs are logged as `experiment = study name`, `method = key`, `dataset = problem.dataset_name(split)`,
`instance = Instance.name`, `seed`, `config = condition ∪ knobs`. Re-running is a resume: settings
already logged as completed are skipped, failed ones are recomputed. The runner never deletes anything.

## 4. Why the spec is the boundary

The GUI's future "New study" page authors these specs from the declared knob spaces and shows the
command; the repo runs them where the data and GPUs are. The spec is committable and reproducible,
and the database it fills is the same one the GUI reads. Direct launching from the GUI is a
local-only convenience to add last.

## 5. The toy (`results_tracker.recipe.toy`)

`ToyDeblurring` blurs 64×64 synthetic phantoms; `Wiener` and `TikhonovGD` are the baselines, and
`AdaptiveGD` ("Ours") minimises an edge-preserving Huber-TV objective (`prior`, vs a quadratic
smoothness prior) with a Barzilai–Borwein step (`adaptive`) from a Wiener warm start (`warm_start`);
those three knobs are the ablation arms and `reg` is the sweep. `toy_studies()`
is the reference for spec shapes; `tests/test_recipe.py` is the contract.
