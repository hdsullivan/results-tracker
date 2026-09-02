"""Synthetic fixture data: one fake paper with a comparison, a sweep, and an ablation.

Deliberately includes the messy bits a real paper has:
- a baseline copied from another paper (`source="reported"`) that only covers one dataset,
  so the comparison audit reports a missing cell;
- one failed run in the sweep, so aggregation shows n=2 for that value;
- artifact folders with reconstruction / error-map images and a log for the Run detail gallery.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional

from sqlmodel import Session, select

from .api import _resolve_engine, define_method, define_metric, log_run
from .models import Project

PROJECT = "demo-paper"
METHODS = ["TV", "PnP-BM3D", "Ours"]
DATASETS = {"Set12": 12, "CBSD68": 68}
SEEDS = [0, 1, 2]
LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0]
BASE_CONFIG = {"denoiser": "drunet", "adaptive": True, "warm_start": True, "lambda": 0.1}


def demo_exists(db=None, engine=None) -> bool:
    engine = _resolve_engine(engine, db)
    with Session(engine) as s:
        return s.exec(select(Project).where(Project.name == PROJECT)).first() is not None


def _scene(dataset: str, seed: int, n: int = 96):
    """Deterministic 'ground truth' per dataset (smooth background, bright square, thin line, small defect).

    The scene depends on the dataset only; `seed` seeds the returned generator for the noise realisation,
    so different seeds share a ground truth like real repeated runs do."""
    import numpy as np

    rng = np.random.default_rng(hash(dataset) % (2**32))
    yy, xx = np.mgrid[0:n, 0:n] / n
    gt = 0.5 + 0.3 * np.sin(2 * np.pi * (xx * 1.3 + 0.2 * rng.random())) * np.cos(2 * np.pi * (yy * 0.9 + 0.1 * rng.random()))
    gt[n // 3:2 * n // 3, n // 3:2 * n // 3] += 0.25          # square (edge recovery)
    gt[int(0.8 * n), int(0.1 * n):int(0.9 * n)] = 1.0          # thin line
    gt[int(0.2 * n):int(0.2 * n) + 3, int(0.75 * n):int(0.75 * n) + 3] = 0.0  # small dark defect
    return np.clip(gt, 0, 1), np.random.default_rng(hash((dataset, seed)) % (2**32))


def _write_artifacts(root: Path, method: str, dataset: str, seed: int, psnr: float, rng: random.Random) -> str:
    """Reconstruction / error map / measurement / ground truth PNGs plus a log, all from one shared scene.

    The measurement and ground truth are identical across methods for a given (dataset, seed), so the
    Visual page can show a fair comparison with a shared error scale. Needs numpy + Pillow.
    """
    d = root / "main-comparison" / f"{method}_{dataset}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        import numpy as np
        from PIL import Image
    except ImportError:  # pragma: no cover
        (d / "log.txt").write_text(f"{method} on {dataset}, seed {seed}\nfinal psnr {psnr:.2f}\n")
        return str(d)

    def save(arr, name):
        Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype("uint8")).save(d / name)

    gt, nrng = _scene(dataset, seed)
    meas = gt + nrng.normal(0, 0.12, gt.shape)
    sigma = 10 ** (-psnr / 20)  # noise level implied by the logged psnr (data range 1)
    mrng = np.random.default_rng(hash((method, dataset, seed)) % (2**32))
    recon = gt + mrng.normal(0, sigma, gt.shape)
    if method == "TV":  # oversmoothing: blur the edges a little
        k = np.array([1, 2, 1]) / 4
        recon = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, recon)
        recon = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, recon)
    save(gt, "ground_truth.png")
    save(meas, "measurement.png")
    save(recon, "reconstruction.png")
    save(np.abs(recon - gt) / 0.3, "error_map.png")
    lines = [f"method={method} dataset={dataset} seed={seed}", "iter  psnr"]
    for it in range(0, 51, 10):
        lines.append(f"{it:4d}  {psnr - (50 - it) * 0.05:.2f}")
    (d / "train.log").write_text("\n".join(lines) + "\n")
    return str(d)


def seed_demo(db=None, engine=None, rng_seed: int = 0, artifacts_dir: Optional[str] = None) -> dict[str, int]:
    """Populate the database. Returns run counts per experiment. Call `demo_exists` first to avoid duplicates."""
    engine = _resolve_engine(engine, db)
    rng = random.Random(rng_seed)
    common = dict(engine=engine, git_commit=None, hostname="demo")
    define_metric("psnr", unit="dB", higher_is_better=True, fmt=".2f", engine=engine)
    define_metric("ssim", higher_is_better=True, fmt=".3f", engine=engine)
    define_metric("runtime_s", unit="s", higher_is_better=False, fmt=".1f", engine=engine)
    define_method("TV", label="TV [1]", is_baseline=True, engine=engine)
    define_method("PnP-BM3D", label="PnP-BM3D [2]", is_baseline=True, engine=engine)
    define_method("DPIR", label="DPIR [3] (reported)", is_baseline=True, engine=engine)
    define_method("Ours", label="Ours", engine=engine)
    art_root = Path(artifacts_dir).expanduser() if artifacts_dir else None

    counts = {"comparison": 0, "sweep": 0, "ablation": 0}
    base_psnr = {"TV": 27.5, "PnP-BM3D": 29.8, "Ours": 30.9}
    base_time = {"TV": 2.0, "PnP-BM3D": 14.0, "Ours": 6.5}

    # 1. Comparison: methods x datasets x seeds (+ artifacts)
    for m in METHODS:
        for ds in DATASETS:
            for s in SEEDS:
                psnr = base_psnr[m] + (0.4 if ds == "Set12" else -0.3) + rng.gauss(0, 0.15)
                art = _write_artifacts(art_root, m, ds, s, psnr, rng) if art_root else None
                log_run(
                    "main-comparison", project=PROJECT, experiment_type="comparison",
                    method=m, dataset=ds, seed=s, config={"lambda": 0.1, "iters": 50},
                    metrics={"psnr": psnr, "ssim": 0.75 + (psnr - 27) * 0.03 + rng.gauss(0, 0.005),
                             "runtime_s": base_time[m] * (1 + rng.uniform(-0.1, 0.1))},
                    artifacts_dir=art, **common,
                )
                counts["comparison"] += 1
    # a baseline copied from its paper: one dataset only, no seeds, no runtime
    log_run("main-comparison", project=PROJECT, method="DPIR", dataset="Set12", source="reported",
            config={}, metrics={"psnr": 30.4, "ssim": 0.85}, notes="Table 2 of [3]", **common)
    counts["comparison"] += 1

    # 2. Sweep: lambda for Ours, peak at 0.1; one run failed
    for lam in LAMBDAS:
        for s in SEEDS:
            failed = lam == 1.0 and s == 2
            psnr = 31.3 - 1.8 * (math.log10(lam) - math.log10(0.1)) ** 2 + rng.gauss(0, 0.1)
            log_run(
                "lambda-sweep", project=PROJECT, experiment_type="sweep",
                method="Ours", dataset="Set12", seed=s, config={**BASE_CONFIG, "lambda": lam},
                metrics={} if failed else {"psnr": psnr, "runtime_s": 6.5 + rng.gauss(0, 0.3)},
                status="failed" if failed else "completed",
                notes="diverged: NaN in iterate 37" if failed else "", **common,
            )
            counts["sweep"] += 1

    # 3. Ablation: base plus one-at-a-time removals
    variants = {
        "full": (BASE_CONFIG, 0.0, ["base"]),
        "no-adaptive": ({**BASE_CONFIG, "adaptive": False}, -0.9, []),
        "no-warm-start": ({**BASE_CONFIG, "warm_start": False}, -0.3, []),
        "dncnn": ({**BASE_CONFIG, "denoiser": "dncnn"}, -0.6, []),
    }
    for _, (cfg, drop, tags) in variants.items():
        for s in SEEDS:
            log_run(
                "ablation", project=PROJECT, experiment_type="ablation",
                method="Ours", dataset="Set12", seed=s, config=cfg, tags=tags,
                metrics={"psnr": 31.3 + drop + rng.gauss(0, 0.12), "runtime_s": 6.5 + rng.gauss(0, 0.3)},
                **common,
            )
            counts["ablation"] += 1
    return counts
