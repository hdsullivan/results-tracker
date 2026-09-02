"""Synthetic fixture data: one fake paper with a comparison, a sweep, and an ablation."""

from __future__ import annotations

import math
import random

from .api import define_method, define_metric, log_run

PROJECT = "demo-paper"
METHODS = ["TV", "PnP-BM3D", "Ours"]
DATASETS = {"Set12": 12, "CBSD68": 68}
SEEDS = [0, 1, 2]
LAMBDAS = [0.01, 0.03, 0.1, 0.3, 1.0]
BASE_CONFIG = {"denoiser": "drunet", "adaptive": True, "warm_start": True, "lambda": 0.1}


def seed_demo(db=None, engine=None, rng_seed: int = 0) -> dict[str, int]:
    rng = random.Random(rng_seed)
    common = dict(db=db, engine=engine, git_commit=None, hostname="demo")
    define_metric("psnr", unit="dB", higher_is_better=True, fmt=".2f", db=db, engine=engine)
    define_metric("ssim", higher_is_better=True, fmt=".3f", db=db, engine=engine)
    define_metric("runtime_s", unit="s", higher_is_better=False, fmt=".1f", db=db, engine=engine)
    define_method("TV", label="TV [1]", is_baseline=True, db=db, engine=engine)
    define_method("PnP-BM3D", label="PnP-BM3D [2]", is_baseline=True, db=db, engine=engine)
    define_method("Ours", label="Ours", db=db, engine=engine)

    counts = {"comparison": 0, "sweep": 0, "ablation": 0}
    base_psnr = {"TV": 27.5, "PnP-BM3D": 29.8, "Ours": 30.9}
    base_time = {"TV": 2.0, "PnP-BM3D": 14.0, "Ours": 6.5}

    # 1. Comparison: methods x datasets x seeds
    for m in METHODS:
        for ds, offset in DATASETS.items():
            for s in SEEDS:
                psnr = base_psnr[m] + (0.4 if ds == "Set12" else -0.3) + rng.gauss(0, 0.15)
                log_run(
                    "main-comparison", project=PROJECT, experiment_type="comparison",
                    method=m, dataset=ds, seed=s, config={"lambda": 0.1, "iters": 50},
                    metrics={"psnr": psnr, "ssim": 0.75 + (psnr - 27) * 0.03 + rng.gauss(0, 0.005),
                             "runtime_s": base_time[m] * (1 + rng.uniform(-0.1, 0.1))},
                    **common,
                )
                counts["comparison"] += 1

    # 2. Sweep: lambda for Ours, peak at 0.1
    for lam in LAMBDAS:
        for s in SEEDS:
            psnr = 31.3 - 1.8 * (math.log10(lam) - math.log10(0.1)) ** 2 + rng.gauss(0, 0.1)
            log_run(
                "lambda-sweep", project=PROJECT, experiment_type="sweep",
                method="Ours", dataset="Set12", seed=s, config={**BASE_CONFIG, "lambda": lam},
                metrics={"psnr": psnr, "runtime_s": 6.5 + rng.gauss(0, 0.3)},
                **common,
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
