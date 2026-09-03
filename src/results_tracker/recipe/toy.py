"""A toy recipe: 2-D phantom deblurring with three numpy methods. The demo of the recipe layer.

- `ToyDeblurring`: 64×64 synthetic phantoms (smooth background, square, disk, fine stripes, a thin line)
  blurred by a Gaussian and corrupted by white noise; conditions `blur` (std, px) and `noise` (std).
  Metrics PSNR and SSIM computed exactly as the Visual page recomputes them from saved panels, so
  the panel-metrics audit agrees with the tables; every method clips its estimate to the known [0, 1] range.
- `Wiener` (baseline): closed-form Tikhonov deconvolution with an identity penalty.
- `TikhonovGD` (baseline): the same objective by fixed-step gradient descent from the measurement.
- `AdaptiveGD` ("Ours"): an edge-preserving Huber-TV objective minimised by gradient descent with a
  Barzilai–Borwein step, warm-started from Wiener. Its `adaptive` and `warm_start` switches and its
  `prior` choice (Huber-TV vs quadratic smoothness) are the ablation arms; `reg` is the sweep.

`toy_studies()` returns the three studies (comparison, sweep, ablation) `results-tracker recipe demo` runs.
Everything is deterministic in (instance index, seed).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np

from .core import Estimate, Instance, Method, Problem, registry
from .knobs import Knob
from .study import Ablation, Arm, Report, Study, Sweep, run_study

PROJECT = "toy-paper"
N = 64


# --------------------------------------------------------------------------- forward model and metrics

class GaussianBlur:
    """Circular Gaussian blur on an n×m grid via the FFT; symmetric, so it is its own adjoint."""

    def __init__(self, sigma: float, shape: tuple[int, int]):
        n, m = shape
        yy, xx = np.mgrid[0:n, 0:m]
        yy, xx = yy - n // 2, xx - m // 2
        kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel /= kernel.sum()
        self.otf = np.fft.fft2(np.fft.ifftshift(kernel))  # max |H| = 1: the operator norm is one
        fy = 2 * np.pi * np.fft.fftfreq(n)[:, None]
        fx = 2 * np.pi * np.fft.fftfreq(m)[None, :]
        self.laplacian_sym = (2 - 2 * np.cos(fy)) + (2 - 2 * np.cos(fx))  # |D|^2 of the discrete gradient

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.otf))

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        return np.real(np.fft.ifft2(np.fft.fft2(y) * np.conj(self.otf)))

    def wiener(self, y: np.ndarray, reg: float, smooth: bool = False) -> np.ndarray:
        """argmin ||Hx - y||² + reg·||x||² (or reg·||∇x||² with `smooth`), in closed form."""
        penalty = self.laplacian_sym if smooth else 1.0
        return np.real(np.fft.ifft2(np.conj(self.otf) * np.fft.fft2(y) / (np.abs(self.otf) ** 2 + reg * penalty)))

    def grad_penalty(self, x: np.ndarray) -> np.ndarray:
        """DᵀD x: the gradient of ½||∇x||²."""
        return np.real(np.fft.ifft2(np.fft.fft2(x) * self.laplacian_sym))


def _dx(x: np.ndarray) -> np.ndarray:
    return np.roll(x, -1, axis=1) - x


def _dy(x: np.ndarray) -> np.ndarray:
    return np.roll(x, -1, axis=0) - x


def huber_tv_grad(x: np.ndarray, delta: float) -> np.ndarray:
    """Gradient of Σ huber_δ(∇x): quadratic below δ, linear above, so edges survive and noise does not."""
    gx, gy = _dx(x), _dy(x)
    hx, hy = gx / np.maximum(np.abs(gx), delta), gy / np.maximum(np.abs(gy), delta)
    return (np.roll(hx, 1, axis=1) - hx) + (np.roll(hy, 1, axis=0) - hy)


def psnr(x: np.ndarray, ref: np.ndarray) -> float:
    """PSNR exactly as the Visual page recomputes it from a saved panel (10 px border crop, range 1)."""
    from ..export.visual import panel_psnr

    return panel_psnr(x, ref)


def ssim(x: np.ndarray, ref: np.ndarray) -> float:
    """SSIM exactly as the Visual page recomputes it (Gaussian window σ = 1.5, 10 px border crop)."""
    from ..export.visual import panel_ssim

    return panel_ssim(x, ref)


def phantom(index: int, split: str, n: int = N) -> np.ndarray:
    """Deterministic scene `index` of `split`: what over-smoothing destroys and what noise hides."""
    rng = np.random.default_rng([abs(hash(split)) % 2**31, index])
    yy, xx = np.mgrid[0:n, 0:n] / n
    img = 0.45 + 0.25 * np.sin(2 * np.pi * (1.5 * xx + rng.random())) * np.cos(2 * np.pi * (1.1 * yy + rng.random()))
    r0, c0 = rng.integers(n // 6, n // 2, size=2)
    img[r0:r0 + n // 4, c0:c0 + n // 4] += 0.3                                  # square: edges
    cy, cx = rng.uniform(0.3, 0.7, size=2) * n
    img[(yy * n - cy) ** 2 + (xx * n - cx) ** 2 < (n / 8) ** 2] = 0.15         # disk: a flat dark region
    r0, r1, c0, c1 = int(0.65 * n), int(0.95 * n), int(0.05 * n), int(0.35 * n)
    img[r0:r1, c0:c1] = 0.5 + 0.3 * np.sign(np.sin(2 * np.pi * xx[r0:r1, c0:c1] * n / 4))  # period-4 stripes
    row = int(rng.integers(n // 10, n - n // 10))
    img[row, n // 10:n - n // 10] = 1.0                                          # thin bright line
    return np.clip(img, 0.0, 1.0)


# --------------------------------------------------------------------------- the problem

@registry.problem
class ToyDeblurring(Problem):
    key = "toy-deblur"
    label = "Toy deblurring (synthetic phantoms)"
    conditions = (
        Knob("blur", "float", 1.5, bounds=(0.3, 5.0), doc="Gaussian blur std (px)"),
        Knob("noise", "float", 0.02, bounds=(0.0, 0.3), doc="white noise std"),
    )
    splits = ("test", "validation")
    metric_definitions = {"psnr": ("dB", True, ".2f"), "ssim": ("", True, ".3f")}

    def dataset_name(self, split: str) -> str:
        return "Phantoms" if split == "test" else "Phantoms-val"

    def instances(self, condition: Mapping[str, Any], split: str, n: int, seed: int) -> Iterable[Instance]:
        op = GaussianBlur(float(condition["blur"]), (N, N))
        for i in range(n):
            x = phantom(i, split)
            noise_rng = np.random.default_rng([seed, abs(hash(split)) % 2**31, i])
            y = op(x) + float(condition["noise"]) * noise_rng.standard_normal(x.shape)
            yield Instance(name=f"phantom_{i:02d}", measurement=y, reference=x, forward=op, condition=dict(condition))

    def metrics(self, estimate: Estimate, instance: Instance) -> dict[str, float]:
        return {"psnr": psnr(estimate.x, instance.reference), "ssim": ssim(estimate.x, instance.reference)}


# --------------------------------------------------------------------------- the methods

@registry.method
class Wiener(Method):
    key = "wiener"
    label = "Wiener"
    is_baseline = True
    knobs = (Knob("reg", "float", 0.01, bounds=(1e-5, 1.0), log=True, doc="Tikhonov weight"),)

    def reconstruct(self, instance: Instance, config: Mapping[str, Any], state: Any) -> Estimate:
        return Estimate(instance.forward.wiener(instance.measurement, config["reg"]).clip(0, 1), {"iterations": 1})


@registry.method
class TikhonovGD(Method):
    key = "gd"
    label = "Tikhonov-GD"
    is_baseline = True
    knobs = (
        Knob("reg", "float", 0.01, bounds=(1e-5, 1.0), log=True, doc="Tikhonov weight"),
        Knob("step", "float", 1.0, bounds=(0.01, 2.0), doc="fixed gradient step"),
        Knob("iters", "int", 30, bounds=(1, 1000)),
    )

    def reconstruct(self, instance: Instance, config: Mapping[str, Any], state: Any) -> Estimate:
        op, y = instance.forward, instance.measurement
        x = y.copy()
        for _ in range(config["iters"]):
            x = x - config["step"] * (op.adjoint(op(x) - y) + config["reg"] * x)
        return Estimate(x.clip(0, 1), {"iterations": config["iters"]})


@registry.method
class AdaptiveGD(Method):
    key = "adaptive-gd"
    label = "Ours"
    knobs = (
        Knob("reg", "float", 0.003, bounds=(1e-5, 1.0), log=True, doc="prior weight"),
        Knob("iters", "int", 30, bounds=(1, 1000)),
        Knob("prior", "choice", "huber_tv", choices=("huber_tv", "tikhonov"), doc="edge-preserving vs quadratic smoothness"),
        Knob("delta", "float", 0.05, bounds=(1e-3, 1.0), doc="Huber transition (gradient magnitude)"),
        Knob("adaptive", "bool", True, doc="Barzilai–Borwein step instead of the fixed 1/L step"),
        Knob("warm_start", "bool", True, doc="start from the Wiener estimate instead of the measurement"),
    )

    def reconstruct(self, instance: Instance, config: Mapping[str, Any], state: Any) -> Estimate:
        op, y, reg, delta = instance.forward, instance.measurement, config["reg"], config["delta"]
        if config["prior"] == "huber_tv":
            prior_grad = lambda z: huber_tv_grad(z, delta)
            lipschitz = 1.0 + reg * 8.0 / delta
        else:
            prior_grad = op.grad_penalty
            lipschitz = 1.0 + reg * 8.0
        grad = lambda z: op.adjoint(op(z) - y) + reg * prior_grad(z)
        x = op.wiener(y, 0.01) if config["warm_start"] else y.copy()
        g, step, steps = grad(x), 1.0 / lipschitz, []
        for _ in range(config["iters"]):
            x_new = x - step * g
            g_new = grad(x_new)
            if config["adaptive"]:
                s, d = x_new - x, g_new - g
                step = float(np.clip((s * s).sum() / max((s * d).sum(), 1e-12), 0.05, 50.0))
            x, g = x_new, g_new
            steps.append(step)
        return Estimate(x.clip(0, 1), {"iterations": config["iters"], "final_step": step, "step_sizes": steps})


# --------------------------------------------------------------------------- the demo paper

def toy_studies(artifacts_dir: Optional[str] = None) -> list[Study]:
    common = dict(project=PROJECT, problem=ToyDeblurring.key, imports=["results_tracker.recipe.toy"])
    return [
        Study(name="main-comparison", kind="comparison", methods=[Arm("wiener"), Arm("gd"), Arm("adaptive-gd")],
              conditions={"blur": [1.0, 2.0], "noise": [0.01, 0.05]}, n_instances=4, seeds=[0, 1],
              artifacts_dir=artifacts_dir,
              description="Three methods on the 2×2 blur × noise grid, four phantoms, two noise seeds.", **common),
        Study(name="reg-sweep", kind="sweep", methods=[Arm("adaptive-gd")],
              sweep=Sweep("reg", [0.0003, 0.001, 0.003, 0.01, 0.03]), conditions={"blur": [1.5], "noise": [0.05]},
              n_instances=3, seeds=[0, 1, 2], description="Prior weight of Ours: where is the plateau?", **common),
        Study(name="ablation", kind="ablation", methods=[Arm("adaptive-gd")],
              ablation=Ablation(arms=[{"adaptive": False}, {"warm_start": False}, {"prior": "tikhonov"}]),
              conditions={"blur": [1.0, 2.0], "noise": [0.05]}, n_instances=3, seeds=[0, 1],
              description="Each component of Ours switched off in turn, pooled over two blur widths.", **common),
    ]


def run_toy_demo(db=None, engine=None, artifacts_dir: Optional[str] = None, log=print) -> list[Report]:
    return [run_study(s, db=db, engine=engine, log=log) for s in toy_studies(artifacts_dir)]
