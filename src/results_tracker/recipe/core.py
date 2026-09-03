"""The two abstractions a paper repo implements: a `Method` that reconstructs, a `Problem` that poses.

Deliberately at the altitude of "an estimator for an inverse problem": a method takes a measurement and
returns an estimate on some grid, and its behaviour is fully described by a configuration drawn from the
knobs it declares. Iterations, denoisers, step sizes, kernels, doses: none of that is here. A problem
owns the forward model, the data, the conditions it is posed under, and the metrics. Arrays are opaque
to this package (numpy, torch, anything with a shape works); only the problem's metric code reads them.
"""

from __future__ import annotations

import importlib
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Hashable, Iterable, Mapping, Optional, Sequence

from .knobs import Knob, KnobSpace


@dataclass
class Instance:
    """One realisation of a problem: what a method sees and what it is scored against.

    `reference` is None for real data without ground truth; `forward` is whatever operator or geometry
    object the method may use (or None for methods that do not need one); `condition` records the
    problem knobs that generated it."""

    name: str
    measurement: Any
    reference: Any = None
    forward: Any = None
    condition: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Estimate:
    """What a method returns: the estimate on the reference grid plus free-form diagnostics.

    Numeric scalars in `diagnostics` (iterations, a final step size) become metrics of the run; anything
    else (curves, arrays) is written to the run's artifacts folder as `diagnostics.json` and never parsed."""

    x: Any = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    message: str = ""

    @classmethod
    def failed(cls, message: str) -> "Estimate":
        return cls(None, {}, False, message)


class Method(ABC):
    """A reconstruction method. Subclass, set the class attributes, implement `reconstruct`.

    `setup` runs once per study for each distinct `setup_key` (by default once per method) and returns
    shared state such as loaded weights; `reconstruct` runs once per instance. A method that cannot be
    applied to a problem says so in `supports`, so a GUI can grey the combination out."""

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    citation: ClassVar[str] = ""  # BibTeX key; rendered as label~\cite{key} in LaTeX tables
    is_baseline: ClassVar[bool] = False
    knobs: ClassVar[Sequence[Knob]] = ()

    @classmethod
    def space(cls) -> KnobSpace:
        return KnobSpace(cls.knobs)

    @classmethod
    def display_label(cls) -> str:
        label = cls.label or cls.key
        return f"{label}~\\cite{{{cls.citation}}}" if cls.citation else label

    def supports(self, problem: "Problem") -> bool:
        return True

    def setup_key(self, config: Mapping[str, Any]) -> Hashable:
        """Which configs share one `setup` state; override to depend on e.g. a `denoiser` knob."""
        return self.key

    def setup(self, problem: "Problem", config: Mapping[str, Any]) -> Any:
        return None

    @abstractmethod
    def reconstruct(self, instance: Instance, config: Mapping[str, Any], state: Any) -> Estimate:
        """Estimate `instance.reference` from `instance.measurement` under a full `config`."""


class Problem(ABC):
    """An inverse problem: conditions it can be posed under, instances, metrics, and how to show an estimate.

    `conditions` are knobs (kernel, noise level, dose, undersampling ...); a study picks a grid over them
    and the runner records each instance's condition in the run config next to the method's knobs, so the
    two sets of names must not collide. `splits` name the data partitions (test, validation ...)."""

    key: ClassVar[str] = ""
    label: ClassVar[str] = ""
    conditions: ClassVar[Sequence[Knob]] = ()
    splits: ClassVar[Sequence[str]] = ("test",)
    #: metric name -> (unit, higher_is_better, fmt); seeded into the database once, never overwritten
    metric_definitions: ClassVar[Mapping[str, tuple[str, bool, str]]] = {}
    #: shared display range for saved images; None = each image's own min/max (avoid for comparisons)
    display_range: ClassVar[Optional[tuple[float, float]]] = (0.0, 1.0)

    @classmethod
    def condition_space(cls) -> KnobSpace:
        return KnobSpace(cls.conditions)

    def dataset_name(self, split: str) -> str:
        """The tracker `dataset` for a split: the problem key, suffixed for non-default splits."""
        return self.key if split == self.splits[0] else f"{self.key}-{split}"

    @abstractmethod
    def instances(self, condition: Mapping[str, Any], split: str, n: int, seed: int) -> Iterable[Instance]:
        """`n` instances of `split` under `condition`; `seed` fixes the random realisation (noise)."""

    @abstractmethod
    def metrics(self, estimate: Estimate, instance: Instance) -> dict[str, float]:
        """Scores of `estimate.x` against `instance.reference` (only called when both exist)."""

    def view(self, x: Any) -> Any:
        """A 2-D (or H×W×3) array to display for `x`: itself for images, the middle slice for volumes.

        A 3-D array whose last axis has 3 or 4 entries and whose first two axes are at least 8 long is
        taken to be a colour image; anything else 3-D is a volume. Override when that guess is wrong."""
        ndim = getattr(x, "ndim", None)
        shape = tuple(getattr(x, "shape", ()) or ())
        if ndim == 2 or (ndim == 3 and shape[-1] in (3, 4) and min(shape[:2]) >= 8):
            return x
        if ndim == 3:
            return x[shape[0] // 2]
        if ndim == 4:
            return x[shape[0] // 2, shape[1] // 2]
        return None

    def save_artifacts(self, run_dir: Path, instance: Instance, estimate: Estimate) -> None:
        """Write `reconstruction.png`, `ground_truth.png`, `measurement.png` (the names the Visual page knows).

        Uses `view` and the shared `display_range`; needs matplotlib, skips with one warning without it.
        Override for problems whose measurement lives on another grid (a sinogram, k-space)."""
        try:
            from matplotlib import image as mpimg
        except ImportError:  # pragma: no cover
            warnings.warn("matplotlib is not installed; recipe artifacts are not written", stacklevel=2)
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        lo, hi = self.display_range if self.display_range else (None, None)
        for name, arr in (("reconstruction", estimate.x), ("ground_truth", instance.reference),
                          ("measurement", instance.measurement)):
            img = self.view(arr) if arr is not None else None
            if img is None:
                continue
            if getattr(img, "ndim", 2) == 3:
                if lo is not None:
                    img = (img - lo) / (hi - lo)
                img = img.clip(0, 1)
                mpimg.imsave(str(run_dir / f"{name}.png"), img)
            else:
                mpimg.imsave(str(run_dir / f"{name}.png"), img, cmap="gray", vmin=lo, vmax=hi)


# --------------------------------------------------------------------------- registry

class Registry:
    """Maps keys to Method / Problem classes so study specs can name them; also resolves `module:Class`."""

    def __init__(self) -> None:
        self.methods: dict[str, type[Method]] = {}
        self.problems: dict[str, type[Problem]] = {}

    def method(self, cls: type[Method]) -> type[Method]:
        if not cls.key:
            raise ValueError(f"{cls.__name__} needs a `key`")
        self.methods[cls.key] = cls
        return cls

    def problem(self, cls: type[Problem]) -> type[Problem]:
        if not cls.key:
            raise ValueError(f"{cls.__name__} needs a `key`")
        self.problems[cls.key] = cls
        return cls

    def resolve_method(self, ref: str) -> type[Method]:
        return self._resolve(ref, self.methods, Method)

    def resolve_problem(self, ref: str) -> type[Problem]:
        return self._resolve(ref, self.problems, Problem)

    def _resolve(self, ref: str, table: Mapping[str, type], base: type) -> Any:
        if ref in table:
            return table[ref]
        cls = import_ref(ref) if (":" in ref or "." in ref) else None
        if cls is None or not (isinstance(cls, type) and issubclass(cls, base)):
            raise KeyError(f"{ref!r} is not a registered {base.__name__} key ({sorted(table)}) nor an importable "
                           f"`module:Class` reference to a {base.__name__} subclass")
        return cls


def import_ref(ref: str) -> Any:
    """`pkg.module:Name` or `pkg.module.Name` -> the object."""
    module_name, _, attr = ref.partition(":") if ":" in ref else ref.rpartition(".")
    if not module_name or not attr:
        raise KeyError(f"cannot parse {ref!r} as module:Name")
    return getattr(importlib.import_module(module_name), attr)


#: the default registry `run_study` consults; plugin modules register into it on import
registry = Registry()
