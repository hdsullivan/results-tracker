"""Per-iteration curves from a run's `diagnostics.json`, aggregated over runs.

The recipe runner writes one `diagnostics.json` per run (in the run's `artifacts_dir`, or under
`<database>.diagnostics/` when a study has no artifacts folder). Anything non-scalar a method returned in
`Estimate.diagnostics` lands there: a `curves` dict of name -> list (the adaptivePnP convention: psnr, ssim,
rho_k, sigma_hat_over_true, ... per iteration) or bare top-level lists (the toy's `step_sizes`). This module
reads them lazily (cached on file mtime) and averages them over runs, so the Curves page and the curves figure
never touch a database or an array library: plain lists, NaN-aware.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import aggregate as agg

Record = dict[str, Any]
DIAGNOSTICS = "diagnostics.json"


@lru_cache(maxsize=4096)
def _read(path: str, mtime: float) -> dict[str, list[float]]:
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    curves: dict[str, list[float]] = {}
    for name, values in (d.get("curves") or {}).items():
        if isinstance(values, list):
            curves[name] = [_num(v) for v in values]
    for name, values in d.items():
        if name != "curves" and isinstance(values, list) and values and all(isinstance(v, (int, float)) or v is None for v in values):
            curves.setdefault(name, [_num(v) for v in values])
    return curves


def _num(v: Any) -> float:
    return float("nan") if v is None or isinstance(v, bool) else float(v)


def record_curves(record: Mapping[str, Any]) -> dict[str, list[float]]:
    """name -> per-iteration values of one run; {} when it has no diagnostics file."""
    root = record.get("artifacts_dir")
    if not root:
        return {}
    path = os.path.join(os.path.expanduser(str(root)), DIAGNOSTICS)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    return _read(path, mtime)


def curve_names(records: Iterable[Record], limit: int = 50) -> list[str]:
    """Curve names available across (the first `limit`) completed runs, in first-seen order."""
    names: dict[str, None] = {}
    for i, r in enumerate(agg.completed(records)):
        if i >= limit:
            break
        for n in record_curves(r):
            names.setdefault(n, None)
    return list(names)


@dataclass
class CurveStat:
    """A curve averaged over runs: per-iteration mean, std (n-1) and count; iterations run 0..len-1."""

    mean: list[float]
    std: list[float]
    n: list[int]
    runs: int  # runs that contributed at least one value
    members: list[list[float]] = field(default_factory=list, repr=False)  # the individual curves (for a strip view)

    @property
    def x(self) -> list[int]:
        return list(range(len(self.mean)))

    def final(self) -> Optional[float]:
        return next((m for m in reversed(self.mean) if not math.isnan(m)), None)


def _pool(curves: Sequence[Sequence[float]]) -> CurveStat:
    length = max((len(c) for c in curves), default=0)
    mean, std, n = [], [], []
    for i in range(length):
        vals = [c[i] for c in curves if i < len(c) and not math.isnan(c[i])]
        k = len(vals)
        if k == 0:
            mean.append(float("nan")), std.append(0.0), n.append(0)
            continue
        m = sum(vals) / k
        mean.append(m)
        std.append(math.sqrt(sum((v - m) ** 2 for v in vals) / (k - 1)) if k > 1 else 0.0)
        n.append(k)
    return CurveStat(mean, std, n, len(curves), [list(c) for c in curves])


def curve_series(records: Iterable[Record], curve: str, group_by: Sequence[str] = ()) -> dict[tuple, CurveStat]:
    """`curve` of every completed run that has it, pooled per group (mean ± std at each iteration).

    Groups are tuples of `group_by` field values (`method`, `config.noise`, `derived.kernel_type`, ...); with
    no grouping the single key is `()`. Runs without the curve are skipped, so `runs` may be below the group size."""
    out: dict[tuple, CurveStat] = {}
    for g, rs in agg.group_records(agg.completed(records), list(group_by)).items():
        members = [c[curve] for c in (record_curves(r) for r in rs) if curve in c and c[curve]]
        if members:
            out[g] = _pool(members)
    return out


def normalise(stat: CurveStat, mode: str) -> CurveStat:
    """`mode`: "value" (as is), "delta" (minus the first iteration), "ratio" (divided by the first iteration)."""
    if mode == "value" or not stat.mean:
        return stat

    def transform(c: Sequence[float]) -> list[float]:
        first = next((v for v in c if not math.isnan(v)), None)
        if first is None:
            return list(c)
        if mode == "delta":
            return [v - first for v in c]
        return [v / first if first else float("nan") for v in c]

    return _pool([transform(c) for c in stat.members])
