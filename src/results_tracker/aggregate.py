"""Pure-Python aggregation over run records (dicts from `run_records`).

No ORM, no pandas: easy to test and reusable by the GUI and the exporters.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional, Sequence

Record = dict[str, Any]
GroupKey = tuple[Hashable, ...]


# --------------------------------------------------------------------------- basics

@dataclass
class Stat:
    mean: float
    std: float  # sample std (n-1); 0.0 when n == 1
    n: int
    min: float
    max: float
    values: list[float] = field(default_factory=list, repr=False)

    def format(self, fmt: str = ".2f", with_std: bool = True) -> str:
        s = format(self.mean, fmt)
        if with_std and self.n > 1:
            s += f" ± {format(self.std, fmt)}"
        return s


def summarize(values: Iterable[Optional[float]]) -> Optional[Stat]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return Stat(mean=statistics.fmean(vals), std=std, n=len(vals), min=min(vals), max=max(vals), values=vals)


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """{'a': {'b': 1}} -> {'a.b': 1}. Lists are left as values."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def get_field(record: Record, key: str) -> Any:
    """'method' -> record['method']; 'config.lambda' -> record['config']['lambda'];
    'metrics.psnr' -> record['metrics']['psnr']."""
    if key.startswith("config."):
        return flatten(record.get("config", {})).get(key[len("config."):])
    if key.startswith("metrics."):
        return record.get("metrics", {}).get(key[len("metrics."):])
    return record.get(key)


def group_records(records: Iterable[Record], keys: Sequence[str]) -> dict[GroupKey, list[Record]]:
    groups: dict[GroupKey, list[Record]] = {}
    for r in records:
        k = tuple(get_field(r, key) for key in keys)
        groups.setdefault(k, []).append(r)
    return groups


def completed(records: Iterable[Record]) -> list[Record]:
    return [r for r in records if r.get("status", "completed") == "completed"]


def metric_names(records: Iterable[Record]) -> list[str]:
    seen: dict[str, None] = {}
    for r in records:
        for m in r.get("metrics", {}):
            seen.setdefault(m, None)
    return list(seen)


def aggregate_metrics(
    records: Iterable[Record],
    group_by: Sequence[str] = ("method",),
    metrics: Optional[Sequence[str]] = None,
    only_completed: bool = True,
) -> dict[GroupKey, dict[str, Optional[Stat]]]:
    """For each group (e.g. each method) summarise each metric over its runs."""
    recs = list(records)
    if only_completed:
        recs = completed(recs)
    names = list(metrics) if metrics else metric_names(recs)
    out: dict[GroupKey, dict[str, Optional[Stat]]] = {}
    for key, rs in group_records(recs, group_by).items():
        out[key] = {m: summarize(r["metrics"].get(m) for r in rs) for m in names}
    return out


# --------------------------------------------------------------------------- comparison

@dataclass
class ComparisonTable:
    group_by: tuple[str, ...]
    rows: list[GroupKey]  # one per group, in insertion order
    metrics: list[str]
    cells: dict[GroupKey, dict[str, Optional[Stat]]]
    rank: dict[str, dict[GroupKey, int]]  # metric -> row -> 1 (best), 2 (second), ...
    higher_is_better: dict[str, bool]

    def is_best(self, row: GroupKey, metric: str) -> bool:
        return self.rank.get(metric, {}).get(row) == 1

    def is_second(self, row: GroupKey, metric: str) -> bool:
        return self.rank.get(metric, {}).get(row) == 2

    def row_label(self, row: GroupKey) -> str:
        return " / ".join(str(v) for v in row)


def comparison_table(
    records: Iterable[Record],
    group_by: Sequence[str] = ("method",),
    metrics: Optional[Sequence[str]] = None,
    higher_is_better: Optional[Mapping[str, bool]] = None,
    row_order: Optional[Callable[[GroupKey], Any]] = None,
) -> ComparisonTable:
    """Methods (rows) x metrics (columns), mean ± std over seeds/instances, with ranks.

    `higher_is_better` maps metric -> bool; unknown metrics default to True.
    """
    agg = aggregate_metrics(records, group_by, metrics)
    names = list(metrics) if metrics else sorted({m for cells in agg.values() for m in cells})
    hib = {m: (higher_is_better or {}).get(m, True) for m in names}
    rows = list(agg)
    if row_order is not None:
        rows.sort(key=row_order)
    rank: dict[str, dict[GroupKey, int]] = {}
    for m in names:
        scored = [(agg[r][m].mean, r) for r in rows if agg[r].get(m) is not None]
        scored.sort(key=lambda t: t[0], reverse=hib[m])
        rank[m] = {}
        last, pos = None, 0
        for i, (val, r) in enumerate(scored):
            if val != last:  # ties share a rank
                pos = i + 1
                last = val
            rank[m][r] = pos
    return ComparisonTable(tuple(group_by), rows, names, agg, rank, hib)


# --------------------------------------------------------------------------- sweeps

def sweep_series(
    records: Iterable[Record],
    param: str,
    metric: str,
    group_by: Sequence[str] = (),
    only_completed: bool = True,
) -> dict[GroupKey, list[tuple[Any, Stat]]]:
    """metric vs one swept parameter, aggregated over seeds, one series per group.

    `param` is a key inside `config` ('lambda' or nested 'solver.step').
    Returns {group: [(x, Stat), ...]} sorted by x.
    """
    recs = list(records)
    if only_completed:
        recs = completed(recs)
    pkey = f"config.{param}"
    out: dict[GroupKey, list[tuple[Any, Stat]]] = {}
    for g, rs in group_records(recs, list(group_by)).items():
        series = []
        for (x,), xs in group_records(rs, [pkey]).items():
            if x is None:
                continue
            st = summarize(r["metrics"].get(metric) for r in xs)
            if st is not None:
                series.append((x, st))
        series.sort(key=lambda t: (isinstance(t[0], str), t[0]))
        out[g] = series
    return out


def best_sweep_value(series: Sequence[tuple[Any, Stat]], higher_is_better: bool = True) -> Optional[Any]:
    if not series:
        return None
    pick = max if higher_is_better else min
    return pick(series, key=lambda t: t[1].mean)[0]


# --------------------------------------------------------------------------- ablations

def config_diff(base: Mapping[str, Any], other: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Flattened key -> (base_value, other_value) for every key that differs.
    Keys missing on one side show up with None on that side."""
    fb, fo = flatten(base), flatten(other)
    out: dict[str, tuple[Any, Any]] = {}
    for k in sorted(set(fb) | set(fo)):
        if fb.get(k) != fo.get(k):
            out[k] = (fb.get(k), fo.get(k))
    return out


def describe_diff(diff: Mapping[str, tuple[Any, Any]]) -> str:
    if not diff:
        return "full model"
    parts = []
    for k, (b, o) in diff.items():
        if b is True and o is False:
            parts.append(f"w/o {k}")
        elif b is None:
            parts.append(f"+ {k}={o}")
        elif o is None:
            parts.append(f"- {k}")
        else:
            parts.append(f"{k}: {b}→{o}")
    return "; ".join(parts)


@dataclass
class AblationRow:
    label: str
    diff: dict[str, tuple[Any, Any]]
    n: int
    stats: dict[str, Optional[Stat]]
    delta: dict[str, Optional[float]]  # variant mean - base mean
    is_base: bool = False


def _diff_signature(diff: Mapping[str, tuple[Any, Any]]) -> tuple:
    return tuple((k, repr(v[1])) for k, v in sorted(diff.items()))


def ablation_table(
    records: Iterable[Record],
    base_config: Optional[Mapping[str, Any]] = None,
    base_run_id: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
    only_completed: bool = True,
) -> list[AblationRow]:
    """Group variants by how their config differs from the base; report metric deltas.

    Base is chosen by (in order): `base_config`, `base_run_id`, a record tagged
    'base', else the config shared by the most runs.
    """
    recs = list(records)
    if only_completed:
        recs = completed(recs)
    if not recs:
        return []

    if base_config is None:
        cand = None
        if base_run_id is not None:
            cand = next((r for r in recs if r.get("run_id") == base_run_id), None)
        if cand is None:
            cand = next((r for r in recs if "base" in (r.get("tags") or [])), None)
        if cand is not None:
            base_config = cand["config"]
        else:
            counts: dict[tuple, tuple[int, Mapping]] = {}
            for r in recs:
                sig = tuple(sorted(flatten(r["config"]).items(), key=lambda kv: kv[0]))
                sig = tuple((k, repr(v)) for k, v in sig)
                n, _ = counts.get(sig, (0, r["config"]))
                counts[sig] = (n + 1, r["config"])
            base_config = max(counts.values(), key=lambda t: t[0])[1]

    names = list(metrics) if metrics else metric_names(recs)
    groups: dict[tuple, tuple[dict, list[Record]]] = {}
    for r in recs:
        d = config_diff(base_config, r["config"])
        sig = _diff_signature(d)
        groups.setdefault(sig, (d, []))[1].append(r)

    base_stats = {m: summarize(r["metrics"].get(m) for r in groups.get((), ({}, []))[1]) for m in names}

    rows: list[AblationRow] = []
    for sig, (d, rs) in groups.items():
        stats = {m: summarize(r["metrics"].get(m) for r in rs) for m in names}
        delta = {}
        for m in names:
            b, v = base_stats.get(m), stats.get(m)
            delta[m] = (v.mean - b.mean) if (b is not None and v is not None) else None
        rows.append(AblationRow(describe_diff(d), d, len(rs), stats, delta, is_base=(sig == ())))
    rows.sort(key=lambda r: (not r.is_base, len(r.diff), r.label))
    return rows
