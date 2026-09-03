"""Pure-Python aggregation over run records (dicts from `run_records`).

No ORM, no pandas: easy to test and reusable by the GUI and the exporters.
"""

from __future__ import annotations

import json
import re
from numbers import Number

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


def parse_where(items: Iterable[str]) -> dict[str, Any]:
    """`["config.K=5", "method=dpir"]` -> `{"config.K": 5, "method": "dpir"}`.

    Values are parsed as JSON when possible (5, 0.03, true, "x"), otherwise kept as strings."""
    out: dict[str, Any] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"expected field=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def same_value(a: Any, b: Any) -> bool:
    """Loose equality for filters: 5 matches 5.0 and '5'; True matches 'true'."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b or str(a).lower() == str(b).lower()
    if isinstance(a, Number) and isinstance(b, Number):
        return a == b  # 5 matches 5.0
    return a == b or str(a) == str(b)


def filter_records(records: Iterable[Record], where: Mapping[str, Any]) -> list[Record]:
    """Runs whose fields equal every value in `where`: method, dataset, seed, instance, status, config.<key>.

    Numbers compare numerically (5 matches 5.0); everything else by equality or by string form, so a
    value typed on a command line matches the stored one. A list value keeps runs matching *any* of its
    elements (`config.K=[2,5]`), which is how the GUI filter and a multi-valued `--where` are expressed."""
    recs = list(records)
    for k, v in where.items():
        wanted = list(v) if isinstance(v, (list, tuple, set)) else [v]
        recs = [r for r in recs if any(same_value(get_field(r, k), w) for w in wanted)]
    return recs


def group_records(records: Iterable[Record], keys: Sequence[str]) -> dict[GroupKey, list[Record]]:
    groups: dict[GroupKey, list[Record]] = {}
    for r in records:
        k = tuple(get_field(r, key) for key in keys)
        groups.setdefault(k, []).append(r)
    return groups


_CITE = re.compile(r"\s*~?\\cite[tp]?\{[^}]*\}")


def plain_label(label: Optional[str]) -> Optional[str]:
    """A method label without its LaTeX citation: `DPIR~\cite{zhang2021}` -> `DPIR`.

    Labels may carry a `\cite{}` so LaTeX tables cite the baseline; figures, panel titles and the GUI
    cannot render it and show the bare name instead."""
    if label is None:
        return None
    return _CITE.sub("", label).strip() or label


def method_labels(records: Iterable[Record], latex: bool = False) -> dict[Any, str]:
    """method name -> display label (from Method.label), for table row labels.

    With `latex=False` (figures, HTML, panel titles) citations are stripped; LaTeX table exports pass
    `latex=True` to keep `label~\cite{key}` intact."""
    out: dict[Any, str] = {}
    for r in records:
        if r.get("method") is not None and r.get("method_label"):
            label = r["method_label"] if latex else plain_label(r["method_label"])
            out.setdefault(r["method"], label)
    return out


def select_runs(
    records: Iterable[Record],
    dataset: Optional[Any] = None,
    seed: Optional[Any] = None,
    instance: Optional[Any] = None,
    methods: Optional[Sequence[Any]] = None,
) -> list[Record]:
    """One completed run per method for a visual comparison: baselines first, proposed last,
    or in the order of `methods` when given.

    Fairness: when `seed` (or `instance`) is not given and several exist, the smallest value shared by
    *every* shown method is used, so all panels show the same realisation. If no common value exists,
    each method falls back to its own smallest value; `selection_notes` reports that."""
    recs = completed(records)
    if dataset is not None:
        recs = [r for r in recs if r.get("dataset") == dataset]
    if seed is not None:
        recs = [r for r in recs if r.get("seed") == seed]
    if instance is not None:
        recs = [r for r in recs if r.get("instance") == instance]
    if methods:
        recs = [r for r in recs if r.get("method") in set(methods)]
    for key, given in (("seed", seed), ("instance", instance)):
        if given is not None:
            continue
        per_method = {}
        for r in recs:
            per_method.setdefault(r.get("method"), set()).add(r.get(key))
        if per_method and any(len(v) > 1 for v in per_method.values()):
            common = set.intersection(*per_method.values()) - {None}
            if common:
                pick = min(common, key=_sort_key)
                recs = [r for r in recs if r.get(key) == pick]
    recs.sort(key=lambda r: (r.get("seed") is None, _sort_key(r.get("seed")) if r.get("seed") is not None else (),
                             r.get("instance") is None, _sort_key(r.get("instance")) if r.get("instance") is not None else (),
                             r.get("run_id") or 0))
    by_method: dict[Any, Record] = {}
    for r in recs:
        by_method.setdefault(r.get("method"), r)
    if methods:
        return [by_method[m] for m in methods if m in by_method]
    ordered = sorted(by_method.values(), key=lambda r: (not r.get("method_is_baseline", False)))
    return ordered


def selection_notes(chosen: Sequence[Record]) -> list[str]:
    """Warn when the runs shown side by side are not the same realisation (different seeds or instances)."""
    notes = []
    for key in ("seed", "instance"):
        vals = {r.get(key) for r in chosen if r.get(key) is not None}
        if len(vals) > 1:
            who = ", ".join(f"{plain_label(r.get('method_label')) or r.get('method')}: {key} {r.get(key)}" for r in chosen if r.get(key) is not None)
            notes.append(f"panels show different {key}s ({who}); pick one {key} for a fair comparison")
    return notes


def omitted_methods(records: Iterable[Record], chosen: Sequence[Record], dataset: Optional[Any] = None) -> dict[Any, str]:
    """Methods that exist in the experiment (for this dataset) but are not in the visual comparison, with why.
    Surfacing these keeps a qualitative figure honest about who was left out."""
    recs = completed(records)
    if dataset is not None:
        recs = [r for r in recs if r.get("dataset") == dataset]
    shown = {r.get("method") for r in chosen}
    out: dict[Any, str] = {}
    for m in {r.get("method") for r in recs} - shown:
        runs = [r for r in recs if r.get("method") == m]
        label = plain_label(runs[0].get("method_label")) or str(m)
        if not any(r.get("artifacts_dir") for r in runs):
            out[label] = "no artifacts_dir (e.g. numbers reported from a paper)"
        else:
            out[label] = "no completed run for the selected seed / instance"
    return out


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


# --------------------------------------------------------------------------- ranking

def rank_values(scored: Sequence[tuple[float, Hashable]], higher_is_better: bool = True) -> dict[Hashable, int]:
    """1 = best. Ties share a rank (1, 1, 3). Uses the unrounded values passed in."""
    ordered = sorted(scored, key=lambda t: t[0], reverse=higher_is_better)
    out: dict[Hashable, int] = {}
    last, pos = None, 0
    for i, (val, key) in enumerate(ordered):
        if val != last:
            pos = i + 1
            last = val
        out[key] = pos
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
        rank[m] = rank_values([(agg[r][m].mean, r) for r in rows if agg[r].get(m) is not None], hib[m])
    return ComparisonTable(tuple(group_by), rows, names, agg, rank, hib)


# --------------------------------------------------------------------------- pivot (paper layout)

@dataclass
class PivotTable:
    """rows = one key (methods), column groups = another key (datasets), metrics under each group.

    Ranks are computed within each (column, metric) from unrounded means.
    """

    row_key: str
    col_key: Optional[str]
    rows: list[Any]
    cols: list[Any]  # [None] when col_key is None
    metrics: list[str]
    cells: dict[tuple[Any, Any], dict[str, Optional[Stat]]]
    rank: dict[tuple[Any, str], dict[Any, int]]
    higher_is_better: dict[str, bool]

    def stat(self, row: Any, col: Any, metric: str) -> Optional[Stat]:
        return self.cells.get((row, col), {}).get(metric)

    def is_best(self, row: Any, col: Any, metric: str) -> bool:
        return self.rank.get((col, metric), {}).get(row) == 1

    def is_second(self, row: Any, col: Any, metric: str) -> bool:
        return self.rank.get((col, metric), {}).get(row) == 2

    def n_values(self) -> set[int]:
        return {st.n for cell in self.cells.values() for st in cell.values() if st is not None}


def pivot_table(
    records: Iterable[Record],
    row_key: str = "method",
    col_key: Optional[str] = "dataset",
    metrics: Optional[Sequence[str]] = None,
    higher_is_better: Optional[Mapping[str, bool]] = None,
    row_order: Optional[Sequence[Any]] = None,
    col_order: Optional[Sequence[Any]] = None,
) -> PivotTable:
    recs = completed(records)
    keys = [row_key] + ([col_key] if col_key else [])
    agg = aggregate_metrics(recs, keys, metrics)
    names = list(metrics) if metrics else metric_names(recs)
    hib = {m: (higher_is_better or {}).get(m, True) for m in names}
    rows_seen: list[Any] = []
    cols_seen: list[Any] = []
    cells: dict[tuple[Any, Any], dict[str, Optional[Stat]]] = {}
    for key, stats in agg.items():
        r = key[0]
        c = key[1] if col_key else None
        if r not in rows_seen:
            rows_seen.append(r)
        if c not in cols_seen:
            cols_seen.append(c)
        cells[(r, c)] = stats
    rows = [r for r in row_order if r in rows_seen] + [r for r in rows_seen if not row_order or r not in row_order] if row_order else rows_seen
    cols = [c for c in col_order if c in cols_seen] + [c for c in cols_seen if not col_order or c not in col_order] if col_order else cols_seen
    if col_key:
        cols = sorted(cols, key=lambda c: cols.index(c)) if col_order else sorted(cols, key=_sort_key)
    rank: dict[tuple[Any, str], dict[Any, int]] = {}
    for c in cols:
        for m in names:
            scored = [(cells[(r, c)][m].mean, r) for r in rows if (r, c) in cells and cells[(r, c)].get(m) is not None]
            rank[(c, m)] = rank_values(scored, hib[m])
    return PivotTable(row_key, col_key, rows, cols, names, cells, rank, hib)


# --------------------------------------------------------------------------- audit

@dataclass
class GridAudit:
    keys: tuple[str, ...]
    expected: int
    present: int
    missing: list[tuple]  # combinations with no completed run
    failed: dict[tuple, int]  # combinations with failed runs (count)
    n_per_cell: dict[tuple, int]
    coverage: list[str] = field(default_factory=list)  # rows pooled over different hidden-key sets

    @property
    def uneven(self) -> bool:
        return len(set(self.n_per_cell.values())) > 1

    @property
    def ok(self) -> bool:
        return not self.missing and not self.failed and not self.coverage

    def summary(self) -> str:
        parts = [f"{self.present}/{self.expected} cells present"]
        if self.missing:
            parts.append(f"{len(self.missing)} missing")
        if self.failed:
            parts.append(f"{sum(self.failed.values())} failed run(s) in {len(self.failed)} cell(s)")
        if self.uneven:
            ns = sorted(set(self.n_per_cell.values()))
            parts.append(f"n varies ({ns[0]}–{ns[-1]})")
        if self.coverage:
            parts.append("rows pooled over different " + " / ".join(c.split(":")[0] for c in self.coverage))
        return ", ".join(parts)


def coverage_audit(records: Iterable[Record], keys: Sequence[str], hidden: Sequence[str] = ("dataset", "instance")) -> list[str]:
    """For each hidden key with several values that is *not* in `keys`, check every row covers the same set.

    A row pooled over fewer datasets (e.g. a baseline reported on one dataset) is not comparable to rows
    pooled over all of them; the messages say who covers what."""
    recs = completed(records)
    out: list[str] = []
    for h in hidden:
        if h in keys:
            continue
        all_vals = {get_field(r, h) for r in recs if get_field(r, h) is not None}
        if len(all_vals) < 2:
            continue
        per_row: dict[tuple, set] = {}
        for r in recs:
            v = get_field(r, h)
            if v is not None:
                per_row.setdefault(tuple(get_field(r, k) for k in keys), set()).add(v)
        if len({frozenset(v) for v in per_row.values()}) > 1:
            who = "; ".join(f"{' / '.join(map(str, row))}: {', '.join(sorted(map(str, vals)))}" for row, vals in per_row.items())
            out.append(f"{h}s: {who} — pooled means are not comparable across rows; group by {h} or restrict to common {h}s")
    return out


def audit_grid(records: Iterable[Record], keys: Sequence[str]) -> GridAudit:
    """Check that every combination of observed key values has completed runs.

    E.g. keys=("method", "dataset"): every method should have been run on every dataset.
    """
    from itertools import product

    recs = list(records)
    values = [sorted({get_field(r, k) for r in recs if get_field(r, k) is not None}, key=_sort_key) for k in keys]
    n_per: dict[tuple, int] = {}
    failed: dict[tuple, int] = {}
    for r in recs:
        combo = tuple(get_field(r, k) for k in keys)
        if any(v is None for v in combo):
            continue
        if r.get("status", "completed") == "completed":
            n_per[combo] = n_per.get(combo, 0) + 1
        elif r.get("status") == "failed":
            failed[combo] = failed.get(combo, 0) + 1
    expected = list(product(*values)) if values and all(values) else []
    missing = [c for c in expected if c not in n_per]
    return GridAudit(tuple(keys), len(expected), len(n_per), missing, failed, n_per, coverage_audit(recs, keys))


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


def select_best(
    records: Iterable[Record],
    param: str,
    metric: str,
    group_by: Sequence[str] = (),
    higher_is_better: bool = True,
) -> dict[GroupKey, Any]:
    """The swept value with the best mean metric, per group: {group: value}.

    The tuning rule a fixed-step baseline needs ("the rho that maximises mean PSNR pooled over kernels and
    noise levels, per scale factor") is `select_best(recs, "rho", "psnr", group_by=["config.scale"])`."""
    return {
        g: agg_best
        for g, series in sweep_series(records, param, metric, group_by=group_by).items()
        if (agg_best := best_sweep_value(series, higher_is_better)) is not None
    }


def _sort_key(x: Any) -> tuple:
    return (isinstance(x, str), x)


@dataclass
class Plateau:
    """Which swept values are 'as good as the best' within a tolerance."""

    best: Any
    best_stat: Stat
    tolerance: float
    members: list[Any]  # swept values whose mean is within tolerance of the best mean (sorted)
    worst: Any
    worst_stat: Stat

    @property
    def span(self) -> tuple[Any, Any]:
        return (self.members[0], self.members[-1])

    @property
    def drop(self) -> float:
        """Best mean minus worst mean, as a positive 'cost of a bad choice'."""
        return abs(self.best_stat.mean - self.worst_stat.mean)


def sweep_plateau(series: Sequence[tuple[Any, Stat]], higher_is_better: bool = True,
                  tolerance: Optional[float] = None) -> Optional[Plateau]:
    """Sensitivity summary of one sweep line. Default tolerance = std of the best point (0 -> 1% of the range)."""
    if not series:
        return None
    pick = max if higher_is_better else min
    best_x, best_st = pick(series, key=lambda t: t[1].mean)
    worst_x, worst_st = (min if higher_is_better else max)(series, key=lambda t: t[1].mean)
    if tolerance is None:
        rng = abs(best_st.mean - worst_st.mean)
        tolerance = best_st.std if best_st.std > 0 else 0.01 * (rng if rng > 0 else abs(best_st.mean) or 1.0)
    members = [x for x, st in series if (best_st.mean - st.mean if higher_is_better else st.mean - best_st.mean) <= tolerance]
    members.sort(key=_sort_key)
    return Plateau(best_x, best_st, tolerance, members, worst_x, worst_st)


@dataclass
class SweepGrid:
    xs: list[Any]
    ys: list[Any]
    cells: dict[tuple[Any, Any], Stat]  # (x, y) -> Stat

    def matrix(self) -> list[list[Optional[float]]]:
        """rows = ys, cols = xs, mean or None."""
        return [[(self.cells[(x, y)].mean if (x, y) in self.cells else None) for x in self.xs] for y in self.ys]

    def best(self, higher_is_better: bool = True) -> Optional[tuple[Any, Any]]:
        if not self.cells:
            return None
        pick = max if higher_is_better else min
        return pick(self.cells, key=lambda k: self.cells[k].mean)


def sweep_grid(
    records: Iterable[Record], param_x: str, param_y: str, metric: str, only_completed: bool = True
) -> SweepGrid:
    """Two swept parameters -> one aggregated cell per (x, y). Runs missing either parameter are skipped."""
    recs = list(records)
    if only_completed:
        recs = completed(recs)
    cells: dict[tuple[Any, Any], Stat] = {}
    for (x, y), rs in group_records(recs, [f"config.{param_x}", f"config.{param_y}"]).items():
        if x is None or y is None:
            continue
        st = summarize(r["metrics"].get(metric) for r in rs)
        if st is not None:
            cells[(x, y)] = st
    xs = sorted({k[0] for k in cells}, key=_sort_key)
    ys = sorted({k[1] for k in cells}, key=_sort_key)
    return SweepGrid(xs, ys, cells)


def varying_config_keys(records: Iterable[Record]) -> list[str]:
    """Flattened config keys that take more than one value across the records."""
    seen: dict[str, set] = {}
    for r in records:
        for k, v in flatten(r.get("config", {})).items():
            seen.setdefault(k, set()).add(repr(v))
    return [k for k, vals in seen.items() if len(vals) > 1]


def condition_keys(records: Iterable[Record]) -> list[str]:
    """Config keys that vary *among the runs tagged 'base'*.

    An ablation repeated over a grid of conditions (blur kernel, noise level, scale factor, image size ...)
    logs those conditions in `config` alongside the settings it varies. The conditions are not settings:
    the full model was run at every one of them, so they are exactly the keys that differ between base
    runs. `ablation_table` leaves them out of the diff so every arm pools over the same grid. With fewer
    than two base runs nothing can vary and the list is empty."""
    base = [r for r in records if "base" in (r.get("tags") or [])]
    return varying_config_keys(base) if len(base) > 1 else []


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


class AmbiguousBaseError(ValueError):
    """No run is tagged 'base', no base was given, and the most common config is not unique."""


@dataclass
class AblationRow:
    label: str
    diff: dict[str, tuple[Any, Any]]
    n: int
    stats: dict[str, Optional[Stat]]
    delta: dict[str, Optional[float]]  # variant mean - base mean
    is_base: bool = False
    base_mean: dict[str, Optional[float]] = field(default_factory=dict)

    def rel_delta(self, metric: str) -> Optional[float]:
        """(variant - base) / |base| as a fraction; None when undefined."""
        d, b = self.delta.get(metric), self.base_mean.get(metric)
        if d is None or b is None or b == 0:
            return None
        return d / abs(b)


def _diff_signature(diff: Mapping[str, tuple[Any, Any]]) -> tuple:
    return tuple((k, repr(v[1])) for k, v in sorted(diff.items()))


def ablation_table(
    records: Iterable[Record],
    base_config: Optional[Mapping[str, Any]] = None,
    base_run_id: Optional[int] = None,
    metrics: Optional[Sequence[str]] = None,
    only_completed: bool = True,
    ignore_keys: Optional[Sequence[str]] = None,
) -> list[AblationRow]:
    """Group variants by how their config differs from the base; report metric deltas.

    Base is chosen by (in order): `base_config`, `base_run_id`, a record tagged
    'base', else the config shared by the most runs.

    `ignore_keys` are config keys left out of the diff. By default these are the `condition_keys`:
    whatever varies among the runs tagged 'base' (kernel, noise level, ...) is a condition the whole
    ablation was repeated over, not a setting, so arms pool over it instead of splitting on it.
    """
    recs = list(records)
    if only_completed:
        recs = completed(recs)
    if not recs:
        return []
    ignored = set(ignore_keys) if ignore_keys is not None else set(condition_keys(recs))

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
            top = max(n for n, _ in counts.values())
            leaders = [cfg for n, cfg in counts.values() if n == top]
            if len(leaders) > 1:
                raise AmbiguousBaseError(
                    f"{len(leaders)} configs share the highest run count ({top}); tag the full model's runs with "
                    "'base' or pass base_run_id/base_config — the base cannot be guessed")
            base_config = leaders[0]

    names = list(metrics) if metrics else metric_names(recs)
    groups: dict[tuple, tuple[dict, list[Record]]] = {}
    for r in recs:
        d = {k: v for k, v in config_diff(base_config, r["config"]).items() if k not in ignored}
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
        rows.append(AblationRow(
            describe_diff(d), d, len(rs), stats, delta, is_base=(sig == ()),
            base_mean={m: (b.mean if b is not None else None) for m, b in base_stats.items()},
        ))
    rows.sort(key=lambda r: (not r.is_base, len(r.diff), r.label))
    return rows


# --------------------------------------------------------------------------- ablation effect sizes

@dataclass
class AblationEffect:
    label: str
    n: int
    delta: float            # variant mean − base mean
    rel: Optional[float]    # delta / |base mean|
    pooled_std: float       # sqrt((s_base² + s_variant²)/2), 0 when both n == 1
    d: Optional[float]      # delta / pooled_std (Cohen's d against the full model); None when pooled_std == 0
    improves: bool          # the change moves the metric in the "better" direction
    verdict: str            # 'clear' | 'likely' | 'within noise' | 'n = 1'


def ablation_effects(rows: Sequence[AblationRow], metric: str, higher_is_better: bool = True) -> list[AblationEffect]:
    """Per variant: absolute and relative delta against the full model and a standardised effect size.

    Verdict thresholds on |d|: >= 2 'clear', >= 1 'likely', else 'within noise'; single runs cannot be judged.
    Sorted with the most harmful change first."""
    base = next((r for r in rows if r.is_base), None)
    if base is None or base.stats.get(metric) is None:
        return []
    bs = base.stats[metric]
    out: list[AblationEffect] = []
    for r in rows:
        if r.is_base or r.delta.get(metric) is None or r.stats.get(metric) is None:
            continue
        vs = r.stats[metric]
        delta = r.delta[metric]
        pooled = ((bs.std**2 + vs.std**2) / 2) ** 0.5
        d = (delta / pooled) if pooled > 0 else None
        improves = (delta > 0) if higher_is_better else (delta < 0)
        if bs.n < 2 and vs.n < 2:
            verdict = "n = 1"
        elif d is None:
            verdict = "clear" if delta != 0 else "within noise"
        elif abs(d) >= 2:
            verdict = "clear"
        elif abs(d) >= 1:
            verdict = "likely"
        else:
            verdict = "within noise"
        out.append(AblationEffect(r.label, r.n, delta, r.rel_delta(metric), pooled, d, improves, verdict))
    sign = 1 if higher_is_better else -1
    out.sort(key=lambda e: e.delta * sign)
    return out
