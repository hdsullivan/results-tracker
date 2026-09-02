"""Flat CSV of runs: one row per run, config keys and metrics as columns."""

from __future__ import annotations

import csv
import io
from typing import Any, Iterable

from .. import aggregate as agg

FIXED = ["run_id", "experiment", "method", "dataset", "instance", "seed", "status", "source", "git_commit", "timestamp", "artifacts_dir"]


def runs_csv(records: Iterable[dict[str, Any]]) -> str:
    recs = list(records)
    cfg_keys = sorted({k for r in recs for k in agg.flatten(r.get("config", {}))})
    met_keys = agg.metric_names(recs)
    header = FIXED + [f"config.{k}" for k in cfg_keys] + [f"metric.{m}" for m in met_keys]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in recs:
        flat = agg.flatten(r.get("config", {}))
        ts = r.get("timestamp")
        row = [r.get(k) if k != "timestamp" else (ts.isoformat() if ts else "") for k in FIXED]
        row += [flat.get(k) for k in cfg_keys]
        row += [r.get("metrics", {}).get(m) for m in met_keys]
        w.writerow(row)
    return buf.getvalue()
