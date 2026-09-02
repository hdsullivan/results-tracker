"""Overview page: what is in the database, in the paper look.

Three IEEE-style tables: the experiments (with run counts, methods, datasets, metrics), the results at a
glance (each experiment's headline read off the data: best method, best swept value, largest ablation drop),
and the most recent runs. The sortable data grids stay available in expanders.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from .. import aggregate as agg
from ..export.latex import display_metric_name
from .common import hib_map, load_catalog, load_metric_defs, load_records, sidebar_db
from .tables import fmt_stat, generic_html


def _fmt_ts(ts: Any, with_time: bool = True) -> str:
    try:
        return ts.strftime("%Y-%m-%d %H:%M" if with_time else "%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return "—"


def _val(name: str, st_html: str, unit: str = "", suffix: str = "") -> str:
    """Value-column HTML: escaped text around the (trusted) fmt_stat markup."""
    return f"{html.escape(name)} {st_html}" + (f" {html.escape(unit)}" if unit else "") + (f" {html.escape(suffix)}" if suffix else "")


def experiments_rows(cat: dict, recs: list[dict]) -> list[list[Any]]:
    rows = []
    for e in cat["experiments"]:
        rs = [r for r in recs if r["experiment"] == e["experiment"]]
        done = [r for r in rs if r["status"] == "completed"]
        failed = sum(r["status"] == "failed" for r in rs)
        methods = list(dict.fromkeys(r["method"] for r in rs if r.get("method")))
        datasets = list(dict.fromkeys(r["dataset"] for r in rs if r.get("dataset")))
        metrics = agg.metric_names(rs)
        last = max((r["timestamp"] for r in rs if r.get("timestamp")), default=None)
        rows.append([
            e["experiment"], e["type"], f"{len(done)}" + (f" (+{failed} failed)" if failed else ""),
            ", ".join(map(str, methods)) or "—", ", ".join(map(str, datasets)) or "—",
            ", ".join(display_metric_name(m) for m in metrics) or "—", _fmt_ts(last, with_time=False) if last else "—",
        ])
    return rows


def glance_rows(cat: dict, recs: list[dict], defs: dict) -> list[list[Any]]:
    """One line per experiment: what a reader would want to know first."""
    hib = hib_map(defs)
    out = []
    for e in cat["experiments"]:
        rs = agg.completed([r for r in recs if r["experiment"] == e["experiment"]])
        if not rs:
            out.append([e["experiment"], e["type"], "no completed runs", ""])
            continue
        metrics = agg.metric_names(rs)
        primary = next((m for m in ("psnr", "ssim") if m in metrics), metrics[0] if metrics else None)
        if primary is None:
            out.append([e["experiment"], e["type"], "no metrics", ""])
            continue
        fmt = defs.get(primary, {}).get("fmt", ".2f")
        unit = defs.get(primary, {}).get("unit", "")
        name = display_metric_name(primary)
        if e["type"] == "sweep":
            keys = agg.varying_config_keys(rs) or sorted({k for r in rs for k in agg.flatten(r["config"])})
            if keys:
                series = agg.sweep_series(rs, keys[0], primary)
                s = series.get((), [])
                best = agg.best_sweep_value(s, hib.get(primary, True))
                st_ = dict(s).get(best)
                out.append([e["experiment"], "sweep", f"best {keys[0]} = {best:g}" if isinstance(best, (int, float)) else f"best {keys[0]} = {best}",
                            _val(name, fmt_stat(st_, fmt), unit)])
                continue
        if e["type"] == "ablation":
            rows = agg.ablation_table(rs, metrics=[primary])
            variants = [r for r in rows if not r.is_base and r.delta.get(primary) is not None]
            base = next((r for r in rows if r.is_base), None)
            if variants and base:
                sign = 1 if hib.get(primary, True) else -1
                worst = min(variants, key=lambda r: r.delta[primary] * sign)
                delta = format(worst.delta[primary], "+" + fmt).replace("-", "−")
                out.append([e["experiment"], "ablation", f"largest drop: {worst.label}",
                            _val(f"{name} {delta}", "", unit).rstrip() + " vs full model " + fmt_stat(base.stats.get(primary), fmt)])
                continue
        labels = agg.method_labels(rs)
        datasets = list(dict.fromkeys(r["dataset"] for r in rs if r.get("dataset") is not None))
        col_key = "dataset" if len(datasets) > 1 else None  # never pool across datasets a method was not run on
        pt = agg.pivot_table(rs, "method", col_key, metrics=[primary], higher_is_better=hib)
        heads, vals = [], []
        for c in pt.cols:
            best = next((r for r in pt.rows if pt.is_best(r, c, primary)), None)
            if best is None:
                continue
            heads.append(f"{labels.get(best, best)}" + (f" on {c}" if c is not None else ""))
            vals.append(fmt_stat(pt.stat(best, c, primary), fmt))
        if not heads:
            out.append([e["experiment"], e["type"], "—", ""])
            continue
        out.append([e["experiment"], e["type"], "best method: " + "; ".join(heads),
                    " · ".join(_val(name, v, unit) for v in vals) + html.escape(f" · {len(pt.rows)} methods")])
    return out


def recent_rows(recs: list[dict], defs: dict, n: int = 15) -> tuple[list[str], list[list[Any]]]:
    recent = sorted(recs, key=lambda r: r["timestamp"] or 0, reverse=True)[:n]
    metrics = agg.metric_names(recent)[:4]
    headers = ["#", "logged", "experiment", "method", "dataset", "seed", "status"] + [display_metric_name(m) for m in metrics]
    rows = []
    for r in recent:
        vals = []
        for m in metrics:
            v = r["metrics"].get(m)
            vals.append("—" if v is None else format(v, defs.get(m, {}).get("fmt", ".2f")).replace("-", "−"))
        rows.append([r["run_id"], _fmt_ts(r["timestamp"]), r["experiment"], r["method"] or "—", r["dataset"] or "—",
                     "—" if r["seed"] is None else r["seed"], r["status"], *vals])
    return headers, rows


def render() -> None:
    st.title("Results Tracker")
    sidebar_db()
    cat = load_catalog()
    recs = load_records()
    defs = load_metric_defs()

    failed = [r for r in recs if r["status"] == "failed"]
    running = [r for r in recs if r["status"] == "running"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", len(cat["projects"]))
    c2.metric("Experiments", len(cat["experiments"]))
    c3.metric("Runs", len(recs))
    c4.metric("Failed / running", f"{len(failed)} / {len(running)}")

    if not recs:
        st.info("Empty database. Seed a demo with `results-tracker demo`, or log runs with `results_tracker.log_run`.")
        return

    st.subheader("Experiments")
    projects = ", ".join(p["name"] for p in cat["projects"])
    st.markdown(generic_html(
        ["Experiment", "Type", "Runs", "Methods", "Datasets", "Metrics", "Last run"], experiments_rows(cat, recs),
        caption=f"Experiments in {projects}. Run counts are completed runs; failed runs in parentheses.", number=1, left_cols=2,
    ), unsafe_allow_html=True)

    st.subheader("Results at a glance")
    st.markdown(generic_html(
        ["Experiment", "Type", "Headline", "Value"], glance_rows(cat, recs, defs),
        caption="One line per experiment, read off the completed runs: best method (mean ± std over datasets and seeds), "
                "best swept value, or the setting whose removal costs the most. Details on the respective pages.",
        number=2, left_cols=4, raw_html_cols=[3],
    ), unsafe_allow_html=True)

    st.subheader("Recent runs")
    headers, rows = recent_rows(recs, defs)
    st.markdown(generic_html(headers, rows, caption="Most recent runs, newest first; metrics as logged.", number=3, left_cols=5),
                unsafe_allow_html=True)

    with st.expander("Sortable grids"):
        per_exp = pd.DataFrame(cat["experiments"])
        run_df = pd.DataFrame([{"experiment": r["experiment"]} for r in recs])
        per_exp["runs"] = per_exp["experiment"].map(run_df["experiment"].value_counts()).fillna(0).astype(int)
        st.dataframe(per_exp, width="stretch", hide_index=True)
        recent = sorted(recs, key=lambda r: r["timestamp"] or 0, reverse=True)[:50]
        st.dataframe(pd.DataFrame([
            {"id": r["run_id"], "time": r["timestamp"], "experiment": r["experiment"], "method": r["method"],
             "dataset": r["dataset"], "seed": r["seed"], "status": r["status"], **r["metrics"]} for r in recent
        ]), width="stretch", hide_index=True)
