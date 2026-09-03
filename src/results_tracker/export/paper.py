"""The paper: pinned assets rendered from the database into stable file names.

An `Asset` row (models.Asset) says what one table or figure of the manuscript is: its LaTeX label, the
experiment, the `where` filter and the rendering options of one of the KINDS below. `render_asset` turns
that spec plus the experiment's records into files; `render_paper` does it for every asset of a project;
`write_paper` / `zip_paper` lay them out as

    tables/<label>.tex      figures/<label>.pdf + .tex snippet (+ .json provenance for visuals)
    data/<label>.csv        preamble.tex   MANIFEST.json

so the manuscript can `\\input{tables/tab-main}` and `\\includegraphics{figures/fig-beta}` and one command
refreshes everything. `records_fingerprint` hashes the runs an asset was rendered from; when the current
records hash differently the asset is stale.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

from .. import aggregate as agg
from .csv import runs_csv
from .figures import (ablation_figure, comparison_figure, curves_figure, distribution_figure, figure_bytes, figure_tex,
                      ieee_preamble, sweep_figure, tradeoff_figure)
from .latex import ablation_latex, comparison_latex, provenance_note, selection_latex, sweep_latex, width_hint
from .visual import make_visual

Record = dict[str, Any]

KINDS = ("comparison-table", "ablation-table", "sweep-table", "selection-table",
         "sweep-figure", "ablation-figure", "comparison-figure", "visual-figure",
         "curves-figure", "tradeoff-figure", "distribution-figure", "runs-csv")
TABLE_KINDS = ("comparison-table", "ablation-table", "sweep-table", "selection-table")
FIGURE_KINDS = ("sweep-figure", "ablation-figure", "comparison-figure", "visual-figure", "curves-figure", "tradeoff-figure",
                "distribution-figure")
#: which GUI page configures (and restores) an asset kind; kinds not listed live on the Export page
KIND_PAGE = {"curves-figure": "curves", "tradeoff-figure": "tradeoff", "distribution-figure": "comparison", "selection-table": "sweep"}
EXPORT_STATUSES = ("planned", "draft", "final")  # dropped assets are kept in the database but not rendered

KIND_TITLES = {
    "comparison-table": "Comparison table (LaTeX)", "ablation-table": "Ablation table (LaTeX)",
    "sweep-table": "Sweep table (LaTeX)", "sweep-figure": "Sweep figure", "ablation-figure": "Ablation figure",
    "comparison-figure": "Comparison figure", "visual-figure": "Visual comparison figure", "runs-csv": "Runs (CSV)",
    "selection-table": "Selection table (LaTeX)", "curves-figure": "Curves figure", "tradeoff-figure": "Trade-off figure",
    "distribution-figure": "Distribution figure",
}


def label_slug(label: str) -> str:
    """`tab:main` -> `tab-main`; anything that is not a file-name character becomes `-`."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in str(label)).strip("-") or "asset"


def default_label(kind: str, experiment: str) -> str:
    prefix = "tab" if kind in TABLE_KINDS else "data" if kind == "runs-csv" else "fig"
    return f"{prefix}:{label_slug(experiment)}"


def records_fingerprint(records: Iterable[Record]) -> str:
    """Short hash of everything a rendering depends on: run ids, status, timestamps, configs, metrics, labels."""
    rows = sorted(
        (r.get("run_id"), r.get("status"), str(r.get("timestamp")), r.get("method_label"),
         json.dumps(r.get("config", {}), sort_keys=True, default=str), json.dumps(r.get("metrics", {}), sort_keys=True, default=str))
        for r in records
    )
    return hashlib.sha1(json.dumps(rows, default=str).encode()).hexdigest()[:12]


def _width(w: Any) -> Union[str, float]:
    try:
        return float(w)
    except (TypeError, ValueError):
        return w or "single"


@dataclass
class RenderedAsset:
    label: str
    kind: str
    experiment: str
    status: str = "planned"
    files: list[tuple[str, bytes]] = field(default_factory=list)  # (relative path, content)
    runs: int = 0  # completed runs the asset was rendered from
    note: str = ""  # audit summary, omitted methods, width hints
    fingerprint: str = ""
    error: str = ""  # set when nothing could be rendered
    filters: list[str] = field(default_factory=list)

    @property
    def main_file(self) -> Optional[str]:
        return self.files[0][0] if self.files else None

    def manifest_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["files"] = [f for f, _ in self.files]
        return d


def render_asset(spec: Mapping[str, Any], records: Sequence[Record], defs: Mapping[str, Mapping[str, Any]], *,
                 source: str = "") -> RenderedAsset:
    """Render one asset spec (`label`, `kind`, `experiment`, `filters`, `options`, `caption`, `status`) from the
    experiment's records. Errors end up in `.error`, never raised: one broken figure must not stop the paper."""
    label, kind = spec["label"], spec["kind"]
    if kind not in KINDS:
        raise ValueError(f"unknown asset kind {kind!r}; expected one of {KINDS}")
    experiment = " + ".join(asset_experiments(spec)) if spec.get("extra_experiments") else (spec.get("experiment") or "")
    filters = dict(spec.get("filters") or {})
    o = dict(spec.get("options") or {})
    caption = spec.get("caption") or None
    out = RenderedAsset(label, kind, experiment, status=str(spec.get("status") or "planned"), filters=agg.where_items(filters))
    recs_all = agg.filter_records(records, filters) if filters else list(records)
    recs = agg.completed(recs_all)
    out.fingerprint = records_fingerprint(recs_all)
    out.runs = len(recs)
    if not recs:
        out.error = "no completed runs match" + (f" {agg.where_text(filters)}" if filters else "")
        return out
    slug = label_slug(label)
    hib = {k: v["higher_is_better"] for k, v in defs.items()}
    prov = provenance_note(source, experiment, len(recs_all), extra=f"Filter: {agg.where_cli(filters)}" if filters else "")
    metric = o.get("metric") or (spec.get("primary_metric") if spec.get("primary_metric") in agg.metric_names(recs) else None) or _primary_metric(recs)
    unit = defs.get(metric, {}).get("unit", "") if metric else ""
    default_ylabel = (f"{metric} ({unit})" if unit else metric) if metric else ""
    width = _width(o.get("width", "double" if kind == "visual-figure" else "single"))
    env = o.get("env", "table")
    env = None if env in (None, "none", "tabular") else env

    def add_figure(fig, extra_note: str = "", snippet_caption: Optional[str] = None) -> None:
        out.files.append((f"figures/{slug}.pdf", figure_bytes(fig, "pdf")))
        out.files.append((f"figures/{slug}.tex", figure_tex(f"figures/{slug}.pdf", caption=caption or snippet_caption or "TODO",
                                                            label=label, width=width).encode()))
        out.note = extra_note

    try:
        if kind == "comparison-table":
            rows_key, cols_key = o.get("rows", "method"), o.get("cols", "dataset")
            cols_key = None if cols_key in (None, "none") else cols_key
            pt = agg.pivot_table(recs, rows_key, cols_key, metrics=o.get("metrics") or None, higher_is_better=hib,
                                 row_order=agg.method_order(recs) if rows_key == "method" else agg.value_order(recs, rows_key),
                                 col_order=agg.value_order(recs, cols_key) if cols_key else None)
            audit = agg.audit_grid(recs_all, [rows_key] + ([cols_key] if cols_key else []))
            hint = width_hint(pt, o.get("std", "pm"), env, o.get("font"))
            tex = comparison_latex(pt, defs, caption=caption, label=label, env=env, font=o.get("font"), std=o.get("std", "pm"),
                                   underline_second=o.get("underline", True),
                                   row_labels=agg.method_labels(recs, latex=True) if rows_key == "method" else None,
                                   audit=audit, provenance=prov)
            out.files.append((f"tables/{slug}.tex", tex.encode()))
            out.note = audit.summary() + (f" · {hint}" if hint else "")
        elif kind == "ablation-table":
            rows = agg.ablation_table(recs, base_run_id=o.get("base_run_id"), metrics=o.get("metrics") or None)
            metrics = o.get("metrics") or (list(rows[0].stats) if rows else [])
            tex = ablation_latex(rows, metrics, defs, caption=caption, label=label, env=env, font=o.get("font"), std=o.get("std", "pm"),
                                 show_delta=o.get("show_delta", True), setting_columns=o.get("setting_columns", True), provenance=prov)
            out.files.append((f"tables/{slug}.tex", tex.encode()))
            if not any(r.is_base for r in rows):
                out.note = "no run matches the base config; deltas missing"
        elif kind in ("sweep-table", "sweep-figure"):
            param = o.get("param") or (agg.varying_config_keys(recs) or [None])[0]
            if not param or not metric:
                raise ValueError("a sweep asset needs `param` and `metric` options")
            series = {g: s for g, s in agg.sweep_series(recs, param, metric, group_by=o.get("by") or []).items() if s}
            if not series:
                raise ValueError(f"no runs have `{param}` in their config")
            if kind == "sweep-table":
                tex = sweep_latex(series, param, metric, defs, caption=caption, label=label, env=env, font=o.get("font"),
                                  std=o.get("std", "pm"), param_label=o.get("param_label"), provenance=prov)
                out.files.append((f"tables/{slug}.tex", tex.encode()))
            else:
                best = {g: agg.best_sweep_value(s, hib.get(metric, True)) for g, s in series.items()}
                fig = sweep_figure(series, param, metric, xlabel=o.get("xlabel") or param, ylabel=o.get("ylabel") or default_ylabel,
                                   band=o.get("band", True), best_by_group=best, width=width, height=o.get("height"),
                                   emphasize=o.get("emphasize") or (), caption=o.get("panel_label") or None, log_x=o.get("log_x"))
                add_figure(fig)
        elif kind == "ablation-figure":
            if not metric:
                raise ValueError("an ablation figure needs a `metric` option")
            rows = agg.ablation_table(recs, base_run_id=o.get("base_run_id"), metrics=[metric])
            if not any(r.is_base for r in rows):
                raise ValueError("no run matches the base config; tag the full model's runs `base`")
            d = defs.get(metric, {})
            fig = ablation_figure(rows, metric, higher_is_better=d.get("higher_is_better", True), fmt=d.get("fmt", ".2f"),
                                  xlabel=o.get("xlabel") or (f"$\\Delta$ {metric} vs. full model" + (f" ({unit})" if unit else "")),
                                  width=width, height=o.get("height"), caption=o.get("panel_label") or None)
            add_figure(fig)
        elif kind == "comparison-figure":
            if not metric:
                raise ValueError("a comparison figure needs a `metric` option")
            rows_key, cols_key = o.get("rows", "method"), o.get("cols", "dataset")
            cols_key = None if cols_key in (None, "none") else cols_key
            pt = agg.pivot_table(recs, rows_key, cols_key, metrics=[metric], higher_is_better=hib,
                                 row_order=agg.method_order(recs) if rows_key == "method" else agg.value_order(recs, rows_key),
                                 col_order=agg.value_order(recs, cols_key) if cols_key else None)
            fig = comparison_figure(pt, metric, ylabel=o.get("ylabel") or default_ylabel, width=width, height=o.get("height"),
                                    emphasize=o.get("emphasize") or (), zero_based=o.get("zero_based", False),
                                    hatch=o.get("hatch", False), caption=o.get("panel_label") or None,
                                    row_labels=agg.method_labels(recs) if rows_key == "method" else None)
            add_figure(fig)
        elif kind == "visual-figure":
            zc = o.get("zoom_center") or (0.5, 0.5)
            crop = o.get("crop_box")
            vr = make_visual(recs, defs, experiment=experiment, dataset=o.get("dataset"), seed=o.get("seed"), instance=o.get("instance"),
                             image=o.get("image"), reference=o.get("reference"), measurement=o.get("measurement"), kernel=o.get("kernel"),
                             methods=o.get("methods") or None, metrics=o.get("metrics") or ("psnr", "ssim"), mode=o.get("mode", "image"),
                             zoom=o.get("zoom", True), zoom_fraction=o.get("zoom_fraction", 0.3), zoom_center=(float(zc[0]), float(zc[1])),
                             crop_box=tuple(int(v) for v in crop) if crop else None, rows=o.get("rows"), width=width,
                             auto_roles=o.get("image") is None, data_range=o.get("data_range"))
            out.files.append((f"figures/{slug}.pdf", figure_bytes(vr.fig, "pdf")))
            out.files.append((f"figures/{slug}.tex", figure_tex(f"figures/{slug}.pdf", caption=caption or vr.spec.caption_stub(),
                                                                label=label, width=width).encode()))
            out.files.append((f"figures/{slug}.json", json.dumps(asdict(vr.spec), indent=2, default=str).encode()))
            out.note = "; ".join(vr.problems + [f"not shown: {k} — {why}" for k, why in vr.omitted.items()])
        elif kind == "curves-figure":
            from ..curves import curve_series, normalise

            curve = o.get("curve")
            if not curve:
                raise ValueError("a curves figure needs a `curve` option")
            series = {g: normalise(cs, o.get("normalise", "value")) for g, cs in curve_series(recs, curve, o.get("by") or []).items()}
            if not series:
                raise ValueError(f"no run has a `{curve}` curve in its diagnostics.json")
            fig = curves_figure(series, curve, xlabel=o.get("xlabel") or "iteration", ylabel=o.get("ylabel") or curve,
                                band=o.get("band", True), log_y=o.get("log_y", False), width=width, height=o.get("height"),
                                emphasize=o.get("emphasize") or (), caption=o.get("panel_label") or None, guide=o.get("guide"))
            add_figure(fig, f"{sum(cs.runs for cs in series.values())} runs with curves")
        elif kind == "tradeoff-figure":
            x_metric, y_metric = o.get("x_metric") or "runtime_s", o.get("y_metric") or metric
            pts = agg.tradeoff_points(recs, x_metric, y_metric, series_key=o.get("series", "method"), path_key=o.get("path"))
            if not pts:
                raise ValueError(f"no run has both `{x_metric}` and `{y_metric}`")
            hollow = set(o.get("hollow") or []) | ({r["method"] for r in recs if r.get("method_is_baseline") or r.get("source") == "reported"}
                                                  if o.get("hollow_baselines", True) else set())
            fig = tradeoff_figure(pts, x_metric, y_metric, xlabel=o.get("xlabel"), ylabel=o.get("ylabel"), log_x=o.get("log_x", True),
                                  hollow=hollow, annotate=o.get("annotate", True), width=width, height=o.get("height"),
                                  emphasize=o.get("emphasize") or (), labels=agg.method_labels(recs) if o.get("series", "method") == "method" else None,
                                  caption=o.get("panel_label") or None)
            add_figure(fig)
        elif kind == "distribution-figure":
            if not metric:
                raise ValueError("a distribution figure needs a `metric` option")
            table = agg.instance_table(recs, metric, methods=o.get("methods") or None, higher_is_better=hib.get(metric, True))
            if not table.methods:
                raise ValueError(f"no per-instance runs with `{metric}` (runs need an `instance`)")
            fig = distribution_figure({m: table.values(m) for m in table.methods}, metric, ylabel=o.get("ylabel") or default_ylabel,
                                      width=width, height=o.get("height"), emphasize=o.get("emphasize") or (), labels=agg.method_labels(recs),
                                      show_points=o.get("points", True), caption=o.get("panel_label") or None)
            add_figure(fig, f"{len(table.instances)} instances")
        elif kind == "selection-table":
            param = o.get("param")
            if not param or not metric:
                raise ValueError("a selection table needs `param` and `metric` options")
            sel = agg.selection_table(recs, param, metric, group_by=o.get("by") or [], higher_is_better=hib.get(metric, True))
            if not sel:
                raise ValueError(f"no runs have `{param}` in their config")
            tex = selection_latex(sel, param, metric, o.get("by") or [], defs, caption=caption, label=label, env=env, font=o.get("font"),
                                  param_label=o.get("param_label"), provenance=prov)
            out.files.append((f"tables/{slug}.tex", tex.encode()))
            out.note = f"{sum(s.at_boundary for s in sel)} of {len(sel)} winners at a grid boundary" if any(s.at_boundary for s in sel) else ""
        elif kind == "runs-csv":
            out.files.append((f"data/{slug}.csv", runs_csv(recs_all).encode()))
            out.runs = len(recs_all)
    except (ValueError, agg.AmbiguousBaseError) as e:
        out.error = str(e)
        out.files = []
    return out


def _primary_metric(recs: Sequence[Record]) -> Optional[str]:
    names = agg.metric_names(recs)
    return next((m for m in ("psnr", "ssim") if m in names), names[0] if names else None)


# --------------------------------------------------------------------------- whole paper

def asset_spec(asset) -> dict[str, Any]:
    """The render spec of a models.Asset row."""
    status = asset.status.value if hasattr(asset.status, "value") else str(asset.status)
    return {"label": asset.label, "kind": asset.kind, "experiment": asset.experiment,
            "extra_experiments": list(asset.extra_experiments or []), "filters": dict(asset.filters or {}),
            "options": dict(asset.options or {}), "caption": asset.caption, "status": status}


def asset_experiments(asset_or_spec) -> list[str]:
    """The experiment(s) an asset pools: its own first, then `extra_experiments`."""
    if isinstance(asset_or_spec, Mapping):
        return [asset_or_spec.get("experiment") or ""] + list(asset_or_spec.get("extra_experiments") or [])
    return [asset_or_spec.experiment] + list(asset_or_spec.extra_experiments or [])


def render_paper(engine, project: str, *, source: str = "", statuses: Sequence[str] = EXPORT_STATUSES,
                 records_for=None) -> list[RenderedAsset]:
    """Render every asset of `project` whose status is in `statuses`, in manuscript order.

    `records_for(experiment) -> records` overrides how an experiment's runs are loaded (the GUI passes its cache)."""
    from ..api import get_metric_defs, get_runs, list_assets, list_projects, run_records

    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt} for k, m in get_metric_defs(engine=engine).items()}
    primary = next((p.primary_metric for p in list_projects(engine=engine) if p.name == project), "") or None
    cache: dict[str, list[Record]] = {}

    def load(exp: str) -> list[Record]:
        if exp not in cache:
            cache[exp] = records_for(exp) if records_for else run_records(get_runs(experiment=exp, project=project, engine=engine), engine=engine)
        return cache[exp]

    wanted = set(statuses)
    out = []
    for a in list_assets(project, engine=engine):
        status = a.status.value if hasattr(a.status, "value") else str(a.status)
        if status not in wanted:
            continue
        records = [r for exp in asset_experiments(a) for r in load(exp)]
        out.append(render_asset({**asset_spec(a), "primary_metric": primary}, records, defs, source=source))
    return out


def mark_exported(engine, project: str, rendered: Iterable[RenderedAsset]) -> None:
    """Record on each rendered asset when it was exported and from which records."""
    from ..api import update_asset

    now = datetime.now(timezone.utc)
    for r in rendered:
        if not r.error:
            update_asset(project, r.label, engine=engine, exported_at=now, fingerprint=r.fingerprint)


def paper_files(rendered: Sequence[RenderedAsset], *, project: str, source: str) -> list[tuple[str, bytes]]:
    """Every file of the paper directory, including preamble.tex, README.txt and MANIFEST.json."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    files: list[tuple[str, bytes]] = []
    for r in rendered:
        files.extend(r.files)
    files.append(("preamble.tex", ieee_preamble().encode()))
    files.append(("README.txt", (
        f"results-tracker paper assets for project '{project}'\nGenerated {stamp} from {source}.\n\n"
        "tables/   booktabs tables, one per pinned table asset (\\input them; needs preamble.tex packages)\n"
        "figures/  vector PDFs + figure environment snippets (.tex) + provenance sidecars (.json) for visuals\n"
        "data/     CSV assets\n"
        "MANIFEST.json  every asset: label, kind, status, experiment, filter, files, runs, fingerprint\n"
        "Do not edit numbers by hand; regenerate with `results-tracker export paper`.\n").encode()))
    manifest = {"project": project, "source": source, "generated": stamp, "assets": [r.manifest_row() for r in rendered]}
    files.append(("MANIFEST.json", json.dumps(manifest, indent=2, default=str).encode()))
    return files


def write_paper(rendered: Sequence[RenderedAsset], out_dir: Union[str, Path], *, project: str, source: str) -> list[Path]:
    """Write the paper directory (stable names: an asset's files replace the previous export). Returns the paths."""
    root = Path(out_dir)
    written = []
    for rel, data in paper_files(rendered, project=project, source=source):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        written.append(p)
    return written


def zip_paper(rendered: Sequence[RenderedAsset], *, project: str, source: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel, data in paper_files(rendered, project=project, source=source):
            zf.writestr(rel, data)
    return buf.getvalue()


def staleness(asset, records: Sequence[Record]) -> tuple[str, str]:
    """(state, detail) of an asset against the current records of its experiment:
    `never exported` / `current` / `stale` (the data changed since the last export) / `no data`."""
    filters = dict(asset.filters or {})
    recs = agg.filter_records(records, filters) if filters else list(records)
    if not agg.completed(recs):
        return "no data", "no completed runs match" + (f" {agg.where_text(filters)}" if filters else "")
    now = records_fingerprint(recs)
    if asset.exported_at is None:
        moved = " (data changed since it was pinned)" if asset.fingerprint and asset.fingerprint != now else ""
        return "never exported", f"{len(agg.completed(recs))} completed runs{moved}"
    if asset.fingerprint == now:
        return "current", f"{len(agg.completed(recs))} completed runs, unchanged since the export"
    return "stale", f"runs changed since the export on {asset.exported_at:%Y-%m-%d}; re-export"
