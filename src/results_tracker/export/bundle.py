"""One-shot paper bundle: every table, figure, CSV and a provenance manifest for a project, zipped.

Regenerating the whole results section from the database in one command is the point of the tool:
    results-tracker export bundle -p my-paper -o paper_assets.zip
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

from .. import aggregate as agg
from .csv import runs_csv
from .figures import ablation_figure, comparison_figure, figure_bytes, figure_tex, ieee_preamble, sweep_figure
from .latex import ablation_latex, comparison_latex, provenance_note, sweep_latex, width_hint
from .visual import make_visual

Record = dict[str, Any]


def _primary_metric(recs: Sequence[Record]) -> Optional[str]:
    names = agg.metric_names(recs)
    return next((m for m in ("psnr", "ssim") if m in names), names[0] if names else None)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text))


def build_bundle(
    experiments: Mapping[str, tuple[str, Sequence[Record]]],
    defs: Mapping[str, Mapping[str, Any]],
    *,
    project: str,
    source: str,
    width: str = "single",
    visual: bool = True,
) -> tuple[bytes, list[dict[str, Any]]]:
    """`experiments` maps name -> (type, records). Returns (zip bytes, manifest rows)."""
    hib = {k: v["higher_is_better"] for k, v in defs.items()}
    buf = io.BytesIO()
    manifest: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def add(zf: zipfile.ZipFile, path: str, data: bytes | str, kind: str, exp: str, n: int, note: str = "") -> None:
        zf.writestr(path, data)
        manifest.append({"file": path, "kind": kind, "experiment": exp, "runs": n, "note": note})

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, (etype, recs) in experiments.items():
            recs = list(recs)
            done = agg.completed(recs)
            slug = _slug(name)
            prov = provenance_note(source, name, len(recs))
            add(zf, f"runs/{slug}.csv", runs_csv(recs), "runs-csv", name, len(recs))
            if not done:
                continue
            metric = _primary_metric(done)
            unit = defs.get(metric, {}).get("unit", "") if metric else ""
            ylabel = (f"{metric} ({unit})" if unit else metric) if metric else ""
            labels = agg.method_labels(done)

            if etype == "comparison":
                has_ds = any(r.get("dataset") is not None for r in done)
                pt = agg.pivot_table(done, "method", "dataset" if has_ds else None, higher_is_better=hib)
                audit = agg.audit_grid(recs, ["method"] + (["dataset"] if has_ds else []))
                hint = width_hint(pt, "pm", "table", None)
                add(zf, f"tables/{slug}.tex",
                    comparison_latex(pt, defs, label=f"tab:{slug}", row_labels=agg.method_labels(done, latex=True),
                                     audit=audit, provenance=prov,
                                     env="table*" if hint else "table"),
                    "comparison-table", name, len(done), audit.summary() + (" · table* (wide)" if hint else ""))
                if metric:
                    fig = comparison_figure(pt, metric, ylabel=ylabel, width=width, row_labels=labels)
                    add(zf, f"figures/{slug}_{metric}.pdf", figure_bytes(fig, "pdf"), "comparison-figure", name, len(done))
                    add(zf, f"figures/{slug}_{metric}.tex", figure_tex(f"figures/{slug}_{metric}.pdf", label=f"fig:{slug}", width=width),
                        "figure-tex", name, len(done))
                if visual and any(r.get("artifacts_dir") for r in done):
                    datasets = list(dict.fromkeys(r["dataset"] for r in done if r.get("dataset") is not None and r.get("artifacts_dir")))
                    for ds in (datasets or [None]):
                        try:
                            vr = make_visual(done, defs, experiment=name, dataset=ds, zoom=True, width="double")
                        except ValueError as e:
                            manifest.append({"file": "", "kind": "visual-figure", "experiment": name, "runs": 0, "note": f"skipped: {e}"})
                            continue
                        vs = f"{slug}_{_slug(ds)}" if ds is not None else slug
                        add(zf, f"figures/{vs}_visual.pdf", figure_bytes(vr.fig, "pdf"), "visual-figure", name, len(vr.spec.panels),
                            "; ".join(vr.problems + [f"not shown: {k}" for k in vr.omitted]))
                        add(zf, f"figures/{vs}_visual.json", json.dumps(asdict(vr.spec), indent=2, default=str), "provenance", name, 0)
                        add(zf, f"figures/{vs}_visual.tex", figure_tex(f"figures/{vs}_visual.pdf", caption=vr.spec.caption_stub(),
                                                                        label=f"fig:{vs}_visual", width="double"), "figure-tex", name, 0)
                        if vr.spec.reference:
                            try:
                                er = make_visual(done, defs, experiment=name, dataset=ds, mode="error", width="double")
                                add(zf, f"figures/{vs}_error.pdf", figure_bytes(er.fig, "pdf"), "visual-error-figure", name, len(er.spec.panels))
                            except ValueError:
                                pass

            elif etype == "sweep":
                params = agg.varying_config_keys(done) or sorted({k for r in done for k in agg.flatten(r["config"])})
                if params and metric:
                    param = params[0]
                    series = agg.sweep_series(done, param, metric)
                    add(zf, f"tables/{slug}_{_slug(param)}.tex",
                        sweep_latex(series, param, metric, defs, label=f"tab:{slug}", provenance=prov), "sweep-table", name, len(done))
                    best = {g: agg.best_sweep_value(s_, hib.get(metric, True)) for g, s_ in series.items()}
                    fig = sweep_figure(series, param, metric, ylabel=ylabel, best_by_group=best, width=width)
                    add(zf, f"figures/{slug}_{_slug(param)}.pdf", figure_bytes(fig, "pdf"), "sweep-figure", name, len(done))
                    add(zf, f"figures/{slug}_{_slug(param)}.tex", figure_tex(f"figures/{slug}_{_slug(param)}.pdf", label=f"fig:{slug}", width=width),
                        "figure-tex", name, len(done))

            elif etype == "ablation":
                try:
                    rows = agg.ablation_table(done)
                except agg.AmbiguousBaseError as e:
                    manifest.append({"file": "", "kind": "ablation-table", "experiment": name, "runs": 0, "note": f"skipped: {e}"})
                    continue
                metrics = list(rows[0].stats) if rows else []
                if rows and metrics:
                    add(zf, f"tables/{slug}.tex", ablation_latex(rows, metrics, defs, label=f"tab:{slug}", provenance=prov),
                        "ablation-table", name, len(done))
                    if metric:
                        d = defs.get(metric, {})
                        fig = ablation_figure(rows, metric, higher_is_better=d.get("higher_is_better", True),
                                              fmt=d.get("fmt", ".2f"), width=width)
                        add(zf, f"figures/{slug}_{metric}.pdf", figure_bytes(fig, "pdf"), "ablation-figure", name, len(done))
                        add(zf, f"figures/{slug}_{metric}.tex", figure_tex(f"figures/{slug}_{metric}.pdf", label=f"fig:{slug}", width=width),
                            "figure-tex", name, len(done))

        zf.writestr("preamble.tex", ieee_preamble())
        readme = (f"results-tracker paper bundle for project '{project}'\nGenerated {stamp} from {source}.\n\n"
                  "tables/   booktabs tables (\\input them; needs preamble.tex packages)\n"
                  "figures/  vector PDFs + figure environment snippets (.tex) + provenance sidecars (.json)\n"
                  "runs/     every run as CSV\n"
                  "MANIFEST.json  what was generated from which experiment and how many runs\n"
                  "Do not edit numbers by hand; regenerate with `results-tracker export bundle`.\n")
        zf.writestr("README.txt", readme)
        zf.writestr("MANIFEST.json", json.dumps({"project": project, "source": source, "generated": stamp, "files": manifest},
                                                 indent=2, default=str))
    return buf.getvalue(), manifest
