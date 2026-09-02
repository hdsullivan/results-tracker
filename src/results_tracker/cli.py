"""Command-line interface: `results-tracker --help`."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import aggregate as agg
from .api import define_metric, get_metric_defs, get_runs, list_experiments, list_projects, log_run, run_records
from .db import get_engine, resolve_db_path

app = typer.Typer(help="Track paper results: comparisons, sweeps, ablations.", no_args_is_help=True)


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version

        typer.echo(f"results-tracker {version('results-tracker')}")
        raise typer.Exit()


@app.callback()
def _main(
    version: bool = typer.Option(False, "--version", "-V", callback=_version_callback, is_eager=True, help="Show version."),
):
    """Track paper results: comparisons, sweeps, ablations."""

metric_app = typer.Typer(help="Define how metrics are displayed and ranked.")
app.add_typer(metric_app, name="metric")
export_app = typer.Typer(help="Paper-ready exports: booktabs LaTeX tables, IEEE-sized figures, CSV.", no_args_is_help=True)
app.add_typer(export_app, name="export")
console = Console()
err_console = Console(stderr=True)

DbOpt = typer.Option(None, "--db", help="SQLite file (default: $RESULTS_TRACKER_DB or ./results.db).")


def _parse_kv(items: list[str]) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise typer.BadParameter(f"expected key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


@app.command()
def init(db: Optional[Path] = DbOpt):
    """Create the database file and tables."""
    get_engine(db)
    console.print(f"[green]ready:[/] {resolve_db_path(db)}")


@app.command()
def projects(db: Optional[Path] = DbOpt):
    """List projects (papers)."""
    for p in list_projects(db=db):
        console.print(f"{p.name}  [dim]{p.description}[/]")


@app.command()
def experiments(project: Optional[str] = typer.Option(None, "--project", "-p"), db: Optional[Path] = DbOpt):
    """List experiments, optionally within one project."""
    t = Table("project", "experiment", "type", "runs")
    engine = get_engine(db)
    projs = {p.id: p.name for p in list_projects(engine=engine)}
    for e in list_experiments(project, engine=engine):
        n = len(get_runs(experiment=e.name, project=projs[e.project_id], engine=engine))
        t.add_row(projs[e.project_id], e.name, e.type.value, str(n))
    console.print(t)


@app.command()
def runs(
    experiment: Optional[str] = typer.Option(None, "--experiment", "-e"),
    project: Optional[str] = typer.Option(None, "--project", "-p"),
    limit: int = typer.Option(50, "--limit", "-n"),
    db: Optional[Path] = DbOpt,
):
    """List runs, newest last."""
    engine = get_engine(db)
    rs = get_runs(experiment=experiment, project=project, engine=engine)
    recs = run_records(rs, engine=engine)[-limit:]
    t = Table("id", "experiment", "method", "dataset", "seed", "status", "metrics", "config")
    for r in recs:
        t.add_row(
            str(r["run_id"]), r["experiment"], str(r["method"]), str(r["dataset"]), str(r["seed"]),
            r["status"], json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r["metrics"].items()}),
            json.dumps(r["config"]),
        )
    console.print(t)


@app.command()
def log(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    project: str = typer.Option("default", "--project", "-p"),
    method: Optional[str] = typer.Option(None, "--method", "-m"),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    experiment_type: str = typer.Option("comparison", "--type", help="comparison | sweep | ablation"),
    metric: list[str] = typer.Option([], "--metric", help="name=value (repeatable)"),
    param: list[str] = typer.Option([], "--param", help="config key=value (repeatable)"),
    metrics_json: Optional[Path] = typer.Option(None, "--metrics-json", help="JSON file with metrics"),
    config_json: Optional[Path] = typer.Option(None, "--config-json", help="JSON file with config"),
    tag: list[str] = typer.Option([], "--tag"),
    source: str = typer.Option("logged", "--source", help="logged | imported | reported"),
    notes: str = typer.Option("", "--notes"),
    artifacts_dir: Optional[Path] = typer.Option(None, "--artifacts"),
    db: Optional[Path] = DbOpt,
):
    """Record a single run from the shell."""
    metrics = json.loads(metrics_json.read_text()) if metrics_json else {}
    metrics.update(_parse_kv(metric))
    config = json.loads(config_json.read_text()) if config_json else {}
    config.update(_parse_kv(param))
    r = log_run(
        experiment, project=project, method=method, dataset=dataset, seed=seed,
        experiment_type=experiment_type, config=config, metrics=metrics, tags=tag,
        source=source, notes=notes, artifacts_dir=str(artifacts_dir) if artifacts_dir else None, db=db,
    )
    console.print(f"[green]logged run {r.id}[/] -> {experiment}: {metrics}")


@metric_app.command("define")
def metric_define(
    name: str,
    unit: str = typer.Option("", "--unit"),
    lower: bool = typer.Option(False, "--lower", help="Lower is better (default: higher)."),
    fmt: str = typer.Option(".2f", "--fmt"),
    db: Optional[Path] = DbOpt,
):
    define_metric(name, unit=unit, higher_is_better=not lower, fmt=fmt, db=db)
    console.print(f"[green]metric {name}[/]: {'lower' if lower else 'higher'} is better, fmt={fmt}")


@metric_app.command("list")
def metric_list(db: Optional[Path] = DbOpt):
    t = Table("metric", "unit", "direction", "fmt")
    for m in get_metric_defs(db=db).values():
        t.add_row(m.name, m.unit, "↑" if m.higher_is_better else "↓", m.fmt)
    console.print(t)


@app.command()
def table(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    project: Optional[str] = typer.Option(None, "--project", "-p"),
    by: list[str] = typer.Option(["method"], "--by", help="Row grouping (repeatable): method, dataset, config.<k>"),
    metrics: list[str] = typer.Option([], "--metric", help="Metrics to show (default: all)."),
    db: Optional[Path] = DbOpt,
):
    """Comparison table: rows grouped by --by, mean ± std over the rest. Best is bold."""
    engine = get_engine(db)
    recs = run_records(get_runs(experiment=experiment, project=project, engine=engine), engine=engine)
    defs = get_metric_defs(engine=engine)
    ct = agg.comparison_table(
        recs, group_by=by, metrics=metrics or None,
        higher_is_better={k: m.higher_is_better for k, m in defs.items()},
    )
    t = Table(" / ".join(by), *[f"{m} {'↑' if ct.higher_is_better[m] else '↓'}" for m in ct.metrics], "n")
    for row in ct.rows:
        cells = []
        for m in ct.metrics:
            st = ct.cells[row][m]
            s = st.format(defs[m].fmt if m in defs else ".2f") if st else "—"
            if ct.is_best(row, m):
                s = f"[bold]{s}[/]"
            elif ct.is_second(row, m):
                s = f"[underline]{s}[/]"
            cells.append(s)
        n = max((st.n for st in ct.cells[row].values() if st), default=0)
        t.add_row(ct.row_label(row), *cells, str(n))
    console.print(t)


@app.command()
def sweep(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    param: str = typer.Option(..., "--param", help="Config key that was swept, e.g. lambda"),
    metric: str = typer.Option(..., "--metric"),
    project: Optional[str] = typer.Option(None, "--project", "-p"),
    by: list[str] = typer.Option([], "--by", help="One series per group (e.g. method)."),
    db: Optional[Path] = DbOpt,
):
    """Print metric vs parameter, aggregated over seeds."""
    engine = get_engine(db)
    recs = run_records(get_runs(experiment=experiment, project=project, engine=engine), engine=engine)
    defs = get_metric_defs(engine=engine)
    hib = defs[metric].higher_is_better if metric in defs else True
    for g, series in agg.sweep_series(recs, param, metric, group_by=by).items():
        best = agg.best_sweep_value(series, hib)
        title = " / ".join(map(str, g)) if g else experiment
        t = Table(title=f"{title}: {metric} vs {param}  (best {param}={best})")
        t.add_column(param); t.add_column(metric); t.add_column("n")
        for x, st in series:
            s = st.format(defs[metric].fmt if metric in defs else ".2f")
            t.add_row(str(x), f"[bold]{s}[/]" if x == best else s, str(st.n))
        console.print(t)


@app.command()
def ablation(
    experiment: str = typer.Option(..., "--experiment", "-e"),
    project: Optional[str] = typer.Option(None, "--project", "-p"),
    metrics: list[str] = typer.Option([], "--metric"),
    db: Optional[Path] = DbOpt,
):
    """Ablation table: each config variant vs the base, with deltas."""
    engine = get_engine(db)
    recs = run_records(get_runs(experiment=experiment, project=project, engine=engine), engine=engine)
    defs = get_metric_defs(engine=engine)
    rows = agg.ablation_table(recs, metrics=metrics or None)
    names = rows[0].stats.keys() if rows else []
    t = Table("variant", *[f"{m} (Δ)" for m in names], "n")
    for r in rows:
        cells = []
        for m in names:
            st = r.stats[m]
            fmt = defs[m].fmt if m in defs else ".2f"
            if st is None:
                cells.append("—"); continue
            d = r.delta[m]
            ds = "" if r.is_base or d is None else f" ({d:+{fmt}})"
            cells.append(st.format(fmt) + ds)
        t.add_row(f"[bold]{r.label}[/]" if r.is_base else r.label, *cells, str(r.n))
    console.print(t)


@app.command()
def demo(
    db: Optional[Path] = DbOpt,
    reset: bool = typer.Option(False, "--reset", help="Delete the database file first."),
    artifacts: Optional[Path] = typer.Option(None, "--artifacts", help="Write demo reconstruction images/logs here (for Run detail)."),
):
    """Populate the database with a synthetic paper (comparison + sweep + ablation). Safe to re-run."""
    from .demo import PROJECT, demo_exists, seed_demo

    path = resolve_db_path(db)
    if reset and path != ":memory:" and Path(path).exists():
        Path(path).unlink()
        console.print(f"[yellow]deleted[/] {path}")
    engine = get_engine(db)
    if demo_exists(engine=engine):
        console.print(f"[yellow]project '{PROJECT}' already exists in {path}[/]; use --reset to start over.")
        raise typer.Exit(code=0)
    counts = seed_demo(engine=engine, artifacts_dir=str(artifacts) if artifacts else None)
    console.print(f"[green]seeded project '{PROJECT}'[/] into {path}: {counts}")
    if artifacts:
        console.print(f"artifacts written under {artifacts}")
    console.print("try:  results-tracker ui --db " + str(db or path))


@app.command("import")
def import_cmd(
    path: Path = typer.Argument(..., help="CSV file, JSON file, or directory of JSON run files."),
    experiment: str = typer.Option(..., "--experiment", "-e"),
    project: str = typer.Option("default", "--project", "-p"),
    experiment_type: str = typer.Option("comparison", "--type", help="comparison | sweep | ablation"),
    method: Optional[str] = typer.Option(None, "--method", "-m", help="Constant method for every row."),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Constant dataset for every row."),
    method_col: str = typer.Option("method", "--method-col"),
    dataset_col: str = typer.Option("dataset", "--dataset-col"),
    seed_col: str = typer.Option("seed", "--seed-col"),
    instance_col: str = typer.Option("instance", "--instance-col"),
    metric_col: list[str] = typer.Option([], "--metric-col", help="Columns that are metrics (repeatable). Default: numeric columns."),
    config_col: list[str] = typer.Option([], "--config-col", help="Columns that are config (repeatable). Default: the rest."),
    tag: list[str] = typer.Option([], "--tag"),
    source: str = typer.Option("imported", "--source", help="imported | reported"),
    keep_duplicates: bool = typer.Option(False, "--keep-duplicates", help="Import rows even if an identical run exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse and count, write nothing."),
    db: Optional[Path] = DbOpt,
):
    """Bulk-import existing results from CSV or JSON."""
    from .importer import ImportSpec, import_path, normalize, read_records

    spec = ImportSpec(
        experiment=experiment, project=project, experiment_type=experiment_type, method=method, dataset=dataset,
        method_col=method_col, dataset_col=dataset_col, seed_col=seed_col, instance_col=instance_col,
        metric_cols=metric_col, config_cols=config_col, tags=tag, source=source, skip_duplicates=not keep_duplicates,
    )
    if dry_run:
        raws = read_records(path)
        console.print(f"[cyan]{len(raws)} rows[/] in {path}. First row maps to:")
        if raws:
            console.print_json(json.dumps(normalize(raws[0], spec), default=str))
        res = import_path(path, spec, db=db, dry_run=True)
        console.print(f"would import {res.imported}, skip {res.skipped}")
        return
    res = import_path(path, spec, db=db)
    console.print(f"[green]{res}[/] -> {project}/{experiment}")
    for e in res.errors[:10]:
        console.print(f"[red]{e}[/]")


def _free_port(start: int, tries: int = 50) -> int:
    import socket

    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise typer.BadParameter(f"no free port in {start}-{start + tries}")


def _wait_for_port(port: int, timeout: float = 20.0) -> bool:
    import socket
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.25)
    return False


@app.command()
def ui(
    db: Optional[Path] = DbOpt,
    port: int = typer.Option(8501, "--port", help="Preferred port; the next free one is used if busy."),
    headless: bool = typer.Option(False, "--headless", help="Don't open a browser tab."),
):
    """Launch the Streamlit GUI."""
    import webbrowser

    try:
        import streamlit  # noqa: F401
    except ImportError:
        console.print("[red]Streamlit is not installed.[/] Run: pip install 'results-tracker[ui]'")
        raise typer.Exit(code=1)
    chosen = _free_port(port)
    if chosen != port:
        console.print(f"[yellow]port {port} is busy, using {chosen}[/]")
    app_path = Path(__file__).parent / "ui" / "app.py"
    env = {**os.environ, "RESULTS_TRACKER_DB": resolve_db_path(db)}
    # Always run headless: it skips Streamlit's first-run email prompt. We open the browser ourselves.
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(chosen), "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    url = f"http://localhost:{chosen}"
    console.print(f"[green]starting GUI[/] at {url}  (db: {env['RESULTS_TRACKER_DB']})  Ctrl-C to stop")
    proc = subprocess.Popen(cmd, env=env)
    try:
        if not headless and _wait_for_port(chosen):
            webbrowser.open(url)
        raise typer.Exit(code=proc.wait())
    except KeyboardInterrupt:
        proc.terminate()
        raise typer.Exit(code=0)



# --------------------------------------------------------------------------- export

def _load_experiment(experiment: str, project: Optional[str], db: Optional[Path]):
    engine = get_engine(db)
    recs = run_records(get_runs(experiment=experiment, project=project, engine=engine), engine=engine)
    if not recs:
        console.print(f"[red]no runs found for experiment {experiment!r}[/]")
        raise typer.Exit(code=1)
    defs = {k: {"unit": m.unit, "higher_is_better": m.higher_is_better, "fmt": m.fmt}
            for k, m in get_metric_defs(engine=engine).items()}
    return recs, defs


def _emit(text: str, out: Optional[Path]) -> None:
    if out is None:
        typer.echo(text, nl=False)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    console.print(f"[green]wrote[/] {out}")


def _env(env: str) -> Optional[str]:
    return None if env in ("none", "tabular") else env


def _save_fig(fig, out: Path, png: bool, tex: bool = False, width: str = "single", label: Optional[str] = None) -> None:
    from .export.figures import figure_tex, save_figure

    for pth in save_figure(fig, out, also_png=png):
        console.print(f"[green]wrote[/] {pth}")
    if tex:
        t = out.with_suffix(".tex")
        t.write_text(figure_tex(out.name, caption="TODO", label=label or f"fig:{out.stem}", width=_width(width)))
        console.print(f"[green]wrote[/] {t}  (figure environment; edit the caption)")


ExpOpt = typer.Option(..., "--experiment", "-e")
ProjOpt = typer.Option(None, "--project", "-p")
OutOpt = typer.Option(None, "--out", "-o", help="Output file; prints to stdout when omitted.")
EnvOpt = typer.Option("table", "--env", help="table | table* | none (bare tabular)")
StdOpt = typer.Option("pm", "--std", help="pm | small | none")
FontOpt = typer.Option(None, "--font", help="e.g. small, footnotesize")
WidthOpt = typer.Option("single", "--width", help="single (5.0 in) | double (10.5 in) | ieee-single (3.5) | ieee-double (7.16) | inches")
CaptionOpt = typer.Option(None, "--panel-label", help='Bold caption under the panel, e.g. "a. PSNR vs lambda"')
PngOpt = typer.Option(False, "--png", help="Also write a 300-dpi PNG next to the PDF.")
TexOpt = typer.Option(False, "--tex", help="Also write a figure/figure* environment snippet next to the file.")


def _width(w: str):
    try:
        return float(w)
    except ValueError:
        return w


@export_app.command("table")
def export_table(
    experiment: str = ExpOpt,
    project: Optional[str] = ProjOpt,
    rows: str = typer.Option("method", "--rows", help="Row key: method, dataset, config.<k>"),
    cols: str = typer.Option("dataset", "--cols", help="Column-group key, or 'none'."),
    metric: list[str] = typer.Option([], "--metric", help="Metrics (repeatable). Default: all."),
    caption: Optional[str] = typer.Option(None, "--caption"),
    label: Optional[str] = typer.Option(None, "--label"),
    env: str = EnvOpt,
    font: Optional[str] = FontOpt,
    std: str = StdOpt,
    no_underline: bool = typer.Option(False, "--no-underline", help="Don't underline second best."),
    out: Optional[Path] = OutOpt,
    db: Optional[Path] = DbOpt,
):
    """Comparison table: rows x (column group x metric), best bold, second underlined."""
    from . import aggregate as agg
    from .export.latex import comparison_latex, provenance_note, width_hint

    recs, defs = _load_experiment(experiment, project, db)
    col_key = None if cols == "none" else cols
    pt = agg.pivot_table(recs, rows, col_key, metrics=metric or None,
                         higher_is_better={k: v["higher_is_better"] for k, v in defs.items()})
    audit = agg.audit_grid(recs, [rows] + ([col_key] if col_key else []))
    if audit.missing or audit.failed or audit.uneven:
        err_console.print(f"[yellow]audit:[/] {audit.summary()}")
    hint = width_hint(pt, std, _env(env), font)
    if hint:
        err_console.print(f"[yellow]width:[/] {hint}")
    text = comparison_latex(
        pt, defs, caption=caption, label=label, env=_env(env), font=font, std=std,
        underline_second=not no_underline, row_labels=agg.method_labels(recs) if rows == "method" else None,
        audit=audit, provenance=provenance_note(resolve_db_path(db), experiment, len(recs)),
    )
    _emit(text, out)


@export_app.command("ablation-table")
def export_ablation_table(
    experiment: str = ExpOpt,
    project: Optional[str] = ProjOpt,
    metric: list[str] = typer.Option([], "--metric"),
    caption: Optional[str] = typer.Option(None, "--caption"),
    label: Optional[str] = typer.Option(None, "--label"),
    env: str = EnvOpt,
    font: Optional[str] = FontOpt,
    std: str = StdOpt,
    no_delta: bool = typer.Option(False, "--no-delta"),
    no_settings: bool = typer.Option(False, "--no-settings", help="Omit the per-setting ✓/✗ columns."),
    out: Optional[Path] = OutOpt,
    db: Optional[Path] = DbOpt,
):
    """Ablation table: variants x (settings, metrics with deltas vs the full model)."""
    from . import aggregate as agg
    from .export.latex import ablation_latex, provenance_note

    recs, defs = _load_experiment(experiment, project, db)
    rows = agg.ablation_table(recs, metrics=metric or None)
    metrics = metric or (list(rows[0].stats) if rows else [])
    text = ablation_latex(rows, metrics, defs, caption=caption, label=label, env=_env(env), font=font, std=std,
                          show_delta=not no_delta, setting_columns=not no_settings,
                          provenance=provenance_note(resolve_db_path(db), experiment, len(recs)))
    _emit(text, out)


@export_app.command("sweep-table")
def export_sweep_table(
    experiment: str = ExpOpt,
    param: str = typer.Option(..., "--param"),
    metric: str = typer.Option(..., "--metric"),
    project: Optional[str] = ProjOpt,
    by: list[str] = typer.Option([], "--by"),
    param_label: Optional[str] = typer.Option(None, "--param-label", help=r"LaTeX label, e.g. '$\lambda$'"),
    caption: Optional[str] = typer.Option(None, "--caption"),
    label: Optional[str] = typer.Option(None, "--label"),
    env: str = EnvOpt,
    std: str = StdOpt,
    out: Optional[Path] = OutOpt,
    db: Optional[Path] = DbOpt,
):
    """Sweep table: swept values x groups, best per column in bold."""
    from . import aggregate as agg
    from .export.latex import provenance_note, sweep_latex

    recs, defs = _load_experiment(experiment, project, db)
    series = agg.sweep_series(recs, param, metric, group_by=by)
    text = sweep_latex(series, param, metric, defs, caption=caption, label=label, env=_env(env), std=std,
                       param_label=param_label, provenance=provenance_note(resolve_db_path(db), experiment, len(recs)))
    _emit(text, out)


@export_app.command("sweep-fig")
def export_sweep_fig(
    experiment: str = ExpOpt,
    param: str = typer.Option(..., "--param"),
    metric: str = typer.Option(..., "--metric"),
    out: Path = typer.Option(..., "--out", "-o", help=".pdf (vector) or .png"),
    project: Optional[str] = ProjOpt,
    by: list[str] = typer.Option([], "--by", help="One line per group (method, dataset)."),
    xlabel: Optional[str] = typer.Option(None, "--xlabel", help=r"e.g. '$\lambda$'"),
    ylabel: Optional[str] = typer.Option(None, "--ylabel", help="e.g. 'PSNR (dB)'"),
    width: str = WidthOpt,
    height: Optional[float] = typer.Option(None, "--height", help="inches"),
    error_bars: bool = typer.Option(False, "--error-bars", help="Capped error bars instead of the shaded ± std band."),
    emphasize: list[str] = typer.Option([], "--emphasize", help="Group label(s) drawn heavier (the proposed method), e.g. Ours"),
    panel_label: Optional[str] = CaptionOpt,
    png: bool = PngOpt,
    tex: bool = TexOpt,
    db: Optional[Path] = DbOpt,
):
    """Metric vs swept parameter (IEEE single/double column, PDF)."""
    from . import aggregate as agg
    from .export.figures import sweep_figure

    recs, defs = _load_experiment(experiment, project, db)
    series = agg.sweep_series(recs, param, metric, group_by=by)
    hib = defs.get(metric, {}).get("higher_is_better", True)
    best = {g: agg.best_sweep_value(s, hib) for g, s in series.items()}
    unit = defs.get(metric, {}).get("unit", "")
    fig = sweep_figure(series, param, metric, xlabel=xlabel, ylabel=ylabel or (f"{metric} ({unit})" if unit else metric),
                       band=not error_bars, best_by_group=best, width=_width(width), height=height,
                       emphasize=emphasize, caption=panel_label)
    _save_fig(fig, out, png, tex, width)


@export_app.command("ablation-fig")
def export_ablation_fig(
    experiment: str = ExpOpt,
    metric: str = typer.Option(..., "--metric"),
    out: Path = typer.Option(..., "--out", "-o"),
    project: Optional[str] = ProjOpt,
    xlabel: Optional[str] = typer.Option(None, "--xlabel"),
    width: str = WidthOpt,
    height: Optional[float] = typer.Option(None, "--height"),
    panel_label: Optional[str] = CaptionOpt,
    png: bool = PngOpt,
    tex: bool = TexOpt,
    db: Optional[Path] = DbOpt,
):
    """Horizontal bars of each variant's change vs the full model."""
    from . import aggregate as agg
    from .export.figures import ablation_figure

    recs, defs = _load_experiment(experiment, project, db)
    rows = agg.ablation_table(recs, metrics=[metric])
    d = defs.get(metric, {})
    fig = ablation_figure(rows, metric, higher_is_better=d.get("higher_is_better", True), fmt=d.get("fmt", ".2f"),
                          xlabel=xlabel, width=_width(width), height=height, caption=panel_label)
    _save_fig(fig, out, png, tex, width)


@export_app.command("comparison-fig")
def export_comparison_fig(
    experiment: str = ExpOpt,
    metric: str = typer.Option(..., "--metric"),
    out: Path = typer.Option(..., "--out", "-o"),
    project: Optional[str] = ProjOpt,
    rows: str = typer.Option("method", "--rows"),
    cols: str = typer.Option("dataset", "--cols"),
    ylabel: Optional[str] = typer.Option(None, "--ylabel"),
    width: str = WidthOpt,
    height: Optional[float] = typer.Option(None, "--height"),
    emphasize: list[str] = typer.Option([], "--emphasize"),
    zero_based: bool = typer.Option(False, "--zero-based", help="Start the y axis at 0 (default: data-tight)."),
    ylim: Optional[str] = typer.Option(None, "--ylim", help="lo,hi y-axis limits"),
    hatch: bool = typer.Option(False, "--hatch", help="Add hatching to bars (grayscale print safety)."),
    panel_label: Optional[str] = CaptionOpt,
    png: bool = PngOpt,
    tex: bool = TexOpt,
    db: Optional[Path] = DbOpt,
):
    """Grouped bars: x = column key, one bar per row entity, error bar = std. Missing cells stay empty."""
    from . import aggregate as agg
    from .export.figures import comparison_figure

    recs, defs = _load_experiment(experiment, project, db)
    pt = agg.pivot_table(recs, rows, None if cols == "none" else cols, metrics=[metric],
                         higher_is_better={k: v["higher_is_better"] for k, v in defs.items()})
    unit = defs.get(metric, {}).get("unit", "")
    lim = tuple(float(v) for v in ylim.split(",")) if ylim else None
    fig = comparison_figure(pt, metric, ylabel=ylabel or (f"{metric} ({unit})" if unit else metric),
                            width=_width(width), height=height, emphasize=emphasize, zero_based=zero_based, ylim=lim,
                            hatch=hatch, caption=panel_label,
                            row_labels=agg.method_labels(recs) if rows == "method" else None)
    _save_fig(fig, out, png, tex, width)



@export_app.command("visual")
def export_visual(
    experiment: str = ExpOpt,
    out: Path = typer.Option(..., "--out", "-o", help=".pdf or .png (a .json provenance sidecar is written too)"),
    project: Optional[str] = ProjOpt,
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    instance: Optional[str] = typer.Option(None, "--instance"),
    image: str = typer.Option("reconstruction.png", "--image", help="File name inside each run's artifacts_dir."),
    reference: Optional[str] = typer.Option(None, "--reference", help="Ground-truth file name (in a run dir) or absolute path."),
    measurement: Optional[str] = typer.Option(None, "--measurement", help="Degraded input file name or path."),
    kernel: Optional[str] = typer.Option(None, "--kernel", help="Kernel/PSF thumbnail shown on the Measurement panel."),
    method: list[str] = typer.Option([], "--method", "-m", help="Methods in display order (default: baselines first)."),
    metric: list[str] = typer.Option(["psnr", "ssim"], "--metric", help="Metrics stamped on each panel."),
    mode: str = typer.Option("image", "--mode", help="image | error (|luminance error| on a pooled scale)"),
    zoom: bool = typer.Option(False, "--zoom", help="Yellow zoom box + lower-right magnified inset on every panel."),
    zoom_fraction: float = typer.Option(0.30, "--zoom-fraction", help="Box side as a fraction of the short side."),
    zoom_center: str = typer.Option("0.5,0.5", "--zoom-center", help="cx,cy in [0,1]"),
    crop: Optional[str] = typer.Option(None, "--crop", help="Explicit x,y,w,h zoom box (overrides fraction/centre)."),
    rows: Optional[str] = typer.Option(None, "--rows", help="One row per value of: seed | instance | config.<key>"),
    width: str = typer.Option("double", "--width", help="double (IEEE text width 7.16 in) | single (3.5 in) | inches"),
    tex: bool = TexOpt,
    db: Optional[Path] = DbOpt,
):
    """Qualitative comparison in the lab's IEEE style: GT | Measurement | baselines | proposed (+ zoom / error)."""
    from . import aggregate as agg
    from .export.figures import figure_tex
    from .export.visual import build_panels, build_rows, reconstruction_figure, save_visual

    recs, defs = _load_experiment(experiment, project, db)
    pool = agg.completed(recs)
    if dataset is not None:
        pool = [r for r in pool if r.get("dataset") == dataset]
    if instance is not None and rows != "instance":
        pool = [r for r in pool if r.get("instance") == instance]
    if seed is not None and rows != "seed":
        pool = [r for r in pool if r.get("seed") == seed]
    chosen = agg.select_runs(pool, methods=method or None)
    if not chosen:
        console.print("[red]no completed runs match[/]")
        raise typer.Exit(code=1)
    shown = [r for r in pool if r.get("artifacts_dir")] if rows else chosen
    for label, why in agg.omitted_methods(recs, shown, dataset=dataset).items():
        err_console.print(f"[yellow]not shown:[/] {label} — {why}", soft_wrap=True)
    if rows:
        # reference block from any run directory; method panels come from build_rows
        aux, ref_panel, problems = build_panels(shown[:1], image, defs, reference=reference, measurement=measurement, kernel=kernel)
        problems = [pr for pr in problems if pr.startswith(("reference", "measurement", "kernel"))]
        row_specs, probs = build_rows(pool, rows, image, defs, metrics=metric, methods=method or None, reference=reference)
        problems += probs
        panel_arg = row_specs
    else:
        aux, ref_panel, problems = build_panels(chosen, image, defs, metrics=metric, reference=reference,
                                                measurement=measurement, kernel=kernel)
        panel_arg = [p for p in aux if p.kind == "method"]
    meas_panel = next((p for p in aux if p.kind == "measurement"), None)
    ker_panel = next((p for p in aux if p.kind == "kernel"), None)
    for pr in problems:
        err_console.print(f"[yellow]{pr}[/]", soft_wrap=True)
    if not panel_arg:
        raise typer.Exit(code=1)
    box = tuple(int(v) for v in crop.split(",")) if crop else None
    if box is not None and len(box) != 4:
        raise typer.BadParameter("--crop expects x,y,w,h")
    cx, cy = (float(v) for v in zoom_center.split(","))
    fig, spec = reconstruction_figure(panel_arg, reference=ref_panel, measurement=meas_panel, kernel=ker_panel,
                                      mode=mode, zoom=zoom, zoom_fraction=zoom_fraction, zoom_center=(cx, cy),
                                      crop_box=box, width=_width(width))
    spec.experiment, spec.dataset, spec.instance, spec.seed, spec.image = experiment, dataset, instance, seed, image
    for pth in save_visual(fig, out, spec):
        console.print(f"[green]wrote[/] {pth}")
    typer.echo("caption material: " + spec.caption_stub())
    if tex:
        t = out.with_suffix(".tex")
        t.write_text(figure_tex(out.name, caption=spec.caption_stub(), label=f"fig:{out.stem}", width=_width(width)))
        console.print(f"[green]wrote[/] {t}")


@export_app.command("preamble")
def export_preamble():
    """Print the LaTeX packages the generated tables and figures need."""
    from .export.figures import ieee_preamble

    typer.echo(ieee_preamble(), nl=False)


@export_app.command("runs-csv")
def export_runs_csv(
    experiment: str = ExpOpt,
    project: Optional[str] = ProjOpt,
    out: Optional[Path] = OutOpt,
    db: Optional[Path] = DbOpt,
):
    """Every run as one CSV row (config keys and metrics as columns)."""
    from .export.csv import runs_csv

    recs, _ = _load_experiment(experiment, project, db)
    _emit(runs_csv(recs), out)



if __name__ == "__main__":  # pragma: no cover
    app()
