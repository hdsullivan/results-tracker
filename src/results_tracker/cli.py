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
metric_app = typer.Typer(help="Define how metrics are displayed and ranked.")
app.add_typer(metric_app, name="metric")
console = Console()

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
def demo(db: Optional[Path] = DbOpt):
    """Populate the database with a synthetic paper (comparison + sweep + ablation)."""
    from .demo import PROJECT, seed_demo

    counts = seed_demo(db=db)
    console.print(f"[green]seeded project '{PROJECT}'[/] into {resolve_db_path(db)}: {counts}")
    console.print("try:  results-tracker table -e main-comparison --by method --by dataset")


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


if __name__ == "__main__":  # pragma: no cover
    app()
