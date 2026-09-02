"""HTML renderings of results tables in the IEEEtran/booktabs look, for the GUI.

Mirrors export/latex.py one-to-one (same ranking, same highlighting, same captions) so the on-screen
table is a faithful preview of the compiled one: Times serif, white paper, small-caps caption above,
\\toprule / \\midrule / \\bottomrule only (no vertical rules), \\cmidrule under column groups,
bold best, underlined second, `—` for missing cells.
"""

from __future__ import annotations

import html
import re
from typing import Any, Mapping, Optional, Sequence

from .. import aggregate as agg
from ..export.latex import default_comparison_caption, display_metric_name

MetricDefs = Mapping[str, Mapping[str, Any]]

CSS = """
<style>
.ieee-paper { background:#fff; color:#000; display:inline-block; padding:16px 26px 14px 26px; margin:4px 0 10px 0;
              border:1px solid #d8d8d8; border-radius:2px; max-width:100%; overflow-x:auto; }
.ieee-paper * { font-family:"Times New Roman", Times, STIXGeneral, serif; }
.ieee-cap { text-align:center; font-variant:small-caps; font-size:13px; letter-spacing:.02em; line-height:1.35;
            margin:0 auto 8px auto; max-width:560px; }
.ieee-cap .num { display:block; margin-bottom:3px; }
table.ieee { border-collapse:collapse; margin:0 auto; font-size:15px; }
table.ieee th, table.ieee td { padding:3px 11px; text-align:center; white-space:nowrap; border:0; font-weight:normal; }
table.ieee th { font-weight:normal; }
table.ieee td:first-child, table.ieee th:first-child { text-align:left; }
table.ieee thead tr.top th { border-top:1.6px solid #000; padding-top:5px; }
table.ieee thead tr.head th { border-bottom:1px solid #000; padding-bottom:4px; }
table.ieee tbody tr:last-child td { border-bottom:1.6px solid #000; padding-bottom:5px; }
table.ieee th.group span { display:block; border-bottom:.8px solid #000; margin:0 6px; padding-bottom:2px; }
table.ieee td.base { font-weight:normal; }
table.ieee small { font-size:12px; }
table.ieee .std { font-size:15px; }
.ieee-figcap { font-size:14px; line-height:1.4; text-align:justify; margin:6px auto 0 auto; max-width:900px; }
.ieee-figcap b { font-weight:bold; }
</style>
"""

_ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def _esc(x: Any) -> str:
    return html.escape(str(x))


def delatex(text: str) -> str:
    """Turn the LaTeX captions from export/latex into readable text for the screen."""
    t = text
    t = re.sub(r"\$n=(.+?)\$", lambda m: "n = " + m.group(1).replace("--", "–"), t)
    for a, b in [("$\\pm$", "±"), ("$\\uparrow$", "↑"), ("$\\downarrow$", "↓"), ("$\\times$", "×"),
                 ("\\checkmark{}", "✓"), ("\\checkmark", "✓"), ("$\\rightarrow$", "→"), ("\\_", "_"), ("\\&", "&"),
                 ("\\%", "%"), ("--", "–"), ("$", "")]:
        t = t.replace(a, b)
    return t


def _num(v: float, fmt: str) -> str:
    s = format(v, fmt)
    return s.replace("-", "−")


def fmt_stat(st: Optional[agg.Stat], fmt: str, show_std: bool = True) -> str:
    if st is None:
        return "—"
    s = _num(st.mean, fmt)
    if show_std and st.n > 1:
        s += f' <span class="std">± {format(st.std, fmt)}</span>'
    return s


def header(metric: str, defs: MetricDefs, arrows: bool = True) -> str:
    d = defs.get(metric, {})
    name = _esc(display_metric_name(metric))
    if d.get("unit"):
        name += f" ({_esc(d['unit'])})"
    if arrows:
        name += " ↑" if d.get("higher_is_better", True) else " ↓"
    return name


def _mark(s: str, best: bool, second: bool, underline_second: bool = True) -> str:
    if best:
        return f"<b>{s}</b>"
    if second and underline_second:
        return f"<u>{s}</u>"
    return s


def _wrap(table_html: str, caption: Optional[str], number: Optional[int]) -> str:
    cap = ""
    if caption:
        num = f'<span class="num">TABLE {_ROMAN[(number or 1) - 1]}</span>' if number else ""
        cap = f'<div class="ieee-cap">{num}{_esc(caption)}</div>'
    return CSS + f'<div class="ieee-paper">{cap}{table_html}</div>'


# --------------------------------------------------------------------------- comparison (pivot)

def comparison_html(
    pt: agg.PivotTable,
    defs: MetricDefs,
    *,
    caption: Optional[str] = None,
    number: Optional[int] = 1,
    show_std: bool = True,
    underline_second: bool = True,
    row_labels: Optional[Mapping[Any, str]] = None,
    row_header: Optional[str] = None,
    show_n: bool = False,
) -> str:
    nm = len(pt.metrics)
    grouped = pt.col_key is not None and not (len(pt.cols) == 1 and pt.cols[0] is None)
    rows_html: list[str] = []
    if grouped:
        cells = ["<th></th>"] + [f'<th class="group" colspan="{nm}"><span>{_esc(c)}</span></th>' for c in pt.cols]
        if show_n:
            cells.append("<th></th>")
        rows_html.append(f'<tr class="top">{"".join(cells)}</tr>')
    head = [f"<th>{_esc(row_header or pt.row_key.capitalize())}</th>"]
    for _ in pt.cols:
        head += [f"<th>{header(m, defs)}</th>" for m in pt.metrics]
    if show_n:
        head.append("<th>n</th>")
    rows_html.append(f'<tr class="{"head" if grouped else "top head"}">{"".join(head)}</tr>')
    body: list[str] = []
    for r in pt.rows:
        cells = [f"<td>{_esc((row_labels or {}).get(r, r))}</td>"]
        ns = []
        for c in pt.cols:
            for m in pt.metrics:
                st = pt.stat(r, c, m)
                s = fmt_stat(st, defs.get(m, {}).get("fmt", ".2f"), show_std)
                if st is not None:
                    ns.append(st.n)
                    s = _mark(s, pt.is_best(r, c, m), pt.is_second(r, c, m), underline_second)
                cells.append(f"<td>{s}</td>")
        if show_n:
            cells.append(f"<td>{max(ns) if ns else 0}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="ieee"><thead>{"".join(rows_html)}</thead><tbody>{"".join(body)}</tbody></table>'
    if caption is None:
        caption = delatex(default_comparison_caption(pt, defs, "pm" if show_std else "none", underline_second))
    return _wrap(table, caption, number)


# --------------------------------------------------------------------------- flat (any grouping)

def flat_html(
    ct: agg.ComparisonTable,
    defs: MetricDefs,
    *,
    caption: Optional[str] = None,
    number: Optional[int] = 1,
    show_std: bool = True,
    show_n: bool = True,
    underline_second: bool = True,
) -> str:
    """Rows = arbitrary group keys joined with ' / ', columns = metrics. Used when there are 3+ keys."""
    head = [f"<th>{_esc(' / '.join(ct.group_by))}</th>"] + [f"<th>{header(m, defs)}</th>" for m in ct.metrics]
    if show_n:
        head.append("<th>n</th>")
    body = []
    for row in ct.rows:
        cells = [f"<td>{_esc(ct.row_label(row))}</td>"]
        for m in ct.metrics:
            st = ct.cells[row].get(m)
            s = fmt_stat(st, defs.get(m, {}).get("fmt", ".2f"), show_std)
            if st is not None:
                s = _mark(s, ct.is_best(row, m), ct.is_second(row, m), underline_second)
            cells.append(f"<td>{s}</td>")
        if show_n:
            cells.append(f"<td>{max((c.n for c in ct.cells[row].values() if c), default=0)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="ieee"><thead><tr class="top head">{"".join(head)}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    if caption is None:
        ms = ", ".join(delatex(header(m, defs, arrows=False)) for m in ct.metrics)
        caption = f"{ms} by {' and '.join(ct.group_by)}. Mean{' ± std' if show_std else ''} over the remaining runs. Best in bold" + (", second best underlined." if underline_second else ".")
    return _wrap(table, caption, number)


# --------------------------------------------------------------------------- ablation

def ablation_html(
    rows: Sequence[agg.AblationRow],
    metrics: Sequence[str],
    defs: MetricDefs,
    *,
    caption: Optional[str] = None,
    number: Optional[int] = 2,
    show_std: bool = True,
    relative: bool = False,
    setting_columns: bool = True,
    base_label: str = "Full model",
) -> str:
    keys = sorted({k for r in rows for k in r.diff}) if setting_columns else []
    base_vals = {k: next(r.diff[k][0] for r in rows if k in r.diff) for k in keys}

    def setting(v: Any) -> str:
        return "✓" if v is True else ("×" if v is False else ("—" if v is None else _esc(v)))

    ranks = {
        m: agg.rank_values([(r.stats[m].mean, i) for i, r in enumerate(rows) if r.stats.get(m) is not None],
                           defs.get(m, {}).get("higher_is_better", True))
        for m in metrics
    }
    thead = []
    if keys:
        thead.append(f'<tr class="top"><th></th><th class="group" colspan="{len(keys)}"><span>Setting</span></th>'
                     f'<th class="group" colspan="{len(metrics)}"><span>Result</span></th></tr>')
    head = ["<th>Variant</th>"] + [f"<th>{_esc(k)}</th>" for k in keys] + [f"<th>{header(m, defs)}</th>" for m in metrics]
    thead.append(f'<tr class="{"head" if keys else "top head"}">{"".join(head)}</tr>')
    body = []
    for i, r in enumerate(rows):
        cells = [f"<td>{_esc(base_label) if r.is_base else _esc(r.label)}</td>"]
        cells += [f"<td>{setting(r.diff[k][1] if k in r.diff else base_vals[k])}</td>" for k in keys]
        for m in metrics:
            st = r.stats.get(m)
            fmt = defs.get(m, {}).get("fmt", ".2f")
            s = fmt_stat(st, fmt, show_std)
            if st is not None:
                s = _mark(s, ranks[m].get(i) == 1, False, False)
                if not r.is_base and r.delta.get(m) is not None:
                    if relative:
                        rd = r.rel_delta(m)
                        d = "" if rd is None else f"{rd * 100:+.1f}%"
                    else:
                        d = format(r.delta[m], f"+{fmt}")
                    if d:
                        s += f" <small>({d.replace('-', '−')})</small>"
            cells.append(f"<td>{s}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="ieee"><thead>{"".join(thead)}</thead><tbody>{"".join(body)}</tbody></table>'
    if caption is None:
        ns = sorted({r.n for r in rows})
        n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
        caption = ("Ablation of the full model. Each row changes one setting"
                   + (" (✓ on, × off)" if any(isinstance(v, bool) for v in base_vals.values()) else "")
                   + f". Mean{' ± std' if show_std else ''} over {n_txt} runs per row; parentheses give the change "
                   + ("in percent of the full model" if relative else "relative to the full model") + ". Best per column in bold.")
    return _wrap(table, caption, number)


# --------------------------------------------------------------------------- sweep

def sweep_html(
    series_by_group: Mapping[tuple, Sequence[tuple[Any, agg.Stat]]],
    param: str,
    metric: str,
    defs: MetricDefs,
    *,
    caption: Optional[str] = None,
    number: Optional[int] = 3,
    show_std: bool = True,
    param_label: Optional[str] = None,
) -> str:
    hib = defs.get(metric, {}).get("higher_is_better", True)
    fmt = defs.get(metric, {}).get("fmt", ".2f")
    groups = list(series_by_group)
    xs = sorted({x for s in series_by_group.values() for x, _ in s}, key=lambda x: (isinstance(x, str), x))
    lookups = {g: dict(s) for g, s in series_by_group.items()}
    best = {g: agg.best_sweep_value(list(series_by_group[g]), hib) for g in groups}
    single = groups == [()]
    thead = []
    if not single:
        thead.append(f'<tr class="top"><th></th><th class="group" colspan="{len(groups)}"><span>{header(metric, defs)}</span></th></tr>')
    head = [f"<th>{_esc(param_label or param)}</th>"]
    head += [f"<th>{header(metric, defs)}</th>"] if single else [f"<th>{_esc(' / '.join(map(str, g)))}</th>" for g in groups]
    thead.append(f'<tr class="{"top head" if single else "head"}">{"".join(head)}</tr>')
    body = []
    for x in xs:
        cells = [f"<td>{_esc(f'{x:g}' if isinstance(x, float) else x)}</td>"]
        for g in groups:
            st = lookups[g].get(x)
            s = fmt_stat(st, fmt, show_std)
            if st is not None and best[g] == x:
                s = f"<b>{s}</b>"
            cells.append(f"<td>{s}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="ieee"><thead>{"".join(thead)}</thead><tbody>{"".join(body)}</tbody></table>'
    if caption is None:
        ns = sorted({st.n for s in series_by_group.values() for _, st in s})
        n_txt = f"n = {ns[0]}" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]}"
        caption = (f"{delatex(header(metric, defs, arrows=False))} as a function of {param_label or param}. "
                   f"Mean{' ± std' if show_std else ''} over {n_txt} runs per value. Best in bold ({'higher' if hib else 'lower'} is better).")
    return _wrap(table, caption, number)


# --------------------------------------------------------------------------- generic

def generic_html(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    caption: Optional[str] = None,
    number: Optional[int] = None,
    left_cols: int = 1,
    raw_html_cols: Sequence[int] = (),
) -> str:
    """Any small table in the same booktabs look: first `left_cols` columns left-aligned, rest centred.
    Cells are escaped unless their column index is in `raw_html_cols`."""
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for r in rows:
        cells = []
        for i, v in enumerate(r):
            txt = str(v) if i in raw_html_cols else _esc("—" if v is None else v)
            style = ' style="text-align:left"' if i < left_cols else ""
            cells.append(f"<td{style}>{txt}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    table = f'<table class="ieee"><thead><tr class="top head">{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
    return _wrap(table, caption, number)


def figure_caption_html(text: str, number: int = 1) -> str:
    """IEEEtran-style figure caption ("Fig. 1. ...") in the paper look, placed under a rendered figure."""
    return CSS + f'<div class="ieee-paper" style="display:block"><div class="ieee-figcap"><b>Fig. {number}.</b> {_esc(text)}</div></div>'
