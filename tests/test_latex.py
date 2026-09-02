from results_tracker import aggregate as agg
from results_tracker.export.latex import (
    ablation_latex, comparison_latex, fmt_signed, fmt_stat, latex_escape, sweep_latex, width_hint,
)

DEFS = {
    "psnr": {"unit": "dB", "higher_is_better": True, "fmt": ".2f"},
    "rmse": {"unit": "", "higher_is_better": False, "fmt": ".3f"},
}


def rec(method, dataset, seed, cfg=None, status="completed", tags=(), **metrics):
    return {"run_id": 0, "method": method, "method_label": method, "dataset": dataset, "seed": seed,
            "config": cfg or {}, "metrics": metrics, "status": status, "tags": list(tags)}


def comparison_records():
    recs = []
    for s in range(3):
        recs += [
            rec("TV", "Set12", s, psnr=27.0 + s * 0.1, rmse=0.30),
            rec("TV", "CBSD68", s, psnr=26.0 + s * 0.1, rmse=0.35),
            rec("Ours", "Set12", s, psnr=31.0 + s * 0.1, rmse=0.10),
            rec("Ours", "CBSD68", s, psnr=30.0 + s * 0.1, rmse=0.12),
            rec("PnP", "Set12", s, psnr=29.0 + s * 0.1, rmse=0.20),
        ]
    return recs  # PnP never run on CBSD68 -> missing cell


def test_helpers():
    assert latex_escape("warm_start & 5%") == r"warm\_start \& 5\%"
    assert latex_escape(r"TV~\cite{rudin}") == r"TV~\cite{rudin}"  # already LaTeX: untouched
    assert latex_escape("denoiser: drunet→dncnn") == r"denoiser: drunet$\rightarrow$dncnn"
    assert fmt_signed(-0.63) == "$-$0.63" and fmt_signed(0.5) == "$+$0.50"
    st = agg.summarize([1.0, 1.2])
    assert fmt_stat(st, ".2f", "pm") == "1.10 $\\pm$ 0.14"
    assert fmt_stat(st, ".2f", "small") == "1.10 {\\scriptsize$\\pm$0.14}"
    assert fmt_stat(st, ".2f", "none") == "1.10"
    assert fmt_stat(None, ".2f") == "--"
    assert fmt_stat(agg.summarize([-0.5]), ".1f") == "$-$0.5"


def test_comparison_latex_structure_and_highlighting():
    recs = comparison_records()
    pt = agg.pivot_table(recs, "method", "dataset", higher_is_better={"psnr": True, "rmse": False})
    audit = agg.audit_grid(recs, ["method", "dataset"])
    tex = comparison_latex(pt, DEFS, label="tab:x", audit=audit, provenance="prov line")
    assert tex.startswith("% prov line")
    assert "\\begin{table}[!t]" in tex and "\\end{table}" in tex and "\\label{tab:x}" in tex
    assert "\\toprule" in tex and "\\midrule" in tex and "\\bottomrule" in tex
    assert "\\multicolumn{2}{c}{CBSD68}" in tex and "\\multicolumn{2}{c}{Set12}" in tex
    assert "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}" in tex
    assert "PSNR (dB) $\\uparrow$" in tex and "RMSE $\\downarrow$" in tex
    # Ours best on both metrics in both datasets, TV second on Set12 psnr is PnP... check bold/underline per column
    ours_row = next(l for l in tex.splitlines() if l.startswith("Ours"))
    assert ours_row.count("\\textbf") == 4
    pnp_row = next(l for l in tex.splitlines() if l.startswith("PnP"))
    assert "\\underline{29.10" in pnp_row  # second best psnr on Set12
    assert pnp_row.count("--") == 2  # missing CBSD68 cells
    assert "missing: method=PnP, dataset=CBSD68" in tex
    # caption auto-generated with n
    assert "\\caption{" in tex and "$n=3$" in tex and "Best in bold" in tex


def test_comparison_latex_bare_tabular_and_labels():
    recs = comparison_records()
    pt = agg.pivot_table(recs, "method", None, metrics=["psnr"])
    tex = comparison_latex(pt, DEFS, env=None, std="none", row_labels={"TV": r"TV~\cite{rudin}", "Ours": "Ours_v2"},
                           underline_second=False)
    assert "\\begin{table}" not in tex and "\\caption" not in tex
    assert "\\multicolumn" not in tex
    assert r"TV~\cite{rudin} & " in tex and r"Ours\_v2 & \textbf{" in tex
    assert "$\\pm$" not in tex and "\\underline" not in tex
    # ranking across datasets combined: Ours mean over both datasets is best
    assert next(l for l in tex.splitlines() if l.startswith("Ours")).count("\\textbf") == 1


def test_ablation_latex():
    base = {"adaptive": True, "denoiser": "drunet"}
    recs = []
    for s in range(2):
        recs.append(rec("Ours", "D", s, cfg=base, tags=["base"], psnr=31.0, rmse=0.10))
        recs.append(rec("Ours", "D", s, cfg={**base, "adaptive": False}, psnr=30.0, rmse=0.12))
        recs.append(rec("Ours", "D", s, cfg={**base, "denoiser": "dncnn"}, psnr=30.5, rmse=0.11))
    rows = agg.ablation_table(recs)
    tex = ablation_latex(rows, ["psnr", "rmse"], DEFS)
    assert "\\multicolumn{2}{c}{Setting}" in tex and "\\multicolumn{2}{c}{Result}" in tex
    full = next(l for l in tex.splitlines() if l.startswith("Full model"))
    assert "\\checkmark" in full and "drunet" in full and full.count("\\textbf") == 2
    noad = next(l for l in tex.splitlines() if l.startswith("w/o adaptive"))
    assert "$\\times$" in noad and "{\\scriptsize($-$1.00)}" in noad and "{\\scriptsize($+$0.020)}" in noad
    assert "\\checkmark{} on, $\\times$ off" in tex
    tex2 = ablation_latex(rows, ["psnr"], DEFS, setting_columns=False, show_delta=False, env=None)
    assert "Setting" not in tex2 and "scriptsize" not in tex2


def test_sweep_latex():
    recs = [rec("Ours", "D", s, cfg={"lambda": lam}, psnr=30 - abs(lam - 0.1) * 10 + 0.01 * s)
            for lam in [0.01, 0.1, 1.0] for s in range(2)]
    series = agg.sweep_series(recs, "lambda", "psnr")
    tex = sweep_latex(series, "lambda", "psnr", DEFS, param_label=r"$\lambda$", env=None)
    lines = tex.splitlines()
    assert lines[0] == "\\begin{tabular}{l c}"
    assert "$\\lambda$ & PSNR (dB) $\\uparrow$ \\\\" in tex
    best_line = next(l for l in lines if l.startswith("0.1 "))
    assert "\\textbf{" in best_line
    assert sum("\\textbf" in l for l in lines) == 1
    # grouped
    recs2 = recs + [rec("TV", "D", 0, cfg={"lambda": lam}, psnr=25.0) for lam in [0.01, 0.1, 1.0]]
    tex2 = sweep_latex(agg.sweep_series(recs2, "lambda", "psnr", group_by=["method"]), "lambda", "psnr", DEFS)
    assert "\\multicolumn{2}{c}{PSNR (dB) $\\uparrow$}" in tex2 and "Ours & TV" in tex2


def test_width_hint():
    recs = comparison_records()
    wide = agg.pivot_table(recs, "method", "dataset", metrics=["psnr", "rmse", "psnr"])  # 6 columns
    assert width_hint(wide, "pm", "table", None) is not None
    assert width_hint(wide, "pm", "table*", None) is None
    assert width_hint(wide, "none", "table", "footnotesize") is None
    narrow = agg.pivot_table(recs, "method", None, metrics=["psnr", "rmse"])
    assert width_hint(narrow, "pm", "table", None) is None
