from results_tracker import aggregate as agg
from results_tracker.ui.tables import ablation_html, comparison_html, delatex, flat_html, sweep_html

DEFS = {"psnr": {"unit": "dB", "higher_is_better": True, "fmt": ".2f"},
        "rmse": {"unit": "", "higher_is_better": False, "fmt": ".3f"}}


def rec(method, dataset, seed, cfg=None, tags=(), **metrics):
    return {"run_id": 0, "method": method, "method_label": method, "dataset": dataset, "seed": seed,
            "config": cfg or {}, "metrics": metrics, "status": "completed", "tags": list(tags)}


def recs():
    out = []
    for s in range(3):
        out += [rec("TV", "Set12", s, psnr=27.0 + 0.1 * s, rmse=0.30), rec("TV", "CBSD68", s, psnr=26.0, rmse=0.35),
                rec("Ours", "Set12", s, psnr=31.0 + 0.1 * s, rmse=0.10), rec("Ours", "CBSD68", s, psnr=30.0, rmse=0.12),
                rec("PnP", "Set12", s, psnr=29.0, rmse=0.20)]
    return out


def test_comparison_html_structure():
    pt = agg.pivot_table(recs(), "method", "dataset", higher_is_better={"psnr": True, "rmse": False})
    h = comparison_html(pt, DEFS, row_labels={"TV": "TV [1]"})
    assert 'class="ieee-paper"' in h and "TABLE I" in h and 'class="ieee-cap"' in h
    assert h.count('class="group" colspan="2"') == 2 and "<span>Set12</span>" in h  # cmidrule groups
    assert 'tr class="top"' in h and 'tr class="head"' in h  # toprule + midrule
    assert "PSNR (dB) ↑" in h and "RMSE ↓" in h
    ours = h[h.index("<td>Ours</td>"):]
    ours = ours[:ours.index("</tr>")]
    assert ours.count("<b>") == 4  # best in every column
    pnp = h[h.index("<td>PnP</td>"):]
    pnp = pnp[:pnp.index("</tr>")]
    assert pnp.count("<td>—</td>") == 2 and "<u>29.00" in pnp  # missing cells, second best psnr on Set12
    assert "TV [1]" in h
    assert "n = 3" in h and "±" in h and "$" not in h  # caption de-LaTeXed
    # no std, no underline, no numbering
    h2 = comparison_html(pt, DEFS, show_std=False, underline_second=False, number=None)
    assert "±" not in h2.split("</div>")[1] and "<u>" not in h2 and "TABLE" not in h2


def test_flat_html_and_delatex():
    ct = agg.comparison_table(recs(), group_by=["method", "dataset", "seed"], metrics=["psnr"])
    h = flat_html(ct, DEFS)
    assert "method / dataset / seed" in h and "<td>Ours / Set12 / 2</td>" in h
    assert delatex("Mean $\\pm$ std over $n=3$ runs; warm\\_start $\\uparrow$") == "Mean ± std over n = 3 runs; warm_start ↑"


def test_ablation_html():
    base = {"adaptive": True, "denoiser": "drunet"}
    rs = []
    for s in range(2):
        rs.append(rec("Ours", "D", s, cfg=base, tags=["base"], psnr=31.0, rmse=0.10))
        rs.append(rec("Ours", "D", s, cfg={**base, "adaptive": False}, psnr=30.0, rmse=0.12))
    rows = agg.ablation_table(rs)
    h = ablation_html(rows, ["psnr", "rmse"], DEFS)
    assert "<span>Setting</span>" in h and "<span>Result</span>" in h
    assert "<td>Full model</td>" in h and "<td>✓</td>" in h and "<td>×</td>" in h
    assert "<small>(−1.00)</small>" in h and "<small>(+0.020)</small>" in h
    assert h.count("<b>") == 2  # best per metric column (the full model on both)
    h2 = ablation_html(rows, ["psnr"], DEFS, relative=True)
    assert "<small>(−3.2%)</small>" in h2 and "in percent of the full model" in h2


def test_sweep_html():
    rs = [rec("Ours", "D", s, cfg={"lambda": lam}, psnr=30 - abs(lam - 0.1) * 10 + 0.01 * s)
          for lam in [0.01, 0.1, 1.0] for s in range(2)]
    series = agg.sweep_series(rs, "lambda", "psnr")
    h = sweep_html(series, "lambda", "psnr", DEFS, param_label="λ")
    assert "<th>λ</th>" in h and "<th>PSNR (dB) ↑</th>" in h
    assert h.count("<b>") == 1 and "<td>0.1</td>" in h
    grouped = agg.sweep_series(rs + [rec("TV", "D", 0, cfg={"lambda": l}, psnr=25.0) for l in [0.01, 0.1, 1.0]],
                               "lambda", "psnr", group_by=["method"])
    h2 = sweep_html(grouped, "lambda", "psnr", DEFS)
    assert 'colspan="2"' in h2 and "<th>Ours</th>" in h2 and "<th>TV</th>" in h2
