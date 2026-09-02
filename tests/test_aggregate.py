import math

from results_tracker import aggregate as agg


def rec(method, seed, cfg=None, status="completed", tags=(), **metrics):
    return {"run_id": hash((method, seed, str(cfg))) % 10000, "method": method, "dataset": "D",
            "seed": seed, "config": cfg or {}, "metrics": metrics, "status": status, "tags": list(tags)}


def test_summarize_mean_std_and_none_handling():
    st = agg.summarize([1.0, 2.0, 3.0, None])
    assert st.n == 3 and st.mean == 2.0 and math.isclose(st.std, 1.0)
    assert st.min == 1.0 and st.max == 3.0
    assert agg.summarize([5.0]).std == 0.0
    assert agg.summarize([None, None]) is None
    assert agg.summarize([1.234, 2.345]).format(".1f") == "1.8 ± 0.8"
    assert agg.summarize([1.234]).format(".1f") == "1.2"


def test_flatten_and_get_field():
    r = rec("m", 0, cfg={"a": 1, "solver": {"iters": 50, "tol": 1e-3}})
    assert agg.flatten(r["config"]) == {"a": 1, "solver.iters": 50, "solver.tol": 1e-3}
    assert agg.get_field(r, "method") == "m"
    assert agg.get_field(r, "config.solver.iters") == 50
    assert agg.get_field(r, "config.missing") is None
    r["metrics"] = {"psnr": 30}
    assert agg.get_field(r, "metrics.psnr") == 30


def test_comparison_table_ranks_with_direction_and_excludes_failed():
    recs = [
        rec("A", 0, psnr=30.0, rmse=0.10), rec("A", 1, psnr=31.0, rmse=0.12),
        rec("B", 0, psnr=32.0, rmse=0.20), rec("B", 1, psnr=33.0, rmse=0.22),
        rec("C", 0, psnr=28.0, rmse=0.30),
        rec("C", 1, psnr=99.0, rmse=0.0, status="failed"),
    ]
    ct = agg.comparison_table(recs, group_by=["method"], higher_is_better={"psnr": True, "rmse": False})
    assert ct.rows == [("A",), ("B",), ("C",)]
    assert ct.metrics == ["psnr", "rmse"]
    assert ct.cells[("A",)]["psnr"].mean == 30.5 and ct.cells[("A",)]["psnr"].n == 2
    assert ct.cells[("C",)]["psnr"].n == 1  # failed run excluded
    assert ct.is_best(("B",), "psnr") and ct.is_second(("A",), "psnr")
    assert ct.is_best(("A",), "rmse") and ct.is_second(("B",), "rmse")
    assert ct.rank["psnr"][("C",)] == 3


def test_comparison_table_ties_share_rank_and_multi_group():
    recs = [rec("A", 0, psnr=30.0), rec("B", 0, psnr=30.0), rec("C", 0, psnr=29.0)]
    for r in recs:
        r["dataset"] = "D1"
    ct = agg.comparison_table(recs, group_by=["method", "dataset"])
    assert ct.rank["psnr"][("A", "D1")] == 1 and ct.rank["psnr"][("B", "D1")] == 1
    assert ct.rank["psnr"][("C", "D1")] == 3
    assert ct.row_label(("A", "D1")) == "A / D1"


def test_sweep_series_sorted_and_aggregated():
    recs = []
    for lam in [1.0, 0.01, 0.1]:
        for s in range(2):
            recs.append(rec("ours", s, cfg={"lambda": lam}, psnr=30 - abs(math.log10(lam / 0.1)) + s * 0.1))
    recs.append(rec("ours", 9, cfg={}, psnr=0.0))  # no lambda -> ignored
    series = agg.sweep_series(recs, "lambda", "psnr")[()]
    assert [x for x, _ in series] == [0.01, 0.1, 1.0]
    assert all(st.n == 2 for _, st in series)
    assert agg.best_sweep_value(series, higher_is_better=True) == 0.1
    assert agg.best_sweep_value(series, higher_is_better=False) in (0.01, 1.0)


def test_sweep_series_grouped():
    recs = [rec(m, 0, cfg={"k": k}, psnr=v) for m, k, v in [("A", 1, 1.0), ("A", 2, 2.0), ("B", 1, 3.0)]]
    out = agg.sweep_series(recs, "k", "psnr", group_by=["method"])
    assert set(out) == {("A",), ("B",)}
    assert [x for x, _ in out[("A",)]] == [1, 2]


def test_config_diff_and_describe():
    base = {"denoiser": "drunet", "adaptive": True, "solver": {"iters": 50}}
    other = {"denoiser": "dncnn", "adaptive": False, "solver": {"iters": 50}, "extra": 1}
    d = agg.config_diff(base, other)
    assert d == {"adaptive": (True, False), "denoiser": ("drunet", "dncnn"), "extra": (None, 1)}
    assert agg.describe_diff(d) == "w/o adaptive; denoiser: drunet→dncnn; + extra=1"
    assert agg.describe_diff({}) == "full model"


def test_ablation_table_deltas_with_tagged_base():
    base = {"denoiser": "drunet", "adaptive": True, "warm": True}
    recs = []
    for s in range(3):
        recs.append(rec("ours", s, cfg=base, tags=["base"], psnr=31.0 + 0.1 * s))
        recs.append(rec("ours", s, cfg={**base, "adaptive": False}, psnr=30.0 + 0.1 * s))
        recs.append(rec("ours", s, cfg={**base, "denoiser": "dncnn"}, psnr=30.5 + 0.1 * s))
    rows = agg.ablation_table(recs, metrics=["psnr"])
    assert rows[0].is_base and rows[0].label == "full model" and rows[0].n == 3
    by_label = {r.label: r for r in rows}
    assert math.isclose(by_label["w/o adaptive"].delta["psnr"], -1.0)
    assert math.isclose(by_label["denoiser: drunet→dncnn"].delta["psnr"], -0.5)
    assert rows[0].delta["psnr"] == 0.0


def test_ablation_table_falls_back_to_most_common_config():
    base = {"a": 1, "b": 2}
    recs = [rec("m", s, cfg=base, psnr=10.0) for s in range(3)] + [rec("m", 0, cfg={"a": 1, "b": 3}, psnr=9.0)]
    rows = agg.ablation_table(recs)
    assert rows[0].is_base and rows[0].n == 3
    assert rows[1].label == "b: 2→3" and math.isclose(rows[1].delta["psnr"], -1.0)


def test_ablation_table_explicit_base_config_without_base_runs():
    recs = [rec("m", 0, cfg={"a": 1}, psnr=9.0)]
    rows = agg.ablation_table(recs, base_config={"a": 0})
    assert len(rows) == 1 and not rows[0].is_base and rows[0].delta["psnr"] is None
    assert agg.ablation_table([]) == []


def test_sweep_grid_and_best():
    recs = []
    for lam in [0.01, 0.1, 1.0]:
        for it in [10, 50]:
            for s in range(2):
                val = 30 - abs(math.log10(lam / 0.1)) + (0.5 if it == 50 else 0) + 0.01 * s
                recs.append(rec("m", s, cfg={"lambda": lam, "iters": it}, psnr=val))
    recs.append(rec("m", 0, cfg={"lambda": 0.1}, psnr=99.0))  # missing iters -> skipped
    grid = agg.sweep_grid(recs, "lambda", "iters", "psnr")
    assert grid.xs == [0.01, 0.1, 1.0] and grid.ys == [10, 50]
    assert grid.best(True) == (0.1, 50)
    assert grid.best(False) in ((0.01, 10), (1.0, 10))
    m = grid.matrix()
    assert len(m) == 2 and len(m[0]) == 3
    assert grid.cells[(0.1, 50)].n == 2
    assert math.isclose(m[1][1], 30.505)


def test_varying_config_keys():
    recs = [rec("m", 0, cfg={"a": 1, "b": "x", "c": {"d": 1}}), rec("m", 1, cfg={"a": 2, "b": "x", "c": {"d": 2}})]
    assert agg.varying_config_keys(recs) == ["a", "c.d"]


def test_ablation_relative_delta():
    base = {"a": True}
    recs = [rec("m", s, cfg=base, tags=["base"], psnr=20.0) for s in range(2)]
    recs += [rec("m", s, cfg={"a": False}, psnr=18.0) for s in range(2)]
    rows = agg.ablation_table(recs, metrics=["psnr"])
    var = next(r for r in rows if not r.is_base)
    assert math.isclose(var.delta["psnr"], -2.0)
    assert math.isclose(var.rel_delta("psnr"), -0.1)
    assert rows[0].rel_delta("psnr") == 0.0


def test_pivot_table_ranks_within_each_column():
    recs = []
    for s in range(2):
        recs += [
            rec("A", s, psnr=30.0), rec("B", s, psnr=31.0),
        ]
    for r in recs:
        r["dataset"] = "D1"
    recs += [{**rec("A", 0, psnr=35.0), "dataset": "D2"}, {**rec("B", 0, psnr=20.0), "dataset": "D2"}]
    pt = agg.pivot_table(recs, "method", "dataset", metrics=["psnr"])
    assert pt.rows == ["A", "B"] and pt.cols == ["D1", "D2"]
    assert pt.is_best("B", "D1", "psnr") and pt.is_second("A", "D1", "psnr")
    assert pt.is_best("A", "D2", "psnr") and not pt.is_best("B", "D2", "psnr")
    assert pt.stat("A", "D1", "psnr").n == 2 and pt.n_values() == {1, 2}
    pt2 = agg.pivot_table(recs, "method", None, metrics=["psnr"], row_order=["B", "A"])
    assert pt2.rows == ["B", "A"] and pt2.cols == [None]
    assert pt2.is_best("A", None, "psnr")  # A mean over everything = (30+30+35)/3 > B


def test_audit_grid():
    recs = [rec("A", 0, psnr=1), rec("A", 1, psnr=1), rec("B", 0, psnr=1), rec("B", 1, psnr=1, status="failed")]
    for r in recs:
        r["dataset"] = "D1"
    recs.append({**rec("A", 0, psnr=1), "dataset": "D2"})
    a = agg.audit_grid(recs, ["method", "dataset"])
    assert a.expected == 4 and a.present == 3
    assert a.missing == [("B", "D2")]
    assert a.failed == {("B", "D1"): 1}
    assert a.uneven and a.n_per_cell[("A", "D1")] == 2
    assert "1 missing" in a.summary() and "1 failed run" in a.summary()


def test_rank_values_ties():
    assert agg.rank_values([(1.0, "a"), (2.0, "b"), (2.0, "c"), (0.5, "d")]) == {"b": 1, "c": 1, "a": 3, "d": 4}
    assert agg.rank_values([(1.0, "a"), (2.0, "b")], higher_is_better=False) == {"a": 1, "b": 2}


def test_method_labels():
    recs = [{"method": "TV", "method_label": "TV [1]"}, {"method": "Ours", "method_label": "Ours"}, {"method": None}]
    assert agg.method_labels(recs) == {"TV": "TV [1]", "Ours": "Ours"}
