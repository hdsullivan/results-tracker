import json
import os
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from results_tracker.demo import seed_demo  # noqa: E402


@pytest.fixture
def demo_db(tmp_path, monkeypatch):
    db = tmp_path / "demo.db"
    seed_demo(db=db, artifacts_dir=str(tmp_path / "art"))
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    st.cache_data.clear()
    st.cache_resource.clear()
    return db


def _run(page: str):
    at = AppTest.from_string(f"from results_tracker.ui import {page}\n{page}.render()\n", default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_overview_page(demo_db):
    at = _run("overview")
    labels = [m.label for m in at.metric]
    assert "Projects" in labels and "Runs" in labels
    values = {m.label: m.value for m in at.metric}
    assert values["Runs"] == "46"
    md = "\n".join(m.value for m in at.markdown)
    assert md.count('class="ieee-paper"') == 3  # experiments, at a glance, recent runs
    assert "TABLE I" in md and "TABLE II" in md and "TABLE III" in md
    assert "best method: Ours" in md and "best lambda = 0.1" in md and "largest drop: w/o adaptive" in md
    assert "(+1 failed)" in md  # the diverged sweep run is counted, not hidden
    assert "&lt;span" not in md and '<span class="std">' in md  # std markup rendered, not escaped
    assert len(at.dataframe) == 2  # sortable grids still available in the expander


def test_comparison_page_table_and_chart(demo_db):
    at = _run("comparison")
    # make the comparison experiment explicit (the selector's own default is checked separately)
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "TV" in md and "PnP-BM3D" in md and "Ours" in md
    assert "<b>" in md and 'class="ieee-paper"' in md  # IEEE-look table with a bolded best
    assert "PSNR (dB) ↑" in md
    # group by method and dataset -> dataset column groups (cmidrules)
    rows_box = [ms for ms in at.sidebar.multiselect if ms.label == "Rows grouped by"][0]
    rows_box.set_value(["method", "dataset"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "<span>Set12</span>" in md and "<span>CBSD68</span>" in md
    # three keys -> flat layout
    rows_box = [ms for ms in at.sidebar.multiselect if ms.label == "Rows grouped by"][0]
    rows_box.set_value(["method", "dataset", "seed"]).run()
    assert not at.exception
    assert "Ours / Set12 / 0" in "\n".join(m.value for m in at.markdown)


def test_comparison_page_empty_db(tmp_path, monkeypatch):
    """An existing but empty database: the sidebar says there is nothing to select yet."""
    from results_tracker.db import get_engine

    db = tmp_path / "empty.db"
    get_engine(db)  # deliberately created, the way the button below does it
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _run("comparison")
    assert any("No projects yet" in i.value for i in at.sidebar.info)


def test_a_database_that_is_not_there_is_never_created_behind_your_back(tmp_path, monkeypatch):
    """A typo in the Database box used to make an empty database (file and tables), which on screen is
    indistinguishable from a database whose runs have gone missing."""
    missing = tmp_path / "not-mounted" / "resluts.db"
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(missing))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _run("overview")
    assert any("does not exist" in w.value for w in at.warning)
    assert not missing.exists() and not missing.parent.exists()
    assert not at.dataframe and not at.metric  # the page stops rather than reporting an empty database

    create = [b for b in at.button if "Create an empty database" in b.label][0]
    create.click().run()
    assert not at.exception and missing.is_file()
    assert any("Empty database" in i.value for i in at.info)  # now it really is one


def test_run_detail_page(demo_db, tmp_path):
    at = _run("run_detail")
    assert at.selectbox  # run picker exists
    labels = [m.label for m in at.metric]
    assert "Status" in labels and any(lbl.startswith("psnr") for lbl in labels)
    assert any("Config" in h.value for h in at.subheader)
    # pick another run to compare against and check a diff / metric table renders
    assert len(at.dataframe) >= 1
    # artifacts render as the lab-style figure strip (this run + compared run)
    caption = "\n".join(c.value for c in at.caption)
    assert "IEEE text width" in caption and "logged metrics" in caption
    assert not at.error and not at.exception
    # error-map mode
    [r for r in at.radio if r.label == "Mode"][0].set_value("Error map").run()
    assert not at.exception and not at.error
    assert "error scale" in "\n".join(c.value for c in at.caption)


def test_sweep_page(demo_db):
    at = _run("sweep")
    md = "\n".join(m.value for m in at.markdown)
    assert "<th>lambda</th>" in md and "<b>" in md  # best value bolded
    assert "<b>Fig. 1.</b>" in md and "stays within" in md  # caption with the sensitivity statement
    assert "TABLE II" in md and "plateau" in md  # sensitivity table
    caption = "\n".join(c.value for c in at.caption)
    assert "best lambda = **0.1**" in caption
    assert any("\\toprule" in c.value for c in at.code)  # LaTeX expander
    assert not at.warning


def test_sweep_page_heatmap_mode(demo_db):
    at = _run("sweep")
    box = [sb for sb in at.sidebar.selectbox if sb.label.startswith("Second parameter")][0]
    # only lambda varies in the demo sweep, so a 2nd parameter gives a 1-row heatmap; still must not crash
    box.select(box.options[1]).run()
    assert not at.exception
    caption = "\n".join(c.value for c in at.caption)
    assert "best at lambda=0.1" in caption
    md = "\n".join(m.value for m in at.markdown)
    assert "<b>Fig. 1.</b>" in md and 'class="ieee-paper"' in md and "<b>31.26" in md  # paper-look grid with best bold


def test_ablation_page(demo_db):
    at = _run("ablation")
    md = "\n".join(m.value for m in at.markdown)
    assert "<td>Full model</td>" in md and "w/o adaptive" in md and "<td>×</td>" in md
    assert "<b>Fig. 1.</b>" in md and "Clearly needed" in md  # caption with verdicts
    assert "TABLE II" in md and "Cohen" in md and "hurts · clear" in md  # effect-size table
    assert any("\\toprule" in c.value for c in at.code) and any("figure" in c.value for c in at.code)
    caption = "\n".join(c.value for c in at.caption)
    assert "Largest drop" in caption and "w/o adaptive" in caption
    # relative deltas
    [cb for cb in at.sidebar.checkbox if "%" in cb.label][0].check().run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "%)</small>" in md


def test_export_page_latex_and_figure(demo_db):
    at = _run("export")
    # default experiment "ablation" -> ablation table kind preselected
    code = "\n".join(c.value for c in at.code)
    assert "\\begin{tabular}" in code and "\\checkmark" in code
    assert at.sidebar.radio[0].value == "Ablation table (LaTeX)"
    at.sidebar.radio[0].set_value("Ablation figure").run()
    assert not at.exception
    assert not at.warning and not at.error
    at.sidebar.radio[0].set_value("Runs (CSV)").run()
    assert not at.exception
    assert any("runs (including failed)" in c.value for c in at.caption)


def test_export_page_comparison_audit(demo_db):
    at = _run("export")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    assert not at.exception
    assert at.sidebar.radio[0].value == "Comparison table (LaTeX)"
    # DPIR is reported on Set12 only -> one missing cell, flagged before the LaTeX
    assert any("7/8 cells present" in w.value and "1 missing" in w.value for w in at.warning)
    assert "DPIR" in "\n".join(c.value for c in at.code)
    code = "\n".join(c.value for c in at.code)
    assert "\\multicolumn{3}{c}{Set12}" in code and "TV [1]" in code
    assert any('class="ieee-paper"' in m.value for m in at.markdown)  # rendered preview above the LaTeX


def test_visual_page(demo_db):
    at = _run("visual")
    assert not at.error
    caption = "\n".join(c.value for c in at.caption)
    assert "IEEE text width" in caption
    body = "\n".join(m.value for m in at.markdown) + "\n".join(str(t.value) for t in at.text)
    assert "Left to right: Reference, Measurement, TV [1], PnP-BM3D [2], Ours" in body and "Yellow box" in body
    # error-map mode
    at.radio[0].set_value("Error maps").run()
    assert not at.exception and not at.error
    assert "luminance" in "\n".join(m.value for m in at.markdown)
    # DPIR has no artifacts -> warned, not crashed
    assert any("DPIR" in w.value for w in at.warning)
    # grayscale + labels toggles re-render
    [cb for cb in at.checkbox if cb.label == "Grayscale preview"][0].check().run()
    assert not at.exception
    # two comparisons stacked: independent method selections, two numbered figures and metric tables
    at = _run("visual")
    [ni for ni in at.sidebar.number_input if ni.label.startswith("Comparisons stacked")][0].set_value(2).run()
    assert not at.exception and not at.error
    methods = [ms for ms in at.multiselect if ms.label.startswith("Methods")]
    assert len(methods) == 2
    methods[1].set_value(["PnP-BM3D", "Ours"]).run()
    assert not at.exception
    body = "\n".join(m.value for m in at.markdown)
    assert "Left to right: Reference, Measurement, TV [1], PnP-BM3D [2], Ours" in body  # comparison 1 unchanged
    assert "Left to right: Reference, Measurement, PnP-BM3D [2], Ours" in body  # comparison 2
    assert "Fig. 1" in body and "Fig. 2" in body
    assert [h.value for h in at.header] == ["Comparison 1", "Comparison 2"]
    assert len([sh for sh in at.subheader if sh.value == "Panel metrics"]) == 2


def test_export_page_visual_and_bundle(demo_db):
    at = _run("export")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    at.sidebar.radio[0].set_value("Visual comparison figure").run()
    assert not at.exception and not at.error
    body = "\n".join(str(t.value) for t in at.markdown) + "\n".join(str(t.value) for t in at.text)
    assert "Left to right: Reference, Measurement, TV [1], PnP-BM3D [2], Ours" in body
    assert any("DPIR" in w.value for w in at.warning)
    at.sidebar.radio[0].set_value("Runs (CSV)").run()
    assert any('class="ieee-paper"' in m.value for m in at.markdown)  # CSV preview in the paper look
    at.sidebar.radio[0].set_value("Paper bundle (zip)").run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception and not at.error
    md = "\n".join(m.value for m in at.markdown)
    assert "Contents of the paper bundle" in md and "comparison-table" in md and "visual-figure" in md


def test_run_detail_delete_button(demo_db):
    at = _run("run_detail")
    n_runs = len(at.selectbox[0].options)
    first = at.selectbox[0].value
    # button is disabled until the confirmation box is ticked
    btn = [b for b in at.button if b.label.startswith("Delete run")][0]
    assert btn.disabled
    [cb for cb in at.checkbox if cb.label.startswith("Yes, delete run")][0].check().run()
    btn = [b for b in at.button if b.label.startswith("Delete run")][0]
    assert not btn.disabled
    btn.click().run()
    assert not at.exception
    assert any("Deleted run #" in s.value for s in at.success)
    assert len(at.selectbox[0].options) == n_runs - 1 and first not in at.selectbox[0].options


@pytest.fixture
def toy_studies_db(tmp_path, monkeypatch):
    """A database holding one finished toy study and a studies folder with all three toy specs."""
    from results_tracker import get_engine
    from results_tracker.recipe import run_study
    from results_tracker.recipe.toy import toy_studies

    db = tmp_path / "toy.db"
    engine = get_engine(db)
    comparison, sweep, ablation = toy_studies(str(tmp_path / "art"))
    run_study(comparison, engine=engine, log=None)
    studies = tmp_path / "studies"
    for s in (comparison, sweep, ablation):
        s.save(studies / f"{s.name}.json")
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    monkeypatch.setenv("RESULTS_TRACKER_STUDIES", str(studies))
    st.cache_data.clear()
    st.cache_resource.clear()
    return db, studies


def test_studies_page_derives_progress_and_saves_new_specs(toy_studies_db):
    db, studies = toy_studies_db
    at = _run("studies")
    assert not at.error
    table = at.dataframe[0].value
    by_name = {row["experiment"]: row for _, row in table.iterrows()}
    assert by_name["main-comparison"]["progress"] == 1.0 and by_name["main-comparison"]["status"] == "done"
    assert by_name["main-comparison"]["runs done"] == 96 and by_name["main-comparison"]["jobs"] == 24
    assert by_name["reg-sweep"]["progress"] == 0.0 and by_name["reg-sweep"]["status"] == "planned"
    assert by_name["ablation"]["jobs"] == 16 and by_name["ablation"]["expected"] == 48
    # the grid for the selected study (first alphabetically: ablation) shows every cell missing
    md = "\n".join(m.value for m in at.markdown)
    assert "0/6" in md and "missing" in md and "recipe run" in "\n".join(c.value for c in at.code)

    # plan a new comparison from the declared knobs and save it
    at.text_input(key="new_name").input("smoke-plan").run()
    at.selectbox(key="new_arm_0_method").select("adaptive-gd").run()
    at.text_input(key="new_cond_blur").input("1.0, 2.0").run()
    at.text_input(key="new_arm_0_reg").input("0.01").run()  # away from the default 0.003 -> written to the spec
    assert not at.exception
    assert any("2 jobs" in s.value for s in at.success), [s.value for s in at.success]
    at.button(key="new_save").click().run()
    assert not at.exception
    spec = json.loads((studies / "smoke-plan.json").read_text())
    assert spec["conditions"] == {"blur": [1.0, 2.0], "noise": [0.02], "size": [64]} if "size" in spec["conditions"] else spec["conditions"]["blur"] == [1.0, 2.0]
    assert spec["methods"] == [{"method": "adaptive-gd", "config": {"reg": 0.01}}]
    assert spec["imports"] == ["results_tracker.recipe.toy"]
    # the saved plan shows up as planned on the next render
    st.cache_data.clear()
    at = _run("studies")
    names = set(at.dataframe[0].value["experiment"])
    assert "smoke-plan" in names


def _sidebar_multiselect(at, label):
    return [ms for ms in at.sidebar.multiselect if ms.label == label][0]


def test_selection_is_shared_and_mirrored_in_the_url(demo_db):
    # the URL seeds the selection ...
    at = AppTest.from_string("from results_tracker.ui import sweep\nsweep.render()\n", default_timeout=30)
    at.query_params["experiment"] = "main-comparison"
    at.run()
    assert not at.exception
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    assert exp_box.value.startswith("main-comparison")
    # ... a new choice is written back to the URL and to the shared session entry
    exp_box.select([o for o in exp_box.options if o.startswith("lambda-sweep")][0]).run()
    assert not at.exception
    assert at.query_params["experiment"] == ["lambda-sweep"] and at.query_params["project"] == ["demo-paper"]
    assert at.session_state["experiment_name"] == "lambda-sweep"
    assert "db" not in at.query_params  # the default database is not spelled out
    # ... and another page opens on the experiment chosen here
    at2 = AppTest.from_string("from results_tracker.ui import export\nexport.render()\n", default_timeout=30)
    at2.session_state["experiment_name"] = "main-comparison"
    at2.session_state["project_name"] = "demo-paper"
    at2.run()
    assert not at2.exception
    assert [sb for sb in at2.sidebar.selectbox if sb.label == "Experiment"][0].value.startswith("main-comparison")
    assert at2.sidebar.radio[0].value == "Comparison table (LaTeX)"


def test_where_filter_is_shared_and_matches_the_cli(demo_db):
    at = _run("comparison")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    fields = _sidebar_multiselect(at, "Filter on")
    assert "dataset" in fields.options and "status" not in fields.options  # constant fields are not offered
    assert "config.iters" in fields.options  # unset for the reported DPIR run, 50 elsewhere: that does vary
    fields.set_value(["dataset"]).run()
    assert not at.exception
    _sidebar_multiselect(at, "dataset").set_value(["Set12"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "Set12" not in md or "CBSD68" not in md  # one dataset left -> no dataset column groups
    caption = "\n".join(c.value for c in at.sidebar.caption) + "\n".join(c.value for c in at.caption)
    assert "runs match · dataset = Set12" in caption and "filter: dataset = Set12" in caption
    assert at.query_params["where"] == ["dataset=Set12"]  # same grammar as --where
    assert at.session_state["where"] == {"dataset": ["Set12"]}
    # the URL form is understood by the CLI parser and by another page
    from results_tracker import aggregate as agg
    assert agg.parse_where(at.query_params["where"]) == {"dataset": "Set12"}
    at2 = AppTest.from_string("from results_tracker.ui import export\nexport.render()\n", default_timeout=30)
    at2.query_params["experiment"] = "main-comparison"
    at2.query_params["where"] = ["dataset=Set12", "method=[\"TV\",\"Ours\"]"]
    at2.run()
    assert not at2.exception
    code = "\n".join(c.value for c in at2.code)
    assert "Filter: --where 'dataset=Set12' --where 'method=[\"Ours\",\"TV\"]'" in code  # provenance comment
    assert "PnP-BM3D" not in code and "TV [1]" in code
    assert any("Filtered to" in c.value for c in at2.caption)
    # a filter that leaves nothing is reported, not silently emptied
    at3 = AppTest.from_string("from results_tracker.ui import sweep\nsweep.render()\n", default_timeout=30)
    at3.query_params["experiment"] = "lambda-sweep"
    at3.query_params["where"] = "config.lambda=[0.1,0.3]"
    at3.run()
    assert not at3.exception
    assert any("6 of 15 runs match" in c.value for c in at3.sidebar.caption)  # 2 lambdas x 3 seeds, of 5 x 3 (one failed)
    _sidebar_multiselect(at3, "config.lambda").set_value([]).run()
    _sidebar_multiselect(at3, "Filter on").set_value(["seed"]).run()
    _sidebar_multiselect(at3, "seed").set_value(["0"]).run()
    assert not at3.exception
    assert at3.query_params["where"] == ["seed=0"]


def test_pin_paper_page_export_and_reopen(demo_db, tmp_path):
    from results_tracker import log_run

    # pin the filtered comparison table from the Comparison page
    at = _run("comparison")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    _sidebar_multiselect(at, "Filter on").set_value(["dataset"]).run()
    _sidebar_multiselect(at, "dataset").set_value(["Set12"]).run()
    assert at.text_input(key="cmp_pin_label").value == "tab:main-comparison"  # default label from the experiment
    at.text_input(key="cmp_pin_label").input("tab:main").run()
    at.button(key="cmp_pin_button").click().run()
    assert not at.exception and any("Pinned `tab:main`" in s_.value for s_ in at.success)
    # and a sweep figure from the Sweep page (two kinds offered there)
    at2 = _run("sweep")
    assert at2.selectbox(key="sweep_pin_kind").value == "sweep-figure"
    at2.button(key="sweep_pin_button").click().run()
    assert not at2.exception and any("fig:lambda-sweep" in s_.value for s_ in at2.success)

    # the Paper page lists both in manuscript order, never exported, with the filter
    at3 = _run("paper")
    md = "\n".join(m.value for m in at3.markdown)
    assert md.index("tab:main") < md.index("fig:lambda-sweep") and "dataset = Set12" in md and md.count("never exported") >= 2
    assert {m.label: m.value for m in at3.metric}["Assets"] == "2"
    assert 'href="export?project=demo-paper&asset=tab%3Amain"' in md  # the default database is not spelled out
    # bookkeeping through the form: status, position and a rename land in the database and the selection follows the new label
    [sb for sb in at3.selectbox if sb.label == "Status"][0].select("final").run()
    [ti for ti in at3.text_input if ti.label == "Label"][0].input("tab:main-renamed").run()
    [b for b in at3.button if b.label == "Save"][0].click().run()
    assert not at3.exception, at3.exception
    from results_tracker import get_asset as _get_asset
    assert _get_asset("demo-paper", "tab:main-renamed", db=demo_db).status.value == "final" and _get_asset("demo-paper", "tab:main", db=demo_db) is None
    assert at3.selectbox(key="paper_asset").value == "tab:main-renamed"
    [ti for ti in at3.text_input if ti.label == "Label"][0].input("tab:main").run()
    [b for b in at3.button if b.label == "Save"][0].click().run()
    assert not at3.exception and _get_asset("demo-paper", "tab:main", db=demo_db).status.value == "final"
    # export the paper into a directory: stable names, filter in the provenance, assets become current
    at3.text_input(key="paper_out_dir").input(str(tmp_path / "paper")).run()
    [b for b in at3.button if b.label == "Write to directory"][0].click().run()
    assert not at3.exception
    tex = (tmp_path / "paper" / "tables" / "tab-main.tex").read_text()
    assert "\\label{tab:main}" in tex and "Filter: --where 'dataset=Set12'" in tex and "CBSD68" not in tex
    assert (tmp_path / "paper" / "figures" / "fig-lambda-sweep.pdf").exists() and (tmp_path / "paper" / "MANIFEST.json").exists()
    md = "\n".join(m.value for m in at3.markdown)
    assert md.count(">current<") == 2 and any("2 assets rendered, 0 failed" in s_.value for s_ in at3.success)

    # opening an asset restores experiment, filter and options on the Export page
    at4 = AppTest.from_string("from results_tracker.ui import export\nexport.render()\n", default_timeout=30)
    at4.query_params["asset"] = "tab:main"
    at4.query_params["project"] = "demo-paper"
    at4.run()
    assert not at4.exception
    assert [sb for sb in at4.sidebar.selectbox if sb.label == "Experiment"][0].value.startswith("main-comparison")
    assert at4.session_state["where"] == {"dataset": ["Set12"]} and at4.query_params["where"] == ["dataset=Set12"]
    assert at4.sidebar.radio[0].value == "Comparison table (LaTeX)" and at4.text_input(key="exp_label").value == "tab:main"
    assert "asset" not in at4.query_params and any("opened from the Paper page" in c.value for c in at4.caption)
    assert at4.text_input(key="exp_pin_label").value == "tab:main"  # pinning again updates the same asset
    at4.selectbox(key="exp_std").select("small").run()
    at4.button(key="exp_pin_button").click().run()
    assert not at4.exception
    from results_tracker import get_asset
    a = get_asset("demo-paper", "tab:main", db=demo_db)
    assert a.options["std"] == "small" and a.filters == {"dataset": ["Set12"]} and a.exported_at is None  # re-pinned: export forgotten

    # new data -> stale on the Paper page and on the Overview
    log_run("main-comparison", project="demo-paper", method="Ours", dataset="Set12", seed=7, config={"lambda": 0.1, "iters": 50},
            metrics={"psnr": 31.0, "ssim": 0.9, "runtime_s": 6.0}, db=demo_db, git_commit=None)
    st.cache_data.clear()
    at5 = _run("paper")
    md = "\n".join(m.value for m in at5.markdown)
    assert ">never exported<" in md and ">current<" in md  # tab:main was re-pinned, fig:lambda-sweep is untouched
    at6 = _run("overview")
    assert any("Paper:" in c.value and "2 pinned assets" in c.value for c in at6.caption)


def test_export_kind_follows_experiment_unless_an_asset_is_open(demo_db):
    at = _run("export")
    assert at.sidebar.radio[0].value == "Ablation table (LaTeX)"
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("lambda-sweep")][0]).run()
    assert at.sidebar.radio[0].value == "Sweep figure"
    at.sidebar.radio[0].set_value("Sweep table (LaTeX)").run()
    assert at.sidebar.radio[0].value == "Sweep table (LaTeX)"  # an explicit choice sticks within the experiment


def test_page_url_carries_a_non_default_database(demo_db):
    # a link opens a new Streamlit session, so it must say which database it means unless that is the default
    at = AppTest.from_string(
        "import streamlit as st\nfrom results_tracker.ui.common import page_url\n"
        "st.write(page_url('export', project='p q', asset='tab:m'))\n"
        "st.session_state['db'] = '/x/other.db'\nst.write(page_url('export', project='p', asset='tab:m'))\n", default_timeout=30)
    at.run()
    assert not at.exception
    assert at.markdown[0].value == "export?project=p+q&asset=tab%3Am"
    assert at.markdown[1].value == "export?db=%2Fx%2Fother.db&project=p&asset=tab%3Am"


def test_studies_planning_layer(toy_studies_db):
    """Progress with running rows, compute-left estimate, pending-only spec, edit/clone, knob-driven ablation arms,
    feeds, and the knobs.json fallback for specs whose modules cannot be imported."""
    from results_tracker import get_asset, log_run, save_asset
    from results_tracker.recipe import expand, load_study_classes, registry, save_declarations
    from results_tracker.recipe.toy import PROJECT, toy_studies

    db, studies = toy_studies_db
    _, sweep, ablation = toy_studies()
    problem_cls, methods = load_study_classes(sweep)
    jobs = expand(sweep, problem_cls, methods)
    for inst in ("phantom_00", "phantom_01", "phantom_02"):  # the first job done ...
        log_run("reg-sweep", project=PROJECT, experiment_type="sweep", method=jobs[0].method, dataset="Phantoms", instance=inst,
                seed=jobs[0].seed, config={**jobs[0].condition, **jobs[0].config}, metrics={"psnr": 20.0, "runtime_s": 2.0}, db=db, git_commit=None)
    log_run("reg-sweep", project=PROJECT, method=jobs[1].method, dataset="Phantoms", instance="phantom_00", seed=jobs[1].seed,  # ... one in flight
            config={**jobs[1].condition, **jobs[1].config}, metrics={}, status="running", db=db, git_commit=None)
    spec = json.loads((studies / "reg-sweep.json").read_text())
    spec["feeds"] = ["fig:reg"]
    (studies / "reg-sweep.json").write_text(json.dumps(spec))
    save_asset(PROJECT, "fig:reg", kind="sweep-figure", experiment="reg-sweep", options={"param": "reg", "metric": "psnr"}, db=db)
    # a spec whose module does not exist here, plus the declarations that stand in for it
    decl = json.loads((studies / "ablation.json").read_text())
    decl.update(name="declared-ablation", imports=["no_such_module_xyz"])
    (studies / "declared.json").write_text(json.dumps(decl))
    import results_tracker.recipe.toy  # noqa: F401
    save_declarations(registry, studies / "knobs.json")
    st.cache_data.clear()

    at = _run("studies")
    table = at.dataframe[0].value
    by_name = {row["experiment"]: row for _, row in table.iterrows()}
    assert by_name["reg-sweep"]["runs done"] == 3 and by_name["reg-sweep"]["running"] == 1 and by_name["reg-sweep"]["status"] == "running"
    assert by_name["reg-sweep"]["time left"] == "84 s"  # (45 - 3) pending runs x median 2.0 s
    assert by_name["main-comparison"]["time left"] == "" and by_name["ablation"]["time left"] == "—"
    assert by_name["reg-sweep"]["feeds"] == "fig:reg"
    assert by_name["declared-ablation"]["status"] == "planned" and by_name["declared-ablation"]["jobs"] == 16  # expanded from knobs.json
    captions = "\n".join(c.value for c in at.caption)
    assert "Compute left: ~84 s" in captions and "no completed run to time yet" in captions and "knobs.json" in captions
    assert not at.warning  # a spec covered by knobs.json is not "not runnable"

    at.selectbox(key="studies_pick").select("reg-sweep.json").run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "0/9 ▶1" in md and "running" in md and "3/9" in md  # seeds pooled per cell: 3 seeds x 3 instances
    captions = "\n".join(c.value for c in at.caption)
    assert "Feeds paper assets: `fig:reg` (planned)" in captions
    assert "narrowed to the 14 unfinished jobs" in captions and "1 seed" not in captions  # 3 seeds remain, 4 sweep values
    assert "3 seed(s)" in captions and "5 sweep values" in captions  # every value still has pending seeds
    assert {m.label: m.value for m in at.metric}["time left"] == "84 s"

    # edit: the form is prefilled with the spec; clone: same with a new name and no file
    at.button(key="studies_edit").click().run()
    assert not at.exception
    assert at.text_input(key="new_name").value == "reg-sweep" and at.selectbox(key="new_kind").value == "sweep"
    assert at.selectbox(key="new_sweep_knob").value == "reg" and at.text_input(key="new_sweep_values").value == "0.0003, 0.001, 0.003, 0.01, 0.03"
    assert at.text_input(key="new_cond_blur").value == "1.5" and at.text_input(key="new_seeds").value == "0, 1, 2"
    assert at.text_input(key="new_feeds").value == "fig:reg" and at.text_input(key="new_filename").value == "reg-sweep.json"
    assert at.checkbox(key="new_overwrite").value is True and any("15 jobs" in s_.value for s_ in at.success)
    at.button(key="studies_clone").click().run()
    assert at.text_input(key="new_name").value == "reg-sweep-copy" and at.text_input(key="new_filename").value == ""

    # an ablation planned with knob widgets: one single-knob arm and one labelled joint arm
    at.text_input(key="new_name").input("abl-plan").run()
    at.selectbox(key="new_kind").select("ablation").run()
    at.text_input(key="new_feeds").input("fig:abl").run()
    at.number_input(key="new_abl_n").set_value(2).run()
    at.multiselect(key="new_abl_0_knobs").set_value(["adaptive"]).run()
    at.checkbox(key="new_abl_0_adaptive").uncheck().run()
    at.multiselect(key="new_abl_1_knobs").set_value(["prior", "warm_start"]).run()
    at.selectbox(key="new_abl_1_prior").select("tikhonov").run()
    at.checkbox(key="new_abl_1_warm_start").uncheck().run()
    at.text_input(key="new_abl_1_label").input("quadratic, cold start").run()
    assert not at.exception, at.exception
    assert any("9 jobs" in s_.value for s_ in at.success), [s_.value for s_ in at.success]  # (base + 2 arms) x the cloned 3 seeds
    at.button(key="new_save").click().run()
    saved = json.loads((studies / "abl-plan.json").read_text())
    assert saved["ablation"]["arms"] == [{"adaptive": False}, {"label": "quadratic, cold start", "set": {"prior": "tikhonov", "warm_start": False}}]
    assert saved["feeds"] == ["fig:abl"] and saved["kind"] == "ablation"

    # the Paper page reads readiness from the feeding study
    at2 = _run("paper")
    md = "\n".join(m.value for m in at2.markdown)
    assert "3/45 runs · 1 running" in md and "fig:reg" in md
    assert get_asset(PROJECT, "fig:reg", db=db).experiment == "reg-sweep"


def test_settings_page_drives_tables(demo_db):
    """A value map, a method order and a primary metric set on the Settings page show up on the other pages."""
    from results_tracker import get_metric_defs, list_methods, list_value_maps

    at = _run("settings")
    # value map: dataset -> size class, saved for the project
    at.text_input(key="set_vm_name").input("size").run()
    at.selectbox(key="set_vm_field").select("dataset").run()
    at.text_area(key="set_vm_rules").input("small = Set12\nlarge = CBSD68").run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert 'Set12</td><td style="text-align:left">small</td>' in md and 'CBSD68</td><td style="text-align:left">large</td>' in md  # mapping preview
    at.button(key="set_vm_save").click().run()
    assert not at.exception and any("derived.size" in s_.value for s_ in at.success)
    vm = list_value_maps("demo-paper", db=demo_db)[0]
    assert vm.name == "size" and vm.field == "dataset" and vm.rules[0] == {"label": "small", "values": ["Set12"]}
    # methods: Ours first; metrics: SSIM to 4 decimals
    at.number_input(key="set_method_Ours_pos").set_value(-1).run()
    at.button(key="set_methods_save").click().run()
    assert next(m for m in list_methods(db=demo_db) if m.name == "Ours").position == -1
    at.text_input(key="set_metric_ssim_fmt").input(".4f").run()
    at.button(key="set_metrics_save").click().run()
    assert get_metric_defs(db=demo_db)["ssim"].fmt == ".4f"
    at.selectbox(key="set_primary").select("ssim").run()
    at.button(key="set_project_save").click().run()
    assert not at.exception

    # Comparison: the derived field is a grouping key, columns follow rule order, rows follow method position
    at2 = _run("comparison")
    exp_box = [sb for sb in at2.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    rows_box = [ms for ms in at2.sidebar.multiselect if ms.label == "Rows grouped by"][0]
    assert "derived.size" in rows_box.options and "derived.size" in _sidebar_multiselect(at2, "Filter on").options
    rows_box.set_value(["method", "derived.size"]).run()
    assert not at2.exception
    md = "\n".join(m.value for m in at2.markdown)
    assert md.index("<span>small</span>") < md.index("<span>large</span>")  # rule order, not alphabetical
    table = md[md.index('class="ieee"'):]
    assert table.index("Ours") < table.index("TV")  # method position -1 puts Ours first
    assert "0.8500" in md  # SSIM now printed with 4 decimals
    # Overview headline uses the project's primary metric
    at3 = _run("overview")
    md = "\n".join(m.value for m in at3.markdown)
    assert "best method: Ours" in md and "SSIM" in md.split("Results at a glance")[-1] if "Results at a glance" in md else "SSIM" in md


def test_comparison_pools_several_experiments_and_pins_them(demo_db):
    from results_tracker import get_asset, get_engine
    from results_tracker.export import paper

    at = _run("comparison")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    _sidebar_multiselect(at, "Also include experiments").set_value(["lambda-sweep"]).run()
    assert not at.exception
    rows_box = [ms for ms in at.sidebar.multiselect if ms.label == "Rows grouped by"][0]
    assert rows_box.value == ["method", "experiment"] and "experiment" in _sidebar_multiselect(at, "Filter on").options
    md = "\n".join(m.value for m in at.markdown)
    assert "<span>main-comparison</span>" in md and "<span>lambda-sweep</span>" in md
    assert any("main-comparison + lambda-sweep" in c.value for c in at.caption)
    assert at.query_params["extra"] == ["lambda-sweep"]
    at.text_input(key="cmp_pin_label").input("tab:pooled").run()
    at.button(key="cmp_pin_button").click().run()
    a = get_asset("demo-paper", "tab:pooled", db=demo_db)
    assert a.extra_experiments == ["lambda-sweep"] and a.options["cols"] == "experiment"
    rendered = paper.render_paper(get_engine(demo_db), "demo-paper", source="d.db")
    tex = rendered[0].files[0][1].decode()
    assert "experiment 'main-comparison + lambda-sweep'" in tex and "\\multicolumn{3}{c}{lambda-sweep}" in tex
    assert rendered[0].runs == 33  # 18 + 14 completed runs pooled (the reported DPIR row and the failed run excluded)
    # the Export page restores the pooled experiments when the asset is opened
    at2 = AppTest.from_string("from results_tracker.ui import export\nexport.render()\n", default_timeout=30)
    at2.query_params["asset"] = "tab:pooled"
    at2.query_params["project"] = "demo-paper"
    at2.run()
    assert not at2.exception
    assert _sidebar_multiselect(at2, "Also include experiments").value == ["lambda-sweep"]
    assert at2.selectbox(key="exp_cols").value == "experiment"
    assert any("Pooling" in c.value for c in at2.caption)


@pytest.fixture
def condition_sweep_db(tmp_path, monkeypatch):
    """A sweep over a knob on a 2-noise grid with two arms, plus a categorical sweep, logged like the recipe runner
    does (condition next to the knobs in config) but without recording the swept knob on the experiment; a spec
    of the same name in the studies folder declares it."""
    from results_tracker import log_run

    db = tmp_path / "s.db"
    for noise in (0.01, 0.05):
        for den in ("drunet", "dncnn"):
            for beta in (0.0, 0.5, 1.0):
                for seed in (0, 1):
                    psnr = 30 - 100 * noise - (beta - 0.5) ** 2 * 4 + (1 if den == "drunet" else 0) + 0.1 * seed
                    log_run("ema-beta", project="pnp", experiment_type="sweep", method="adaptive", dataset="CBSD68", seed=seed,
                            config={"kernel": 3, "noise": noise, "denoiser": den, "beta": beta, "K": 20},
                            metrics={"psnr": psnr, "runtime_s": 1.0}, db=db, git_commit=None)
    for floor in ("none", "op_norm", "cr_bound"):
        for seed in (0, 1):
            log_run("rho-floor", project="pnp", experiment_type="sweep", method="adaptive", dataset="CBSD68", seed=seed,
                    config={"kernel": 3, "noise": 0.01, "rho_floor": floor, "K": 20},
                    metrics={"psnr": {"none": 27.0, "op_norm": 28.0, "cr_bound": 28.5}[floor] + 0.1 * seed}, db=db, git_commit=None)
    studies = tmp_path / "studies"
    studies.mkdir()
    (studies / "ema_beta.json").write_text(json.dumps({"name": "ema-beta", "kind": "sweep", "project": "pnp", "problem": "deblurring",
                                                       "methods": [{"method": "adaptive"}], "sweep": {"knob": "beta", "values": [0, 0.5, 1]}}))
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    monkeypatch.setenv("RESULTS_TRACKER_STUDIES", str(studies))
    st.cache_data.clear()
    st.cache_resource.clear()
    return db


def test_sweep_page_defaults_to_the_declared_knob_and_splits_by_condition(condition_sweep_db):
    at0 = _run("overview")
    md = "\n".join(m.value for m in at0.markdown)
    assert "best beta = 0.5" in md and "best denoiser" not in md  # the Overview headline also takes the knob from the spec
    at = _run("sweep")
    param_box = [sb for sb in at.sidebar.selectbox if sb.label == "Parameter (x)"][0]
    assert param_box.value == "beta"  # from the spec, not the alphabetically first varying key (denoiser)
    lines = [ms for ms in at.sidebar.multiselect if ms.label == "One line per"][0]
    assert lines.options == ["config.denoiser", "config.noise"]  # varying conditions and arm knobs, not the swept knob
    assert lines.value == []  # one method only; with several arms the default is one line per method
    caption = "\n".join(c.value for c in at.caption)
    assert "pooled over config.denoiser, config.noise" in caption and "best beta = **0.5**" in caption
    lines.set_value(["config.noise"]).run()
    assert not at.exception
    caption = "\n".join(c.value for c in at.caption)
    assert "best per line: 0.01 → 0.5, 0.05 → 0.5" in caption and "pooled over config.denoiser" in caption
    # re-read the widget from the current tree: the chart's colour pickers are keyed by line, so the previous
    # tree's nodes refer to widgets this run no longer has
    lines = [ms for ms in at.sidebar.multiselect if ms.label == "One line per"][0]
    lines.set_value(["config.denoiser", "config.noise"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "drunet / 0.01" in md and "dncnn / 0.05" in md  # one sensitivity row per line
    assert not at.warning
    # a categorical swept knob plots on a category axis, best value ringed
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("rho-floor")][0]).run()
    assert not at.exception
    param_box = [sb for sb in at.sidebar.selectbox if sb.label == "Parameter (x)"][0]
    assert param_box.value == "rho_floor"  # no spec: the only varying key
    caption = "\n".join(c.value for c in at.caption)
    assert "best rho_floor = **cr_bound**" in caption
    assert [cb for cb in at.sidebar.checkbox if cb.label == "Log x axis"][0].disabled
    # export page offers the same line splits and renders the print figure
    at2 = _run("export")
    at2.sidebar.radio[0].set_value("Sweep figure").run()
    assert not at2.exception
    at2.multiselect(key="exp_by").set_value(["config.noise"]).run()
    assert not at2.exception and not at2.error


def test_curves_page_and_asset_routing(toy_studies_db):
    from results_tracker import get_asset

    db, studies = toy_studies_db
    at = _run("curves")
    assert not at.exception
    assert at.selectbox(key="cur_curve").value == "step_sizes"
    opts = at.multiselect(key="cur_by").options
    assert "config.blur" in opts and "config.noise" in opts and "method" not in opts and at.multiselect(key="cur_by").value == []
    # only adaptive-gd records curves, so `method` does not vary among the plotted runs and is not offered
    caption = "\n".join(c.value for c in at.caption)
    assert "32 runs with curves" in caption and "pooled over config.blur, config.noise" in caption
    at.multiselect(key="cur_by").set_value(["config.noise"]).run()
    at.selectbox(key="cur_norm").select("ratio").run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "<td>16</td>" in md and "divided by the first iteration" in md  # 16 runs per noise level
    at.button(key="cur_pin_button").click().run()
    assert not at.exception
    a = get_asset("toy-paper", "fig:main-comparison-step_sizes", db=db)
    assert a.kind == "curves-figure" and a.options["by"] == ["config.noise"] and a.options["normalise"] == "ratio"
    # the Paper page opens a curves asset on the Curves page, which restores its options
    at2 = _run("paper")
    md = "\n".join(m.value for m in at2.markdown)
    assert 'href="curves?' in md and "Open in Curves" in md
    at3 = AppTest.from_string("from results_tracker.ui import curves\ncurves.render()\n", default_timeout=30)
    at3.query_params["asset"] = "fig:main-comparison-step_sizes"
    at3.query_params["project"] = "toy-paper"
    at3.run()
    assert not at3.exception
    assert at3.multiselect(key="cur_by").value == ["config.noise"] and at3.selectbox(key="cur_norm").value == "ratio"
    # the Export page points such kinds back to their page
    at4 = _run("export")
    at4.sidebar.radio[0].set_value("Curves figure").run()
    assert any("Curves" in i.value and "page" in i.value for i in at4.info)


def test_tradeoff_page(demo_db):
    from results_tracker import get_asset

    at = _run("tradeoff")
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    assert not at.exception
    assert at.selectbox(key="to_x").value == "runtime_s" and at.selectbox(key="to_y").value == "psnr"
    caption = "\n".join(c.value for c in at.caption)
    assert "hollow: DPIR, PnP-BM3D, TV" in caption  # the demo's two baselines and its reported method
    at.selectbox(key="to_path").select("dataset").run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "CBSD68</td>" in md and "Set12</td>" in md  # one point per dataset along the path
    at.button(key="to_pin_button").click().run()
    a = get_asset("demo-paper", "fig:main-comparison-tradeoff", db=demo_db)
    assert a.kind == "tradeoff-figure" and a.options["path"] == "dataset" and a.options["x_metric"] == "runtime_s"


def test_comparison_per_instance_section(toy_studies_db):
    from results_tracker import get_asset

    db, _ = toy_studies_db
    at = _run("comparison")
    assert not at.exception
    assert any(h.value == "Per instance" for h in at.subheader)
    md = "\n".join(m.value for m in at.markdown)
    assert "phantom_00" in md and "Instances ranked by the gain of" in md and "candidates for a qualitative figure" in md
    assert at.selectbox(key="cmp_gain_ours").value == "adaptive-gd"  # last in method order = ours
    caption = "\n".join(c.value for c in at.caption)
    assert "Largest gain: **phantom_" in caption
    at.button(key="cmp_dist_pin_button").click().run()
    assert not at.exception
    a = get_asset("toy-paper", "fig:main-comparison-psnr-distribution", db=db)
    assert a.kind == "distribution-figure" and a.options["methods"] == ["wiener", "gd", "adaptive-gd"]


def test_sweep_selection_and_materialize(condition_sweep_db, tmp_path):
    from results_tracker import get_asset
    from results_tracker.recipe import Study

    studies = tmp_path / "studies"
    (studies / "compare.json").write_text(json.dumps({"name": "compare", "kind": "comparison", "project": "pnp", "problem": "deblurring",
                                                      "methods": [{"method": "adaptive"}, {"method": "pnp", "config": {"beta": 0.9}}]}))
    at = _run("sweep")
    assert not at.exception and any(h.value == "Selection" for h in at.subheader)
    at.multiselect(key="sel_by").set_value(["config.denoiser"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "drunet</td><td>0.5</td>" in md and "dncnn</td><td>0.5</td>" in md and "interior" in md and "0, 0.5, 1" in md
    assert not at.warning
    at.multiselect(key="sel_by").set_value([]).run()
    at.button(key="sel_pin_button").click().run()
    assert get_asset("pnp", "tab:ema-beta-beta-selection", db=condition_sweep_db).kind == "selection-table"
    # a boundary winner is flagged
    _sidebar_multiselect(at, "Filter on").set_value(["config.beta"]).run()
    _sidebar_multiselect(at, "config.beta").set_value(["0.5", "1"]).run()
    assert any("at a grid boundary" in w.value for w in at.warning)
    _sidebar_multiselect(at, "config.beta").set_value([]).run()
    # write the winner into the comparison spec: only the arm without beta receives it
    at.text_input(key="sel_target").input("compare-tuned.json").run()
    at.button(key="sel_write").click().run()
    assert not at.exception and any("compare-tuned.json" in s_.value for s_ in at.success)
    tuned = Study.load(studies / "compare-tuned.json")
    assert tuned.methods[0].config == {"beta": 0.5} and tuned.methods[1].config == {"beta": 0.9} and "from ema-beta" in tuned.description


def test_overview_notes_stage_and_run_links(demo_db, tmp_path):
    from results_tracker import list_notes, set_experiment

    at = _run("overview")
    md = "\n".join(m.value for m in at.markdown)
    assert "<th>Stage</th>" in md and 'href="run?project=demo-paper&experiment=' in md and "&run=" in md  # ids link to Run detail
    assert any("No notes yet" in c.value for c in at.caption)
    # add a note through the form
    at.text_input(key="note_text").input("lambda = 0.1 chosen: plateau").run()
    [b for b in at.button if b.label == "Add note"][0].click().run()
    assert not at.exception
    notes = list_notes("demo-paper", db=demo_db)
    assert len(notes) == 1 and notes[0].text == "lambda = 0.1 chosen: plateau"
    assert "lambda = 0.1 chosen: plateau" in "\n".join(m.value for m in at.markdown)
    # a superseded experiment leaves the selectors unless asked for, and the Overview says so
    set_experiment("ablation", project="demo-paper", stage="superseded", db=demo_db)
    st.cache_data.clear()
    at2 = _run("comparison")
    exp_box = [sb for sb in at2.sidebar.selectbox if sb.label == "Experiment"][0]
    assert not any(o.startswith("ablation") for o in exp_box.options)
    [cb for cb in at2.sidebar.checkbox if cb.label.startswith("Show 1 superseded")][0].check().run()
    exp_box = [sb for sb in at2.sidebar.selectbox if sb.label == "Experiment"][0]
    assert any(o.startswith("ablation") and "superseded" in o for o in exp_box.options)
    at3 = _run("overview")
    md = "\n".join(m.value for m in at3.markdown)
    assert "superseded</td>" in md
    # the recent-databases file lives under $RESULTS_TRACKER_HOME (set by conftest), never in the real home
    import json as _json
    import os

    recent = _json.loads((Path(os.environ["RESULTS_TRACKER_HOME"]) / "recent.json").read_text())
    assert recent == [str(demo_db)]


def test_recent_databases_with_the_same_file_name_stay_distinct(demo_db, tmp_path):
    """Two remembered databases called results.db must not collapse onto one dropdown label: Streamlit returns the
    first option whose label matches, and the GUI then switched to the wrong file on the next rerun."""
    import json as _json

    from results_tracker import get_engine
    from results_tracker.ui.common import _canonical_db, db_labels, recent_dbs

    other = tmp_path / "elsewhere" / "results.db"
    other.parent.mkdir()
    get_engine(other).dispose()
    home = Path(os.environ["RESULTS_TRACKER_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    # a relative spelling of the empty file, a directory typed by mistake, and the real database
    (home / "recent.json").write_text(_json.dumps([os.path.relpath(other), ".", str(demo_db)]))

    assert recent_dbs() == [str(other), str(demo_db)]  # canonical, the directory dropped
    labels = db_labels(recent_dbs())
    assert labels[str(other)] != labels[str(demo_db)]
    assert all(lbl.endswith(os.path.basename(pth)) for pth, lbl in labels.items())
    assert _canonical_db("~/x.db") == os.path.expanduser("~/x.db")

    at = _run("comparison")
    box = [sb for sb in at.sidebar.selectbox if sb.label == "Recent databases"][0]
    assert box.value == _canonical_db(str(demo_db)) and len(set(box.format_func(o) for o in box.options)) == len(box.options)
    at.run()  # a rerun must not drift to the other file
    assert not at.exception
    assert [t for t in at.sidebar.text_input if t.label == "Database"][0].value == _canonical_db(str(demo_db))
    assert [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"], "selectors vanished: the database changed"


def test_run_detail_opens_the_run_in_the_url(demo_db):
    from results_tracker import get_runs

    runs = get_runs(experiment="main-comparison", db=demo_db)
    target = runs[5]
    at = AppTest.from_string("from results_tracker.ui import run_detail\nrun_detail.render()\n", default_timeout=30)
    at.query_params["experiment"] = "main-comparison"
    at.query_params["run"] = str(target.id)
    at.run()
    assert not at.exception
    assert at.selectbox(key="run_pick").value.startswith(f"#{target.id} ")
    assert at.query_params["run"] == [str(target.id)]
    at.query_params["run"] = "999999"
    at2 = AppTest.from_string("from results_tracker.ui import run_detail\nrun_detail.render()\n", default_timeout=30)
    at2.query_params["experiment"] = "main-comparison"
    at2.query_params["run"] = "999999"
    at2.run()
    assert any("not in this experiment" in w.value for w in at2.warning)


def test_settings_experiments_tab_and_studies_dir(toy_studies_db, tmp_path):
    from results_tracker import list_experiments

    db, studies = toy_studies_db
    other = tmp_path / "elsewhere"
    other.mkdir()
    (studies / "ablation.json").rename(other / "ablation.json")
    at = _run("settings")
    at.selectbox(key="set_exp_main-comparison_stage").select("exploratory").run()  # the only experiment with runs so far
    at.text_input(key="set_exp_main-comparison_desc").input("scratch: where is the plateau?").run()
    at.button(key="set_exps_save").click().run()
    assert not at.exception
    e = next(x for x in list_experiments("toy-paper", db=db) if x.name == "main-comparison")
    assert e.stage == "exploratory" and e.description == "scratch: where is the plateau?"
    at.text_input(key="set_studies_dir").input(str(other)).run()
    at.button(key="set_project_save").click().run()
    assert not at.exception
    # the Studies page now reads the project's directory instead of $RESULTS_TRACKER_STUDIES
    at2 = _run("studies")
    assert set(at2.dataframe[0].value["experiment"]) == {"ablation"}


def _chart(at, index: int = 0) -> dict:
    """The figure a `st.plotly_chart` sent to the browser, as its JSON spec (AppTest has no plotly element)."""
    return json.loads(at.get("plotly_chart")[index].proto.spec)


def test_plot_style_controls_write_to_the_project_and_reach_the_chart(demo_db):
    """The sidebar sizes and a chart's colour picker are the only writers of `Project.plot_style`; a change must
    land on the project (so every page and every export sees it) and redraw the chart in the same run."""
    from results_tracker.api import get_plot_style
    from results_tracker.db import get_engine
    from results_tracker.plotstyle import SCREEN_FONT_SCALE

    engine = get_engine(demo_db)
    at = _run("comparison")
    project = [sb for sb in at.sidebar.selectbox if sb.label == "Project"][0].value
    assert get_plot_style(project, engine=engine).is_default

    at.sidebar.number_input(key="stylew_legend").set_value(22.0).run()
    assert not at.exception
    assert get_plot_style(project, engine=engine).legend == 22.0  # saved on the project, not just in the session
    assert _chart(at)["layout"]["legend"]["font"]["size"] == pytest.approx(22.0 * SCREEN_FONT_SCALE)  # drawn at once

    keys = [str(k) for k in at.session_state.filtered_state]
    colour_keys = [k for k in keys if k.startswith("cmp_bars:") and "_color_" in k]
    assert colour_keys, "the chart offers a colour per series"
    at.color_picker(key=colour_keys[0]).set_value("#00aa00").run()
    assert not at.exception
    style = get_plot_style(project, engine=engine)
    assert "#00aa00" in style.colors.values() and style.legend == 22.0
    assert "#00aa00" in [t["marker"]["color"] for t in _chart(at)["data"]]

    # a fixed y range travels with the view, not with the project
    ylim_key = [k for k in keys if k.startswith("cmp_bars:") and k.endswith("_ylim")][0]
    at.text_input(key=ylim_key).set_value("28,34").run()
    assert not at.exception
    assert _chart(at)["layout"]["yaxis"]["range"] == [28.0, 34.0]
    assert set(get_plot_style(project, engine=engine).to_dict()) == {"legend", "colors"}


def test_a_project_level_label_order_reaches_the_charts(demo_db):
    """Order is stored per grouping key on the project, so a page picks it up with no interaction -- and the
    palette follows it (first drawn is blue), which is why a colour can be pinned separately."""
    from results_tracker.api import set_project
    from results_tracker.db import get_engine
    from results_tracker.plotstyle import PALETTE, PlotStyle

    engine = get_engine(demo_db)
    at = _run("comparison")
    default = [t["name"] for t in _chart(at)["data"]]
    assert len(default) > 2
    backwards = list(reversed(default))
    set_project("demo-paper", plot_style=PlotStyle().with_order("method", backwards).to_dict(), engine=engine)
    st.cache_data.clear()
    at = _run("comparison")
    assert not at.exception
    bars = _chart(at)["data"]
    assert [t["name"] for t in bars] == backwards
    assert [t["marker"]["color"] for t in bars] == PALETTE[:len(bars)]


def test_a_page_says_why_it_has_nothing_to_show(demo_db):
    """A filter that keeps only failed runs used to leave the page blank: the sidebar said "1 of 15 runs
    match" and the body rendered nothing at all."""
    at = _run("sweep")
    box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    box.select([o for o in box.options if o.startswith("lambda-sweep")][0]).run()
    [ms for ms in at.sidebar.multiselect if ms.label == "Filter on"][0].set_value(["status"]).run()
    [ms for ms in at.sidebar.multiselect if ms.label == "status"][0].set_value(["failed"]).run()
    assert not at.exception
    said = "\n".join(i.value for i in at.info)
    assert "none of them completed" in said and "1 failed" in said
    assert "Run detail" in said  # where the failure message is

    # a filter that matches no run at all names the filter and says how many runs are hidden
    [ms for ms in at.sidebar.multiselect if ms.label == "Filter on"][0].set_value(["status", "seed"]).run()
    [ms for ms in at.sidebar.multiselect if ms.label == "seed"][0].set_value(["0"]).run()  # the failed run has seed 2
    assert not at.exception
    said = "\n".join(i.value for i in at.info)
    assert "No run matches the filter" in said and "15 runs" in said


def test_an_experiment_whose_runs_all_failed_says_so(demo_db):
    """It used to blame the reader: with no completed runs there are no metric names, so the Metrics
    multiselect came up empty and the page asked for a grouping key instead of reporting the failures."""
    from results_tracker import log_run

    for seed in (0, 1):
        log_run("all-failed", project="demo-paper", method="Ours", dataset="Set12", seed=seed, config={"K": 5},
                metrics={}, status="failed", notes="RuntimeError: CUDA out of memory", db=demo_db, git_commit=None)
    st.cache_data.clear()
    at = _run("comparison")
    box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    box.select([o for o in box.options if o.startswith("all-failed")][0]).run()
    assert not at.exception
    said = "\n".join(i.value for i in at.info)
    assert "2 run(s) in this experiment, none of them completed: 2 failed" in said
    assert not [w for w in at.warning if "grouping key" in w.value]


def test_failed_runs_are_named_in_every_page_caption(demo_db):
    """Pages aggregate completed runs only; the ones they drop have to be named where the reader looks."""
    for page, experiment in (("sweep", "lambda-sweep"), ("comparison", "lambda-sweep")):
        at = _run(page)
        box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
        box.select([o for o in box.options if o.startswith(experiment)][0]).run()
        assert not at.exception
        caption = "\n".join(c.value for c in at.caption)
        assert "1 failed excluded" in caption, f"{page} does not say the failed run was dropped"


def test_a_settings_save_keeps_the_record_cache_but_still_shows_the_change(demo_db):
    """Saving used to clear every cache. Method labels ride on each record, so those must reload; a metric's
    unit does not, and the catalog/metric caches come back on their own (they are keyed on the file mtime)."""
    from results_tracker.api import define_metric, list_methods
    from results_tracker.db import get_engine
    from results_tracker.ui import common

    engine = get_engine(demo_db)
    at = _run("comparison")
    assert "TV [1]" in "\n".join(m.value for m in at.markdown)

    name = next(m.name for m in list_methods(engine=engine) if m.name == "TV")
    from results_tracker.api import define_method

    define_method(name, label="Total Variation", is_baseline=True, position=0, engine=engine)
    common.invalidate_records()
    at = _run("comparison")
    assert "Total Variation" in "\n".join(m.value for m in at.markdown)

    define_metric("psnr", unit="decibel", higher_is_better=True, fmt=".2f", engine=engine)
    at = _run("comparison")  # no explicit invalidation: the metric cache is keyed on the database's mtime
    assert "decibel" in "\n".join(m.value for m in at.markdown)


def test_the_experiment_selector_prefers_one_with_results(tmp_path, monkeypatch):
    """The first click should not land on an experiment that has nothing to show just because its name sorts
    first, and a numbered family should read K2, K5, K10 rather than K10, K2, K5."""
    from results_tracker import log_run

    db = tmp_path / "rank.db"
    for seed in (0, 1):
        log_run("aaa-all-failed", project="p", method="m", dataset="D", seed=seed, config={}, metrics={},
                status="failed", db=db, git_commit=None)
    for k in (10, 2, 5):
        log_run(f"compare-K{k}", project="p", method="m", dataset="D", seed=0, config={"K": k},
                metrics={"psnr": 30.0}, db=db, git_commit=None)
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _run("comparison")
    box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    assert [o.split(" ")[0] for o in box.options] == ["compare-K2", "compare-K5", "compare-K10", "aaa-all-failed"]
    assert box.value.startswith("compare-K2")


def test_a_missing_database_is_never_swapped_for_a_recent_one(tmp_path, monkeypatch):
    """With other databases in the recent list, the picker used to fall back to index 0 and silently open the
    most recent one instead of the database that was asked for — hiding the missing-file warning entirely."""
    from results_tracker.demo import seed_demo

    other = tmp_path / "other.db"
    seed_demo(db=other, artifacts_dir=str(tmp_path / "art"))
    home = Path(os.environ["RESULTS_TRACKER_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "recent.json").write_text(json.dumps([str(other), str(tmp_path / "older.db")]))
    missing = tmp_path / "not-mounted" / "results.db"
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(missing))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _run("overview")
    assert any("does not exist" in w.value for w in at.warning)
    assert at.session_state["db"] == str(missing)  # still the database that was asked for
    picker = [sb for sb in at.sidebar.selectbox if sb.label == "Recent databases"][0]
    assert picker.value == str(missing)  # the picker shows it rather than pointing somewhere else
    assert picker.options[0].endswith("(missing)")


def test_a_missing_database_is_marked_in_the_recent_list(demo_db, tmp_path):
    """Picking a database that has been moved or unmounted used to recreate it empty; it is named as missing
    and, if picked, lands on the create offer instead."""
    home = Path(os.environ["RESULTS_TRACKER_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    gone = tmp_path / "unplugged" / "results.db"
    (home / "recent.json").write_text(json.dumps([str(demo_db), str(gone)]))
    at = _run("comparison")
    picker = [sb for sb in at.sidebar.selectbox if sb.label == "Recent databases"][0]
    assert any("(missing)" in o for o in picker.options)
    assert not gone.exists()


@pytest.fixture
def messy_db(tmp_path, monkeypatch):
    """A demo database plus what a real week leaves behind: a crashed run with its message and running rows
    from jobs that were killed."""
    from datetime import datetime, timedelta, timezone

    from results_tracker import log_run

    db = tmp_path / "messy.db"
    seed_demo(db=db, artifacts_dir=str(tmp_path / "art"))
    now = datetime.now(timezone.utc)
    log_run("main-comparison", project="demo-paper", method="Ours", dataset="Set12", seed=41, config={"iters": 50},
            metrics={}, status="failed", notes="RuntimeError: CUDA out of memory; node gpu-07", db=db, git_commit=None)
    for hours, seed in ((30.0, 51), (26.0, 52), (0.2, 53)):
        log_run("main-comparison", project="demo-paper", method="Ours", dataset="Set12", seed=seed, config={"iters": 50},
                metrics={}, status="running", timestamp=now - timedelta(hours=hours), db=db, git_commit=None)
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    st.cache_data.clear()
    st.cache_resource.clear()
    return db


def test_runs_page_surfaces_what_failed_and_why(messy_db):
    """The failure message the runner recorded used to be reachable only by guessing which run to open."""
    at = _run("runs")
    values = {m.label: m.value for m in at.metric}
    assert values["Failed"] == "2" and values["Running"] == "3"  # the demo's diverged run plus the crash above
    frame = at.dataframe[0].value
    assert "RuntimeError: CUDA out of memory" in " ".join(frame["message"].astype(str))
    assert frame["run"].iloc[0].startswith("run?") and "&run=" in frame["run"].iloc[0]  # links to Run detail

    [ms for ms in at.sidebar.multiselect if ms.label == "Status"][0].set_value(["failed"]).run()
    assert not at.exception
    frame = at.dataframe[0].value
    assert len(frame) == 2 and set(frame["status"]) == {"failed"}

    # and the message is searchable across experiments
    [ms for ms in at.sidebar.multiselect if ms.label == "Status"][0].set_value([]).run()
    at.text_input(key="runs_search").set_value("gpu-07").run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 1


def test_runs_page_clears_running_rows_a_killed_job_left_behind(messy_db):
    """A row logged at job start that nothing will ever complete: the Overview counts it forever and the CLI
    was the only way to remove it."""
    from results_tracker.api import query_runs
    from results_tracker.db import get_engine

    engine = get_engine(messy_db)
    at = _run("runs")
    assert any("running for more than 12 h" in w.value for w in at.warning)
    listed = "\n".join(m.value for m in at.markdown)
    assert listed.count("← stale") == 2 and "min ago" in listed  # the one that started minutes ago is not marked

    at.checkbox(key="runs_stale_confirm").check().run()
    at.button(key="runs_stale_delete").click().run()
    assert not at.exception
    rows, counts = query_runs(project="demo-paper", statuses=["running"], engine=engine)
    assert counts["total"] == 1  # only the one that started minutes ago survives
    assert not any("running for more than" in w.value for w in at.warning)


def test_overview_points_at_the_runs_page_when_something_failed(messy_db):
    at = _run("overview")
    said = "\n".join(c.value for c in at.caption)
    assert "2 failed and 3 running" in said  # the demo's own diverged run plus the one above
    assert 'href="runs"' in said  # a link, not just a count


def test_a_crowded_chart_says_its_colours_repeat(demo_db, tmp_path):
    """Past the palette two series share a hue. That is fine (the marker differs), but the reader has to be
    told -- silently repeating blue is how two methods get confused in a paper figure."""
    from results_tracker import log_run
    from results_tracker.plotstyle import PALETTE

    for i in range(len(PALETTE) + 3):
        for seed in (0, 1):
            log_run("crowded", project="demo-paper", experiment_type="sweep", method=f"arm-{i:02d}", dataset="Set12",
                    seed=seed, config={"lambda": 0.1}, metrics={"psnr": 30.0 + i}, db=demo_db, git_commit=None)
    st.cache_data.clear()
    at = _run("sweep")
    box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    box.select([o for o in box.options if o.startswith("crowded")][0]).run()
    lines = [ms for ms in at.sidebar.multiselect if ms.label == "One line per"][0]
    lines.set_value(["method"]).run()
    assert not at.exception
    said = "\n".join(c.value for c in at.caption)
    assert f"11 series share {len(PALETTE)} colours" in said


def test_a_plot_style_can_be_copied_from_another_project(demo_db):
    """Paper number two should not start from the lab default again."""
    from results_tracker.api import get_plot_style, log_run, set_project
    from results_tracker.db import get_engine
    from results_tracker.plotstyle import PlotStyle

    engine = get_engine(demo_db)
    log_run("first", project="paper-two", method="Ours", dataset="Set12", seed=0, config={}, metrics={"psnr": 30.0},
            db=demo_db, git_commit=None)
    styled = PlotStyle(tick=21.0).with_colors({"Ours": "#00aa00"}).with_order("method", ["Ours", "TV"])
    set_project("demo-paper", plot_style=styled.to_dict(), engine=engine)
    st.cache_data.clear()

    at = _run("settings")
    [sb for sb in at.sidebar.selectbox if sb.label == "Project"][0].select("paper-two").run()
    assert get_plot_style("paper-two", engine=engine).is_default
    at.selectbox(key="set_style_source").select("demo-paper").run()
    at.button(key="set_style_copy").click().run()
    assert not at.exception
    assert get_plot_style("paper-two", engine=engine).to_dict() == styled.to_dict()


@pytest.fixture
def instance_db(tmp_path, monkeypatch):
    """Per-instance runs for two methods: ours wins on most images but not all, which is the case the
    paired statistics exist for."""
    from results_tracker import log_run

    db = tmp_path / "instances.db"
    gains = [0.8, 0.5, 0.9, -0.3, 0.4, 0.7, 0.2, 1.1, -0.1, 0.6]  # 8 wins, 2 losses
    for i, gain in enumerate(gains):
        for seed in (0, 1):
            base = 29.0 + 0.05 * i
            log_run("per-image", project="p", method="baseline", dataset="Set12", instance=f"img{i:02d}", seed=seed,
                    config={"K": 5}, metrics={"psnr": base + 0.01 * seed}, db=db, git_commit=None)
            log_run("per-image", project="p", method="ours", dataset="Set12", instance=f"img{i:02d}", seed=seed,
                    config={"K": 5}, metrics={"psnr": base + gain + 0.01 * seed}, db=db, git_commit=None)
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(db))
    st.cache_data.clear()
    st.cache_resource.clear()
    return db


def test_comparison_answers_how_often_ours_wins(instance_db):
    """A mean ± std does not say on how many images the method improved, which is the first thing a reviewer
    asks; the page now states it with a paired test and a sentence to paste."""
    at = _run("comparison")
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Instances won"] == "8 / 10"
    assert values["Median gain (psnr)"].startswith("+0.5")
    assert values["Wilcoxon signed-rank"].startswith("p = 0.0") or values["Wilcoxon signed-rank"] == "p < 0.001"
    sentence = "\n".join(c.value for c in at.code)
    assert "improves psnr on 8 of 10 instances over baseline" in sentence
    assert "Wilcoxon signed-rank" in sentence
    caveat = "\n".join(c.value for c in at.caption)
    assert "multiple-comparison correction" in caveat  # the page says what it does not do


def test_a_table_cell_can_be_opened_to_the_runs_behind_it(instance_db):
    """`30.69 ± 0.06` is a mean over runs the reader cannot otherwise see."""
    at = _run("comparison")
    rows = [ms for ms in at.sidebar.multiselect if ms.label == "Rows grouped by"][0]
    rows.set_value(["method"]).run()
    assert not at.exception
    picker = at.selectbox(key="cmp_cell_row")
    assert picker is not None
    picker.select("ours").run()
    assert not at.exception
    frame = [d.value for d in at.dataframe if "run" in getattr(d.value, "columns", [])][0]
    assert len(frame) == 20 and set(frame["instance"]) == {f"img{i:02d}" for i in range(10)}
    assert frame["run"].iloc[0].startswith("run?")
    said = "\n".join(c.value for c in at.caption)
    assert "over these 20 run(s)" in said and "min" in said and "max" in said


def test_an_experiment_can_be_renamed_and_deleted_from_settings(demo_db):
    """A name typed into a script used to be permanent: `experiment set` only edits the stage."""
    from results_tracker.api import get_asset, list_experiments, save_asset
    from results_tracker.db import get_engine

    engine = get_engine(demo_db)
    save_asset("demo-paper", "fig:beta", kind="sweep-figure", experiment="lambda-sweep",
               options={"param": "lambda", "metric": "psnr"}, engine=engine)
    st.cache_data.clear()

    at = _run("settings")
    at.tabs[2].run()  # Experiments
    at.selectbox(key="set_exp_admin").select("lambda-sweep").run()
    at.text_input(key="set_exp_newname").set_value("lambda-sweep-v2").run()
    at.button(key="set_exp_rename").click().run()
    assert not at.exception
    assert {e.name for e in list_experiments("demo-paper", engine=engine)} >= {"lambda-sweep-v2"}
    assert get_asset("demo-paper", "fig:beta", engine=engine).experiment == "lambda-sweep-v2"  # the pin followed

    at.selectbox(key="set_exp_admin").select("lambda-sweep-v2").run()
    at.checkbox(key="set_exp_delete_confirm").check().run()
    at.button(key="set_exp_delete").click().run()
    assert not at.exception
    assert "lambda-sweep-v2" not in {e.name for e in list_experiments("demo-paper", engine=engine)}


def test_selected_runs_can_be_moved_and_deleted_from_the_runs_page(messy_db):
    """The ticked rows' actions. AppTest cannot tick a dataframe row (the selection is the frontend's), so the
    page's action block is driven with a known selection — the part that moves and deletes."""
    from results_tracker.api import query_runs
    from results_tracker.db import get_engine

    script = (
        "import os\n"
        "from results_tracker.api import query_runs, run_records\n"
        "from results_tracker.db import get_engine\n"
        "from results_tracker.ui import runs\n"
        "e = get_engine(os.environ['RESULTS_TRACKER_DB'])\n"
        "rows, _ = query_runs(project='demo-paper', statuses=['running'], limit=2, engine=e)\n"
        "runs._selected_actions(run_records(rows, engine=e), 'demo-paper', ['main-comparison'], e)\n"
    )
    engine = get_engine(messy_db)
    before = query_runs(project="demo-paper", statuses=["running"], engine=engine)[1]["total"]
    assert before == 3

    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert any("2 run(s) selected" in m.value for m in at.markdown)

    at.text_input(key="runs_move_to").set_value("quarantine").run()
    at.button(key="runs_move").click().run()
    assert not at.exception
    assert query_runs(project="demo-paper", experiment="quarantine", engine=engine)[1]["total"] == 2

    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    at.checkbox(key="runs_delete_confirm").check().run()
    at.button(key="runs_delete").click().run()
    assert not at.exception
    assert query_runs(project="demo-paper", statuses=["running"], engine=engine)[1]["total"] == before - 2


def test_the_import_page_reads_a_file_the_way_the_cli_does(tmp_path, monkeypatch):
    """The page's own path: upload -> records -> spec -> rows in the database, without a second CSV reader."""
    from results_tracker.api import get_runs, run_records
    from results_tracker.db import get_engine
    from results_tracker.importer import ImportSpec, import_records
    from results_tracker.ui import import_runs

    csv = ("method,dataset,instance,seed,K,psnr,ssim\n"
           "ours,Set12,img01,0,5,31.2,0.88\n"
           "ours,Set12,img02,0,5,30.8,0.87\n"
           "baseline,Set12,img01,0,5,29.9,0.85\n")

    class Upload:  # what st.file_uploader hands the page
        name = "results.csv"

        def getvalue(self):
            return csv.encode()

    raws = import_runs._read_upload(Upload())
    assert len(raws) == 3 and import_runs._columns(raws) == ["method", "dataset", "instance", "seed", "K", "psnr", "ssim"]

    db = tmp_path / "imported.db"
    engine = get_engine(db)
    spec = ImportSpec(experiment="from-csv", project="p", metric_cols=["psnr", "ssim"], config_cols=["K"])
    dry = import_records(raws, spec, engine=engine, dry_run=True)
    assert dry.imported == 3 and not get_runs(experiment="from-csv", engine=engine)  # a dry run writes nothing
    result = import_records(raws, spec, engine=engine)
    assert result.imported == 3
    recs = run_records(get_runs(experiment="from-csv", engine=engine), engine=engine)
    assert {r["method"] for r in recs} == {"ours", "baseline"}
    assert recs[0]["metrics"]["psnr"] and recs[0]["config"]["K"] == 5  # K stayed a setting, not a result
    assert import_records(raws, spec, engine=engine).skipped == 3  # re-importing the same file adds nothing


def test_the_export_page_composes_pinned_figures_into_one(demo_db):
    """Panels of one IEEE figure, drawn by the same code at the same size rather than pasted together in LaTeX."""
    from results_tracker.api import save_asset
    from results_tracker.db import get_engine

    engine = get_engine(demo_db)
    save_asset("demo-paper", "fig:beta", kind="sweep-figure", experiment="lambda-sweep",
               options={"param": "lambda", "metric": "psnr"}, engine=engine)
    save_asset("demo-paper", "fig:bars", kind="comparison-figure", experiment="main-comparison",
               options={"metric": "psnr", "rows": "method", "cols": "dataset"}, engine=engine)
    st.cache_data.clear()

    at = _run("export")
    at.sidebar.radio[0].set_value("Multi-panel figure").run()
    assert not at.exception
    at.multiselect(key="exp_panels").set_value(["fig:beta", "fig:bars"]).run()
    assert not at.exception and not at.error
    assert at.image  # the composed preview
    assert any("Download PDF" in b.label for b in at.download_button)
    assert any("each panel is the pinned asset" in c.value.lower() for c in at.caption)
