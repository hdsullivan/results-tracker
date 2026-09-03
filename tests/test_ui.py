import json
import os

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
    # default experiment is alphabetical ("ablation"); switch to the comparison experiment
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
    monkeypatch.setenv("RESULTS_TRACKER_DB", str(tmp_path / "empty.db"))
    st.cache_data.clear()
    st.cache_resource.clear()
    at = _run("comparison")
    assert any("No projects yet" in i.value for i in at.sidebar.info)


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
