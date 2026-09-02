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
    at.sidebar.multiselect[0].set_value(["method", "dataset"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "<span>Set12</span>" in md and "<span>CBSD68</span>" in md
    # three keys -> flat layout
    at.sidebar.multiselect[0].set_value(["method", "dataset", "seed"]).run()
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
    caption = "\n".join(c.value for c in at.caption)
    assert "best lambda = **0.1**" in caption
    assert not at.warning


def test_sweep_page_heatmap_mode(demo_db):
    at = _run("sweep")
    box = [sb for sb in at.sidebar.selectbox if sb.label.startswith("Second parameter")][0]
    # only lambda varies in the demo sweep, so a 2nd parameter gives a 1-row heatmap; still must not crash
    box.select(box.options[1]).run()
    assert not at.exception
    caption = "\n".join(c.value for c in at.caption)
    assert "best at lambda=0.1" in caption


def test_ablation_page(demo_db):
    at = _run("ablation")
    md = "\n".join(m.value for m in at.markdown)
    assert "<td>Full model</td>" in md and "w/o adaptive" in md and "<td>×</td>" in md
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
