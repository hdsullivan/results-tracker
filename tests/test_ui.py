import os

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from results_tracker.demo import seed_demo  # noqa: E402


@pytest.fixture
def demo_db(tmp_path, monkeypatch):
    db = tmp_path / "demo.db"
    seed_demo(db=db)
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
    assert values["Runs"] == "45"
    assert len(at.dataframe) == 2


def test_comparison_page_table_and_chart(demo_db):
    at = _run("comparison")
    # default experiment is alphabetical ("ablation"); switch to the comparison experiment
    exp_box = [sb for sb in at.sidebar.selectbox if sb.label == "Experiment"][0]
    exp_box.select([o for o in exp_box.options if o.startswith("main-comparison")][0]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "TV" in md and "PnP-BM3D" in md and "Ours" in md
    assert "**" in md  # something is bolded as best
    assert "psnr" in md
    # group by method and dataset -> 6 rows
    at.sidebar.multiselect[0].set_value(["method", "dataset"]).run()
    assert not at.exception
    md = "\n".join(m.value for m in at.markdown)
    assert "Ours / Set12" in md and "TV / CBSD68" in md


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
