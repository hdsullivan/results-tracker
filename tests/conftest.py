import pytest

from results_tracker.db import get_engine


@pytest.fixture
def engine(tmp_path):
    return get_engine(tmp_path / "test.db")


@pytest.fixture(autouse=True)
def _tracker_home(tmp_path, monkeypatch):
    """The GUI remembers recent databases under $RESULTS_TRACKER_HOME; keep that out of the real home directory."""
    monkeypatch.setenv("RESULTS_TRACKER_HOME", str(tmp_path / "tracker-home"))
