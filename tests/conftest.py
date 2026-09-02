import pytest

from results_tracker.db import get_engine


@pytest.fixture
def engine(tmp_path):
    return get_engine(tmp_path / "test.db")
