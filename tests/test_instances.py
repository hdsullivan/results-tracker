import os
import socket

import pytest
from typer.testing import CliRunner

from results_tracker import instances
from results_tracker.cli import app

runner = CliRunner()


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv(instances.ENV_VAR, str(tmp_path / "reg"))
    return tmp_path / "reg"


@pytest.fixture
def listening():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(128)  # never accepts; a large backlog keeps repeated probes from filling the queue
        yield s.getsockname()[1]


def test_register_running_and_prune(registry, listening, tmp_path):
    db = tmp_path / "a.db"
    inst = instances.register(listening, str(db))
    assert inst.pid == os.getpid() and inst.url == f"http://localhost:{listening}"
    assert [i.port for i in instances.running()] == [listening]
    # a dead process or a closed port is not "running" and its entry is removed
    instances.register(listening + 1, str(db), pid=os.getpid())        # port not open
    instances.register(listening + 2, str(db), pid=2**22 + 12345)      # no such process
    (registry / "junk.json").write_text("not json")
    assert [i.port for i in instances.running()] == [listening]
    assert sorted(p.name for p in registry.iterdir()) == [f"{listening}.json"]
    instances.unregister(listening)
    instances.unregister(listening)  # idempotent
    assert instances.running() == []


def test_same_db_resolves_paths(tmp_path):
    a = tmp_path / "x.db"
    assert instances.same_db(str(a), str(tmp_path / "." / "x.db"))
    assert not instances.same_db(str(a), str(tmp_path / "y.db"))
    assert instances.same_db(":memory:", ":memory:") and not instances.same_db(":memory:", str(a))


def test_ui_reuses_a_gui_already_running_on_the_same_database(registry, listening, tmp_path, monkeypatch):
    db = tmp_path / "toy.db"
    instances.register(listening, str(db))
    r = runner.invoke(app, ["ui", "--db", str(db), "--headless"])
    assert r.exit_code == 0, r.output
    assert "already running" in r.output and f"localhost:{listening}" in r.output and "streamlit" not in r.output.lower()


def test_free_port_sees_a_dual_stack_ipv6_listener():
    # Streamlit/uvicorn listen on [::] with dual stack; an IPv4-only bind test used to miss that.
    from results_tracker.cli import _free_port, _port_free

    try:
        s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        s6.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        s6.bind(("::", 0))
    except OSError:
        pytest.skip("no IPv6 on this host")
    with s6:
        s6.listen(128)
        port = s6.getsockname()[1]
        assert not _port_free(port)
        assert _free_port(port) != port
    assert _port_free(port)  # released
