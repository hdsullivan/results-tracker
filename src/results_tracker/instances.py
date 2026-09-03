"""Which GUIs are running, on which port, on which database.

`results-tracker ui` records itself here so a second launch can tell you that a GUI is already up,
and on which file. Otherwise the second launch silently lands on the next free port while the browser
tab you are looking at still shows the first database.

One JSON file per port in a cache directory (`$RESULTS_TRACKER_UI_REGISTRY`, else
`$XDG_CACHE_HOME/results-tracker/ui`, else `~/.cache/results-tracker/ui`). An entry counts as live only
while its process exists and its port accepts connections; anything else is pruned on the next look.
"""

from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ENV_VAR = "RESULTS_TRACKER_UI_REGISTRY"


@dataclass(frozen=True)
class UiInstance:
    port: int
    db: str
    pid: int
    started: float

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"


def registry_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "results-tracker" / "ui"


def _entry_path(port: int) -> Path:
    return registry_dir() / f"{port}.json"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def register(port: int, db: str, pid: Optional[int] = None) -> UiInstance:
    """Record a GUI serving `db` on `port` (default: this process)."""
    inst = UiInstance(port=int(port), db=str(Path(db).expanduser().resolve()) if db != ":memory:" else db,
                      pid=int(pid if pid is not None else os.getpid()), started=time.time())
    path = _entry_path(inst.port)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(inst)))
    return inst


def unregister(port: int) -> None:
    try:
        _entry_path(port).unlink()
    except FileNotFoundError:
        pass


def running() -> list[UiInstance]:
    """Live GUI instances, oldest first. Stale entries (dead process or closed port) are removed."""
    out: list[UiInstance] = []
    d = registry_dir()
    if not d.is_dir():
        return out
    for path in sorted(d.glob("*.json")):
        try:
            inst = UiInstance(**json.loads(path.read_text()))
        except (OSError, ValueError, TypeError):
            path.unlink(missing_ok=True)
            continue
        if _pid_alive(inst.pid) and port_open(inst.port):
            out.append(inst)
        else:
            path.unlink(missing_ok=True)
    return sorted(out, key=lambda i: i.started)


def same_db(a: str, b: str) -> bool:
    if a == ":memory:" or b == ":memory:":
        return a == b
    return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
