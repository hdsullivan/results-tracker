"""Engine and session helpers. One SQLite file per lab / paper collection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from . import models  # noqa: F401  (registers tables on SQLModel.metadata)

ENV_VAR = "RESULTS_TRACKER_DB"
DEFAULT_DB = "results.db"
MEMORY = ":memory:"

PathLike = Union[str, os.PathLike]


def resolve_db_path(path: Optional[PathLike] = None) -> str:
    """Explicit arg > $RESULTS_TRACKER_DB > ./results.db."""
    p = path or os.environ.get(ENV_VAR) or DEFAULT_DB
    if str(p) == MEMORY:
        return MEMORY
    return str(Path(p).expanduser())


def get_engine(path: Optional[PathLike] = None, echo: bool = False):
    """Create (or open) the database and make sure all tables exist."""
    p = resolve_db_path(path)
    if p == MEMORY:
        engine = create_engine(
            "sqlite://",
            echo=echo,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{p}", echo=echo)
    SQLModel.metadata.create_all(engine)
    return engine


@contextmanager
def session_scope(engine) -> Iterator[Session]:
    """Commit on success, roll back on error. Objects stay usable after commit."""
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
