"""Engine and session helpers. One SQLite file per lab / paper collection."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from sqlalchemy import inspect, text
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
    add_missing_columns(engine)
    return engine


def add_missing_columns(engine) -> list[str]:
    """Add columns the models declare but an older database lacks (`create_all` only creates whole tables).

    SQLite's ALTER TABLE ADD COLUMN is all that is needed: new columns get the model's scalar default, or NULL
    (which the models read back as an empty list/dict). Returns `table.column` for every column added."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    added = []
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for col in table.columns:
                if col.name in present:
                    continue
                ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {col.type.compile(engine.dialect)}'
                default = getattr(col.default, "arg", None)
                if isinstance(default, (int, float, str, bool)) and not callable(default):
                    literal = repr(default).replace("'", "''") if isinstance(default, str) else str(int(default) if isinstance(default, bool) else default)
                    ddl += f" DEFAULT {literal!s}" if not isinstance(default, str) else f" DEFAULT '{default}'"
                conn.execute(text(ddl))
                added.append(f"{table.name}.{col.name}")
    return added


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
