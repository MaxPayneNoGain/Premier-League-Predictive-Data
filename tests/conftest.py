from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from plpd.config import Settings
from plpd.db.tables import Base


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, snapshot_root=tmp_path)


@pytest.fixture
def session() -> Iterator[Session]:
    # SQLite rather than the docker-compose Postgres: every column uses a
    # generic type, so the DDL is the same and the suite needs no services.
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as active:
        yield active
    engine.dispose()
