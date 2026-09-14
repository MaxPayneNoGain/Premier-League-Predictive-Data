from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from plpd.config import Settings
from plpd.db.tables import Base
from plpd.ingest.storage import FilesystemStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, snapshot_root=tmp_path)


@pytest.fixture
def store(tmp_path: Path) -> FilesystemStore:
    return FilesystemStore(tmp_path)


@pytest.fixture
def session() -> Iterator[Session]:
    # SQLite rather than the docker-compose Postgres: every column uses a
    # generic type, so the DDL is the same and the suite needs no services.
    # TestClient serves on another thread, and an in-memory database is per connection.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as active:
        yield active
    engine.dispose()
