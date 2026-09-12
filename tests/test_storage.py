from pathlib import Path

import pytest

from plpd.ingest.storage import FilesystemStore


def test_put_returns_a_uri_that_get_reads_back(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path)

    uri = store.put("fpl_core/teams/abc.parquet", b"bytes")

    assert store.get(uri) == b"bytes"


def test_exists_reports_what_was_put(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path)

    assert not store.exists("fpl_core/teams/abc.parquet")
    store.put("fpl_core/teams/abc.parquet", b"bytes")
    assert store.exists("fpl_core/teams/abc.parquet")


def test_put_creates_intermediate_directories(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path)

    store.put("a/b/c/d.parquet", b"bytes")

    assert (tmp_path / "a" / "b" / "c" / "d.parquet").read_bytes() == b"bytes"


def test_get_rejects_a_missing_uri(tmp_path: Path) -> None:
    store = FilesystemStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.get((tmp_path / "gone.parquet").as_posix())
