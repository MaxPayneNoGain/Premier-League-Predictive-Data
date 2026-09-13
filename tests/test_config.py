import pytest
from pydantic import ValidationError

from plpd.config import Settings


def test_defaults_load() -> None:
    assert Settings(_env_file=None).current_season == "2026-2027"


@pytest.mark.parametrize("season", ["2026", "26-27", "2026/2027"])
def test_malformed_season_is_rejected(season: str) -> None:
    with pytest.raises(ValidationError, match="season"):
        Settings(_env_file=None, current_season=season)


def test_the_archive_defaults_to_the_filesystem() -> None:
    assert Settings(_env_file=None).archive_backend == "filesystem"


def test_an_unknown_archive_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PLPD_ARCHIVE_BACKEND", "s3")

    with pytest.raises(ValidationError, match="archive_backend"):
        Settings(_env_file=None)


def test_the_github_backend_needs_a_repo_and_a_token() -> None:
    with pytest.raises(ValidationError, match="archive_repo"):
        Settings(_env_file=None, archive_backend="github")
