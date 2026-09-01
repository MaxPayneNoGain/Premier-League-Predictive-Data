import pytest
from pydantic import ValidationError

from plpd.config import Settings


def test_defaults_load() -> None:
    assert Settings(_env_file=None).current_season == "2026-2027"


@pytest.mark.parametrize("season", ["2026", "26-27", "2026/2027"])
def test_malformed_season_is_rejected(season: str) -> None:
    with pytest.raises(ValidationError, match="season"):
        Settings(_env_file=None, current_season=season)
