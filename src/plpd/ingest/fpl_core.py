"""Client for the FPL Core Insights dataset.

https://github.com/olbauday/FPL-Core-Insights - refreshed at 07:30 and 17:30 UTC.
"""

from dataclasses import dataclass
from io import BytesIO
from urllib.parse import quote

import httpx
import pandas as pd

from plpd.config import Settings

RAW_BASE = "https://raw.githubusercontent.com/olbauday/FPL-Core-Insights"

SEASON_FILES = ("players.csv", "teams.csv", "playerstats.csv")
GAMEWEEK_FILES = ("fixtures.csv", "playermatchstats.csv")

# Only the columns something downstream reads. Upstream ships far more, and
# demanding all of them would break on a harmless addition.
REQUIRED_COLUMNS = {
    "players.csv": {"player_code", "player_id", "web_name", "team_code", "position"},
    "teams.csv": {"code", "id", "name", "short_name"},
    "playerstats.csv": {"id", "status", "chance_of_playing_next_round", "news"},
    "fixtures.csv": {"gameweek", "kickoff_time", "home_team", "away_team", "match_id"},
    "playermatchstats.csv": {"player_id", "match_id", "minutes_played"},
    "matches.csv": {
        "gameweek",
        "kickoff_time",
        "home_team",
        "away_team",
        "match_id",
        "home_score",
        "away_score",
        "finished",
    },
}


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceFile:
    name: str
    raw: bytes
    frame: pd.DataFrame


class FplCoreSource:
    name = "fpl_core"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.http_timeout_seconds)

    def fetch(self, season: str, gameweek: int | None = None) -> list[SourceFile]:
        # season files always, gameweek files only if one was asked for
        files = [self._fetch(self._season_url(season, name), name) for name in SEASON_FILES]
        if gameweek is not None:
            files += [
                self._fetch(self._gameweek_url(season, gameweek, name), f"GW{gameweek}/{name}")
                for name in GAMEWEEK_FILES
            ]
        return files

    def _season_url(self, season: str, filename: str) -> str:
        return f"{RAW_BASE}/{self.settings.fpl_core_ref}/data/{season}/{filename}"

    def _gameweek_url(self, season: str, gameweek: int, filename: str) -> str:
        directory = quote(f"By Gameweek/GW{gameweek}")
        return f"{RAW_BASE}/{self.settings.fpl_core_ref}/data/{season}/{directory}/{filename}"

    def _fetch(self, url: str, name: str) -> SourceFile:
        response = self.client.get(url)
        if response.status_code == httpx.codes.NOT_FOUND:
            raise SourceError(f"{url} returned 404 - season or gameweek may not exist yet")
        response.raise_for_status()

        frame = read_csv(response.content)
        required = REQUIRED_COLUMNS.get(name.rsplit("/", 1)[-1], set())
        missing = required - set(frame.columns)
        if missing:
            raise SourceError(f"{name} is missing column(s): {', '.join(sorted(missing))}")

        return SourceFile(name=name, raw=response.content, frame=frame)


class FplCoreLegacySource(FplCoreSource):
    """Seasons upstream stored before it moved to the `By Gameweek` layout.

    One file holds the whole finished season, so there is no gameweek to ask
    for and no separate fixtures file to stitch together.
    """

    name = "fpl_core_legacy"

    def fetch(self, season: str, gameweek: int | None = None) -> list[SourceFile]:
        if gameweek is not None:
            raise SourceError(f"{season} is stored as one whole-season file, not by gameweek")
        return [self._fetch(self._matches_url(season), "matches.csv")]

    def _matches_url(self, season: str) -> str:
        return f"{RAW_BASE}/{self.settings.fpl_core_ref}/data/{season}/matches/matches.csv"


def read_csv(raw: bytes) -> pd.DataFrame:
    """Read upstream CSV with every column left as text.

    Unplayed fixtures leave most columns blank, so inferred dtypes shift between
    pulls and the Parquet archives stop being comparable. dtype=str on its own
    still converts blanks to NaN, hence keep_default_na. Casting happens later,
    where the intended type is known.
    """
    return pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False)
