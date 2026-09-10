"""Client for a mirror of football-data.co.uk's results and bookmaker prices.

https://github.com/xgabora/Club-Football-Match-Data - the same data behind a host
that stays up, with ClubElo ratings joined on. One CSV covers 27 countries, so a
pull is narrowed to the English top flight and a single season before anything is
archived, and hashed over what is kept rather than over the whole download.
"""

import httpx
import pandas as pd

from plpd.config import Settings
from plpd.ingest.fpl_core import SourceError, SourceFile, read_csv

ODDS_BASE = "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data"

DIVISION = "E0"
BOOKMAKER = "bet365"

REQUIRED_COLUMNS = {
    "Division",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "OddHome",
    "OddDraw",
    "OddAway",
}


def season_window(season: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = int(season[:4])
    return pd.Timestamp(f"{start}-07-01"), pd.Timestamp(f"{start + 1}-07-01")


class ClubFootballOddsSource:
    name = "club_football_odds"

    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.client = client or httpx.Client(timeout=settings.http_timeout_seconds)

    def fetch(self, season: str, gameweek: int | None = None) -> list[SourceFile]:
        if gameweek is not None:
            raise SourceError("odds are published for whole seasons, not by gameweek")

        url = f"{ODDS_BASE}/{self.settings.odds_ref}/data/Matches.csv"
        response = self.client.get(url, follow_redirects=True)
        if response.status_code == httpx.codes.NOT_FOUND:
            raise SourceError(f"{url} returned 404")
        response.raise_for_status()

        frame = read_csv(response.content)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SourceError(f"Matches.csv is missing column(s): {', '.join(sorted(missing))}")

        start, end = season_window(season)
        played = pd.to_datetime(frame["MatchDate"], errors="coerce")
        kept = frame[(frame["Division"] == DIVISION) & (played >= start) & (played < end)]
        kept = kept.reset_index(drop=True)
        if kept.empty:
            raise SourceError(f"no {DIVISION} matches for {season} in {url}")

        return [SourceFile(name="odds.csv", raw=kept.to_csv(index=False).encode(), frame=kept)]
