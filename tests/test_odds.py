import httpx
import pytest

from plpd.config import Settings
from plpd.ingest import ClubFootballOddsSource, SourceError
from plpd.ingest.odds import season_window

BODY = b"""Division,MatchDate,HomeTeam,AwayTeam,OddHome,OddDraw,OddAway
E0,2025-08-15,Liverpool,Bournemouth,1.30,6.00,8.50
E0,2026-05-24,Arsenal,Everton,1.45,4.75,7.00
E0,2024-08-16,Man United,Fulham,1.60,4.00,5.50
D1,2025-08-15,Bayern Munich,Leipzig,1.40,5.00,7.00
"""


def ok(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=BODY)


def source(handler: object, settings: Settings) -> ClubFootballOddsSource:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return ClubFootballOddsSource(settings, client=httpx.Client(transport=transport))


def test_a_season_runs_from_july_to_july() -> None:
    start, end = season_window("2025-2026")

    assert (start.year, start.month) == (2025, 7)
    assert (end.year, end.month) == (2026, 7)


def test_only_the_english_top_flight_is_kept(settings: Settings) -> None:
    files = source(ok, settings).fetch("2025-2026")

    assert [file.name for file in files] == ["odds.csv"]
    assert set(files[0].frame["Division"]) == {"E0"}


def test_only_the_season_asked_for_is_kept(settings: Settings) -> None:
    frame = source(ok, settings).fetch("2025-2026")[0].frame

    assert sorted(frame["MatchDate"]) == ["2025-08-15", "2026-05-24"]


def test_an_earlier_season_picks_up_its_own_matches(settings: Settings) -> None:
    frame = source(ok, settings).fetch("2024-2025")[0].frame

    assert list(frame["HomeTeam"]) == ["Man United"]


def test_the_hash_covers_only_the_rows_kept(settings: Settings) -> None:
    files = source(ok, settings).fetch("2024-2025")

    assert b"Bayern Munich" not in files[0].raw
    assert b"Man United" in files[0].raw


def test_a_season_with_no_matches_is_an_error(settings: Settings) -> None:
    with pytest.raises(SourceError, match="no E0 matches"):
        source(ok, settings).fetch("2019-2020")


def test_odds_are_published_by_season_not_gameweek(settings: Settings) -> None:
    with pytest.raises(SourceError, match="whole seasons"):
        source(ok, settings).fetch("2025-2026", gameweek=3)


def test_a_missing_column_names_itself(settings: Settings) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"Division,MatchDate\nE0,2025-08-15\n")

    with pytest.raises(SourceError, match="OddHome"):
        source(handler, settings).fetch("2025-2026")
