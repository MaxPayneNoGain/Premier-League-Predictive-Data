import httpx
import pytest

from plpd.config import Settings
from plpd.ingest import FplCoreSource, SourceError

BODIES = {
    "players.csv": b"player_code,player_id,web_name,team_code,position\n118748,1,Vieira,3,MID\n",
    "teams.csv": b"code,id,name,short_name\n3,1,Arsenal,ARS\n",
    "playerstats.csv": b"id,status,chance_of_playing_next_round,news\n1,a,,\n",
    "fixtures.csv": (
        b"gameweek,kickoff_time,home_team,away_team,match_id\n"
        b"3,2026-09-05T16:30:00,88.0,7.0,26-27-prem-hull-city-vs-aston-villa\n"
    ),
    "playermatchstats.csv": b"player_id,match_id,minutes_played\n1,26-27-prem-hull,90.0\n",
}


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=BODIES[request.url.path.rsplit("/", 1)[-1]])


def source(handler: object, settings: Settings) -> FplCoreSource:
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return FplCoreSource(settings, client=httpx.Client(transport=transport))


def test_gameweek_files_are_only_fetched_when_asked(settings: Settings) -> None:
    assert len(source(ok, settings).fetch("2026-2027")) == 3
    assert len(source(ok, settings).fetch("2026-2027", gameweek=3)) == 5


def test_gameweek_directory_space_is_encoded(settings: Settings) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return ok(request)

    source(handler, settings).fetch("2026-2027", gameweek=3)
    assert any("By%20Gameweek/GW3" in url for url in seen)


def test_blank_fields_stay_empty_strings(settings: Settings) -> None:
    files = source(ok, settings).fetch("2026-2027")
    stats = next(f for f in files if f.name == "playerstats.csv")
    assert stats.frame["news"].iloc[0] == ""


def test_missing_column_names_itself(settings: Settings) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("players.csv"):
            return httpx.Response(200, content=b"player_code,player_id\n1,2\n")
        return ok(request)

    with pytest.raises(SourceError, match="web_name"):
        source(handler, settings).fetch("2026-2027")
