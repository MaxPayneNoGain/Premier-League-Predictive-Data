from datetime import UTC, datetime

import pandas as pd
import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from plpd.db.tables import Fixture, Odds, Player, Snapshot, SnapshotFile, Team
from plpd.ingest.load import (
    _club_code,
    archived_snapshots,
    has_frame,
    load_fixtures,
    load_matches,
    load_odds,
    load_players,
    load_teams,
    read_frame,
    to_bool,
    to_datetime,
    to_int,
    tournament_of,
)
from plpd.ingest.snapshots import file_hash
from plpd.ingest.storage import FilesystemStore

FETCHED_AT = datetime(2026, 8, 31, 8, 0, tzinfo=UTC)

TEAMS = pd.DataFrame(
    [
        {"code": "3", "id": "1", "name": "Arsenal", "short_name": "ARS"},
        {"code": "91", "id": "2", "name": "AFC Bournemouth", "short_name": "BOU"},
    ]
)

# Upstream writes the team columns as floats and the flags as words, so the
# archived text carries both spellings verbatim.
FIXTURES = pd.DataFrame(
    [
        {
            "gameweek": "3",
            "tournament": "prem",
            "match_id": "26-27-prem-ipswich-town-vs-liverpool",
            "kickoff_time": "2026-09-04T19:00:00",
            "home_team": "40.0",
            "away_team": "14.0",
            "home_team_elo": "",
            "away_team_elo": "",
            "home_score": "0.0",
            "away_score": "2.0",
            "finished": "True",
        },
        {
            "gameweek": "3",
            "tournament": "prem",
            "match_id": "26-27-prem-arsenal-vs-chelsea",
            "kickoff_time": "2026-09-06T15:30:00",
            "home_team": "3.0",
            "away_team": "8.0",
            "home_team_elo": "",
            "away_team_elo": "",
            "home_score": "",
            "away_score": "",
            "finished": "False",
        },
    ]
)

PLAYERS = pd.DataFrame(
    [
        {
            "player_code": "208706",
            "player_id": "452",
            "first_name": "Bruno",
            "second_name": "Guimaraes",
            "web_name": "Bruno G.",
            "team_code": "4",
            "position": "Midfielder",
        }
    ]
)


# The older layout: uppercase flags, plain integer codes, a space in the
# timestamp, and no tournament column at all.
MATCHES = pd.DataFrame(
    [
        {
            "gameweek": "8",
            "match_id": "24-25-prem-afc-bournemouth-vs-arsenal",
            "kickoff_time": "2024-10-19 16:30:00",
            "home_team": "91",
            "away_team": "3",
            "home_team_elo": "1702.37",
            "away_team_elo": "1981.59",
            "home_score": "2",
            "away_score": "0",
            "finished": "TRUE",
        }
    ]
)


def make_snapshot(
    session: Session,
    store: FilesystemStore,
    frames: dict[str, pd.DataFrame],
    *,
    tag: str = "a",
    fetched_at: datetime = FETCHED_AT,
    source: str = "fpl_core",
    season: str = "2026-2027",
) -> Snapshot:
    snapshot = Snapshot(
        source=source,
        season=season,
        fetched_at=fetched_at,
        source_ref="main",
        content_hash=tag * 64,
        row_count=sum(len(frame) for frame in frames.values()),
    )
    session.add(snapshot)
    session.flush()

    for name, frame in frames.items():
        data = frame.to_parquet(index=False)
        assert data is not None
        session.add(
            SnapshotFile(
                snapshot_id=snapshot.id,
                name=name,
                content_hash=file_hash(data),
                storage_uri=store.put(f"{tag}/{name}.parquet", data),
                row_count=len(frame),
            )
        )
    session.flush()
    return snapshot


def test_teams_load_with_the_snapshot_season(session: Session, store: FilesystemStore) -> None:
    snapshot = make_snapshot(session, store, {"teams": TEAMS})

    assert load_teams(session, snapshot, store=store) == 2

    arsenal = session.get(Team, ("2026-2027", 1))
    assert arsenal is not None
    assert (arsenal.name, arsenal.code) == ("Arsenal", 3)


def test_players_keep_id_and_code_apart(session: Session, store: FilesystemStore) -> None:
    snapshot = make_snapshot(session, store, {"players": PLAYERS})

    load_players(session, snapshot, store=store)

    player = session.get(Player, ("2026-2027", 452))
    assert player is not None
    assert player.code == 208706
    assert player.team_code == 4


def test_a_later_snapshot_overwrites_rather_than_duplicates(
    session: Session, store: FilesystemStore
) -> None:
    load_teams(session, make_snapshot(session, store, {"teams": TEAMS}), store=store)

    renamed = TEAMS.copy()
    renamed.loc[0, "short_name"] = "ARE"
    load_teams(session, make_snapshot(session, store, {"teams": renamed}, tag="b"), store=store)

    assert len(session.scalars(select(Team)).all()) == 2
    arsenal = session.get(Team, ("2026-2027", 1))
    assert arsenal is not None
    assert arsenal.short_name == "ARE"


def test_a_missing_file_names_the_snapshot_and_the_file(
    session: Session, store: FilesystemStore
) -> None:
    snapshot = make_snapshot(session, store, {"teams": TEAMS})

    with pytest.raises(FileNotFoundError, match="snapshot 1 has no players"):
        load_players(session, snapshot, store=store)


def test_float_formatted_team_codes_become_integers(
    session: Session, store: FilesystemStore
) -> None:
    snapshot = make_snapshot(session, store, {"GW3__fixtures": FIXTURES})

    assert load_fixtures(session, snapshot, store=store) == 2

    arsenal = session.get(Fixture, "26-27-prem-arsenal-vs-chelsea")
    assert arsenal is not None
    assert (arsenal.home_team_code, arsenal.away_team_code) == (3, 8)
    assert arsenal.season == "2026-2027"


def test_an_unplayed_fixture_is_not_marked_finished(
    session: Session, store: FilesystemStore
) -> None:
    load_fixtures(session, make_snapshot(session, store, {"GW3__fixtures": FIXTURES}), store=store)

    played = session.get(Fixture, "26-27-prem-ipswich-town-vs-liverpool")
    unplayed = session.get(Fixture, "26-27-prem-arsenal-vs-chelsea")
    assert played is not None and unplayed is not None
    assert played.finished is True
    assert unplayed.finished is False


def test_blank_scores_and_elo_load_as_null_not_zero(
    session: Session, store: FilesystemStore
) -> None:
    load_fixtures(session, make_snapshot(session, store, {"GW3__fixtures": FIXTURES}), store=store)

    unplayed = session.get(Fixture, "26-27-prem-arsenal-vs-chelsea")
    assert unplayed is not None
    assert unplayed.home_score is None
    assert unplayed.home_team_elo is None


def test_kickoff_times_are_read_as_utc() -> None:
    # SQLite drops the offset on the way back out, so this checks the cast
    # rather than a round trip.
    assert to_datetime("2026-09-05T14:00:00") == datetime(2026, 9, 5, 14, 0, tzinfo=UTC)
    assert to_datetime("") is None


def test_a_fractional_code_is_rejected_rather_than_truncated() -> None:
    with pytest.raises(ValueError, match="whole number"):
        to_int("3.5")


def test_a_snapshot_without_fixtures_is_named_in_the_error(
    session: Session, store: FilesystemStore
) -> None:
    snapshot = make_snapshot(session, store, {"teams": TEAMS})

    with pytest.raises(FileNotFoundError, match="no gameweek fixtures"):
        load_fixtures(session, snapshot, store=store)


def test_a_replay_runs_oldest_first_so_the_newest_pull_wins(
    session: Session, store: FilesystemStore
) -> None:
    older = make_snapshot(session, store, {"teams": TEAMS})
    renamed = TEAMS.copy()
    renamed.loc[0, "short_name"] = "ARE"
    newer = make_snapshot(
        session,
        store,
        {"teams": renamed},
        tag="b",
        fetched_at=datetime(2026, 8, 31, 18, 0, tzinfo=UTC),
    )

    replay = archived_snapshots(session, source="fpl_core", season="2026-2027")
    assert [snapshot.id for snapshot in replay] == [older.id, newer.id]

    for snapshot in replay:
        load_teams(session, snapshot, store=store)

    arsenal = session.get(Team, ("2026-2027", 1))
    assert arsenal is not None
    assert arsenal.short_name == "ARE"


def test_a_whole_season_file_loads_with_its_competition_recovered(
    session: Session, store: FilesystemStore
) -> None:
    snapshot = make_snapshot(
        session, store, {"matches": MATCHES}, source="fpl_core_legacy", season="2024-2025"
    )

    assert load_matches(session, snapshot, store=store) == 1

    played = session.get(Fixture, "24-25-prem-afc-bournemouth-vs-arsenal")
    assert played is not None
    assert played.season == "2024-2025"
    assert played.tournament == "prem"
    assert (played.home_team_code, played.away_team_code) == (91, 3)
    assert (played.home_score, played.away_score) == (2, 0)
    assert played.finished


def test_a_space_separated_kickoff_reads_as_utc() -> None:
    assert to_datetime("2024-10-19 16:30:00") == datetime(2024, 10, 19, 16, 30, tzinfo=UTC)


def test_an_uppercase_flag_is_read_as_a_boolean() -> None:
    assert to_bool("TRUE") is True
    assert to_bool("FALSE") is False


def test_a_flag_that_is_neither_spelling_is_rejected() -> None:
    with pytest.raises(ValueError, match="boolean spelling"):
        to_bool("yes")


def test_the_league_is_recovered_from_a_match_id() -> None:
    assert tournament_of("24-25-prem-afc-bournemouth-vs-arsenal", "2024-2025") == "prem"


def test_a_competition_whose_name_has_a_hyphen_survives() -> None:
    cup = tournament_of("25-26-efl-cup-bristol-city-vs-fulham", "2025-2026")
    europe = tournament_of("25-26-champions-league-atletico-madrid-vs-liverpool", "2025-2026")

    assert (cup, europe) == ("efl-cup", "champions-league")


def test_a_match_id_from_another_season_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not belong"):
        tournament_of("24-25-prem-arsenal-vs-chelsea", "2025-2026")


def test_an_unknown_competition_is_rejected() -> None:
    with pytest.raises(ValueError, match="no known competition"):
        tournament_of("24-25-friendly-arsenal-vs-chelsea", "2024-2025")


def test_a_snapshot_reports_which_frames_it_holds(session: Session, store: FilesystemStore) -> None:
    snapshot = make_snapshot(
        session, store, {"matches": MATCHES}, source="fpl_core_legacy", season="2024-2025"
    )

    assert has_frame(session, snapshot, "matches")
    assert not has_frame(session, snapshot, "teams")


def test_read_frame_uses_the_index(session: Session, store: FilesystemStore) -> None:
    snapshot = make_snapshot(session, store, {"teams": TEAMS})
    session.execute(delete(SnapshotFile).where(SnapshotFile.name == "teams"))

    with pytest.raises(FileNotFoundError, match="has no teams"):
        read_frame(session, snapshot, "teams", store=store)


ODDS_TEAMS = pd.DataFrame(
    [
        {"code": "3", "id": "1", "name": "Arsenal", "short_name": "ARS"},
        {"code": "6", "id": "2", "name": "Spurs", "short_name": "TOT"},
    ]
)

ODDS_FIXTURES = pd.DataFrame(
    [
        {
            "gameweek": "4",
            "tournament": "prem",
            "match_id": "26-27-prem-arsenal-vs-tottenham-hotspur",
            "kickoff_time": "2026-09-12T16:30:00",
            "home_team": "3.0",
            "away_team": "6.0",
            "home_team_elo": "",
            "away_team_elo": "",
            "home_score": "2.0",
            "away_score": "1.0",
            "finished": "True",
        }
    ]
)

ODDS_ROWS = pd.DataFrame(
    [
        {
            "Division": "E0",
            "MatchDate": "2026-09-12",
            "HomeTeam": "Arsenal",
            "AwayTeam": "Tottenham",
            "OddHome": "1.75",
            "OddDraw": "3.90",
            "OddAway": "4.20",
        }
    ]
)


def load_a_season(session: Session, store: FilesystemStore) -> None:
    load_teams(session, make_snapshot(session, store, {"teams": ODDS_TEAMS}, tag="t"), store=store)
    load_fixtures(
        session,
        make_snapshot(session, store, {"GW4__fixtures": ODDS_FIXTURES}, tag="f"),
        store=store,
    )


def test_odds_reach_their_fixture_through_the_club_aliases(
    session: Session, store: FilesystemStore
) -> None:
    load_a_season(session, store)

    assert (
        load_odds(session, make_snapshot(session, store, {"odds": ODDS_ROWS}, tag="o"), store=store)
        == 1
    )

    priced = session.get(Odds, "26-27-prem-arsenal-vs-tottenham-hotspur")
    assert priced is not None
    assert (priced.home_win, priced.draw, priced.away_win) == (1.75, 3.90, 4.20)
    assert priced.bookmaker == "bet365"


def test_a_club_the_odds_source_renamed_stops_the_load(
    session: Session, store: FilesystemStore
) -> None:
    load_a_season(session, store)
    strange = ODDS_ROWS.copy()
    strange.loc[0, "AwayTeam"] = "Real Madrid"

    with pytest.raises(ValueError, match="no team code"):
        load_odds(session, make_snapshot(session, store, {"odds": strange}, tag="o"), store=store)


def test_odds_need_the_season_teams_loaded_first(session: Session, store: FilesystemStore) -> None:
    with pytest.raises(ValueError, match="no teams loaded"):
        load_odds(session, make_snapshot(session, store, {"odds": ODDS_ROWS}, tag="o"), store=store)


def test_a_match_with_no_price_is_left_alone(session: Session, store: FilesystemStore) -> None:
    load_a_season(session, store)
    blank = ODDS_ROWS.copy()
    blank.loc[0, "OddDraw"] = ""

    assert (
        load_odds(session, make_snapshot(session, store, {"odds": blank}, tag="o"), store=store)
        == 0
    )


def test_a_priced_match_we_never_loaded_is_skipped(
    session: Session, store: FilesystemStore
) -> None:
    load_a_season(session, store)
    elsewhere = ODDS_ROWS.copy()
    elsewhere.loc[0, "MatchDate"] = "2026-09-19"

    assert (
        load_odds(session, make_snapshot(session, store, {"odds": elsewhere}, tag="o"), store=store)
        == 0
    )


def test_a_season_spelling_a_club_the_odds_way_skips_the_alias() -> None:
    assert _club_code("Ipswich", {"Ipswich": 40}) == 40


def test_the_promoted_clubs_keep_the_suffix_the_odds_source_drops() -> None:
    codes = {"Coventry City": 3, "Hull City": 8, "Ipswich Town": 40}

    assert [_club_code(name, codes) for name in ("Coventry", "Hull", "Ipswich")] == [3, 8, 40]
