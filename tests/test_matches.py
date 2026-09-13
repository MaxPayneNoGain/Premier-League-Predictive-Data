from datetime import UTC, datetime

from sqlalchemy.orm import Session

from plpd.db.tables import Fixture, Odds
from plpd.features import finished_matches, match_odds, next_round, outcomes, round_fixtures

LATER = datetime(2025, 8, 17, 14, 0, tzinfo=UTC)
LATEST = datetime(2025, 8, 18, 14, 0, tzinfo=UTC)


def add(session: Session, match_id: str, **overrides: object) -> None:
    row = {
        "season": "2025-2026",
        "gameweek": 1,
        "tournament": "prem",
        "kickoff_time": datetime(2025, 8, 16, 14, 0, tzinfo=UTC),
        "home_team_code": 3,
        "away_team_code": 8,
        "home_score": 2,
        "away_score": 1,
        "finished": True,
    }
    row.update(overrides)
    session.add(Fixture(match_id=match_id, **row))
    session.flush()


def test_unplayed_matches_are_left_out(session: Session) -> None:
    add(session, "played")
    add(session, "upcoming", finished=False, home_score=None, away_score=None)

    assert list(finished_matches(session)["match_id"]) == ["played"]


def test_clubs_outside_fpl_are_left_out(session: Session) -> None:
    add(session, "league")
    add(session, "european", tournament="champions-league", away_team_code=None)

    assert list(finished_matches(session)["match_id"]) == ["league"]


def test_rows_come_back_oldest_first(session: Session) -> None:
    add(session, "later", kickoff_time=datetime(2025, 12, 1, 15, 0, tzinfo=UTC))
    add(session, "earlier", kickoff_time=datetime(2025, 8, 16, 15, 0, tzinfo=UTC))

    assert list(finished_matches(session)["match_id"]) == ["earlier", "later"]


def test_matches_kicking_off_together_come_back_in_a_fixed_order(session: Session) -> None:
    # A Saturday afternoon puts several matches on the same clock, and an order
    # that leaves those tied is free to change between two databases holding the
    # same rows.
    together = datetime(2025, 8, 16, 15, 0, tzinfo=UTC)
    add(session, "second", kickoff_time=together)
    add(session, "first", kickoff_time=together)
    add(session, "third", kickoff_time=together)

    assert list(finished_matches(session)["match_id"]) == ["first", "second", "third"]


def test_scores_arrive_as_goals(session: Session) -> None:
    add(session, "played", home_score=3, away_score=0)

    match = finished_matches(session).iloc[0]
    assert (match["home_goals"], match["away_goals"]) == (3, 0)
    assert match["home_team_code"] == 3


def test_an_empty_result_still_has_the_columns(session: Session) -> None:
    frame = finished_matches(session)

    assert frame.empty
    assert list(frame.columns) == [
        "match_id",
        "season",
        "gameweek",
        "kickoff_time",
        "home_team_code",
        "away_team_code",
        "home_goals",
        "away_goals",
    ]


def test_outcomes_label_home_draw_and_away_in_order(session: Session) -> None:
    add(session, "home-win", home_score=2, away_score=1)
    add(session, "draw", home_score=1, away_score=1, kickoff_time=LATER)
    add(session, "away-win", home_score=0, away_score=3, kickoff_time=LATEST)

    assert outcomes(finished_matches(session)).tolist() == [0, 1, 2]


def price(session: Session, match_id: str, **overrides: object) -> None:
    row = {"bookmaker": "bet365", "home_win": 1.9, "draw": 3.6, "away_win": 4.2}
    row.update(overrides)
    session.add(Odds(match_id=match_id, **row))
    session.flush()


def test_prices_come_back_named_for_the_outcome(session: Session) -> None:
    add(session, "played")
    price(session, "played")

    odds = match_odds(session)
    assert list(odds.columns) == ["match_id", "home_odds", "draw_odds", "away_odds"]
    assert odds.iloc[0].tolist() == ["played", 1.9, 3.6, 4.2]


def test_prices_for_another_competition_are_left_out(session: Session) -> None:
    add(session, "league")
    add(session, "cup", tournament="efl-cup")
    price(session, "league")
    price(session, "cup")

    assert list(match_odds(session)["match_id"]) == ["league"]


def test_an_unpriced_match_has_no_row(session: Session) -> None:
    add(session, "played")
    add(session, "unpriced", kickoff_time=LATER)
    price(session, "played")

    assert list(match_odds(session)["match_id"]) == ["played"]


def test_an_empty_price_result_still_has_the_columns(session: Session) -> None:
    frame = match_odds(session)

    assert frame.empty
    assert list(frame.columns) == ["match_id", "home_odds", "draw_odds", "away_odds"]


def test_a_round_keeps_fixtures_that_have_already_been_played(session: Session) -> None:
    add(session, "played")
    add(session, "upcoming", finished=False, home_score=None, away_score=None, kickoff_time=LATER)

    assert list(round_fixtures(session, season="2025-2026", gameweek=1)["match_id"]) == [
        "played",
        "upcoming",
    ]


def test_another_gameweek_is_left_out(session: Session) -> None:
    add(session, "this-week")
    add(session, "next-week", gameweek=2, kickoff_time=LATER)

    assert list(round_fixtures(session, season="2025-2026", gameweek=1)["match_id"]) == [
        "this-week"
    ]


def test_the_same_round_of_another_season_is_left_out(session: Session) -> None:
    add(session, "this-season")
    add(session, "last-season", season="2024-2025")

    assert list(round_fixtures(session, season="2025-2026", gameweek=1)["match_id"]) == [
        "this-season"
    ]


def test_a_round_leaves_out_clubs_outside_fpl(session: Session) -> None:
    add(session, "league")
    add(session, "european", tournament="champions-league", away_team_code=None)

    assert list(round_fixtures(session, season="2025-2026", gameweek=1)["match_id"]) == ["league"]


def test_a_round_comes_back_oldest_first(session: Session) -> None:
    add(session, "sunday", kickoff_time=LATEST)
    add(session, "saturday", kickoff_time=LATER)

    assert list(round_fixtures(session, season="2025-2026", gameweek=1)["match_id"]) == [
        "saturday",
        "sunday",
    ]


def test_an_empty_round_still_has_the_columns(session: Session) -> None:
    frame = round_fixtures(session, season="2025-2026", gameweek=1)

    assert frame.empty
    assert list(frame.columns) == [
        "match_id",
        "season",
        "gameweek",
        "kickoff_time",
        "home_team_code",
        "away_team_code",
    ]


def test_the_next_round_is_the_one_kicking_off_soonest(session: Session) -> None:
    add(session, "played")
    add(session, "upcoming", gameweek=2, finished=False, kickoff_time=LATER)
    add(session, "later-still", gameweek=3, finished=False, kickoff_time=LATEST)

    assert next_round(session, season="2025-2026") == 2


def test_a_postponed_round_does_not_jump_the_queue(session: Session) -> None:
    add(session, "postponed", gameweek=2, finished=False, kickoff_time=LATEST)
    add(session, "on-time", gameweek=3, finished=False, kickoff_time=LATER)

    assert next_round(session, season="2025-2026") == 3


def test_no_round_is_next_once_everything_is_played(session: Session) -> None:
    add(session, "played")

    assert next_round(session, season="2025-2026") is None


def test_a_fixture_without_a_kickoff_cannot_be_next(session: Session) -> None:
    add(session, "unscheduled", gameweek=2, finished=False, kickoff_time=None)
    add(session, "scheduled", gameweek=3, finished=False, kickoff_time=LATER)

    assert next_round(session, season="2025-2026") == 3
