from datetime import UTC, datetime

from sqlalchemy.orm import Session

from plpd.db.tables import Fixture, Odds
from plpd.features import finished_matches, match_odds, outcomes

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
