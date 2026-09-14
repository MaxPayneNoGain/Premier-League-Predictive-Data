from datetime import UTC, datetime

from sqlalchemy.orm import Session

from plpd.api import latest_predictions
from plpd.db.tables import Fixture, Prediction, Team
from plpd.predictions import MODEL

KICKOFF = datetime(2026, 8, 15, 14, 0, tzinfo=UTC)
BEFORE = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 8, 14, 18, 0, tzinfo=UTC)
AFTER = datetime(2026, 8, 16, 9, 0, tzinfo=UTC)

SEASON = "2026-2027"


def team(session: Session, code: int, name: str, *, season: str = SEASON, fpl_id: int = 1) -> None:
    session.add(
        Team(
            season=season,
            fpl_id=fpl_id,
            code=code,
            name=name,
            short_name=name[:3].upper(),
        )
    )
    session.flush()


def fixture(session: Session, match_id: str, **overrides: object) -> None:
    row = {
        "season": SEASON,
        "gameweek": 1,
        "tournament": "prem",
        "kickoff_time": KICKOFF,
        "home_team_code": 3,
        "away_team_code": 8,
        "finished": False,
    }
    row.update(overrides)
    session.add(Fixture(match_id=match_id, **row))
    session.flush()


def prediction(session: Session, match_id: str, **overrides: object) -> None:
    row = {
        "model": MODEL,
        "predicted_at": BEFORE,
        "trained_through": BEFORE,
        "home_win": 0.5,
        "draw": 0.3,
        "away_win": 0.2,
        "home_xg": 1.6,
        "away_xg": 1.1,
        "rho": -0.05,
    }
    row.update(overrides)
    session.add(Prediction(match_id=match_id, **row))
    session.flush()


def clubs(session: Session) -> None:
    team(session, 3, "Arsenal", fpl_id=1)
    team(session, 8, "Chelsea", fpl_id=2)


def test_only_the_newest_run_for_a_match_comes_back(session: Session) -> None:
    clubs(session)
    fixture(session, "match")
    prediction(session, "match", predicted_at=BEFORE, home_win=0.5)
    prediction(session, "match", predicted_at=LATER, home_win=0.7)

    rows = latest_predictions(session, season=SEASON, gameweek=1)

    assert len(rows) == 1
    assert rows[0].home_win == 0.7


def test_a_club_in_several_seasons_does_not_multiply_the_row(session: Session) -> None:
    clubs(session)
    team(session, 3, "Arsenal", season="2024-2025", fpl_id=1)
    team(session, 3, "Arsenal", season="2025-2026", fpl_id=1)
    team(session, 8, "Chelsea", season="2024-2025", fpl_id=2)
    fixture(session, "match")
    prediction(session, "match")

    assert len(latest_predictions(session, season=SEASON, gameweek=1)) == 1


def test_team_names_replace_the_codes(session: Session) -> None:
    clubs(session)
    fixture(session, "match")
    prediction(session, "match")

    row = latest_predictions(session, season=SEASON, gameweek=1)[0]

    assert (row.home_team, row.away_team) == ("Arsenal", "Chelsea")


def test_a_prediction_made_after_kickoff_is_a_retrodiction(session: Session) -> None:
    clubs(session)
    fixture(session, "forecast")
    fixture(session, "retrodiction", home_team_code=8, away_team_code=3)
    prediction(session, "forecast", predicted_at=BEFORE)
    prediction(session, "retrodiction", predicted_at=AFTER)

    rows = latest_predictions(session, season=SEASON, gameweek=1)

    assert {row.match_id: row.retrodiction for row in rows} == {
        "forecast": False,
        "retrodiction": True,
    }


def test_another_model_is_left_out_unless_it_is_asked_for(session: Session) -> None:
    clubs(session)
    fixture(session, "match")
    prediction(session, "match", model=MODEL, home_win=0.5)
    prediction(session, "match", model="elo", home_win=0.9, predicted_at=LATER)

    default = latest_predictions(session, season=SEASON, gameweek=1)
    named = latest_predictions(session, season=SEASON, gameweek=1, model="elo")

    assert [row.home_win for row in default] == [0.5]
    assert [row.home_win for row in named] == [0.9]


def test_a_round_with_no_predictions_is_empty(session: Session) -> None:
    clubs(session)
    fixture(session, "match")

    assert latest_predictions(session, season=SEASON, gameweek=1) == []


def test_rows_come_back_in_kickoff_order(session: Session) -> None:
    clubs(session)
    fixture(session, "later", kickoff_time=datetime(2026, 8, 16, 14, 0, tzinfo=UTC))
    fixture(session, "earlier", kickoff_time=KICKOFF)
    prediction(session, "later")
    prediction(session, "earlier")

    rows = latest_predictions(session, season=SEASON, gameweek=1)

    assert [row.match_id for row in rows] == ["earlier", "later"]
