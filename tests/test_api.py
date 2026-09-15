from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from plpd.api.app import app, get_session, settings
from plpd.db.tables import Fixture, Prediction, Team
from plpd.predictions import MODEL

SEASON = settings.current_season

KICKOFF = datetime(2026, 8, 15, 14, 0, tzinfo=UTC)
BEFORE = datetime(2026, 8, 14, 9, 0, tzinfo=UTC)


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_session] = lambda: session
    yield TestClient(app)
    app.dependency_overrides.clear()


def clubs(session: Session) -> None:
    session.add(Team(season=SEASON, fpl_id=1, code=3, name="Arsenal", short_name="ARS"))
    session.add(Team(season=SEASON, fpl_id=2, code=8, name="Chelsea", short_name="CHE"))
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


def test_health_answers_without_touching_the_database() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_a_round_returns_one_entry_per_fixture(client: TestClient, session: Session) -> None:
    clubs(session)
    fixture(session, "match")
    prediction(session, "match")

    body = client.get(f"/predictions/{SEASON}/1").json()

    assert body["season"] == SEASON
    assert body["gameweek"] == 1
    assert len(body["matches"]) == 1
    assert body["matches"][0]["home_team"] == "Arsenal"
    assert body["matches"][0]["away_team"] == "Chelsea"
    assert body["matches"][0]["retrodiction"] is False


def test_an_unpredicted_round_is_empty_not_missing(client: TestClient, session: Session) -> None:
    clubs(session)
    fixture(session, "match")

    response = client.get(f"/predictions/{SEASON}/1")

    assert response.status_code == 200
    assert response.json()["matches"] == []


def test_next_takes_the_soonest_unplayed_round(client: TestClient, session: Session) -> None:
    clubs(session)
    fixture(session, "played", gameweek=1, finished=True)
    fixture(session, "upcoming", gameweek=2)
    prediction(session, "upcoming")

    body = client.get("/predictions/next").json()

    assert body["gameweek"] == 2
    assert [match["match_id"] for match in body["matches"]] == ["upcoming"]


def test_next_is_404_once_the_season_is_over(client: TestClient, session: Session) -> None:
    clubs(session)
    fixture(session, "played", finished=True)

    assert client.get("/predictions/next").status_code == 404
