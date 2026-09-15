from collections.abc import Iterator
from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from plpd.api.queries import latest_predictions
from plpd.config import Settings
from plpd.db import build_engine, build_session_factory
from plpd.features import next_round

settings = Settings()
session_factory = build_session_factory(build_engine(settings))

app = FastAPI(title="Premier League Predictive Data")


def get_session() -> Iterator[Session]:
    with session_factory() as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


class MatchPrediction(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    match_id: str
    kickoff_time: datetime | None
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    home_xg: float
    away_xg: float
    predicted_at: datetime
    finished: bool
    retrodiction: bool


class Round(BaseModel):
    season: str
    gameweek: int
    matches: list[MatchPrediction]


def _round(session: Session, season: str, gameweek: int) -> Round:
    rows = latest_predictions(session, season=season, gameweek=gameweek)
    return Round(
        season=season,
        gameweek=gameweek,
        matches=[MatchPrediction.model_validate(row) for row in rows],
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/predictions/next")
def predictions_next(session: SessionDep) -> Round:
    gameweek = next_round(session, season=settings.current_season)
    if gameweek is None:
        raise HTTPException(404, f"no unplayed round left in {settings.current_season}")
    return _round(session, settings.current_season, gameweek)


@app.get("/predictions/{season}/{gameweek}")
def predictions_round(season: str, gameweek: int, session: SessionDep) -> Round:
    return _round(session, season, gameweek)
