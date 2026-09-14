from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session, aliased

from plpd.db.tables import Fixture, Prediction, Team
from plpd.predictions import MODEL


@dataclass(frozen=True)
class PredictedMatch:
    match_id: str
    season: str
    gameweek: int
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


def latest_predictions(
    session: Session,
    *,
    season: str,
    gameweek: int,
    model: str = MODEL,
    tournament: str = "prem",
) -> list[PredictedMatch]:
    rank = (
        func.row_number()
        .over(
            partition_by=Prediction.match_id,
            order_by=(Prediction.predicted_at.desc(), Prediction.id.desc()),
        )
        .label("rank")
    )
    ranked = select(Prediction, rank).where(Prediction.model == model).subquery()

    home = aliased(Team)
    away = aliased(Team)
    statement = (
        select(
            Fixture.match_id,
            Fixture.season,
            Fixture.gameweek,
            Fixture.kickoff_time,
            home.name.label("home_team"),
            away.name.label("away_team"),
            ranked.c.home_win,
            ranked.c.draw,
            ranked.c.away_win,
            ranked.c.home_xg,
            ranked.c.away_xg,
            ranked.c.predicted_at,
            Fixture.finished,
        )
        .join(ranked, ranked.c.match_id == Fixture.match_id)
        # teams holds one row per club per season, so code alone multiplies the fixture.
        .join(home, (home.code == Fixture.home_team_code) & (home.season == Fixture.season))
        .join(away, (away.code == Fixture.away_team_code) & (away.season == Fixture.season))
        .where(
            Fixture.season == season,
            Fixture.gameweek == gameweek,
            Fixture.tournament == tournament,
            ranked.c.rank == 1,
        )
        .order_by(Fixture.kickoff_time, Fixture.match_id)
    )

    return [
        PredictedMatch(
            match_id=row.match_id,
            season=row.season,
            gameweek=row.gameweek,
            kickoff_time=row.kickoff_time,
            home_team=row.home_team,
            away_team=row.away_team,
            home_win=row.home_win,
            draw=row.draw,
            away_win=row.away_win,
            home_xg=row.home_xg,
            away_xg=row.away_xg,
            predicted_at=row.predicted_at,
            finished=row.finished,
            retrodiction=row.kickoff_time is not None and row.predicted_at > row.kickoff_time,
        )
        for row in session.execute(statement)
    ]
