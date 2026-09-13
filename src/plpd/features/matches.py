"""Model-ready views over the loaded tables.

The tables mirror upstream; this is where that shape is turned into something a
model can be fitted on. Frames start here — nothing below this layer handles a
DataFrame, which keeps the database code independent of the dataframe library.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from plpd.db.tables import Fixture, Odds

COLUMNS = [
    "match_id",
    "season",
    "gameweek",
    "kickoff_time",
    "home_team_code",
    "away_team_code",
    "home_goals",
    "away_goals",
]

ROUND_COLUMNS = [
    "match_id",
    "season",
    "gameweek",
    "kickoff_time",
    "home_team_code",
    "away_team_code",
]

PRICE_COLUMNS = ["home_odds", "draw_odds", "away_odds"]

ODDS_COLUMNS = ["match_id", *PRICE_COLUMNS]


def finished_matches(session: Session, *, tournament: str = "prem") -> pd.DataFrame:
    """Played matches with a result and both clubs identified, oldest first.

    Rows missing a team code are dropped rather than filled: upstream leaves the
    column empty for clubs outside FPL, so a European tie against Barcelona has
    no code to fit a strength for. Ordering by kickoff is what later lets a
    backtest cut the frame at a point in time.
    """
    statement = (
        select(
            Fixture.match_id,
            Fixture.season,
            Fixture.gameweek,
            Fixture.kickoff_time,
            Fixture.home_team_code,
            Fixture.away_team_code,
            Fixture.home_score.label("home_goals"),
            Fixture.away_score.label("away_goals"),
        )
        .where(
            Fixture.finished.is_(True),
            Fixture.tournament == tournament,
            Fixture.home_team_code.is_not(None),
            Fixture.away_team_code.is_not(None),
            Fixture.home_score.is_not(None),
            Fixture.away_score.is_not(None),
        )
        .order_by(Fixture.kickoff_time, Fixture.match_id)
    )
    rows = [dict(row) for row in session.execute(statement).mappings()]
    return pd.DataFrame(rows, columns=COLUMNS)


def round_fixtures(
    session: Session, *, season: str, gameweek: int, tournament: str = "prem"
) -> pd.DataFrame:
    """Every fixture in one round, played or not, oldest first."""
    statement = (
        select(
            Fixture.match_id,
            Fixture.season,
            Fixture.gameweek,
            Fixture.kickoff_time,
            Fixture.home_team_code,
            Fixture.away_team_code,
        )
        .where(
            Fixture.season == season,
            Fixture.gameweek == gameweek,
            Fixture.tournament == tournament,
            Fixture.home_team_code.is_not(None),
            Fixture.away_team_code.is_not(None),
        )
        .order_by(Fixture.kickoff_time, Fixture.match_id)
    )
    rows = [dict(row) for row in session.execute(statement).mappings()]
    return pd.DataFrame(rows, columns=ROUND_COLUMNS)


def next_round(session: Session, *, season: str, tournament: str = "prem") -> int | None:
    """The gameweek holding the soonest unplayed fixture, None once none are left.

    Read off kickoff order, not the lowest unplayed gameweek number. A match
    postponed into December would otherwise hold its round open until then.
    """
    statement = (
        select(Fixture.gameweek)
        .where(
            Fixture.season == season,
            Fixture.tournament == tournament,
            Fixture.finished.is_(False),
            Fixture.kickoff_time.is_not(None),
        )
        .order_by(Fixture.kickoff_time, Fixture.match_id)
        .limit(1)
    )
    gameweek = session.scalar(statement)
    return None if gameweek is None else int(gameweek)


def match_odds(session: Session, *, tournament: str = "prem") -> pd.DataFrame:
    """Decimal prices for one competition, at most one row per match.

    Joined through the fixtures so the tournament filter means the same thing
    here as it does for the matches themselves.
    """
    statement = (
        select(
            Odds.match_id,
            Odds.home_win.label("home_odds"),
            Odds.draw.label("draw_odds"),
            Odds.away_win.label("away_odds"),
        )
        .join(Fixture, Fixture.match_id == Odds.match_id)
        .where(Fixture.tournament == tournament)
    )
    rows = [dict(row) for row in session.execute(statement).mappings()]
    return pd.DataFrame(rows, columns=ODDS_COLUMNS)


def outcomes(matches: pd.DataFrame) -> npt.NDArray[np.int_]:
    """Label each match 0 home win, 1 draw, 2 away win."""
    # RPS needs draw between the two wins.
    margin = matches["home_goals"] - matches["away_goals"]
    return np.where(margin > 0, 0, np.where(margin == 0, 1, 2)).astype(np.int_)
