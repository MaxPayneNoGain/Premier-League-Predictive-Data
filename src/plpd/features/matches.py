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

from plpd.db.tables import Fixture

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
        .order_by(Fixture.kickoff_time)
    )
    rows = [dict(row) for row in session.execute(statement).mappings()]
    return pd.DataFrame(rows, columns=COLUMNS)


def outcomes(matches: pd.DataFrame) -> npt.NDArray[np.int_]:
    """Label each match 0 home win, 1 draw, 2 away win.

    The order is deliberate: these are ordinal, and the ranked probability score
    depends on a draw sitting between the two wins rather than beside them.
    """
    margin = matches["home_goals"] - matches["away_goals"]
    return np.where(margin > 0, 0, np.where(margin == 0, 1, 2)).astype(np.int_)
