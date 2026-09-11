"""Predicting one matchweek and recording what was predicted.

The predictor here is the same one the backtest scores, imported rather than
rebuilt, so a recorded prediction and a reported score come from the same model.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy.orm import Session

from plpd.db.tables import Prediction
from plpd.evaluation import base_rates
from plpd.features import outcomes
from plpd.models import (
    HALF_LIFE_DAYS,
    SHRINKAGE_WEIGHT,
    PoissonModel,
    expected_goals,
    fit,
    predict,
    shrink,
)
from plpd.models.poisson import Vector

MODEL = "dixon-coles shrunk"

OUTCOME_COLUMNS = ["home_win", "draw", "away_win"]


def fit_and_predict(history: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[PoissonModel, Vector]:
    model = fit(history, correlation=True, half_life=HALF_LIFE_DAYS)
    blended = shrink(
        predict(model, fixtures),
        base_rates(outcomes(history), len(fixtures)),
        SHRINKAGE_WEIGHT,
    )
    return model, blended


def probabilities(history: pd.DataFrame, fixtures: pd.DataFrame) -> Vector:
    return fit_and_predict(history, fixtures)[1]


def predict_round(history: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    model, blended = fit_and_predict(history, fixtures)
    rates = [
        expected_goals(model, int(home), int(away))
        for home, away in zip(fixtures["home_team_code"], fixtures["away_team_code"], strict=True)
    ]

    frame = pd.DataFrame(blended, columns=OUTCOME_COLUMNS)
    frame.insert(0, "match_id", list(fixtures["match_id"]))
    frame["home_xg"] = [rate[0] for rate in rates]
    frame["away_xg"] = [rate[1] for rate in rates]
    frame["rho"] = model.rho
    return frame


def record(
    session: Session,
    predictions: pd.DataFrame,
    *,
    predicted_at: datetime,
    trained_through: datetime,
    model: str = MODEL,
) -> int:
    rows = [
        Prediction(
            match_id=str(row["match_id"]),
            model=model,
            predicted_at=predicted_at,
            trained_through=trained_through,
            home_win=float(row["home_win"]),
            draw=float(row["draw"]),
            away_win=float(row["away_win"]),
            home_xg=float(row["home_xg"]),
            away_xg=float(row["away_xg"]),
            rho=float(row["rho"]),
        )
        for row in predictions.to_dict(orient="records")
    ]
    session.add_all(rows)
    session.flush()
    return len(rows)
