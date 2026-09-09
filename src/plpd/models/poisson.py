"""Independent Poisson goals model fitted by maximum likelihood."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

Vector = npt.NDArray[np.float64]
Indices = npt.NDArray[np.int_]

MAX_GOALS = 10


@dataclass(frozen=True)
class PoissonModel:
    teams: dict[int, int]
    attack: Vector
    # Enters the rate negatively, so a higher value means fewer goals conceded.
    defence: Vector
    home_advantage: float
    intercept: float


def _unpack(params: Vector, teams: int) -> tuple[Vector, Vector, float, float]:
    free = teams - 1
    attack = np.empty(teams)
    defence = np.empty(teams)
    attack[:free] = params[:free]
    defence[:free] = params[free : 2 * free]
    # Both strengths are held to sum to zero. Without that a constant shifts
    # between them and the intercept and the fit has no single solution.
    attack[-1] = -attack[:free].sum()
    defence[-1] = -defence[:free].sum()
    return attack, defence, float(params[-2]), float(params[-1])


def _log_rates(
    attack: Vector,
    defence: Vector,
    home_advantage: float,
    intercept: float,
    home: Indices,
    away: Indices,
) -> tuple[Vector, Vector]:
    return (
        intercept + home_advantage + attack[home] - defence[away],
        intercept + attack[away] - defence[home],
    )


def _negative_log_likelihood(
    params: Vector,
    teams: int,
    home: Indices,
    away: Indices,
    home_goals: Vector,
    away_goals: Vector,
) -> float:
    attack, defence, home_advantage, intercept = _unpack(params, teams)
    log_home, log_away = _log_rates(attack, defence, home_advantage, intercept, home, away)
    # The log(y!) term is left out. It does not depend on the parameters.
    return float(
        np.sum(np.exp(log_home) - home_goals * log_home)
        + np.sum(np.exp(log_away) - away_goals * log_away)
    )


def fit(matches: pd.DataFrame) -> PoissonModel:
    if matches.empty:
        raise ValueError("a Poisson model needs at least one match to fit")

    codes = sorted(set(matches["home_team_code"]) | set(matches["away_team_code"]))
    teams = {int(code): index for index, code in enumerate(codes)}

    home = matches["home_team_code"].map(teams).to_numpy(dtype=np.int_)
    away = matches["away_team_code"].map(teams).to_numpy(dtype=np.int_)
    home_goals = matches["home_goals"].to_numpy(dtype=np.float64)
    away_goals = matches["away_goals"].to_numpy(dtype=np.float64)

    start = np.zeros(2 * len(teams))
    start[-2] = 0.25
    start[-1] = np.log(max(np.mean(np.concatenate([home_goals, away_goals])), 0.1))

    fitted = minimize(
        _negative_log_likelihood,
        start,
        args=(len(teams), home, away, home_goals, away_goals),
        method="L-BFGS-B",
    )
    if not fitted.success:
        raise RuntimeError(f"the Poisson fit did not converge: {fitted.message}")

    attack, defence, home_advantage, intercept = _unpack(fitted.x, len(teams))
    return PoissonModel(teams, attack, defence, home_advantage, intercept)


def expected_goals(model: PoissonModel, home_code: int, away_code: int) -> tuple[float, float]:
    # A club with no matches in the training window sits at league average.
    home = model.teams.get(home_code)
    away = model.teams.get(away_code)
    home_attack = model.attack[home] if home is not None else 0.0
    home_defence = model.defence[home] if home is not None else 0.0
    away_attack = model.attack[away] if away is not None else 0.0
    away_defence = model.defence[away] if away is not None else 0.0
    return (
        float(np.exp(model.intercept + model.home_advantage + home_attack - away_defence)),
        float(np.exp(model.intercept + away_attack - home_defence)),
    )


def score_matrix(home_rate: float, away_rate: float, max_goals: int = MAX_GOALS) -> Vector:
    goals = np.arange(max_goals + 1)
    matrix: Vector = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    # Renormalised so the truncated tail does not leave the outcomes short of 1.
    return matrix / matrix.sum()


def outcome_probabilities(matrix: Vector) -> Vector:
    return np.array(
        [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()],
        dtype=np.float64,
    )


def predict(model: PoissonModel, fixtures: pd.DataFrame) -> Vector:
    rows = [
        outcome_probabilities(score_matrix(*expected_goals(model, int(home), int(away))))
        for home, away in zip(fixtures["home_team_code"], fixtures["away_team_code"], strict=True)
    ]
    return np.vstack(rows)
