"""Poisson goals model fitted by maximum likelihood, with the Dixon-Coles correction."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

Vector = npt.NDArray[np.float64]
Indices = npt.NDArray[np.int_]

MAX_GOALS = 10

SECONDS_PER_DAY = 86400.0

# Swept over the 2024-2025 folds and chosen on out-of-sample RPS. The curve is
# flat from roughly 180 days upward and clearly worse below 90.
HALF_LIFE_DAYS = 240.0

RHO_BOUNDS = (-0.3, 0.3)

# Three matches barely constrain a club, so the first time it fails to score
# its attack runs toward minus infinity.
RIDGE_PENALTY = 1.0

# The optimiser can try a rho that drives tau non-positive on its way to a fit.
TAU_FLOOR = 1e-10


@dataclass(frozen=True)
class PoissonModel:
    teams: dict[int, int]
    attack: Vector
    # Enters the rate negatively, so a higher value means fewer goals conceded.
    defence: Vector
    home_advantage: float
    intercept: float
    rho: float = 0.0


def _unpack(params: Vector, teams: int) -> tuple[Vector, Vector, float, float, float]:
    free = teams - 1
    attack = np.empty(teams)
    defence = np.empty(teams)
    attack[:free] = params[:free]
    defence[:free] = params[free : 2 * free]
    # Both strengths are held to sum to zero. Without that a constant shifts
    # between them and the intercept and the fit has no single solution.
    attack[-1] = -attack[:free].sum()
    defence[-1] = -defence[:free].sum()
    return attack, defence, float(params[-3]), float(params[-2]), float(params[-1])


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


def _tau(
    home_goals: Vector,
    away_goals: Vector,
    home_rate: Vector,
    away_rate: Vector,
    rho: float,
) -> Vector:
    # Dixon and Coles (1997). Independent scoring puts too little probability on
    # 0-0 and 1-1 and too much on 1-0 and 0-1. A negative rho moves it back.
    tau = np.ones_like(home_rate)
    goalless = (home_goals == 0) & (away_goals == 0)
    away_only = (home_goals == 0) & (away_goals == 1)
    home_only = (home_goals == 1) & (away_goals == 0)
    one_each = (home_goals == 1) & (away_goals == 1)
    tau[goalless] = 1 - home_rate[goalless] * away_rate[goalless] * rho
    tau[away_only] = 1 + home_rate[away_only] * rho
    tau[home_only] = 1 + away_rate[home_only] * rho
    tau[one_each] = 1 - rho
    return tau


def _decay_weights(matches: pd.DataFrame, half_life: float) -> Vector:
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    if "kickoff_time" not in matches:
        raise ValueError("time decay needs a kickoff_time column")

    kickoff = pd.to_datetime(matches["kickoff_time"])
    age = (kickoff.max() - kickoff).dt.total_seconds() / SECONDS_PER_DAY
    weights: Vector = np.exp2(-age.to_numpy(dtype=np.float64) / half_life)
    return weights


def _negative_log_likelihood(
    params: Vector,
    teams: int,
    home: Indices,
    away: Indices,
    home_goals: Vector,
    away_goals: Vector,
    weights: Vector,
    penalty: float,
) -> float:
    attack, defence, home_advantage, intercept, rho = _unpack(params, teams)
    log_home, log_away = _log_rates(attack, defence, home_advantage, intercept, home, away)
    home_rate = np.exp(log_home)
    away_rate = np.exp(log_away)
    tau = _tau(home_goals, away_goals, home_rate, away_rate, rho)
    # The log(y!) term is left out. It does not depend on the parameters.
    return float(
        np.sum(weights * (home_rate - home_goals * log_home))
        + np.sum(weights * (away_rate - away_goals * log_away))
        - np.sum(weights * np.log(np.clip(tau, TAU_FLOOR, None)))
        # Taken over the unpacked strengths, so the team whose value is derived
        # from the others is penalised alongside them.
        + penalty * float(attack @ attack + defence @ defence)
    )


def fit(
    matches: pd.DataFrame,
    *,
    correlation: bool = False,
    half_life: float | None = None,
    penalty: float = RIDGE_PENALTY,
) -> PoissonModel:
    if matches.empty:
        raise ValueError("a Poisson model needs at least one match to fit")

    weights = np.ones(len(matches)) if half_life is None else _decay_weights(matches, half_life)

    codes = sorted(set(matches["home_team_code"]) | set(matches["away_team_code"]))
    teams = {int(code): index for index, code in enumerate(codes)}

    home = matches["home_team_code"].map(teams).to_numpy(dtype=np.int_)
    away = matches["away_team_code"].map(teams).to_numpy(dtype=np.int_)
    home_goals = matches["home_goals"].to_numpy(dtype=np.float64)
    away_goals = matches["away_goals"].to_numpy(dtype=np.float64)

    start = np.zeros(2 * len(teams) + 1)
    start[-3] = 0.25
    start[-2] = np.log(max(np.mean(np.concatenate([home_goals, away_goals])), 0.1))

    # Pinned to zero rather than dropped from the vector, so the independent fit
    # and the corrected one run the same code.
    bounds = [(None, None)] * (len(start) - 1) + [RHO_BOUNDS if correlation else (0.0, 0.0)]

    fitted = minimize(
        _negative_log_likelihood,
        start,
        args=(len(teams), home, away, home_goals, away_goals, weights, penalty),
        method="L-BFGS-B",
        bounds=bounds,
    )
    if not fitted.success:
        raise RuntimeError(f"the Poisson fit did not converge: {fitted.message}")

    attack, defence, home_advantage, intercept, rho = _unpack(fitted.x, len(teams))
    return PoissonModel(teams, attack, defence, home_advantage, intercept, rho)


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


def score_matrix(
    home_rate: float, away_rate: float, rho: float = 0.0, max_goals: int = MAX_GOALS
) -> Vector:
    goals = np.arange(max_goals + 1)
    matrix: Vector = np.outer(poisson.pmf(goals, home_rate), poisson.pmf(goals, away_rate))
    matrix[0, 0] *= 1 - home_rate * away_rate * rho
    matrix[0, 1] *= 1 + home_rate * rho
    matrix[1, 0] *= 1 + away_rate * rho
    matrix[1, 1] *= 1 - rho
    # Renormalised so the truncated tail does not leave the outcomes short of 1.
    return matrix / matrix.sum()


def outcome_probabilities(matrix: Vector) -> Vector:
    return np.array(
        [np.tril(matrix, -1).sum(), np.trace(matrix), np.triu(matrix, 1).sum()],
        dtype=np.float64,
    )


def predict(model: PoissonModel, fixtures: pd.DataFrame) -> Vector:
    rows = []
    for home, away in zip(fixtures["home_team_code"], fixtures["away_team_code"], strict=True):
        home_rate, away_rate = expected_goals(model, int(home), int(away))
        rows.append(outcome_probabilities(score_matrix(home_rate, away_rate, model.rho)))
    return np.vstack(rows)
