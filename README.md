# Premier League Predictive Data

Home, draw and away probabilities for English Premier League fixtures, with expected
goals and a full scoreline distribution behind each one. Upstream data is archived
before it is used and every prediction is stored with the time it was made, so what
the model said before a match can be checked afterwards rather than taken on trust.

Fixtures, teams and players come from
[FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights). Bookmaker prices
come from [Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data)
and are used only as a benchmark to measure against, never as an input to the model.

## Where it stands

Ingest, archiving, the loader, the model, the evaluation harness and a read-only HTTP
API are all working and covered by tests. A web page for the upcoming round is being
built, and a scheduled job that records each round before it is played is coming
after that.

## The archive

Upstream overwrites its files in place, so there is no point-in-time history to be had
from it later and no honest backtest without keeping one here. Every pull is stored as
Parquet under its own content hash before anything is loaded, and a pull whose contents
have not changed is skipped rather than stored twice.

Archives keep every column as text and the loader decides each column's type, which is
why fetching and loading are separate commands: a bad cast can be corrected and
replayed against archives that already exist, without going back to the network. The
`ep_this` and `ep_next` columns are deliberately never ingested, because FPL revises
them after matches have been played and they would leak results into a prediction that
is supposed to predate kickoff.

## The model

Goals are modelled as two independent Poisson counts, with an attack and a defence
strength for every club and a single home advantage term, fitted by maximum likelihood.
The Dixon-Coles correction adjusts the four low-scoring cells that independence gets
wrong, which is where draws are otherwise under-predicted. Matches are weighted by age
on a 240 day half life, and team strengths carry a ridge penalty so that a club with very
few matches played cannot run away with an extreme rating, and the outcome probabilities
are shrunk toward the training base rate before they are recorded.

Measured walk-forward on the 700 scored matches of 2024-2025 and 2025-2026, so that every
round is predicted by a model fitted only on matches that kicked off before it, the model
closes 77% of the distance from base rates to de-vigged bet365 prices on ranked probability
score. Picking the single most likely result it is right 48.7% of the time, where bet365
itself manages 51.0% and always backing the home side gives 42.1%.

## Run it

```bash
uv sync --extra dev
docker compose up -d postgres
cp .env.example .env

uv run alembic upgrade head
uv run plpd-snapshot --gameweek 5
uv run plpd-load
uv run plpd-backtest --season 2024-2025 --season 2025-2026
uv run plpd-predict
uv run uvicorn plpd.api.app:app --reload
```

`plpd-predict` defaults to the next round still to be played. The checks are:

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

## Disclaimer

Not affiliated with or endorsed by the Premier League or any of its clubs. Built for
research, learning and entertainment, not betting advice.
