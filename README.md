# Premier League Predictive Data

EPL match, table and player predictions. Data from
[FPL Core Insights](https://github.com/olbauday/FPL-Core-Insights).

**Where it's at:** ingest only. Pulls the upstream CSVs, archives them as
Parquet, records a `snapshots` row. Schema exists for `teams`, `players` and
`fixtures`.

## Run it

```bash
uv sync --extra dev
docker compose up -d postgres
cp .env.example .env

uv run alembic upgrade head
uv run plpd-snapshot --gameweek 3
```

```bash
uv run ruff check . && uv run ruff format .
uv run mypy
uv run pytest
```

## Notes to self

- Snapshots are the whole point. Upstream overwrites its files in place, so
  without our own archive there's no point-in-time history and no honest
  backtest later.
- Archives keep every column as text; casting happens when the feature layer
  knows the intended type.
- Don't ingest `ep_this` / `ep_next` — FPL revises them after matches, which
  leaks results into a pre-kickoff prediction.


