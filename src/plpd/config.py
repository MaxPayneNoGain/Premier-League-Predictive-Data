"""Settings for the ingest pipeline."""

import re
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SEASON_PATTERN = re.compile(r"^\d{4}-\d{4}$")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PLPD_", env_file=".env", extra="forbid")

    database_url: str = "postgresql+psycopg://plpd:plpd@localhost:5432/plpd"
    snapshot_root: Path = Path("data/snapshots")
    current_season: str = "2026-2027"
    http_timeout_seconds: float = 30.0

    # A branch follows upstream; a commit SHA freezes it, which is what a
    # reproducible backfill needs.
    fpl_core_ref: str = "main"
    odds_ref: str = "main"

    @field_validator("current_season")
    @classmethod
    def check_season(cls, value: str) -> str:
        if not SEASON_PATTERN.match(value):
            raise ValueError(f"season must look like '2026-2027', got {value!r}")
        return value
