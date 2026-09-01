from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from plpd.config import Settings


def build_engine(settings: Settings) -> Engine:
    # pre_ping because the pipelines run as short cron jobs against a managed
    # Postgres that drops idle connections between them.
    return create_engine(settings.database_url, pool_pre_ping=True)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)
