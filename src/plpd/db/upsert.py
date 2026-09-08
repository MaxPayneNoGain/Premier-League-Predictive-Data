"""Insert-or-replace against whichever database is behind the session.

Postgres and SQLite both speak ON CONFLICT but expose it through separate
constructs. The pipelines run on Postgres and the tests on SQLite, so choosing
between them here keeps both on the same statement rather than on a portable
fallback that neither uses in anger.
"""

from typing import Any, cast

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import Insert as PostgresInsert
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import Insert as SqliteInsert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from plpd.db.tables import Base


def upsert(session: Session, model: type[Base], rows: list[dict[str, Any]]) -> None:
    """Write rows, overwriting any that collide on the primary key.

    Reloading a snapshot has to be harmless and a later snapshot has to win, so
    a plain insert would fail on the second run and an insert-if-absent would
    quietly serve stale rows.
    """
    if not rows:
        return

    # Declared as a FromClause on the base, but a mapped class always resolves
    # it to a Table.
    table = cast(Table, model.__table__)
    keys = [column.name for column in table.primary_key]

    dialect = session.get_bind().dialect.name
    statement: PostgresInsert | SqliteInsert
    if dialect == "postgresql":
        statement = postgres_insert(table)
    elif dialect == "sqlite":
        statement = sqlite_insert(table)
    else:
        raise ValueError(f"no upsert is defined for the {dialect!r} dialect")

    session.execute(
        statement.on_conflict_do_update(
            index_elements=keys,
            set_={
                column.name: statement.excluded[column.name]
                for column in table.columns
                if column.name not in keys
            },
        ),
        rows,
    )
