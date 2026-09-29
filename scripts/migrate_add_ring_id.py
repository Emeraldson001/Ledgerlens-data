"""Backward-compatible migration: add the nullable `ring_id` column to
`risk_scores`.

New databases get the column automatically from
`detection.persistence.get_session_factory` (which runs `create_all`). This
script handles *existing* databases created before Issue #11: it performs a
plain, idempotent `ALTER TABLE ... ADD COLUMN ring_id`. Existing rows keep
`ring_id = NULL`.

Usage:
    python -m scripts.migrate_add_ring_id                    # uses config.RISK_SCORE_DB_URL
    python -m scripts.migrate_add_ring_id <db_url>            # explicit target
    python -m scripts.migrate_add_ring_id --dry-run <db_url>  # log the ALTER TABLE, don't run it
"""

import argparse

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from detection.persistence import get_engine
from scripts.sandbox import dry_run_guard, sandboxed_execution
from utils.logging import get_logger

logger = get_logger(__name__)

TABLE = "risk_scores"
COLUMN = "ring_id"


def migrate(engine: Engine | None = None, dry_run: bool = False) -> bool:
    """Add the `ring_id` column if it is missing. Returns True if it was added."""
    engine = engine or get_engine()
    inspector = inspect(engine)

    if TABLE not in inspector.get_table_names():
        logger.info("Table %s does not exist yet — nothing to migrate", TABLE)
        return False

    columns = {col["name"] for col in inspector.get_columns(TABLE)}
    if COLUMN in columns:
        logger.info("Column %s.%s already present — no migration needed", TABLE, COLUMN)
        return False

    def _add_column() -> bool:
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR"))
        logger.info("Added column %s.%s (existing rows default to NULL)", TABLE, COLUMN)
        return True

    result = dry_run_guard(dry_run, f"ALTER TABLE {TABLE} ADD COLUMN {COLUMN} VARCHAR", _add_column)
    return bool(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add the ring_id column to risk_scores.")
    parser.add_argument("db_url", nargs="?", default=None, help="Target database URL")
    parser.add_argument(
        "--dry-run", action="store_true", help="Log the migration without applying it"
    )
    args = parser.parse_args()

    with sandboxed_execution(allow_network=False):
        migrate(get_engine(args.db_url), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
