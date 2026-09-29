"""Migration 0001: Add ring_id column to risk_scores.

Backports the idempotent column addition previously handled by
``scripts/migrate_add_ring_id.py`` into the standard migration framework.
New databases created via ``Base.metadata.create_all`` already include this
column; this migration is for databases that pre-date the column's addition.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddRingId(Migration):
    id = "0001"
    description = "Add nullable ring_id column to risk_scores"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        # Table may not exist yet on a brand-new database (create_all handles it)
        if "risk_scores" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("risk_scores")}
        if "ring_id" not in existing:
            conn.execute(text("ALTER TABLE risk_scores ADD COLUMN ring_id VARCHAR"))


migration = AddRingId()
