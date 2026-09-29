"""Migration 0004: Add schema_version column to risk_scores.

Records the ``feature_contract_version`` integer from ``model_metadata.json``
that was current when the score was produced.  This lets downstream consumers
detect stale scores and trigger re-scoring when the feature schema advances.
Existing rows default to NULL (version unknown / pre-migration).
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddSchemaVersion(Migration):
    id = "0004"
    description = "Add nullable schema_version INTEGER column to risk_scores"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "risk_scores" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("risk_scores")}
        if "schema_version" not in existing:
            conn.execute(text("ALTER TABLE risk_scores ADD COLUMN schema_version INTEGER"))


migration = AddSchemaVersion()
