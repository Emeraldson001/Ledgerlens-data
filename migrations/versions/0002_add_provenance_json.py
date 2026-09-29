"""Migration 0002: Add provenance_json column to risk_scores.

Stores a JSON blob mapping ``feature_name → [trade_id, ...]`` for feature
provenance tracking (Issue #244).  NULL when
``FEATURE_PROVENANCE_ENABLED=False`` or not computed.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddProvenanceJson(Migration):
    id = "0002"
    description = "Add nullable provenance_json TEXT column to risk_scores"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "risk_scores" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("risk_scores")}
        if "provenance_json" not in existing:
            conn.execute(text("ALTER TABLE risk_scores ADD COLUMN provenance_json TEXT"))


migration = AddProvenanceJson()
