"""Migration 0003: Add certified_robust column to risk_scores.

Marks scores that have been certified robust via Interval Bound Propagation
(IBP) at the standard evaluation epsilons (ε=0.01 and ε=0.05) — Issue #245.
Existing rows default to NULL (not yet evaluated).
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddCertifiedRobust(Migration):
    id = "0003"
    description = "Add nullable certified_robust BOOLEAN column to risk_scores"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "risk_scores" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("risk_scores")}
        if "certified_robust" not in existing:
            conn.execute(text("ALTER TABLE risk_scores ADD COLUMN certified_robust BOOLEAN"))


migration = AddCertifiedRobust()
