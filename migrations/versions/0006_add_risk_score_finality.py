"""Migration 0006: Add finality column to risk_scores.

Issue #670 (Grand 1): a risk score previously had no concept of finality
beyond ``updated_at`` — a downstream consumer could not distinguish a
provisional score (still changing as more trades arrive in a continuously
updating stream buffer) from a final score (a completed, bounded batch or
replay run over a closed time window).

Existing rows default to ``'provisional'``: they were written by the
continuous streaming/SSE path, which has no window-close event, so
"provisional" is the accurate description of their finality state. Rows
written going forward by a completed batch pipeline run or a completed
stream-replay run explicitly write ``'final'``.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddRiskScoreFinality(Migration):
    id = "0006"
    description = "Add finality VARCHAR column (default 'provisional') to risk_scores"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "risk_scores" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("risk_scores")}
        if "finality" not in existing:
            conn.execute(
                text(
                    "ALTER TABLE risk_scores ADD COLUMN finality VARCHAR(16) "
                    "NOT NULL DEFAULT 'provisional'"
                )
            )


migration = AddRiskScoreFinality()
