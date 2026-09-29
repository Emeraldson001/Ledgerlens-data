"""Migration 0007: Add promotion/rollback actor columns to model_versions.

Grand 2 (issue #671) requires every promotion or rollback of a production
model to record its approving actor so promotion history is queryable and
auditable. Existing rows default to NULL (actor unknown / pre-migration —
these predate authorization enforcement and cannot be attributed).
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddPromotionActorColumns(Migration):
    id = "0007"
    description = (
        "Add nullable promoted_by, rolled_back_by, parent_version_id TEXT "
        "columns to model_versions"
    )

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "model_versions" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("model_versions")}
        if "promoted_by" not in existing:
            conn.execute(text("ALTER TABLE model_versions ADD COLUMN promoted_by VARCHAR"))
        if "rolled_back_by" not in existing:
            conn.execute(text("ALTER TABLE model_versions ADD COLUMN rolled_back_by VARCHAR"))
        if "parent_version_id" not in existing:
            conn.execute(text("ALTER TABLE model_versions ADD COLUMN parent_version_id VARCHAR"))


migration = AddPromotionActorColumns()
