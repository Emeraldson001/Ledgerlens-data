"""Migration 0005: Add leaf-content columns to audit_merkle_roots.

Issue #670 (Grand 1): ``AuditMerkleChain`` previously kept leaf content
(``content_hash``, ``prev_merkle_root``) only in an in-process list, so a
routine process restart made ``verify_chain()`` indistinguishable from real
tampering (it raised ``TamperDetectedError`` for "entry missing from
in-memory chain" on every restart). These columns let the chain rehydrate
its full leaf history from durable storage on startup.

Existing rows (written before this migration) have no leaf content to
recover — ``content_hash``/``prev_merkle_root`` are nullable and stay NULL
for rows that predate this migration. ``AuditMerkleChain`` treats a NULL
``content_hash`` as "not rehydratable" and only rehydrates the contiguous
prefix of rows that have leaf content, starting a fresh append point after
the gap. This is a data-availability limitation of historical rows written
before the fix, not a correctness regression: verification of the
already-persisted root chain for those rows is unaffected, and no forward
migration can recover content that was never durably written.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class AddAuditMerkleLeafContent(Migration):
    id = "0005"
    description = "Add nullable content_hash/prev_merkle_root columns to audit_merkle_roots"

    def up(self, conn: Connection) -> None:
        inspector = inspect(conn)
        if "audit_merkle_roots" not in inspector.get_table_names():
            # Table is created on first AuditMerkleChain() instantiation with
            # these columns already present in the model — nothing to do.
            return
        existing = {col["name"] for col in inspector.get_columns("audit_merkle_roots")}
        if "content_hash" not in existing:
            conn.execute(text("ALTER TABLE audit_merkle_roots ADD COLUMN content_hash VARCHAR(64)"))
        if "prev_merkle_root" not in existing:
            conn.execute(
                text("ALTER TABLE audit_merkle_roots ADD COLUMN prev_merkle_root VARCHAR(64)")
            )


migration = AddAuditMerkleLeafContent()
