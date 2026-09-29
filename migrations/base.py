"""Base class for all LedgerLens database migrations.

Every migration module in ``migrations/versions/`` must expose a module-level
instance named ``migration`` that is a :class:`Migration` subclass.  The
module's file name must start with the four-digit zero-padded ID followed by
an underscore (e.g. ``0001_add_ring_id.py``).

Migrations are expected to be **idempotent**: re-running an already-applied
migration should not change the database and should not raise an error.  The
:class:`MigrationRunner` guards against this by tracking applied migrations
in the ``schema_migrations`` tracking table, but the individual migration
implementations should be safe to call multiple times anyway (use
``IF NOT EXISTS`` / inspect-then-act patterns where possible).

Backwards-compatibility contract:
- Never delete a migration that has ever been applied to a production database.
- Never renumber an existing migration.
- To "undo" a migration, add a new forward-migration that reverses the change.
  Destructive down-migrations are not supported by design.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from sqlalchemy.engine import Connection


class Migration(ABC):
    """Abstract base class for a single schema migration step.

    Subclasses must set :attr:`id` and :attr:`description`, and implement
    :meth:`up`.
    """

    #: Four-digit zero-padded identifier matching the file prefix.
    id: str
    #: Short human-readable description shown in ``runner.status()`` output.
    description: str

    @abstractmethod
    def up(self, conn: Connection) -> None:
        """Apply the migration.

        Receives an active SQLAlchemy ``Connection`` with an open transaction.
        The :class:`MigrationRunner` commits the transaction if ``up`` returns
        normally, and rolls it back if it raises.

        Implementations should be idempotent: inspecting the current schema
        state before executing DDL statements (e.g. using
        ``sqlalchemy.inspect``) is the recommended pattern.
        """

    def __repr__(self) -> str:
        return f"<Migration {self.id}: {self.description}>"
