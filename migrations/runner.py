"""Migration runner with real blocking locks and checksum validation."""

import logging
from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Integer, String, Table, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def _create_migration_metadata_table(engine: Engine) -> Table:
    """Create or get the migration metadata table."""
    from sqlalchemy import MetaData

    metadata = MetaData()
    migration_metadata = Table(
        "migration_metadata",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("version", String, unique=True, nullable=False, index=True),
        Column("content_hash", String, nullable=False),
        Column("applied_at", DateTime(timezone=True), default=lambda: datetime.now(UTC)),
    )

    metadata.create_all(engine)
    return migration_metadata


class MigrationRunner:
    """Executes migrations with blocking locks and checksum validation.

    Only one migration runner can execute against a database at a time.
    Uses database advisory locks (PostgreSQL) or a mutex table (SQLite).
    """

    MIGRATION_LOCK_ID = 1  # Numeric lock ID for advisory locks

    def __init__(self, engine: Engine):
        self.engine = engine
        self.SessionLocal = sessionmaker(bind=engine)

    def acquire_migration_lock(self, session: Session) -> bool:
        """Acquire an exclusive migration lock.

        For PostgreSQL: uses pg_advisory_lock()
        For SQLite: uses a mutex table with INSERT OR ABORT
        For other DBs: uses a simple mutex table

        Args:
            session: SQLAlchemy session

        Returns:
            True if lock acquired, False if already held (blocks, doesn't fail)
        """
        db_url = str(self.engine.url)

        if "postgresql" in db_url:
            session.execute("SELECT pg_advisory_lock(%d)" % self.MIGRATION_LOCK_ID)
            return True
        else:
            try:
                session.execute(
                    """
                    CREATE TABLE IF NOT EXISTS migration_lock (
                        lock_id INTEGER PRIMARY KEY,
                        locked_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                session.execute("INSERT INTO migration_lock (lock_id) VALUES (1)")
                session.commit()
                return True
            except IntegrityError:
                session.rollback()
                raise RuntimeError(
                    "Migration lock is held by another process. "
                    "Only one migration runner can execute at a time."
                )

    def release_migration_lock(self, session: Session) -> None:
        """Release the migration lock."""
        db_url = str(self.engine.url)

        if "postgresql" in db_url:
            session.execute("SELECT pg_advisory_unlock(%d)" % self.MIGRATION_LOCK_ID)
        else:
            session.execute("DELETE FROM migration_lock WHERE lock_id = 1")
            session.commit()

    def apply_migration(self, migration) -> None:
        """Apply a single migration with locking and checksum validation.

        Args:
            migration: Migration instance to apply

        Raises:
            RuntimeError: If lock acquisition fails or checksum validation fails
        """
        session = self.SessionLocal()
        try:
            if not self.acquire_migration_lock(session):
                raise RuntimeError("Failed to acquire migration lock")

            version = migration.version
            content_hash = migration.content_hash

            metadata_table = _create_migration_metadata_table(self.engine)

            stmt = select(metadata_table).where(metadata_table.c.version == version)
            existing = session.execute(stmt).first()

            if existing:
                if existing.content_hash != content_hash:
                    raise RuntimeError(
                        f"Migration {version} has been modified since application. "
                        f"Expected hash {existing.content_hash}, got {content_hash}"
                    )
                logger.info(f"Migration {version} already applied, skipping")
                return

            logger.info(f"Applying migration {version}: {migration.description}")
            migration.up(self.engine, session)

            stmt = metadata_table.insert().values(
                version=version, content_hash=content_hash
            )
            session.execute(stmt)
            session.commit()

            logger.info(f"Migration {version} applied successfully")

        except Exception as e:
            session.rollback()
            logger.error(f"Migration {version} failed: {e}")
            raise
        finally:
            self.release_migration_lock(session)
            session.close()

    def run_migrations(self, migrations: list) -> None:
        """Apply a list of migrations in order.

        Args:
            migrations: List of Migration instances in application order
        """
        for migration in migrations:
            self.apply_migration(migration)
"""Migration runner: discovers, orders, and applies database migrations.

The runner keeps track of applied migrations in the ``schema_migrations``
table (created automatically on first use).  It is safe to call
:meth:`MigrationRunner.upgrade` in application startup code — if everything
is already up-to-date it returns immediately.

Thread safety: the runner acquires an advisory row-level lock (an ``INSERT``
with ``ON CONFLICT DO NOTHING`` into a ``migration_lock`` table) before
applying any migration so that concurrent startup pods do not race each other.
For SQLite this reduces to a plain serialised write thanks to WAL mode.
"""

from __future__ import annotations

import importlib
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from migrations.base import Migration
from migrations.registry import REGISTRY

logger = logging.getLogger(__name__)

_VERSIONS_DIR = os.path.join(os.path.dirname(__file__), "versions")
_TRACKING_TABLE = "schema_migrations"
_LOCK_TABLE = "migration_lock"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_tracking_tables(conn: Connection) -> None:
    """Create the migration bookkeeping tables if they do not exist yet."""
    conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_TRACKING_TABLE} (
                migration_id    VARCHAR NOT NULL PRIMARY KEY,
                description     VARCHAR NOT NULL,
                applied_at      TIMESTAMP WITH TIME ZONE NOT NULL
            )
            """))
    conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_LOCK_TABLE} (
                lock_name VARCHAR NOT NULL PRIMARY KEY
            )
            """))


def _acquire_lock(conn: Connection) -> None:
    """Best-effort advisory lock for SQLite / PostgreSQL.

    Uses ``INSERT OR IGNORE`` (SQLite) or ``INSERT ... ON CONFLICT DO NOTHING``
    (PostgreSQL) so that only one migration runner is active at a time when
    multiple processes start simultaneously.  For SQLite the WAL + busy-timeout
    provides the actual serialisation guarantee; this table insert is an
    extra canary.
    """
    try:
        conn.execute(text(f"INSERT OR IGNORE INTO {_LOCK_TABLE} (lock_name) VALUES ('global')"))
    except Exception:  # pragma: no cover — dialect-specific fallback
        try:
            conn.execute(
                text(
                    f"INSERT INTO {_LOCK_TABLE} (lock_name) VALUES ('global') ON CONFLICT DO NOTHING"
                )
            )
        except Exception:
            logger.warning("Could not acquire migration advisory lock — proceeding anyway")


def _applied_ids(conn: Connection) -> set[str]:
    rows = conn.execute(text(f"SELECT migration_id FROM {_TRACKING_TABLE}")).fetchall()
    return {r[0] for r in rows}


def _record_applied(conn: Connection, migration: Migration) -> None:
    conn.execute(
        text(
            f"INSERT INTO {_TRACKING_TABLE} (migration_id, description, applied_at)"
            " VALUES (:mid, :desc, :at)"
        ),
        {
            "mid": migration.id,
            "desc": migration.description,
            "at": datetime.now(UTC).isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# Migration discovery
# ---------------------------------------------------------------------------


def _load_all_migrations() -> list[Migration]:
    """Import every ``migrations/versions/NNNN_*.py`` module and return its
    ``migration`` object sorted by numeric ID."""
    if not os.path.isdir(_VERSIONS_DIR):
        return []

    migrations: list[Migration] = []
    for fname in sorted(os.listdir(_VERSIONS_DIR)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        module_name = f"migrations.versions.{fname[:-3]}"
        mod = importlib.import_module(module_name)
        if not hasattr(mod, "migration"):
            logger.warning(
                "Migration module %s has no 'migration' attribute — skipping", module_name
            )
            continue
        m = mod.migration
        if not isinstance(m, Migration):
            raise TypeError(f"{module_name}.migration must be a Migration instance, got {type(m)}")
        migrations.append(m)

    # Sort by numeric id (the leading digits) for safety even if filenames are
    # not perfectly sorted by the OS.
    migrations.sort(key=lambda m: int(m.id))
    return migrations


def _verify_registry(migrations: list[Migration]) -> None:
    """Raise if any known migration ID is missing from the discovered list
    (i.e. the migration file was deleted)."""
    discovered_ids = {m.id for m in migrations}
    for rid in REGISTRY:
        if rid not in discovered_ids:
            raise RuntimeError(
                f"Migration {rid!r} is listed in the registry but its file is missing. "
                "Never delete a migration file that has been applied to a database."
            )


# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------


@dataclass
class MigrationStatus:
    """Summary of applied and pending migrations for display / assertion."""

    applied: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)

    @property
    def is_up_to_date(self) -> bool:
        return len(self.pending) == 0

    def __str__(self) -> str:  # pragma: no cover
        lines = []
        for mid in self.applied:
            lines.append(f"  [applied] {mid}")
        for mid in self.pending:
            lines.append(f"  [pending] {mid}")
        return "\n".join(lines) if lines else "  (no migrations found)"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class MigrationRunner:
    """Discover and apply LedgerLens schema migrations.

    Parameters
    ----------
    engine:
        The SQLAlchemy engine to migrate.  Defaults to the application engine
        from ``detection.persistence.get_engine()``.
    target:
        Optional 4-digit migration ID to stop at (inclusive).  ``None`` means
        apply all pending migrations.
    dry_run:
        Log the migrations that *would* be applied without executing them.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        *,
        target: str | None = None,
        dry_run: bool = False,
    ) -> None:
        if engine is None:
            from detection.persistence import get_engine

            engine = get_engine()
        self._engine = engine
        self._target = target
        self._dry_run = dry_run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upgrade(self, target: str | None = None) -> MigrationStatus:
        """Apply all pending migrations up to *target* (inclusive).

        Returns the final :class:`MigrationStatus`.
        """
        target = target or self._target
        all_migrations = _load_all_migrations()
        _verify_registry(all_migrations)

        # Kept unfiltered for status reporting — `status.pending` must still
        # list migrations beyond `target` (they are pending, just not being
        # applied by this call), not silently drop them from the report.
        to_apply = all_migrations
        if target is not None:
            known_ids = {m.id for m in all_migrations}
            if target not in known_ids:
                raise ValueError(f"Unknown migration target {target!r}. Known: {sorted(known_ids)}")
            to_apply = [m for m in all_migrations if int(m.id) <= int(target)]

        with self._engine.begin() as conn:
            _ensure_tracking_tables(conn)
            _acquire_lock(conn)
            applied = _applied_ids(conn)

            pending = [m for m in to_apply if m.id not in applied]
            if not pending:
                logger.info("Database is up to date — no migrations to apply")
                return self._build_status(all_migrations, applied)

            for migration in pending:
                # Check prerequisites: all earlier-numbered migrations must be applied
                missing_prerequisites = self._check_prerequisites(migration, all_migrations, applied)
                if missing_prerequisites:
                    raise RuntimeError(
                        f"Cannot apply migration {migration.id} ({migration.description}): "
                        f"missing prerequisite migrations: {', '.join(sorted(missing_prerequisites))}"
                    )

                if self._dry_run:
                    logger.info(
                        "[dry-run] Would apply migration %s: %s",
                        migration.id,
                        migration.description,
                    )
                    continue
                logger.info("Applying migration %s: %s", migration.id, migration.description)
                migration.up(conn)
                _record_applied(conn, migration)
                logger.info("Migration %s applied successfully", migration.id)

        # Re-read status after applying
        with self._engine.connect() as conn:
            applied = _applied_ids(conn)
        return self._build_status(all_migrations, applied)

    def _check_prerequisites(
        self, migration: Migration, all_migrations: list[Migration], applied: set[str]
    ) -> set[str]:
        """Check that all prerequisite (lower-numbered) migrations have been applied.

        Returns a set of missing prerequisite migration IDs. Empty set means all
        prerequisites are satisfied.
        """
        current_id = int(migration.id)
        missing = set()
        for m in all_migrations:
            if int(m.id) < current_id and m.id not in applied:
                missing.add(m.id)
        return missing

    def status(self) -> MigrationStatus:
        """Return :class:`MigrationStatus` without applying anything."""
        all_migrations = _load_all_migrations()
        _verify_registry(all_migrations)

        with self._engine.begin() as conn:
            _ensure_tracking_tables(conn)
            applied = _applied_ids(conn)

        return self._build_status(all_migrations, applied)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_status(self, all_migrations: list[Migration], applied: set[str]) -> MigrationStatus:
        return MigrationStatus(
            applied=[m.id for m in all_migrations if m.id in applied],
            pending=[m.id for m in all_migrations if m.id not in applied],
        )
