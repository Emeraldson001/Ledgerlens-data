"""Tests for the migration scaffolding (Issue #1).

Exercises MigrationRunner discovery, ordering, idempotency, status tracking,
the dry-run mode, and the individual version migrations using an in-memory
SQLite database so the tests are fully self-contained.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect, text

from migrations import MigrationRunner
from migrations.registry import REGISTRY
from migrations.runner import _load_all_migrations

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sqlite_engine():
    """Fresh in-memory SQLite engine per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    yield engine
    engine.dispose()


@pytest.fixture()
def populated_engine(sqlite_engine):
    """Engine with the risk_scores table already created (simulates an
    existing database that needs to be migrated)."""
    with sqlite_engine.begin() as conn:
        conn.execute(text("""
                CREATE TABLE IF NOT EXISTS risk_scores (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    wallet      VARCHAR NOT NULL,
                    asset_pair  VARCHAR NOT NULL,
                    score       INTEGER NOT NULL,
                    benford_flag BOOLEAN NOT NULL DEFAULT 0,
                    ml_flag     BOOLEAN NOT NULL DEFAULT 0,
                    confidence  INTEGER NOT NULL DEFAULT 0,
                    updated_at  TIMESTAMP NOT NULL
                )
                """))
    return sqlite_engine


# ---------------------------------------------------------------------------
# MigrationRunner — basic upgrade
# ---------------------------------------------------------------------------


class TestMigrationRunnerUpgrade:
    def test_upgrade_creates_tracking_table(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        runner.upgrade()
        inspector = inspect(populated_engine)
        assert "schema_migrations" in inspector.get_table_names()

    def test_upgrade_applies_all_migrations(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        status = runner.upgrade()
        assert status.is_up_to_date, f"Pending: {status.pending}"

    def test_upgrade_is_idempotent(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        runner.upgrade()
        # Second call should be a no-op
        status = runner.upgrade()
        assert status.is_up_to_date
        assert status.pending == []

    def test_applied_ids_recorded(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        status = runner.upgrade()
        assert set(status.applied) == set(REGISTRY)

    def test_columns_added_by_migrations(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        runner.upgrade()
        inspector = inspect(populated_engine)
        col_names = {col["name"] for col in inspector.get_columns("risk_scores")}
        assert "ring_id" in col_names, "0001 should add ring_id"
        assert "provenance_json" in col_names, "0002 should add provenance_json"
        assert "certified_robust" in col_names, "0003 should add certified_robust"
        assert "schema_version" in col_names, "0004 should add schema_version"


# ---------------------------------------------------------------------------
# MigrationRunner — status
# ---------------------------------------------------------------------------


class TestMigrationRunnerStatus:
    def test_status_shows_pending_before_upgrade(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        status = runner.status()
        assert not status.is_up_to_date
        assert len(status.pending) == len(REGISTRY)

    def test_status_shows_up_to_date_after_upgrade(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        runner.upgrade()
        status = runner.status()
        assert status.is_up_to_date

    def test_status_str_contains_applied(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        runner.upgrade()
        status = runner.status()
        text_output = str(status)
        assert "applied" in text_output


# ---------------------------------------------------------------------------
# MigrationRunner — partial upgrade with target
# ---------------------------------------------------------------------------


class TestMigrationRunnerTarget:
    def test_upgrade_to_target_stops_early(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        status = runner.upgrade(target="0002")
        assert "0001" in status.applied
        assert "0002" in status.applied
        assert "0003" in status.pending
        assert "0004" in status.pending

    def test_upgrade_with_invalid_target_raises(self, populated_engine):
        runner = MigrationRunner(populated_engine)
        with pytest.raises(ValueError, match="Unknown migration target"):
            runner.upgrade(target="9999")


# ---------------------------------------------------------------------------
# MigrationRunner — dry-run
# ---------------------------------------------------------------------------


class TestMigrationRunnerDryRun:
    def test_dry_run_does_not_apply_migrations(self, populated_engine):
        runner = MigrationRunner(populated_engine, dry_run=True)
        runner.upgrade()
        # Tracking table may have been created by _ensure_tracking_tables, but
        # no migrations should be recorded as applied.
        with populated_engine.begin() as conn:
            rows = conn.execute(text("SELECT migration_id FROM schema_migrations")).fetchall()
        assert rows == [], "Dry-run should not record any applied migrations"

    def test_dry_run_columns_not_added(self, populated_engine):
        runner = MigrationRunner(populated_engine, dry_run=True)
        runner.upgrade()
        inspector = inspect(populated_engine)
        col_names = {col["name"] for col in inspector.get_columns("risk_scores")}
        assert "ring_id" not in col_names, "Dry-run should not modify the schema"


# ---------------------------------------------------------------------------
# New-database: migrations against fresh engine (no pre-existing tables)
# ---------------------------------------------------------------------------


class TestMigration0007PromotionActorColumns:
    """Grand 2 / issue #671: promoted_by/rolled_back_by/parent_version_id
    columns added to model_versions."""

    @pytest.fixture()
    def engine_with_model_versions(self, sqlite_engine):
        with sqlite_engine.begin() as conn:
            conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS model_versions (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        version_id          VARCHAR NOT NULL,
                        model_artifact_path VARCHAR NOT NULL,
                        status              VARCHAR NOT NULL,
                        trained_at          TIMESTAMP NOT NULL
                    )
                    """))
        return sqlite_engine

    def test_columns_added(self, engine_with_model_versions):
        runner = MigrationRunner(engine_with_model_versions)
        runner.upgrade()
        inspector = inspect(engine_with_model_versions)
        col_names = {col["name"] for col in inspector.get_columns("model_versions")}
        assert "promoted_by" in col_names
        assert "rolled_back_by" in col_names
        assert "parent_version_id" in col_names

    def test_noop_when_table_missing(self, sqlite_engine):
        """No model_versions table (fresh env) — migration must skip gracefully."""
        runner = MigrationRunner(sqlite_engine)
        status = runner.upgrade()
        assert "0007" in status.applied


class TestMigrationsFreshEngine:
    def test_fresh_engine_upgrade_is_noop_for_missing_table(self, sqlite_engine):
        """On a fresh DB with no risk_scores table the column-adding migrations
        skip gracefully because the table does not exist yet."""
        runner = MigrationRunner(sqlite_engine)
        status = runner.upgrade()
        # All migrations should be marked applied (they each check for table
        # existence and return early when it is absent)
        assert status.is_up_to_date


# ---------------------------------------------------------------------------
# Migration discovery helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Migration prerequisites
# ---------------------------------------------------------------------------


class TestMigrationPrerequisites:
    def test_upgrade_detects_missing_prerequisites(self, populated_engine):
        """Attempting to apply 0003 without 0001 and 0002 should raise RuntimeError."""
        # Manually record 0004 as applied to simulate a broken state where a later
        # migration was applied out of order
        with populated_engine.begin() as conn:
            from migrations.runner import _ensure_tracking_tables, _record_applied
            from migrations.versions._0004_add_schema_version import migration as m0004

            _ensure_tracking_tables(conn)
            _record_applied(conn, m0004)

        runner = MigrationRunner(populated_engine)
        # Trying to upgrade should fail because prerequisites 0001, 0002, 0003 are missing
        with pytest.raises(RuntimeError, match="missing prerequisite migrations"):
            runner.upgrade()

    def test_prerequisite_error_names_missing_migrations(self, populated_engine):
        """The error message should name the specific missing prerequisite migrations."""
        # Record only 0001 as applied
        with populated_engine.begin() as conn:
            from migrations.runner import _ensure_tracking_tables, _record_applied
            from migrations.versions._0001_add_ring_id import migration as m0001

            _ensure_tracking_tables(conn)
            _record_applied(conn, m0001)

        runner = MigrationRunner(populated_engine)
        # Trying to apply 0003 should fail because 0002 is missing (0001 is satisfied)
        with pytest.raises(RuntimeError) as exc:
            runner.upgrade()

        error_msg = str(exc.value)
        assert "0002" in error_msg, "Error should name the missing prerequisite 0002"


# ---------------------------------------------------------------------------
# Migration discovery helpers
# ---------------------------------------------------------------------------


class TestMigrationDiscovery:
    def test_migrations_are_sorted_by_id(self):
        migrations = _load_all_migrations()
        ids = [int(m.id) for m in migrations]
        assert ids == sorted(ids), "Migrations must be in ascending ID order"

    def test_all_registry_entries_are_discovered(self):
        migrations = _load_all_migrations()
        discovered = {m.id for m in migrations}
        for rid in REGISTRY:
            assert rid in discovered, f"Registry entry {rid!r} not found in versions/"

    def test_migration_ids_are_four_digits(self):
        migrations = _load_all_migrations()
        import re

        for m in migrations:
            assert re.fullmatch(r"\d{4}", m.id), f"Migration ID {m.id!r} is not 4 digits"
