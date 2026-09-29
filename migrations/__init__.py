"""LedgerLens database migration scaffolding.

Every migration lives in ``migrations/versions/`` as a numbered Python module
(e.g. ``0001_add_ring_id.py``).  Migrations are discovered, ordered, and
applied by :class:`MigrationRunner`.

The registry in ``migrations/registry.py`` is the authoritative list of
known migration IDs — it lets the runner verify that no migration file has
been deleted or re-ordered after it was already applied to a database.

Typical usage::

    from migrations import MigrationRunner
    from detection.persistence import get_engine

    runner = MigrationRunner(get_engine())
    runner.upgrade()          # apply all pending migrations
    runner.status()           # print applied / pending list
    runner.upgrade("0003")    # apply up to (and including) migration 0003
"""

from migrations.runner import MigrationRunner

__all__ = ["MigrationRunner"]
