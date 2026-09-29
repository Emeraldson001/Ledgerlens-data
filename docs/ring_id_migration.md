# `ring_id` backfill runbook

`ring_id` is a nullable column on `risk_scores` (added by
[`migrations/versions/0001_add_ring_id.py`](../migrations/versions/0001_add_ring_id.py))
that groups wallets found in the same wash-trading ring. New databases get
the column automatically via `Base.metadata.create_all`. Existing self-hosted
deployments created before this field was introduced need to run a one-time
backfill migration. This document covers that process end-to-end.

## 1. Back up your database

Take a backup before running any schema change, even an additive one.

```bash
# PostgreSQL
pg_dump "$RISK_SCORE_DB_URL" > risk_scores_backup_$(date +%Y%m%d).sql

# SQLite
cp ledgerlens.db ledgerlens.db.bak
```

## 2. Run the migration

Use the standard migration runner (preferred — it tracks applied state in
`schema_migrations`):

```bash
python -m scripts.migrate
```

This applies every pending migration, including `0001` (`Add nullable
ring_id column to risk_scores`). To apply only up to and including this
migration:

```bash
python -m scripts.migrate --target 0001
```

Alternatively, the standalone script performs the same idempotent
`ALTER TABLE` directly, outside the tracked migration framework:

```bash
python -m scripts.migrate_add_ring_id
```

Both are safe to re-run — they check for the column's existence before
altering the table.

## 3. Verify the migration succeeded

Check the migration runner's status:

```bash
python -m scripts.migrate --status
```

`0001` should be listed under `Applied migrations`, and the command exits
`0` when the database is fully up to date.

You can also confirm the column exists directly:

```bash
# PostgreSQL
psql "$RISK_SCORE_DB_URL" -c "\d risk_scores" | grep ring_id

# SQLite
sqlite3 ledgerlens.db "PRAGMA table_info(risk_scores);" | grep ring_id
```

Existing rows will have `ring_id = NULL` until the next pipeline run
recomputes wallet graph features and populates ring membership.

## 4. Restart services

Once the column is confirmed present, restart the pipeline, API, and any
streaming workers so they pick up the updated schema.

## Rollback guidance

`ring_id` is additive and nullable, so no downstream code depends on it
being absent — leaving the column in place after a rollback of application
code is safe. If you need to remove it (e.g. reverting to a much older
release that predates `ring_id` entirely), do so manually:

```sql
ALTER TABLE risk_scores DROP COLUMN ring_id;
```

There is no automated down-migration for this by design — see
[`migrations/base.py`](../migrations/base.py) for the project's
backwards-compatibility contract (migrations are never deleted or
reversed automatically).
