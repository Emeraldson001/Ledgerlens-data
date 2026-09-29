# Code Review Checklist — Data Assets & Persistence Changes

> **When to use this checklist:**  
> A PR touches `ingestion/data_models.py`, `data/trade_avro_schema.json`,
> `detection/persistence.py`, `detection/risk_score_store.py`, the `data/`
> directory (parquet, CSV, JSON assets), or the `models/` directory.

---

## Before approving, verify ALL items below

### 1. Pydantic / data model changes (`ingestion/data_models.py`)

- [ ] Backward-compatible: existing producers of `Trade`, `OrderBookEvent`, or
      `AccountActivity` records don't need code changes
- [ ] New required fields have `Optional` type or a default value
- [ ] `Asset.pair_id()` format unchanged (`CODE:ISSUER/CODE:ISSUER`)
- [ ] Updated schema documented in `docs/ingestion.md` or inline docstrings
- [ ] Existing serialisation tests pass (`tests/test_factories.py`)
- [ ] If `pair_id` format changed: linked PRs in `ledgerlens-api` and `ledgerlens-contract`

---

### 2. Avro schema changes (`data/trade_avro_schema.json`)

- [ ] Schema evolution is backward-compatible (new fields have `null` default)
- [ ] Field `type` not changed (would break existing Avro messages)
- [ ] `namespace` and `name` preserved
- [ ] `ingestion/avro_codec.py` updated to encode/decode new fields
- [ ] Avro schema version incremented in the schema's doc or a separate version field
- [ ] Dead-letter queue (DLQ) tested against old-format messages

---

### 3. SQLAlchemy persistence model changes (`detection/persistence.py`)

- [ ] New column migration script created: `scripts/migrate_<description>.py`
- [ ] Migration script is backward-compatible (`ALTER TABLE ... ADD COLUMN` not `DROP`)
- [ ] Migration script has been tested on a local empty database
- [ ] `RiskScore` model `__tablename__` unchanged (no table renames)
- [ ] `ring_id`, `score_lower`, `score_upper`, `coverage_guarantee` fields preserved
- [ ] DB connection pool settings unchanged unless tested under load
- [ ] `tests/test_persistence.py` covers new column read/write

---

### 4. Risk score store changes (`detection/risk_score_store.py`)

- [ ] Upsert logic still uses correct primary key (`wallet_id`, `asset_pair`)
- [ ] `ShapQueryCount` per-wallet DP budget tracking not removed
- [ ] `ring_id` assignment logic preserved if rings are detected
- [ ] Read path returns `score_lower`/`score_upper` if requested
- [ ] No N+1 query patterns introduced (batch ops still batched)

---

### 5. Synthetic dataset changes (`data/synthetic_dataset.parquet`)

- [ ] SHA-256 of new file documented in PR (for provenance tracking)
- [ ] `label_distribution_baseline.json` updated if wash-trade ratio changed
- [ ] `POISON_LABEL_RATIO_THRESHOLD` check passes (ratio shift < 15%)
- [ ] Synthetic data generator (`scripts/generate_synthetic_dataset.py`) reproduces the new file
- [ ] Models retrained on new dataset — metrics in CHANGELOG.md
- [ ] `data/dataset_card.md` updated with new dataset statistics

---

### 6. Ground truth events (`data/known_manipulation_events.csv`)

- [ ] New events sourced and verified (include reference/link in PR description)
- [ ] Event format consistent (required columns: campaign_id, wallet_id, start_date, end_date, type)
- [ ] Total event count documented (was N, now N+K)
- [ ] Backtest results regenerated: `python -m scripts.backtest --output reports/backtest_updated.json`
- [ ] Time-averaged AUC ≥ 0.75 maintained
- [ ] `tests/test_backtest.py` updated with new expected campaign count

---

### 7. Model artifact directory (`models/`)

- [ ] No trained model `.joblib` files committed without `.sig` signature files
- [ ] `models/metrics.json` reflects the newly committed models
- [ ] `models/README.md` updated with current model versions
- [ ] Large model files use Git LFS (check `.gitattributes`)
- [ ] Old model versions archived (not deleted) for rollback capability

---

## Migration and rollback plan

For any change that modifies the database schema or replaces model artifacts:

| Item | Status |
|------|--------|
| Migration script tested on empty DB | ☐ |
| Rollback procedure documented | ☐ |
| Backfill script for existing records (if needed) | ☐ |
| Maintenance window required? | ☐ Yes / ☐ No |
| Downtime impact assessed | ☐ |

---

## Reviewer notes

_Document data lineage, migration approach, and rollback plan here._
