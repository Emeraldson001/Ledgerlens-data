# Review checklists for high-risk data and model changes

Some paths in this repo carry contracts that other services depend on, or act
on data that cannot be restored. Changing them is fine — silently changing them
is not.

When a pull request touches one of these paths, CI
(`.github/workflows/review-gates.yml`) requires the description to answer the
matching question below. The gate is defined in
[`.github/review-gates.yml`](review-gates.yml) and enforced by
`scripts/check_review_gates.py`.

## How it works

1. Add the heading exactly as written below to your pull-request description.
2. Write a sentence or two underneath it. **A ticked checkbox does not
   satisfy the gate** — the point is to leave a note a reviewer can evaluate,
   not a one-bit signal.
3. If a gate genuinely does not apply, or you are shipping an urgent fix, add:

   ```
   Review-Gate-Override: <reason>
   ```

   The override stays in the description and is echoed in the bot comment, so
   it is auditable rather than a silently skipped check.

Run it locally before pushing:

```bash
make check-review-gates
```

---

## Kafka wire schema

**Fires on:** `data/trade_avro_schema.json`

Mechanical backward and forward compatibility is already checked by
`make check-schema-compatibility`, which compares against the branch you are
merging into. That check cannot tell whether the *rollout* is safe. State:

- The dual-write window, and when the old field can be removed.
- Which consumers were updated, with links.
- Whether messages already in the topic can still be read.

See [`data/schema_evolution.md`](../data/schema_evolution.md) for the full
procedure.

## Shared contract

**Fires on:** `ingestion/data_models.py`, `detection/model_inference.py`

These shapes are mirrored by `ledgerlens-core` and consumed by
`ledgerlens-api`, `ledgerlens-contract`, and `ledgerlens-dashboard`. State:

- Which fields changed, and whether the change is additive.
- Links to the downstream issues or PRs.
- If errors are raised at an ingestion boundary: that they carry enough
  context to triage a failure without reproducing it — which source, which
  record, and the underlying cause — and follow the module's existing
  error-handling convention.

## Model behaviour

**Fires on:** `detection/feature_engineering.py`, `detection/model_training.py`

Changing feature semantics or the training pipeline alters what the model
learns without necessarily failing a test. State:

- Whether models were retrained.
- What evaluation was run, and how metrics compare to the previous baseline.
- For a new or changed feature: that it is documented per
  [`docs/contributor_feature_guide.md`](../docs/contributor_feature_guide.md).

## Database schema

**Fires on:** `detection/persistence.py`, `scripts/migrate_*.py`

Covers SQLAlchemy table definitions and migration scripts, which act on
existing production data and are not trivially reversible. State:

- The migration path for databases already in the field.
- Whether the change is backward compatible with the running service.
- Whether the migration is idempotent and safe to re-run.

Model-artifact integrity is already enforced by the `verify_chain` CI check;
no need to restate it here.

## Tenant configuration

**Fires on:** `config/tenants.yaml`

Tenant entries govern isolation between customers. State which tenants are
affected and confirm no tenant gained access to another's data.

## Model metadata

**Fires on:** `models/metrics.json`

This file carries signed model-artifact hashes and differential-privacy
parameters. State why it changed and whether the signature was regenerated.

---

## New top-level module

**Fires on:** any new top-level directory (e.g. `contracts/`, `mlops/`, `pipeline/`)

When you add a brand-new top-level module or directory, add a corresponding
pattern to [`.github/CODEOWNERS`](../CODEOWNERS) so the right team is
automatically requested for review on future changes.  For example, adding a
`contracts/` directory for Soroban smart-contract helpers:

```gitignore
/contracts/                      @Ledger-Lenz/contract-team
```

If the new module touches an existing team's area, include both teams
(e.g. `@Ledger-Lenz/contract-team @Ledger-Lenz/infra-team`).

## What is deliberately not gated

Gating paths that carry no contract trains people to acknowledge noise, which
is how a gate loses its meaning. These are excluded on purpose, and the
exclusions are asserted in `tests/test_check_review_gates.py`:

| Path | Why not |
|---|---|
| `data/feature_ranges.json` | Generated statistics from `scripts/compute_feature_ranges.py`, read by nothing at runtime. `FEATURE_RANGES` is hardcoded in `detection/feature_engineering.py`. |
| `models/*.joblib`, `models/*.pkl` | Gitignored — they cannot appear in a diff. |
| `reporting/schemas/model_metadata.json` | Already enforced at runtime by `reporting/model_card_generator.py` (`MetadataValidationError`). |
