# Runbook: Diagnosing and Recovering from a Dedup/Idempotency Incident

Covers the unified exactly-once library (`pipeline/exactly_once.py`) and its
call sites: `streaming/kafka_worker.py::DeduplicationCache`,
`ingestion/trade_deduplicator.py::SeenEventCache`, and (informationally)
`pipeline/idempotency.py::CheckpointStore` and
`utils/idempotency.py::IdempotencyLedger`. Design rationale:
`docs/adr/0001-unified-idempotency-finality.md`.

---

## Symptom: `DedupBackendUnavailableError` in worker logs / `kafka_dedup_backend_degraded_total` rising

**Meaning**: Redis (the dedup backend for the Kafka worker and trade
ingestion dedup) is unreachable. Per invariant 8 (Issue #670), the system
**fails closed** — it does not fall back to "not a duplicate". The affected
worker halts effective progress on that message/partition (offset left
uncommitted) rather than risk a silent duplicate or a silent drop.

**Diagnosis**:
1. Check the `dedup_backend_available` Prometheus gauge (labeled by
   `source`, e.g. `kafka_trade`, `horizon_trade`) — `0` means degraded.
2. Confirm Redis reachability directly: `redis-cli -u $REDIS_URL ping`.
3. Check Redis memory/eviction — a full Redis instance under a
   restrictive `maxmemory-policy` can reject writes, which surfaces the same
   way as an outage.

**Recovery**:
1. Restore Redis (restart the instance, fix networking/DNS, raise
   `maxmemory`, etc.).
2. Workers reconnect automatically on the next message — there is no
   manual "resume" step; `KafkaWorker.run()`'s poll loop keeps retrying the
   same uncommitted message.
3. Once healthy, `dedup_backend_available` returns to `1` and
   `kafka_dedup_backend_degraded_total` stops incrementing. Consumer lag
   (`kafka_lag_by_partition`) will show the backlog draining.

**Do not**: patch around this by making the dedup check optional/best-effort
in code, or by manually committing offsets past the stuck point — either
reintroduces the exact fail-open/silent-drop bug this issue fixed.

---

## Symptom: A trade appears twice in `FeatureBuffer` / a wallet's feature state looks doubled

**Meaning**: This should not happen after Issue #670 —
`FeatureBuffer.update()` is idempotent per `(wallet, trade_id)`. If it does:

**Diagnosis**:
1. Confirm the two occurrences have the **same** `trade_id`. If they have
   **different** `trade_id`s, this is not a dedup bug — it is two genuinely
   distinct trades (possibly a data-quality issue upstream).
2. If genuinely the same `trade_id` appears twice in
   `buf._buffers[wallet]`, check whether the buffer exceeded `max_trades`
   between the two applications — the companion `_seen_trade_ids` index is
   evicted in lock-step with the deque, so a trade that has scrolled out of
   the retention window is legitimately "forgotten" and will be re-applied
   if redelivered after that point. This is expected, not a bug: the
   dedup guarantee is scoped to the current retention window, not
   unbounded history.

**Recovery**: If neither of the above explains it, this is a regression in
`streaming/feature_buffer.py::FeatureBuffer.update()` — file a bug; do not
attempt a live workaround, since feature state is only trustworthy while
this invariant holds.

---

## Symptom: `validation.reconciliation.reconcile_alert_delivery` reports `missing_count > 0`

**Meaning**: A wallet scored at/above the alert threshold has **no** recorded
outcome (`delivered`, `dead_lettered`, or `suppressed_cooldown`) in
`AlertDeliveryLedger`. This is the literal "silently dropped alert" failure
mode reconciliation exists to catch.

**Diagnosis**:
1. Confirm the `AlertDispatcher` instance that scored this wallet was
   constructed with `delivery_ledger=` set — `scripts/stream.py` and
   `scripts/kafka_workers.py` do this by default; a custom entry point that
   omits it will correctly show every alert as "missing" (this is expected
   for entry points that opt out of the ledger, not an incident).
2. If the ledger was wired up, check worker logs around the time the score
   was computed for an unhandled exception in `AlertDispatcher.dispatch()`
   (e.g. a bug that raises before reaching any of the three outcome-recording
   paths).
3. Check `AlertDeliveryLedger`'s backing DB (`RISK_SCORE_DB_URL`,
   `exactly_once_dedup` table, `source='alert_delivery'`) is reachable and
   not full/locked.

**Recovery**: Fix the underlying exception or DB issue; historical missing
entries cannot be retroactively reconstructed (the dispatch attempt is
already in the past) — treat a nonzero `missing_count` as an incident to
investigate, not something to silently re-run.

---

## Symptom: `AuditMerkleChain.verify_chain()` raises `AuditChainIncompleteError`

**Meaning**: An entry in the requested verification range predates migration
`0005` and has no persisted leaf content (`content_hash IS NULL`). This is
**not** tampering — see `TamperDetectedError` below for that. It means the
audit log has a legacy row from before this fix shipped.

**Recovery**: There is no way to recover leaf content that was never
persisted. Restrict `verify_chain(start_index=...)` to a range that starts
after the gap, or treat the pre-migration prefix as verified only by its
original commitment (outside this chain's re-verification mechanism, e.g. via
the separate signed `AuditTrailWriter` NDJSON log if one exists for that
period).

## Symptom: `AuditMerkleChain.verify_chain()` raises `TamperDetectedError`

**Meaning**: A genuine content or root mismatch — this is the real security
signal. As of Issue #670, a routine process restart **cannot** cause this on
its own (entries are rehydrated from durable storage on every
`AuditMerkleChain()` construction) — treat this as an active tampering
investigation, not a transient restart artifact.

**Diagnosis**: The error message names the failing `entry index` and whether
the mismatch is "in-entry" (recomputed root vs. the entry's own stored root)
or "separate-table" (the entry's root vs. the independently-stored
`audit_merkle_roots` row) — a mismatch only in the separate table with the
in-entry check passing suggests the roots table specifically was altered.

**Recovery**: Escalate as a security incident. Do not modify the audit
tables to "fix" the mismatch — that destroys the evidence the tamper-evident
design exists to preserve.

---

## Symptom: A risk score's `finality` looks wrong (stuck `provisional`, or `final` when you expected `provisional`)

**Meaning**: `finality` reflects which code path last wrote the score:
`streaming/kafka_worker.py` and `streaming/pipeline.py` (continuous
streaming) always write `provisional`; `run_pipeline.py`'s persist stage and
`scripts/replay_stream.py` (bounded, closed-window runs) write `final`. If a
wallet is scored by both paths over time, the **last write wins** —
`RiskScoreRecord.finality` is not itself versioned per contributing run.

**Diagnosis**: Check `RiskScoreRecord.updated_at` against your ingestion
timeline to determine which path wrote most recently.

**Recovery**: Not an incident by itself — this is expected behavior. If a
consumer needs to know "was this wallet's *current* score computed by a
completed batch/replay run", `finality == "final"` answers exactly that
question; it does not promise no streaming update has landed since.
