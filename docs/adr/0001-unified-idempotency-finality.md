# ADR 0001: Unified idempotency and finality model for the trade-processing pipeline

Status: Accepted
Related: Issue #670 ("Grand 1 — Establish deterministic exactly-once processing
and defined finality across ingestion → streaming → detection → alerting")

## Context

The trade-processing path accumulated four independent, ad hoc idempotency
mechanisms over separate PRs:

1. `pipeline/idempotency.py::CheckpointStore` — SQL, `(run_id, pair_id, stage)`
   stage-completion cache for the offline/batch pipeline.
2. `utils/idempotency.py::IdempotencyLedger` — SQLite, generic job-result cache
   with lease-based concurrency control.
3. `ingestion/trade_deduplicator.py::SeenEventCache` — Redis sorted-set dedup
   for Horizon SSE trade ingestion. **Fails open**: if Redis is unreachable,
   every event is treated as new.
4. `streaming/kafka_worker.py::DeduplicationCache` — Redis `SETNX` dedup for
   the Kafka consumer. **Fails open** for the same reason, and — critically —
   marks a message's dedup key *before* the message's side effects (feature
   update, scoring, alert dispatch) are known to have completed. If dispatch
   raises for the second wallet in a two-wallet trade, the Kafka offset is
   correctly left uncommitted for redelivery, but the dedup key is already
   set. On redelivery the message is misclassified as a duplicate and its
   offset is committed without reprocessing — the second wallet is silently
   never scored. This is the critical, verified bug in the issue.

None of the four mechanisms agree on a key format, none reserve a slot for a
future `tenant_id` (Grand 4), and two of them (3 and 4) fail open on the exact
dependency outage they exist to guard against.

Separately: `FeatureBuffer.update()` has no dedup by `trade_id`, so any
redelivery — even within a single process, with no crash involved — double
counts a trade into feature state. `AuditMerkleChain` only persists Merkle
*roots*; leaf content lives in an in-process list, so a routine restart makes
`verify_chain()` indistinguishable from real tampering. `scripts/replay_stream.py`
ignores Kafka's committed offset entirely when `--resume` is passed (it always
seeks to the beginning), and commits every 100 messages regardless of the
live path's per-message durability guarantee. `RiskScoreRecord` has no
finality concept beyond `updated_at`, and `validation/reconciliation.py` never
traces a score through to a delivered (or dead-lettered) alert.

## Decision

### 1. One key scheme, two categories of store

We introduce `pipeline/exactly_once.py::DedupKey(tenant_id, source, external_id)`
as the single canonical key format used everywhere in the pipeline that needs
to answer "have I seen this before". `tenant_id` is a reserved, currently-unused
slot (`None` today) so Grand 4 can populate it without a second migration, per
the issue's stated dependency ordering.

Not every current mechanism plays the same role, so we do not force all four
into one physical table:

- **Pure dedup** (`TradeDeduplicator`, Kafka `DeduplicationCache`): both ask
  exactly one question — "have I already committed the side effects for this
  external id?" — under a hard latency budget. These are unified onto one
  concrete class, `pipeline/exactly_once.py::ExactlyOnceStore`, backed by a
  two-phase (`STAGED` → `COMMITTED`) protocol. This is the actual fix for the
  critical bug: the dedup key is staged before processing and **committed
  only after side effects durably succeed**, so a crash or exception between
  staging and commit leaves the record in `STAGED`, which `ExactlyOnceStore`
  reports as "redo this" (not "duplicate"), not silently dropped.
- **Result-caching idempotency** (`CheckpointStore`, `IdempotencyLedger`):
  both need to cache a rich JSON payload alongside the completion marker
  (stage output, job return value) and support longer-lived TTL/lease
  semantics for a batch job, not a hot streaming path. We keep their existing
  public APIs (used by `pipeline/recovery.py`, `scripts/backfill_amm_trades.py`,
  and callers of `utils/idempotency.py`) unchanged, but re-key their storage
  through `DedupKey.canonical()` so the key format itself is shared, and add
  an explicit `TTL_EXPIRED_REVERIFY` status distinguishable from "never
  processed" (previously a stale checkpoint silently looked identical to a
  fresh one).

This is a deliberate, scoped interpretation of "one library": the failure
mode that actually drops data (Redis-backed pure dedup, fail-open on outage)
gets a full rebuild; the batch-pipeline result caches, which are already
durable, transactional, and were not implicated in the verified bug, get a
shared key format and an explicit stale-state signal without a risky wholesale
merge of two functionally different storage contracts in one change. Fully
folding `CheckpointStore`/`IdempotencyLedger` onto `ExactlyOnceStore`'s payload
column is tracked as follow-up work (see PR).

`utils/checkpointing.py::CheckpointStore` and `utils/checkpoint.py::PipelineCheckpoint`
are unrelated, file-based script-resumption utilities (not part of the
trade-processing path the issue audits) and are out of scope.

### 2. Fail-closed dependency behavior

`ExactlyOnceStore`'s Redis backend raises `DedupBackendUnavailableError`
instead of returning "not a duplicate" when Redis is unreachable. Callers
(`TradeDeduplicator`, `KafkaWorker`) catch this, flip a `dedup_backend_available`
Prometheus gauge to 0, and refuse to proceed with that message — the Kafka
offset is left uncommitted (so the worker retries the same message, i.e. it
halts effective throughput on that partition rather than silently disabling
correctness).

### 3. Kafka worker processing order

`KafkaWorker._process_correlated_message` now: (a) stages the dedup key,
(b) if already `COMMITTED`, skips reprocessing and commits the offset,
(c) otherwise (new or `STAGED`-but-not-committed, i.e. a prior crash)
processes both wallets, (d) commits the dedup key, (e) commits the Kafka
offset — in that order. A crash between (d) and (e) is safe: redelivery sees
`COMMITTED` and just advances the offset. A crash before (d) is safe: redelivery
sees `STAGED` and redoes the message; re-scoring/re-upserting is idempotent,
and duplicate alerts within the crash-restart window are bounded by the
existing per-wallet cooldown in `AlertDispatcher`.

### 4. Feature-buffer dedup

`FeatureBuffer.update()` now skips wallets that already contain the trade's
`trade_id` in their buffer, using an O(1) companion index kept in lock-step
with each wallet's deque. This closes the "redo the tail" path from (3) above:
reprocessing a `STAGED` message is safe specifically because buffer state
does not double-count.

### 5. Persistent, rehydrated audit Merkle chain

`audit_merkle_roots` gains `content_hash` and `prev_merkle_root` columns
(migration `0005`). `AuditMerkleChain.__init__` rehydrates `self._entries`
from these columns on startup instead of starting empty, so `verify_chain()`
after a restart only raises `TamperDetectedError` for genuine tampering, not
for "the process restarted".

### 6. Finality marker

`risk_scores` gains a `finality` column (migration `0006`, values `provisional`
/ `final`, default `provisional`). The continuous streaming/SSE path always
writes `provisional` (there is no "window close" event in a continuously
updating buffer). A completed, bounded offline pipeline run (batch scoring
reaching its `persist` checkpoint) or a completed stream-replay run — both of
which process a *defined, closed* time window — write `final`. This is the
only place in the current architecture with a natural "the window is closed"
signal; introducing a new windowing subsystem purely to give the continuous
path a finality transition is out of scope for this issue.

### 7. Alert-delivery ledger + reconciliation

`streaming/alert_ledger.py::AlertDeliveryLedger` is a thin, durable,
SQL-backed record of every dispatched alert's outcome (`delivered` /
`dead_lettered`), keyed through the same `DedupKey` scheme
(`source="alert_delivery"`). `AlertDispatcher` writes to it on every terminal
outcome. `validation/reconciliation.py::reconcile_alert_delivery` traces every
score at or above threshold to exactly one ledger entry, flagging scores with
no accounted outcome as hard errors.

### 8. Replay determinism

`scripts/replay_stream.py`: `--resume` now looks up the consumer group's
committed offsets via `consumer.committed(...)` and seeks there (falling back
to earliest on first run) instead of unconditionally seeking to the beginning
of the topic. Offsets are committed after each processed message, matching
the live path's "commit only after durable side effects" guarantee, instead
of a fixed 100-message batch (a batch commit is safe only because
`FeatureBuffer` is now trade_id-idempotent; without item 4 this would still
risk in-buffer double counting on redelivery of the tail of a batch).

## Consequences

- `TradeDeduplicator`/`SeenEventCache`'s public shape changes: `is_duplicate`
  now raises `DedupBackendUnavailableError` on a Redis outage instead of
  returning `False`. Existing tests asserting fail-open behavior are rewritten
  to assert fail-closed behavior, per invariant 8 of the issue.
- `KafkaWorker` gains a dependency on `pipeline.exactly_once` and a new
  Prometheus gauge; its constructor still accepts an injected consumer for
  tests, unchanged.
- `RiskScoreStore.upsert` gains an optional `finality` keyword (default
  `"provisional"`), backward compatible with every existing caller.
- Full backend consolidation of `CheckpointStore`/`IdempotencyLedger` onto
  `ExactlyOnceStore`, wiring `tenant_id` end-to-end, and a general streaming
  "window close" concept are explicitly deferred (see PR "Follow-Up Work").
