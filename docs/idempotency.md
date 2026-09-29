# Idempotent Job Execution

## Overview

Pipeline jobs get re-invoked for reasons outside the pipeline's control:
at-least-once Kafka delivery, a cron/Celery retry, or an operator re-running
a script after a partial failure. Without an idempotency layer, replays
either duplicate side effects (double alerts, double writes to
`risk_score_store`) or — worse — silently return a cached result computed
for the *wrong* input.

`utils/idempotency.py` provides `IdempotencyLedger`, a small SQLite-backed
contract for exactly-once job completion:

- **Cached completion** — once a key's job succeeds, replay returns the
  cached result without re-running the body.
- **Key-reuse detection** — reusing a key with a different input payload
  raises `IdempotencyConflictError` instead of returning a stale result.
- **Lease-based concurrency control** — a job already `PENDING` is assumed
  to be executing elsewhere and a concurrent duplicate raises
  `ConcurrentExecutionError`, unless the lease has expired (the previous
  attempt crashed), in which case it's reclaimed automatically.

## Contract

- The `key` UNIQUE constraint in SQLite is the actual mutual-exclusion
  mechanism (`INSERT OR IGNORE`), not an application-level
  check-then-insert race.
- `input_payload` is hashed (SHA-256 over canonical JSON) and stored
  alongside the key; a mismatch on replay is treated as a caller bug and
  raises rather than silently proceeding.
- `lease_seconds` bounds how long a `PENDING` job blocks a duplicate before
  it's considered dead and reclaimed — tune this to comfortably exceed the
  job's expected runtime.

## Usage

```python
from utils.idempotency import IdempotencyLedger, idempotent

ledger = IdempotencyLedger("idempotency.db")

result = ledger.run(
    key=f"score_wallet:{wallet_id}:{ledger_close_time}",
    fn=lambda: score_wallet(wallet_id),
    input_payload={"wallet_id": wallet_id, "ledger_close_time": ledger_close_time},
)

# Or as a decorator:
@idempotent(ledger, key_fn=lambda wallet_id, **_: f"score_wallet:{wallet_id}")
def score_wallet(wallet_id: str) -> dict: ...
```

## Validation

```
pytest tests/test_idempotency.py -v
```

Covers: single execution with cached replay, key-reuse conflict detection,
rejection of concurrent in-flight duplicates, reclaiming an expired lease,
retrying a previously-failed job, `reset()`, the decorator form, and
independent keys not interfering with each other.

## Design tradeoffs / follow-ups

- SQLite (WAL mode) was chosen over a JSON file (as used in
  [[checkpointing]]) specifically because the UNIQUE constraint gives real
  atomic claim semantics across processes, which a JSON store cannot
  provide without an external lock.
- Lease expiry is polled at call time (no background reaper); a key whose
  owner crashed and is never retried stays `PENDING` until something calls
  `run` or `reset` again. Acceptable for batch/worker-triggered jobs; a
  background sweeper could be added for long-idle queues.
- Results must be JSON-serialisable. Non-serialisable results should be
  persisted by the caller (e.g. to `risk_score_store`) with only a
  reference/ID passed through the ledger.
