# Workflow Checkpointing

## Overview

Long-running data workflows — historical backfills, large batch scoring runs,
multi-stage ingestion pipelines — can run for hours and are frequently
interrupted (OOM kill, deploy, transient network failure). Without durable
progress tracking, a restart reprocesses everything from the beginning,
wasting compute and, for anything with side effects (writes, alerts,
external API calls), risking duplicate work.

`utils/checkpointing.py` provides a reusable checkpointing capability:

- **`CheckpointStore`** — durable, checksummed, atomically-written JSON
  persistence for workflow progress, keyed by `workflow_id`.
- **`CheckpointedWorkflow`** — step-level resumption (skip steps that already
  finished) and item-level resumption within a step (skip individual items
  already processed, e.g. individual wallets in a batch scoring run).

## Contract

- Checkpoint files are written via a temp-file-plus-`os.replace` sequence, so
  a reader never observes a half-written file, even under a hard kill.
- Every checkpoint carries a `schema_version` and a SHA-256 checksum over its
  body. `CheckpointStore.load` raises `CheckpointCorruptionError` on checksum
  mismatch and `CheckpointVersionError` on an incompatible schema version —
  both carry `workflow_id` and the on-disk `path` so an operator knows
  exactly which file to inspect.
- A step is only marked complete if its `with workflow.step(...)` block exits
  without raising. A failed step's partial item-level progress is still
  persisted, so retrying only reprocesses the items that didn't finish.

## Usage

```python
from utils.checkpointing import CheckpointStore, CheckpointedWorkflow

store = CheckpointStore(".checkpoints")
workflow = CheckpointedWorkflow(store, workflow_id="backfill_amm_2026_07")

for step_name in workflow.remaining_steps(["fetch", "transform", "load"]):
    with workflow.step(step_name):
        run_step(step_name)

# Item-level resumption within a single step:
with workflow.step("score_wallets") as ckpt:
    for wallet_id in ckpt.remaining_items(all_wallet_ids):
        score(wallet_id)
        ckpt.mark_done(wallet_id)
```

To force a full restart (e.g. because upstream source data changed):

```python
workflow.clear()
```

## Retention / stale-checkpoint cleanup

Checkpoint files accumulate one file per `workflow_id` and are never pruned
automatically. For long-lived deployments, purge checkpoints older than the
retention window to avoid unbounded disk usage:

```python
store.cleanup_stale(retention_days=30)   # default CHECKPOINT_RETENTION_DAYS = 30
```

`CheckpointStore.cleanup_stale` deletes every checkpoint whose file mtime is
older than the retention window and returns the list of removed paths.

## Validation

```
pytest tests/test_checkpointing.py -v
```

Covers: fresh-workflow behavior, step-level resumption across a simulated
process restart, failed steps not being marked complete, item-level
resumption, `clear()` discarding progress, corruption detection, schema
version mismatch detection, and confirming atomic writes leave no temp files
behind.

## Design tradeoffs / follow-ups

- Storage is local-filesystem JSON, matching this repo's existing scripts
  (e.g. `scripts/backfill_amm_trades.py`) which already run as single-host
  batch jobs — no distributed lock is implemented. A follow-up could add a
  Redis- or S3-backed `CheckpointStore` implementation behind the same
  interface for multi-worker backfills.
- Item-level progress is stored as a sorted list of string keys; for very
  large item sets (millions) this list can grow large. A follow-up could
  switch to a compact bitmap or range-based representation if a workflow
  needs it.
