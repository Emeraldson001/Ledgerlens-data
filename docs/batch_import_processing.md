# Batch Processing Abstractions for Large Transaction Imports

## Why

The bulk loaders (`ingestion/historical_loader.py`,
`ingestion/account_activity_loader.py`, `ingestion/amm_pool_loader.py`,
`ingestion/orderbook_loader.py`) each import large volumes of records from
Horizon/external sources. Each one independently needs chunking (to bound
memory and transaction size), retry with backoff (upstream APIs and DB
writes are flaky at volume), and resumability (a multi-hour historical
backfill that crashes at 80% shouldn't restart from zero). Prior to this
change that logic would have to be reimplemented per loader.

## What this adds

`ingestion/batch_processor.py` provides `BatchProcessor[T]`, a generic,
typed abstraction over "take an iterable of records, process them in
chunks, retry failed chunks, checkpoint progress, report a summary":

```python
from ingestion.batch_processor import BatchProcessor

processor = BatchProcessor(
    chunk_size=500,
    max_retries=3,
    backoff_seconds=0.5,
    checkpoint_path="var/checkpoints/orderbook_import.json",
)

def import_chunk(chunk: list[dict]) -> None:
    db.bulk_insert(chunk)

summary = processor.run(records, import_chunk, job_id="orderbook-2024-06-backfill")
if not summary.ok:
    for failure in summary.failures:
        log.error("chunk %s failed after %s attempts: %s", failure.chunk_index, failure.attempts, failure.error)
```

Key properties:

- **Chunking** via `chunk_size`, memory-bounded (no buffering the whole
  iterable).
- **Retry with exponential backoff** per chunk (`max_retries`,
  `backoff_seconds`); a chunk that exhausts retries is recorded in
  `summary.failures` with the causing exception, and the job continues —
  one bad chunk does not abort a multi-hour import.
- **Resumable checkpointing**: `checkpoint_path` + `job_id` together track
  which chunk indices completed, atomically (temp file + `os.replace`).
  Re-running the same `job_id` after a crash skips completed chunks. A
  fully successful run clears the checkpoint for that `job_id`.
- **Progress hook** (`on_progress`) for wiring into
  `monitoring/metrics_collector.py` or logging without coupling this module
  to a specific metrics backend.
- **Typed summary** (`BatchSummary`) with `total_items`, `succeeded_items`,
  `failed_items`, `chunks_failed`, `duration_seconds`, and structured
  `failures`, plus `.as_dict()` for JSON logging/reporting.

## Developer commands

Run the focused test suite:

```
pytest tests/test_batch_processor.py -v
```

Covers: chunk totals/counts, transient-failure retry-then-success, a
failing chunk being recorded without aborting the job, resume-skips-
completed-chunks after a partial failure, checkpoint clearing on full
success, the progress callback firing per chunk, and the `chunk_size<=0`
validation error.

## Design tradeoffs

- **Callback-based, not a class to subclass.** `process_chunk` is a plain
  `Callable[[list[T]], None]`, matching the existing loader style (e.g.
  `detection/batch_scorer.py`'s `score_batch`) rather than introducing an
  ABC every loader must inherit from. Keeps adoption a wrapper, not a
  rewrite.
- **JSON checkpoint file, not a DB table.** Consistent with the
  file-backed-manifest pattern used elsewhere in this PR
  (`detection/artifact_lifecycle.py`) and avoids requiring a DB connection
  just to resume a bulk import.
- **Chunk-level retry, not item-level.** Matches how the existing loaders
  batch writes (bulk insert per chunk); item-level retry would require the
  caller's `process_chunk` to be internally partitionable, which isn't true
  of a bulk-insert callback.
- **Follow-up work:** migrating `ingestion/historical_loader.py` and
  `ingestion/amm_pool_loader.py` to use `BatchProcessor` directly (this PR
  adds the abstraction; wiring in the existing loaders is a separate,
  reviewable change to avoid an unrelated broad refactor here).
