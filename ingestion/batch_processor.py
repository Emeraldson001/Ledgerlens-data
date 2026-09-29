"""Batch processing abstractions for large transaction imports.

Historical/bulk loaders (``ingestion/historical_loader.py``,
``ingestion/account_activity_loader.py``, ``ingestion/amm_pool_loader.py``)
each need the same three things when importing large volumes of records:
chunking, resumable checkpointing, and per-item retry with backoff. This
module factors that into a single reusable, typed abstraction so new
importers don't have to reimplement it, and so behavior (retry counts,
checkpoint format, failure reporting) is consistent across all of them.

Typical usage::

    processor = BatchProcessor(
        chunk_size=500,
        max_retries=3,
        checkpoint_path="/tmp/import_checkpoint.json",
    )

    def import_chunk(chunk: list[dict]) -> None:
        db.bulk_insert(chunk)

    summary = processor.run(records_iterable, import_chunk, job_id="orderbook-2024-06")
    print(summary.succeeded, summary.failed, summary.duration_seconds)

Resumability: if the process is interrupted, calling ``run`` again with the
same ``job_id`` and ``checkpoint_path`` skips chunks already marked
complete in the checkpoint file.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class BatchProcessingError(Exception):
    """Base class for batch-processor diagnostics."""


class ChunkFailedError(BatchProcessingError):
    """Raised (and collected, not necessarily propagated) when a chunk
    exhausts its retry budget."""

    def __init__(self, job_id: str, chunk_index: int, attempts: int, cause: Exception):
        self.job_id = job_id
        self.chunk_index = chunk_index
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"Batch job {job_id!r} chunk #{chunk_index} failed after {attempts} attempt(s): "
            f"{type(cause).__name__}: {cause}"
        )


@dataclass
class ChunkOutcome:
    chunk_index: int
    item_count: int
    ok: bool
    attempts: int
    error: str | None = None


@dataclass
class BatchSummary:
    job_id: str
    total_items: int = 0
    succeeded_items: int = 0
    failed_items: int = 0
    chunks_processed: int = 0
    chunks_skipped_resume: int = 0
    chunks_failed: int = 0
    duration_seconds: float = 0.0
    failures: list[ChunkOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.chunks_failed == 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "total_items": self.total_items,
            "succeeded_items": self.succeeded_items,
            "failed_items": self.failed_items,
            "chunks_processed": self.chunks_processed,
            "chunks_skipped_resume": self.chunks_skipped_resume,
            "chunks_failed": self.chunks_failed,
            "duration_seconds": round(self.duration_seconds, 3),
            "ok": self.ok,
            "failures": [
                {
                    "chunk_index": f.chunk_index,
                    "item_count": f.item_count,
                    "attempts": f.attempts,
                    "error": f.error,
                }
                for f in self.failures
            ],
        }


def _chunks(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError(f"chunk_size must be positive, got {size}")
    buf: list[T] = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


class _JsonCheckpoint:
    """Tracks which chunk indices of a job have completed, atomically."""

    def __init__(self, path: str | None):
        self.path = path
        self._completed: dict[str, list[int]] = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self._completed = json.load(f)

    def is_done(self, job_id: str, chunk_index: int) -> bool:
        return chunk_index in self._completed.get(job_id, [])

    def mark_done(self, job_id: str, chunk_index: int) -> None:
        if not self.path:
            return
        self._completed.setdefault(job_id, []).append(chunk_index)
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".batch_checkpoint_", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._completed, f)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def clear(self, job_id: str) -> None:
        self._completed.pop(job_id, None)
        if self.path and os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump(self._completed, f)


class BatchProcessor(Generic[T]):
    """Chunk, retry, checkpoint, and summarize a large import job.

    Parameters
    ----------
    chunk_size:
        Number of items per chunk passed to the processor callback.
    max_retries:
        Number of attempts per chunk before it is recorded as failed
        (attempt 1 is not a "retry", so ``max_retries=3`` means up to 3
        total attempts).
    backoff_seconds:
        Base delay between retries; attempt *n* sleeps
        ``backoff_seconds * (2 ** (n - 1))``.
    checkpoint_path:
        Optional path to a JSON file tracking completed chunk indices per
        ``job_id``, enabling resumable re-runs after a crash.
    on_progress:
        Optional callback invoked with each ``ChunkOutcome`` as chunks
        complete, for wiring into monitoring (e.g.
        ``monitoring/metrics_collector.py``) without coupling this module
        to any specific metrics backend.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        checkpoint_path: str | None = None,
        on_progress: Callable[[ChunkOutcome], None] | None = None,
    ):
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.checkpoint_path = checkpoint_path
        self.on_progress = on_progress

    def run(
        self,
        records: Iterable[T],
        process_chunk: Callable[[list[T]], None],
        job_id: str,
        resume: bool = True,
    ) -> BatchSummary:
        """Process ``records`` in chunks, calling ``process_chunk`` per chunk.

        ``process_chunk`` should raise on failure (any exception); it will
        be retried up to ``max_retries`` times with exponential backoff. If
        all attempts fail, the chunk is recorded in the returned summary's
        ``failures`` and processing continues with the next chunk (a single
        bad chunk does not abort the whole import).
        """
        checkpoint = _JsonCheckpoint(self.checkpoint_path if resume else None)
        summary = BatchSummary(job_id=job_id)
        start = time.monotonic()

        for chunk_index, chunk in enumerate(_chunks(records, self.chunk_size)):
            summary.total_items += len(chunk)

            if resume and checkpoint.is_done(job_id, chunk_index):
                summary.chunks_skipped_resume += 1
                summary.succeeded_items += len(chunk)
                continue

            outcome = self._run_chunk_with_retry(job_id, chunk_index, chunk, process_chunk)
            summary.chunks_processed += 1

            if outcome.ok:
                summary.succeeded_items += outcome.item_count
                checkpoint.mark_done(job_id, chunk_index)
            else:
                summary.failed_items += outcome.item_count
                summary.chunks_failed += 1
                summary.failures.append(outcome)

            if self.on_progress:
                self.on_progress(outcome)

        summary.duration_seconds = time.monotonic() - start

        if summary.ok and resume:
            checkpoint.clear(job_id)

        return summary

    def _run_chunk_with_retry(
        self,
        job_id: str,
        chunk_index: int,
        chunk: list[T],
        process_chunk: Callable[[list[T]], None],
    ) -> ChunkOutcome:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                process_chunk(chunk)
                return ChunkOutcome(chunk_index, len(chunk), ok=True, attempts=attempt)
            except Exception as exc:  # noqa: BLE001 - intentionally broad, retried & reported
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        return ChunkOutcome(
            chunk_index,
            len(chunk),
            ok=False,
            attempts=self.max_retries,
            error=f"{type(last_error).__name__}: {last_error}",
        )
