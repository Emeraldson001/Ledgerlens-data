"""Idempotent job execution semantics for pipelines.

Pipeline jobs (Kafka message processing, retraining triggers, alert
dispatch, backfill chunks) can be re-delivered or re-invoked — at-least-once
delivery, retried Celery/cron jobs, or an operator re-running a script after
a crash. Without an idempotency layer, replays either duplicate side effects
(double alerts, double writes) or silently return stale results for the
wrong input.

``IdempotencyLedger`` gives callers a small, reusable contract:

- Exactly-once *completion*: once a job with a given key succeeds, replaying
  it returns the cached result without re-running the body.
- Key-reuse detection: reusing a key with a *different* input payload is
  almost always a bug (e.g. a key derived from a truncated timestamp) and
  raises ``IdempotencyConflictError`` instead of silently returning a
  result computed for different inputs.
- Lease-based concurrency control: a job already PENDING is assumed to be
  actively executing elsewhere and concurrent duplicate execution is
  rejected — unless the lease has expired (the prior attempt crashed
  without updating status), in which case the lease is reclaimed and the
  job is retried.

Usage:
    ledger = IdempotencyLedger("idempotency.db")
    result = ledger.run(
        key=f"score_wallet:{wallet_id}:{ledger_close_time}",
        fn=lambda: score_wallet(wallet_id),
        input_payload={"wallet_id": wallet_id, "ledger_close_time": ledger_close_time},
    )

    # Or as a decorator:
    @idempotent(ledger, key_fn=lambda wallet_id, **_: f"score_wallet:{wallet_id}")
    def score_wallet(wallet_id: str) -> dict: ...
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import wraps
from typing import Any, TypeVar

from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_LEASE_SECONDS = 300.0


class JobStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"


class IdempotencyError(Exception):
    """Base class for idempotency-ledger failures."""


class IdempotencyConflictError(IdempotencyError):
    """A job key was reused with a different input payload.

    This is almost always a bug in key derivation (e.g. a key that doesn't
    include a field that actually varies between calls). Returning the
    cached result in this situation would silently apply stale output to
    new input.
    """

    def __init__(self, key: str, expected_hash: str, actual_hash: str):
        self.key = key
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"idempotency key {key!r} was already recorded with a different input "
            f"(recorded input_hash={expected_hash}, got input_hash={actual_hash}); "
            "reusing a key across different inputs is not allowed — include the "
            "varying field(s) in the key or use a distinct key per input"
        )


class ConcurrentExecutionError(IdempotencyError):
    """A job with this key is already PENDING and its lease has not expired."""

    def __init__(self, key: str, lease_age_seconds: float, lease_seconds: float):
        self.key = key
        self.lease_age_seconds = lease_age_seconds
        self.lease_seconds = lease_seconds
        super().__init__(
            f"job with idempotency key {key!r} is already in-flight "
            f"(lease age {lease_age_seconds:.1f}s < lease_seconds {lease_seconds:.1f}s); "
            "refusing concurrent duplicate execution — if the other execution is "
            "known dead, wait for the lease to expire or call IdempotencyLedger.reset(key)"
        )


@dataclass
class JobRecord:
    key: str
    status: JobStatus
    input_hash: str
    result: Any
    error: str | None
    attempts: int
    created_at: float
    updated_at: float


def _hash_payload(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class IdempotencyLedger:
    """SQLite-backed ledger recording job execution outcomes by idempotency key.

    The UNIQUE constraint on ``key`` is the actual concurrency guard —
    claiming a key is a single ``INSERT OR IGNORE`` statement, not an
    application-level check-then-act race.
    """

    def __init__(self, db_path: str = "idempotency.db"):
        self.db_path = db_path
        self._local = threading.local()
        self._init_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS idempotency_jobs (
                key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                result TEXT,
                error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """)
        conn.commit()

    @staticmethod
    def _row_to_record(row: tuple) -> JobRecord:
        return JobRecord(
            key=row[0],
            status=JobStatus(row[1]),
            input_hash=row[2],
            result=json.loads(row[3]) if row[3] is not None else None,
            error=row[4],
            attempts=row[5],
            created_at=row[6],
            updated_at=row[7],
        )

    def get(self, key: str) -> JobRecord | None:
        conn = self._conn()
        cur = conn.execute(
            "SELECT key, status, input_hash, result, error, attempts, created_at, updated_at "
            "FROM idempotency_jobs WHERE key = ?",
            (key,),
        )
        row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def _begin_attempt(self, key: str, input_hash: str) -> tuple[JobRecord, bool]:
        conn = self._conn()
        now = time.time()
        cur = conn.execute(
            "INSERT OR IGNORE INTO idempotency_jobs "
            "(key, status, input_hash, result, error, attempts, created_at, updated_at) "
            "VALUES (?, ?, ?, NULL, NULL, 0, ?, ?)",
            (key, JobStatus.PENDING.value, input_hash, now, now),
        )
        conn.commit()
        created = cur.rowcount == 1
        record = self.get(key)
        assert record is not None
        return record, created

    def _touch_pending(self, key: str) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE idempotency_jobs SET status = ?, attempts = attempts + 1, "
            "updated_at = ? WHERE key = ?",
            (JobStatus.PENDING.value, time.time(), key),
        )
        conn.commit()

    def _mark_success(self, key: str, result: Any) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE idempotency_jobs SET status = ?, result = ?, error = NULL, "
            "attempts = attempts + 1, updated_at = ? WHERE key = ?",
            (JobStatus.SUCCESS.value, json.dumps(result), time.time(), key),
        )
        conn.commit()

    def _mark_failed(self, key: str, error: str) -> None:
        conn = self._conn()
        conn.execute(
            "UPDATE idempotency_jobs SET status = ?, error = ?, "
            "attempts = attempts + 1, updated_at = ? WHERE key = ?",
            (JobStatus.FAILED.value, error, time.time(), key),
        )
        conn.commit()

    def reset(self, key: str) -> None:
        """Delete a job record so the key can be reused from scratch."""
        conn = self._conn()
        conn.execute("DELETE FROM idempotency_jobs WHERE key = ?", (key,))
        conn.commit()
        logger.info("Idempotency ledger entry reset for key=%s", key)

    def run(
        self,
        key: str,
        fn: Callable[[], T],
        *,
        input_payload: Any = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
    ) -> T:
        """Execute ``fn`` exactly once for ``key``; replay returns the cached result.

        Raises ``IdempotencyConflictError`` if ``key`` was previously used
        with a different ``input_payload``, and ``ConcurrentExecutionError``
        if another live execution currently holds the lease for ``key``.
        """
        input_hash = _hash_payload(input_payload)
        record, created = self._begin_attempt(key, input_hash)

        if record.input_hash != input_hash:
            raise IdempotencyConflictError(key, record.input_hash, input_hash)

        if record.status == JobStatus.SUCCESS:
            logger.info(
                "Idempotent replay: key=%s already succeeded (attempts=%d), "
                "returning cached result without re-executing",
                key,
                record.attempts,
            )
            return record.result  # type: ignore[return-value]

        if not created and record.status == JobStatus.PENDING:
            lease_age = time.time() - record.updated_at
            if lease_age < lease_seconds:
                raise ConcurrentExecutionError(key, lease_age, lease_seconds)
            logger.warning(
                "Reclaiming expired lease for key=%s (age=%.1fs exceeds "
                "lease_seconds=%.1fs) — prior attempt likely crashed mid-execution",
                key,
                lease_age,
                lease_seconds,
            )
            self._touch_pending(key)

        try:
            result = fn()
        except Exception as exc:
            self._mark_failed(key, repr(exc))
            raise
        else:
            self._mark_success(key, result)
            return result


def idempotent(
    ledger: IdempotencyLedger,
    *,
    key_fn: Callable[..., str],
    lease_seconds: float = DEFAULT_LEASE_SECONDS,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator wrapping a job function with idempotent execution semantics.

    ``key_fn`` receives the same ``*args, **kwargs`` as the wrapped function
    and must return a deterministic idempotency key for that call.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            key = key_fn(*args, **kwargs)
            return ledger.run(
                key,
                lambda: fn(*args, **kwargs),
                input_payload={"args": args, "kwargs": kwargs},
                lease_seconds=lease_seconds,
            )

        return wrapper

    return decorator
