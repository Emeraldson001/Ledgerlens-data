"""Tests for utils/idempotency.py — idempotent job execution semantics."""

import time

import pytest

from utils.idempotency import (
    ConcurrentExecutionError,
    IdempotencyConflictError,
    IdempotencyLedger,
    JobStatus,
    _hash_payload,
    idempotent,
)


@pytest.fixture
def ledger(tmp_path):
    return IdempotencyLedger(str(tmp_path / "idempotency.db"))


def test_run_executes_once_and_caches_result(ledger):
    calls = []

    def job():
        calls.append(1)
        return {"score": 42}

    result1 = ledger.run("job:1", job, input_payload={"wallet": "GABC"})
    result2 = ledger.run("job:1", job, input_payload={"wallet": "GABC"})

    assert result1 == {"score": 42}
    assert result2 == {"score": 42}
    assert len(calls) == 1


def test_conflict_on_key_reuse_with_different_input(ledger):
    ledger.run("job:1", lambda: "a", input_payload={"wallet": "GABC"})

    with pytest.raises(IdempotencyConflictError) as exc_info:
        ledger.run("job:1", lambda: "b", input_payload={"wallet": "GXYZ"})

    assert "job:1" in str(exc_info.value)


def test_concurrent_execution_within_lease_is_rejected(ledger):
    # Simulate another worker holding the lease: begin an attempt but never
    # complete it. _begin_attempt stores the hash of input_payload, not the
    # raw payload, so the lease-held record must be seeded with the same
    # hash run() will compute for "somehash" — otherwise run() sees an
    # input_hash mismatch and raises IdempotencyConflictError instead of the
    # ConcurrentExecutionError this test is actually about.
    ledger._begin_attempt("job:2", _hash_payload("somehash"))

    with pytest.raises(ConcurrentExecutionError):
        ledger.run("job:2", lambda: "result", input_payload="somehash", lease_seconds=300)


def test_stale_lease_is_reclaimed_and_job_runs(ledger):
    ledger._begin_attempt("job:3", _hash_payload("somehash"))
    # Force the lease to look old.
    conn = ledger._conn()
    conn.execute(
        "UPDATE idempotency_jobs SET updated_at = ? WHERE key = ?",
        (time.time() - 1000, "job:3"),
    )
    conn.commit()

    result = ledger.run("job:3", lambda: "done", input_payload="somehash", lease_seconds=1)
    assert result == "done"

    record = ledger.get("job:3")
    assert record.status == JobStatus.SUCCESS


def test_failed_job_can_be_retried(ledger):
    attempts = {"count": 0}

    def flaky():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError("transient failure")
        return "recovered"

    with pytest.raises(ValueError):
        ledger.run("job:4", flaky, input_payload={"x": 1})

    record = ledger.get("job:4")
    assert record.status == JobStatus.FAILED

    result = ledger.run("job:4", flaky, input_payload={"x": 1})
    assert result == "recovered"
    assert attempts["count"] == 2


def test_reset_allows_full_replay(ledger):
    calls = []

    def job():
        calls.append(1)
        return "ok"

    ledger.run("job:5", job, input_payload={"x": 1})
    ledger.reset("job:5")
    ledger.run("job:5", job, input_payload={"x": 1})

    assert len(calls) == 2


def test_idempotent_decorator_dedupes_by_key(ledger):
    calls = []

    @idempotent(ledger, key_fn=lambda wallet_id: f"score:{wallet_id}")
    def score_wallet(wallet_id: str) -> dict:
        calls.append(wallet_id)
        return {"wallet_id": wallet_id, "score": 99}

    result1 = score_wallet("GABC")
    result2 = score_wallet("GABC")

    assert result1 == result2 == {"wallet_id": "GABC", "score": 99}
    assert calls == ["GABC"]


def test_unrelated_keys_do_not_interfere(ledger):
    assert ledger.run("job:a", lambda: "a", input_payload=1) == "a"
    assert ledger.run("job:b", lambda: "b", input_payload=1) == "b"
    assert ledger.get("job:a").status == JobStatus.SUCCESS
    assert ledger.get("job:b").status == JobStatus.SUCCESS
