"""Tests for scripts/backfill_amm_trades.py's --checkpoint-file resume semantics."""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from ingestion.amm_pool_loader import PoolNotFoundError
from scripts.backfill_amm_trades import _load_amm_trades_for_pools
from utils.checkpoint import PipelineCheckpoint

SINCE = datetime(2024, 1, 1, tzinfo=UTC)
UNTIL = datetime(2024, 6, 30, tzinfo=UTC)
POOL_A = "a" * 64
POOL_B = "b" * 64


def _pool_trades(wallet_pair):
    return pd.DataFrame(
        {
            "base_account": [wallet_pair[0]],
            "counter_account": [wallet_pair[1]],
            "amount": [10.0],
        }
    )


def _make_checkpoint(tmp_path, pool_ids, fresh=False):
    return PipelineCheckpoint.load_or_create(
        path=tmp_path / "ckpt.json",
        pipeline="backfill_amm_trades",
        fingerprint_inputs={"pool_ids": sorted(pool_ids), "since": str(SINCE), "until": str(UNTIL)},
        fresh=fresh,
    )


def test_pool_cached_and_reused_without_refetching(tmp_path):
    pool_ids = [POOL_A, POOL_B]
    load_calls = []

    def fake_load(pool_id, since, until):
        load_calls.append(pool_id)
        return _pool_trades(("GA", "GB") if pool_id == POOL_A else ("GC", "GD"))

    ckpt = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=fake_load):
        result = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt, checkpoint_dir=tmp_path
        )

    assert load_calls == [POOL_A, POOL_B]
    assert len(result) == 2
    assert set(result["pool_id"]) == {POOL_A, POOL_B}

    # Resume: a fresh PipelineCheckpoint instance re-loaded from disk must skip
    # both pools and reconstruct the same combined frame from cached Parquet.
    load_calls.clear()
    ckpt2 = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=fake_load):
        result2 = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt2, checkpoint_dir=tmp_path
        )

    assert load_calls == []
    assert len(result2) == 2
    assert set(result2["pool_id"]) == {POOL_A, POOL_B}


def test_pool_failure_recorded_and_retried_on_resume(tmp_path):
    pool_ids = [POOL_A, POOL_B]
    attempt = {"count": 0}

    def flaky_load(pool_id, since, until):
        if pool_id == POOL_A:
            return _pool_trades(("GA", "GB"))
        attempt["count"] += 1
        if attempt["count"] == 1:
            raise ConnectionError("horizon unreachable")
        return _pool_trades(("GC", "GD"))

    ckpt = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=flaky_load):
        result = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt, checkpoint_dir=tmp_path
        )

    assert set(result["pool_id"]) == {POOL_A}
    assert list(ckpt.completed) == [POOL_A]
    assert list(ckpt.failed) == [POOL_B]

    load_calls = []

    def tracking_flaky_load(pool_id, since, until):
        load_calls.append(pool_id)
        return flaky_load(pool_id, since, until)

    ckpt2 = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=tracking_flaky_load):
        result2 = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt2, checkpoint_dir=tmp_path
        )

    assert load_calls == [POOL_B]  # POOL_A was skipped, only the failed one retried
    assert set(result2["pool_id"]) == {POOL_A, POOL_B}
    assert ckpt2.failed == {}


def test_pool_not_found_is_recorded_done_and_not_retried(tmp_path):
    pool_ids = [POOL_A]
    load_calls = []

    def not_found_load(pool_id, since, until):
        load_calls.append(pool_id)
        raise PoolNotFoundError(f"{pool_id} not found")

    ckpt = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=not_found_load):
        result = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt, checkpoint_dir=tmp_path
        )

    assert result.empty
    assert ckpt.is_done(POOL_A)
    assert ckpt.completed[POOL_A]["metadata"] == {"found": False}

    load_calls.clear()
    ckpt2 = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=not_found_load):
        _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt2, checkpoint_dir=tmp_path
        )

    assert load_calls == []  # never retried


def test_empty_pool_recorded_done_without_artifact_file(tmp_path):
    pool_ids = [POOL_A]

    def empty_load(pool_id, since, until):
        return pd.DataFrame()

    ckpt = _make_checkpoint(tmp_path, pool_ids)
    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=empty_load):
        result = _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, checkpoint=ckpt, checkpoint_dir=tmp_path
        )

    assert result.empty
    assert ckpt.is_done(POOL_A)
    assert ckpt.artifact_path(POOL_A) is None
    assert not list(tmp_path.glob("*.parquet"))


def test_without_checkpoint_other_exceptions_propagate(tmp_path):
    pool_ids = [POOL_A]

    def failing_load(pool_id, since, until):
        raise ConnectionError("horizon unreachable")

    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=failing_load):
        with pytest.raises(ConnectionError):
            _load_amm_trades_for_pools(pool_ids, SINCE, UNTIL)


def test_without_checkpoint_pool_not_found_still_skipped(tmp_path):
    """Preserve pre-checkpoint behavior: PoolNotFoundError is always non-fatal."""
    pool_ids = [POOL_A]

    def not_found_load(pool_id, since, until):
        raise PoolNotFoundError("nope")

    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=not_found_load):
        result = _load_amm_trades_for_pools(pool_ids, SINCE, UNTIL)

    assert result.empty


def test_progress_callback_invoked_after_each_pool():
    """Test that progress callback is invoked after each pool is processed."""
    pool_ids = [POOL_A, POOL_B]
    progress_calls = []

    def track_progress(processed, total, rows, since, until):
        progress_calls.append({
            "processed": processed,
            "total": total,
            "rows": rows,
            "since": since,
            "until": until,
        })

    def fake_load(pool_id, since, until):
        return _pool_trades(("GA", "GB") if pool_id == POOL_A else ("GC", "GD"))

    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=fake_load):
        _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, progress_callback=track_progress
        )

    # Should have 2 progress calls, one per pool
    assert len(progress_calls) == 2
    assert progress_calls[0]["processed"] == 1
    assert progress_calls[0]["total"] == 2
    assert progress_calls[0]["rows"] == 1  # First pool has 1 row

    assert progress_calls[1]["processed"] == 2
    assert progress_calls[1]["total"] == 2
    assert progress_calls[1]["rows"] == 2  # Both pools loaded = 2 rows total


def test_progress_callback_with_fixture_scale(tmp_path):
    """Test progress callback is invoked at expected cadence for fixture run."""
    # Simulate a backfill with multiple pools where callback can track progress
    pool_ids = [POOL_A, POOL_B]
    progress_calls = []

    def track_progress(processed, total, rows, since, until):
        progress_calls.append({"processed": processed, "rows": rows})

    def fake_load(pool_id, since, until):
        # Simulate 100 rows per pool
        return pd.DataFrame({
            "base_account": ["GA"] * 100,
            "counter_account": ["GB"] * 100,
            "amount": [10.0] * 100,
        })

    with patch("scripts.backfill_amm_trades.load_amm_pool_trades", side_effect=fake_load):
        _load_amm_trades_for_pools(
            pool_ids, SINCE, UNTIL, progress_callback=track_progress
        )

    # With 2 pools × 100 rows, should see progress grow from 100 to 200
    assert len(progress_calls) == 2
    assert progress_calls[0]["rows"] == 100
    assert progress_calls[1]["rows"] == 200
