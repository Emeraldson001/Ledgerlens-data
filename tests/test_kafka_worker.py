"""Unit tests for streaming/kafka_worker.py (KafkaWorker).

confluent_kafka.Consumer is mocked — no live broker is required. The scorer,
dispatcher, and feature buffer are mocked so the test focuses on offset-commit
and lag-alerting semantics. The dedup cache is backed by fakeredis (falling
back to a plain in-memory fake when fakeredis is unavailable) so the
exactly-once ordering fix from Issue #670 is exercised end-to-end instead of
running in the previous "Redis unreachable → dedup silently disabled" mode
that let the original bug go undetected.
"""

import datetime
import logging
from unittest.mock import MagicMock

import pytest

from ingestion.avro_codec import load_schema, serialize
from pipeline.exactly_once import DedupBackendUnavailableError
from streaming.kafka_worker import DeduplicationCache

try:
    import fakeredis

    _FAKEREDIS_AVAILABLE = True
except ImportError:
    _FAKEREDIS_AVAILABLE = False
    fakeredis = None


def _avro_value(trade_id: str = "trade-001") -> bytes:
    record = {
        "trade_id": trade_id,
        "base_account": "WALLETBASE123",
        "counter_account": "WALLETCOUNTER456",
        "base_amount": 100.5,
        "counter_amount": 50.25,
        "price": 2.0,
        "asset_pair": "USDC:GISSUER/XLM:native",
        "ledger_close_time": datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.UTC),
        "ingestion_timestamp_ms": 1704110400000,
    }
    return serialize(record, load_schema())


def _make_msg(*, offset: int = 5, topic: str = "ledgerlens.trades.USDC_X") -> MagicMock:
    msg = MagicMock()
    msg.topic.return_value = topic
    msg.partition.return_value = 0
    msg.offset.return_value = offset
    msg.value.return_value = _avro_value()
    msg.error.return_value = None
    return msg


def _fake_dedup_cache() -> DeduplicationCache:
    """A real DeduplicationCache backed by fakeredis (in-process, no network)."""
    cache = DeduplicationCache()
    if _FAKEREDIS_AVAILABLE:
        cache._store._backend._redis = fakeredis.FakeStrictRedis(decode_responses=True)
        cache._store._backend._init_error = None
    return cache


def _make_worker(
    consumer,
    *,
    score=None,
    dispatch_side_effect=None,
    lag_high=100,
    dedup_cache=None,
):
    from streaming.kafka_worker import KafkaWorker

    scorer = MagicMock()
    scorer.score_wallet.return_value = score
    dispatcher = MagicMock()
    if dispatch_side_effect is not None:
        dispatcher.dispatch.side_effect = dispatch_side_effect
    buffer = MagicMock()

    consumer.get_watermark_offsets.return_value = (0, lag_high)

    if dedup_cache is None:
        dedup_cache = _fake_dedup_cache()
        if not _FAKEREDIS_AVAILABLE:
            pytest.skip("fakeredis not installed")

    worker = KafkaWorker(scorer, dispatcher, buffer, consumer=consumer, dedup_cache=dedup_cache)
    return worker, scorer, dispatcher


# ---------------------------------------------------------------------------
# 1. Offset committed exactly once per message after scorer + dispatcher
# ---------------------------------------------------------------------------


def test_offset_committed_once_after_dispatch():
    consumer = MagicMock()
    score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
    worker, scorer, dispatcher = _make_worker(consumer, score=score)

    msg = _make_msg()
    worker.process_message(msg)

    # Both accounts scored + dispatched, but the offset is committed once.
    assert dispatcher.dispatch.call_count == 2
    assert consumer.commit.call_count == 1
    consumer.commit.assert_called_once_with(message=msg, asynchronous=False)


def test_offset_committed_once_even_when_below_threshold():
    """Successful processing with no alert still commits exactly once."""
    consumer = MagicMock()
    worker, scorer, dispatcher = _make_worker(consumer, score=None)

    worker.process_message(_make_msg())

    dispatcher.dispatch.assert_not_called()
    assert consumer.commit.call_count == 1


# ---------------------------------------------------------------------------
# 2. Offset NOT committed if AlertDispatcher.dispatch raises
# ---------------------------------------------------------------------------


def test_offset_not_committed_when_dispatch_raises():
    consumer = MagicMock()
    score = {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 80}
    worker, scorer, dispatcher = _make_worker(
        consumer, score=score, dispatch_side_effect=RuntimeError("dispatch boom")
    )

    with pytest.raises(RuntimeError, match="dispatch boom"):
        worker.process_message(_make_msg())

    consumer.commit.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Lag above threshold emits a CRITICAL log (worker does not crash)
# ---------------------------------------------------------------------------


def test_lag_above_threshold_emits_critical(caplog):
    consumer = MagicMock()
    # High watermark 10_000 with offset 0 → lag ~9_999, well above default 500.
    worker, scorer, dispatcher = _make_worker(consumer, score=None, lag_high=10_000)

    with caplog.at_level(logging.CRITICAL, logger="streaming.kafka_worker"):
        worker.process_message(_make_msg(offset=0))

    assert any("exceeds threshold" in r.message for r in caplog.records)
    # The message was still processed and committed — no crash.
    assert consumer.commit.call_count == 1


def test_dlq_topic_is_skipped_not_scored():
    consumer = MagicMock()
    worker, scorer, dispatcher = _make_worker(consumer, score=None)

    msg = _make_msg(topic="ledgerlens.trades.dlq")
    worker.process_message(msg)

    scorer.score_wallet.assert_not_called()
    consumer.commit.assert_called_once_with(message=msg, asynchronous=False)


# ---------------------------------------------------------------------------
# 4. Exactly-once dedup ordering (Issue #670 — the critical bug and its fix)
# ---------------------------------------------------------------------------


def test_redelivery_after_successful_commit_is_skipped_not_reprocessed():
    """A message whose dedup key is already COMMITTED must not be reprocessed."""
    consumer = MagicMock()
    score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
    dedup_cache = _fake_dedup_cache()
    worker, scorer, dispatcher = _make_worker(consumer, score=score, dedup_cache=dedup_cache)

    msg = _make_msg(offset=1)
    worker.process_message(msg)
    assert dispatcher.dispatch.call_count == 2
    assert scorer.score_wallet.call_count == 2

    # Redelivery of the identical message (same trade_id/ledger_sequence).
    dispatcher.dispatch.reset_mock()
    scorer.score_wallet.reset_mock()
    consumer.commit.reset_mock()
    redelivered = _make_msg(offset=1)
    worker.process_message(redelivered)

    scorer.score_wallet.assert_not_called()
    dispatcher.dispatch.assert_not_called()
    consumer.commit.assert_called_once_with(message=redelivered, asynchronous=False)


def test_crash_mid_dispatch_then_redelivery_scores_second_wallet_exactly_once():
    """Reproduces the exact scenario from the issue: dispatch fails for the
    second wallet, so neither the dedup key nor the offset is committed.
    On redelivery, both wallets must be (re)processed — the second wallet
    must not be silently skipped because the first wallet's dispatch already
    ran once before the crash.
    """
    consumer = MagicMock()
    score = {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 80}
    dedup_cache = _fake_dedup_cache()

    # First delivery: dispatch succeeds for wallet A (base_account), then
    # raises for wallet B (counter_account) — simulating a crash/exception
    # mid-processing before the dedup key or offset is committed.
    call_count = {"n": 0}

    def flaky_dispatch(wallet, risk_score, pair_id):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("dispatch boom on second wallet")

    worker, scorer, dispatcher = _make_worker(
        consumer, score=score, dedup_cache=dedup_cache, dispatch_side_effect=flaky_dispatch
    )

    msg = _make_msg(offset=7)
    with pytest.raises(RuntimeError, match="dispatch boom"):
        worker.process_message(msg)

    assert dispatcher.dispatch.call_count == 2  # wallet A succeeded, wallet B raised
    consumer.commit.assert_not_called()  # neither dedup key nor offset committed

    # Redelivery of the SAME message after the "crash" — dispatch now succeeds
    # for both wallets. Both must be scored again (dedup state was STAGED, not
    # COMMITTED), and this time the message completes and commits.
    dispatcher.dispatch.side_effect = None
    dispatcher.dispatch.reset_mock()
    scorer.score_wallet.reset_mock()

    redelivered = _make_msg(offset=7)
    worker.process_message(redelivered)

    assert scorer.score_wallet.call_count == 2  # wallet B was NOT silently dropped
    assert dispatcher.dispatch.call_count == 2
    consumer.commit.assert_called_once_with(message=redelivered, asynchronous=False)


def test_dedup_backend_unavailable_halts_instead_of_failing_open():
    """When the dedup backend is unreachable, the worker must raise and leave
    the offset uncommitted rather than silently treating the message as new
    (invariant 8: fail closed, never fail open, on a dependency outage).
    """
    consumer = MagicMock()
    dedup_cache = DeduplicationCache(redis_url="redis://nonexistent-host-for-tests:9999/0")
    worker, scorer, dispatcher = _make_worker(consumer, score=None, dedup_cache=dedup_cache)

    with pytest.raises(DedupBackendUnavailableError):
        worker.process_message(_make_msg())

    scorer.score_wallet.assert_not_called()
    consumer.commit.assert_not_called()
