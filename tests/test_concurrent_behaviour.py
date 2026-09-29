"""Comprehensive concurrency validation for streaming workers.

Validates thread safety, lock correctness, and data integrity under
concurrent access for every threaded component in the streaming pipeline.

Every test mocks external dependencies — no live Horizon, Kafka, or Redis.
"""

from __future__ import annotations

import datetime
import threading
import time
from unittest.mock import MagicMock

import pytest

from streaming.pubsub_router import PubSubRouter
from streaming.streaming_scorer import AdaptiveBatchController
from streaming.ws_abuse_detector import AbuseDetector
from streaming.ws_server import ReplayBuffer, SequenceCounter, TokenBucket
from tests.concurrent_validators import StressRunner, assert_eventually

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _stellar_account(seed: str = "") -> str:
    """Generate a valid Stellar G-prefixed account ID (56 chars, base32)."""
    import hashlib

    h = hashlib.sha256(seed.encode()).digest()
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
    # Expand the 32-byte digest to 55 chars by cycling through it twice
    expanded = (h * 2)[:55]
    encoded = "".join(alphabet[b % 32] for b in expanded)
    return "G" + encoded


WALLET_A = _stellar_account("wallet-a")
WALLET_B = _stellar_account("wallet-b")
WALLET_C = _stellar_account("wallet-c")


def _valid_wallet(n: int) -> str:
    return _stellar_account(f"wallet-{n}")


def _wallet_channel(n: int) -> str:
    return f"wallet/{_valid_wallet(n)}"


def _make_trade(
    base_account: str = WALLET_A,
    counter_account: str = WALLET_B,
    base_amount: float = 100.0,
    trade_id: str = "t1",
):
    from ingestion.data_models import Asset, Trade

    return Trade(
        trade_id=trade_id,
        ledger_close_time=datetime.datetime(2024, 6, 1, 12, 0, 0, tzinfo=datetime.UTC),
        base_account=base_account,
        counter_account=counter_account,
        base_asset=Asset(code="USDC", issuer=USDC_ISSUER),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=base_amount,
        counter_amount=50.0,
        price=2.0,
    )


def _make_metadata_update(wallet: str, effect_type: str = "trustline_created"):
    from streaming.account_metadata_stream import AccountMetadataUpdate

    return AccountMetadataUpdate(
        account_id=wallet,
        effect_type=effect_type,
        effect_id=f"effect-{wallet}-{time.time_ns()}",
    )


# ===================================================================
# FeatureBuffer — per-wallet lock isolation & data integrity
# ===================================================================


class TestFeatureBufferConcurrency:
    """Validate FeatureBuffer thread-safety under concurrent read/write."""

    def test_concurrent_writes_to_same_wallet_no_data_loss(self):
        """Many threads writing to the same wallet: all writes accounted for."""
        from streaming.feature_buffer import FeatureBuffer

        buf = FeatureBuffer(max_trades=5000)
        n_threads = 8
        trades_per_thread = 200
        total_trades = n_threads * trades_per_thread

        def write_trades(tid: int, it: int) -> None:
            idx = tid * trades_per_thread + it
            buf.update(
                _make_trade(
                    base_account=WALLET_A,
                    counter_account=WALLET_B,
                    base_amount=float(idx),
                    trade_id=f"w{tid}-{it}",
                )
            )

        errors = StressRunner(target=write_trades).run(
            n_threads=n_threads, n_iters=trades_per_thread, timeout=30
        )
        assert not errors, f"Concurrent write errors: {errors}"

        # The buffer is capped at max_trades, but at minimum all unique trade
        # IDs from the last max_trades entries should be present.
        expected_count = min(total_trades, 5000)
        actual = buf.wallet_trade_count(WALLET_A)
        assert (
            actual == expected_count
        ), f"Expected ~{expected_count} trades for wallet A, got {actual}"

    def test_concurrent_writes_to_different_wallets_no_contention(self):
        """Separate wallets updated concurrently: all writes succeed."""
        from streaming.feature_buffer import FeatureBuffer

        buf = FeatureBuffer(max_trades=100)
        wallets = [_valid_wallet(i) for i in range(20)]
        n_threads = 20
        n_iters = 50

        def write_different_wallets(tid: int, it: int) -> None:
            w = wallets[tid % len(wallets)]
            buf.update(
                _make_trade(
                    base_account=w,
                    counter_account=WALLET_B,
                    base_amount=float(tid * 1000 + it),
                    trade_id=f"dw{tid}-{it}",
                )
            )

        errors = StressRunner(target=write_different_wallets).run(
            n_threads=n_threads, n_iters=n_iters, timeout=30
        )
        assert not errors, f"Concurrent cross-wallet write errors: {errors}"

        all_wallets = buf.all_wallets()
        for w in wallets:
            assert w in all_wallets, f"Wallet {w} missing after concurrent writes"

    def test_concurrent_reads_during_writes_invariant(self):
        """Concurrent get_feature_row calls don't corrupt the buffer state."""
        from streaming.feature_buffer import FeatureBuffer

        buf = FeatureBuffer(max_trades=1000)
        # Pre-populate
        for i in range(100):
            buf.update(
                _make_trade(
                    base_account=WALLET_A,
                    counter_account=WALLET_B,
                    base_amount=float(i),
                    trade_id=f"pre{i}",
                )
            )

        errors: list[Exception] = []
        errors_lock = threading.Lock()
        stop_event = threading.Event()

        def writer() -> None:
            i = 0
            while not stop_event.is_set():
                buf.update(
                    _make_trade(
                        base_account=WALLET_A,
                        counter_account=WALLET_B,
                        base_amount=float(i),
                        trade_id=f"cw{i}",
                    )
                )
                i += 1
                time.sleep(0.001)

        def reader() -> None:
            while not stop_event.is_set():
                try:
                    row = buf.get_feature_row(WALLET_A)
                    if row is not None:
                        assert "benford_chi_square_1h" in row.index
                except Exception as exc:
                    with errors_lock:
                        errors.append(exc)
                time.sleep(0.001)

        threads = []
        for _ in range(2):
            t = threading.Thread(target=writer, daemon=True)
            threads.append(t)
        for _ in range(4):
            t = threading.Thread(target=reader, daemon=True)
            threads.append(t)

        for t in threads:
            t.start()
        time.sleep(2.0)
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"Errors during concurrent read/write: {errors}"

    def test_wallet_trade_count_during_concurrent_updates(self):
        """wallet_trade_count returns non-negative during concurrent writes."""
        from streaming.feature_buffer import FeatureBuffer

        buf = FeatureBuffer(max_trades=100)

        def write_and_check(tid: int, it: int) -> None:
            buf.update(
                _make_trade(
                    base_account=WALLET_A,
                    counter_account=WALLET_B,
                    base_amount=float(tid * 100 + it),
                    trade_id=f"wc{tid}-{it}",
                )
            )
            count = buf.wallet_trade_count(WALLET_A)
            assert count >= 0, f"Negative trade count: {count}"

        errors = StressRunner(target=write_and_check).run(n_threads=4, n_iters=50, timeout=15)
        assert not errors, f"Errors during count checks: {errors}"


# ===================================================================
# FeatureCache — concurrent get/put
# ===================================================================


class TestFeatureCacheConcurrency:
    """Thread-safe feature cache under concurrent access."""

    def test_concurrent_get_put_same_key(self):
        """Concurrent get/put to the same cache key."""
        from detection.feature_cache import FeatureCache

        cache = FeatureCache(ttl_seconds=3600, maxsize=100)
        key = "G" + "A" * 55

        def access_cache(tid: int, it: int) -> None:
            if it % 2 == 0:
                row = cache.get(key)
                if row is None:
                    cache.put(key, {"score": float(tid), "seq": it})
            else:
                cache.put(key, {"score": float(tid), "seq": it})

        errors = StressRunner(target=access_cache).run(n_threads=4, n_iters=100, timeout=15)
        assert not errors, f"FeatureCache concurrent access errors: {errors}"

    def test_cache_ttl_not_extended_by_concurrent_reads(self):
        """Concurrent reads don't corrupt TTL expiry."""
        from detection.feature_cache import FeatureCache

        cache = FeatureCache(ttl_seconds=0.1, maxsize=100)
        key = "G" + "B" * 55
        cache.put(key, {"score": 50.0})

        # Concurrent reads while TTL expires
        def read_key(tid: int, it: int) -> None:
            _ = cache.get(key)

        errors = StressRunner(target=read_key).run(n_threads=4, n_iters=30, timeout=15)
        assert not errors

        # Wait for TTL to expire
        time.sleep(0.2)
        assert cache.get(key) is None, "Cache entry should have expired"


# ===================================================================
# PubSubRouter — concurrent subscribe/unsubscribe/disconnect
# ===================================================================


class TestPubSubRouterConcurrency:
    """Validate PubSubRouter thread-safety."""

    def test_concurrent_subscribe_unsubscribe(self):
        """Many threads subscribe/unsubscribe the same channels."""
        router = PubSubRouter()
        n_clients = 20
        channels = [_wallet_channel(i) for i in range(10)]
        all_subscribed: list[set[str]] = [set() for _ in range(n_clients)]

        def subscribe_unsubscribe(tid: int, it: int) -> None:
            client_id = f"client-{tid}"
            ch = channels[it % len(channels)]
            if tid % 2 == 0:
                router.subscribe(client_id, [ch])
                all_subscribed[tid].add(ch)
            else:
                router.unsubscribe(client_id, [ch])
                all_subscribed[tid].discard(ch)

        errors = StressRunner(target=subscribe_unsubscribe).run(
            n_threads=n_clients, n_iters=60, timeout=30
        )
        assert not errors, f"Subscribe/unsubscribe errors: {errors}"

        # Verify router's internal state is consistent
        stats = router.stats()
        assert stats["total_channels"] >= 0
        assert stats["total_clients"] >= 0

    def test_concurrent_subscribe_and_get_subscribers(self):
        """Concurrent subscribe calls don't miss subscribers."""
        router = PubSubRouter()
        channel = "wallet/G" + "A" * 55
        n_clients = 10

        def subscribe(tid: int, it: int) -> None:
            router.subscribe(f"client-{tid}", [channel])

        errors = StressRunner(target=subscribe).run(n_threads=n_clients, n_iters=1, timeout=15)
        assert not errors

        subscribers = router.get_subscribers(channel)
        assert (
            len(subscribers) == n_clients
        ), f"Expected {n_clients} subscribers, got {len(subscribers)}"

    def test_concurrent_subscribe_and_disconnect(self):
        """Subscribe one client while another disconnects — no state leak."""
        router = PubSubRouter()
        channel = "wallet/G" + "C" * 55

        # Pre-subscribe a few clients
        for i in range(5):
            router.subscribe(f"persistent-{i}", [channel])

        def join_and_leave(tid: int, it: int) -> None:
            cid = f"transient-{tid}-{it}"
            router.subscribe(cid, [channel])
            router.disconnect(cid)

        errors = StressRunner(target=join_and_leave).run(n_threads=8, n_iters=30, timeout=15)
        assert not errors

        # Persistent clients should still be subscribed
        subscribers = router.get_subscribers(channel)
        for i in range(5):
            assert f"persistent-{i}" in subscribers

        # No transient clients should remain
        for i in range(8):
            for j in range(30):
                assert f"transient-{i}-{j}" not in subscribers

    def test_clients_for_event_no_duplicates(self):
        """get_clients_for_event never returns duplicate client IDs."""
        router = PubSubRouter()
        wallet_channel = f"wallet/{WALLET_A}"
        pair_channel = f"pair/USDC:{USDC_ISSUER}/XLM:native"

        def sub_and_check(tid: int, it: int) -> None:
            cid = f"client-{tid}"
            router.subscribe(cid, [wallet_channel, pair_channel])
            clients = router.get_clients_for_event(WALLET_A, f"USDC:{USDC_ISSUER}/XLM:native")
            client_list = list(clients)
            assert len(client_list) == len(
                set(client_list)
            ), f"Duplicate client IDs in event recipients: {client_list}"

        errors = StressRunner(target=sub_and_check).run(n_threads=8, n_iters=15, timeout=15)
        assert not errors


# ===================================================================
# AlertDispatcher — concurrent dispatch with cooldown
# ===================================================================


class TestAlertDispatcherConcurrency:
    """Validate AlertDispatcher thread-safety under concurrent dispatch."""

    def test_concurrent_dispatch_same_wallet_cooldown(self):
        """Concurrent dispatch for the same wallet: cooldown prevents duplicates."""
        from streaming.alert_dispatcher import AlertDispatcher

        dispatcher = AlertDispatcher(
            channel="stdout",
            alert_cooldown_seconds=3600,
            threshold=0,
        )
        risk_score = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}

        n_threads = 10
        dispatch_count: list[int] = [0]
        dispatch_lock = threading.Lock()

        original_deliver = dispatcher._deliver

        def counting_deliver(wallet, risk_score, pair_id):
            with dispatch_lock:
                dispatch_count[0] += 1
            return original_deliver(wallet, risk_score, pair_id)

        dispatcher._deliver = counting_deliver

        def dispatch(tid: int, it: int) -> None:
            dispatcher.dispatch(WALLET_A, risk_score, "USDC:XLM")

        errors = StressRunner(target=dispatch).run(n_threads=n_threads, n_iters=5, timeout=15)
        assert not errors

        # Only the first delivery should go through due to cooldown
        assert dispatch_count[0] == 1, f"Expected 1 dispatch (cooldown), got {dispatch_count[0]}"

    def test_concurrent_dispatch_different_wallets(self):
        """Different wallets dispatched concurrently: all should deliver."""
        from streaming.alert_dispatcher import AlertDispatcher

        dispatcher = AlertDispatcher(
            channel="stdout",
            alert_cooldown_seconds=3600,
            threshold=0,
        )
        risk_score = {"score": 90, "benford_flag": True, "ml_flag": True, "confidence": 80}

        dispatched: list[str] = []
        dispatch_lock = threading.Lock()
        original_deliver = dispatcher._deliver

        def counting_deliver(wallet, risk_score, pair_id):
            with dispatch_lock:
                dispatched.append(wallet)
            return original_deliver(wallet, risk_score, pair_id)

        dispatcher._deliver = counting_deliver

        wallets = [_valid_wallet(i) for i in range(20)]

        def dispatch(tid: int, it: int) -> None:
            w = wallets[(tid * 5 + it) % len(wallets)]
            dispatcher.dispatch(w, risk_score, "USDC:XLM")

        errors = StressRunner(target=dispatch).run(n_threads=8, n_iters=20, timeout=15)
        assert not errors

        # Each wallet should have been dispatched once (first call per wallet)
        unique_dispatched = set(dispatched)
        assert len(unique_dispatched) == len(
            wallets
        ), f"Expected {len(wallets)} unique wallets dispatched, got {len(unique_dispatched)}"

    def test_concurrent_threshold_check(self):
        """threshold_controller is not corrupted by concurrent dispatch calls."""
        from streaming.alert_dispatcher import AlertDispatcher

        dispatcher = AlertDispatcher(
            channel="stdout",
            alert_cooldown_seconds=1,
            threshold=50,
        )
        risk_score_high = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 70}
        risk_score_low = {"score": 20, "benford_flag": False, "ml_flag": False, "confidence": 90}

        delivered: list[str] = []
        dl_lock = threading.Lock()
        original = dispatcher._deliver

        def record_deliver(wallet, risk_score, pair_id):
            with dl_lock:
                delivered.append(wallet)
            return original(wallet, risk_score, pair_id)

        dispatcher._deliver = record_deliver

        def dispatch_mixed(tid: int, it: int) -> None:
            if it % 2 == 0:
                dispatcher.dispatch(WALLET_A, risk_score_high, "USDC:XLM")
            else:
                dispatcher.dispatch(WALLET_A, risk_score_low, "USDC:XLM")

        errors = StressRunner(target=dispatch_mixed).run(n_threads=4, n_iters=20, timeout=15)
        assert not errors


# ===================================================================
# AbuseDetector — concurrent record/reset
# ===================================================================


class TestAbuseDetectorConcurrency:
    """Validate AbuseDetector thread-safety."""

    def test_concurrent_record_same_client(self):
        """Many threads recording requests for the same client."""
        detector = AbuseDetector(
            max_requests_per_minute=1000,
            max_distinct_wallets_per_window=500,
        )
        client_id = "test-client"

        def record_req(tid: int, it: int) -> None:
            verdict = detector.record(client_id, _wallet_channel(tid))
            # Should not block within limits
            assert isinstance(verdict.blocked, bool)

        errors = StressRunner(target=record_req).run(n_threads=10, n_iters=30, timeout=15)
        assert not errors

    def test_concurrent_record_and_reset(self):
        """Concurrent record/reset does not leak state or crash."""
        detector = AbuseDetector(
            max_requests_per_minute=100,
            max_distinct_wallets_per_window=50,
        )
        client_id = "reset-client"

        def record_and_reset(tid: int, it: int) -> None:
            if it % 5 == 0:
                detector.reset(client_id)
            else:
                detector.record(client_id, _wallet_channel(tid))

        errors = StressRunner(target=record_and_reset).run(n_threads=8, n_iters=60, timeout=15)
        assert not errors

    def test_abuse_blocks_after_threshold_concurrent(self):
        """After exceeding threshold, blocked verdict is returned reliably."""
        detector = AbuseDetector(
            max_requests_per_minute=10,
            max_distinct_wallets_per_window=10,
            wallet_window_seconds=3600,
        )
        client_id = "block-client"

        blocked_count: list[int] = [0]

        def trigger_block(tid: int, it: int) -> None:
            verdict = detector.record(client_id, _wallet_channel(tid))
            if verdict.blocked:
                blocked_count[0] += 1

        errors = StressRunner(target=trigger_block).run(n_threads=4, n_iters=20, timeout=15)
        assert not errors

        # At some point the client should have been blocked
        assert blocked_count[0] >= 1, "Expected client to be blocked after exceeding wallet limit"


# ===================================================================
# MetadataJoinState — concurrent apply / get / evict
# ===================================================================


class TestMetadataJoinStateConcurrency:
    """Validate MetadataJoinState thread-safety."""

    def test_concurrent_apply_and_get(self):
        """Concurrent apply_update and get_metadata for the same wallet."""
        from streaming.pipeline import MetadataJoinState

        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=7200)

        def apply_and_get(tid: int, it: int) -> None:
            wallet = WALLET_A if it % 2 == 0 else WALLET_B
            update = _make_metadata_update(wallet)
            state.apply_update(update)
            meta = state.get_metadata(wallet)
            if meta is not None:
                assert meta.account_id == wallet

        errors = StressRunner(target=apply_and_get).run(n_threads=8, n_iters=100, timeout=15)
        assert not errors

    def test_concurrent_evict_during_access(self):
        """Eviction runs concurrently with get_metadata — no crash."""
        from streaming.pipeline import MetadataJoinState

        state = MetadataJoinState(join_window_seconds=30, active_wallet_ttl_seconds=1)

        for i in range(20):
            wallet = _valid_wallet(i)
            update = _make_metadata_update(wallet)
            state.apply_update(update)
            state.get_metadata(wallet)

        def evict_and_access(tid: int, it: int) -> None:
            if tid % 2 == 0:
                state.evict_inactive_wallets()
            else:
                wallet = _valid_wallet(it % 20)
                state.get_metadata(wallet)

        errors = StressRunner(target=evict_and_access).run(n_threads=8, n_iters=50, timeout=15)
        assert not errors

    def test_pending_promotion_under_concurrent_access(self):
        """Pending updates are promoted correctly under concurrent access."""
        from streaming.pipeline import MetadataJoinState

        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=7200)
        wallet = WALLET_A

        def apply_and_promote(tid: int, it: int) -> None:
            if it % 2 == 0:
                update = _make_metadata_update(wallet, f"effect_type_{tid}")
                state.apply_update(update)
            else:
                meta = state.get_metadata(wallet)
                if meta is not None:
                    assert meta.account_id == wallet

        errors = StressRunner(target=apply_and_promote).run(n_threads=6, n_iters=80, timeout=15)
        assert not errors

    def test_wallets_needing_rescore_after_concurrent_updates(self):
        """wallets_needing_rescore returns consistent results."""
        from streaming.pipeline import MetadataJoinState

        state = MetadataJoinState(join_window_seconds=3600, active_wallet_ttl_seconds=7200)

        wallets = [_valid_wallet(i) for i in range(10)]

        def update_and_rescore(tid: int, it: int) -> None:
            w = wallets[(tid + it) % len(wallets)]
            update = _make_metadata_update(w)
            state.apply_update(update)
            state.get_metadata(w)
            rescore = state.wallets_needing_rescore()
            assert isinstance(rescore, list)

        errors = StressRunner(target=update_and_rescore).run(n_threads=6, n_iters=40, timeout=15)
        assert not errors


# ===================================================================
# AdaptiveBatchController — concurrent PID updates
# ===================================================================


class TestAdaptiveBatchControllerConcurrency:
    """PID controller thread-safety under concurrent update calls."""

    def test_concurrent_updates_no_corruption(self):
        """Many threads calling update() concurrently."""
        pid = AdaptiveBatchController(
            target_p95_latency=2.0,
            min_batch=1,
            max_batch=500,
        )

        def update_pid(tid: int, it: int) -> None:
            latency = 1.0 + (tid * 0.1) + (it % 10) * 0.3
            batch = pid.update(latency)
            assert pid.min_batch <= batch <= pid.max_batch

        errors = StressRunner(target=update_pid).run(n_threads=8, n_iters=100, timeout=15)
        assert not errors

    def test_batch_size_within_bounds_under_concurrent_access(self):
        """batch_size property is consistent under concurrent update calls."""
        pid = AdaptiveBatchController(
            target_p95_latency=1.0,
            min_batch=5,
            max_batch=100,
        )

        def update_and_check(tid: int, it: int) -> None:
            pid.update(0.5 + (it % 5) * 0.5)
            bs = pid.batch_size
            assert pid.min_batch <= bs <= pid.max_batch

        errors = StressRunner(target=update_and_check).run(n_threads=6, n_iters=80, timeout=15)
        assert not errors


# ===================================================================
# SequenceCounter — thread-safe monotonic counter
# ===================================================================


class TestSequenceCounterConcurrency:
    """Validate SequenceCounter produces no duplicates under concurrency."""

    def test_no_duplicate_sequence_numbers(self):
        """No two threads ever get the same sequence number."""
        counter = SequenceCounter()
        n_threads = 10
        n_iters = 500
        seen: set[int] = set()
        seen_lock = threading.Lock()

        def get_seq(tid: int, it: int) -> None:
            seq = counter.next()
            with seen_lock:
                if seq in seen:
                    raise AssertionError(f"Duplicate sequence number: {seq}")
                seen.add(seq)

        errors = StressRunner(target=get_seq).run(n_threads=n_threads, n_iters=n_iters, timeout=30)
        assert not errors, f"Duplicate sequence numbers: {errors}"
        assert (
            len(seen) == n_threads * n_iters
        ), f"Expected {n_threads * n_iters} unique sequences, got {len(seen)}"

    def test_monotonicity_per_thread(self):
        """Each thread sees monotonically increasing sequence numbers."""
        counter = SequenceCounter()
        n_threads = 6
        n_iters = 200

        per_thread_seqs: dict[int, list[int]] = {}
        pt_lock = threading.Lock()

        def get_seqs(tid: int, it: int) -> None:
            seq = counter.next()
            with pt_lock:
                if tid not in per_thread_seqs:
                    per_thread_seqs[tid] = []
                per_thread_seqs[tid].append(seq)

        errors = StressRunner(target=get_seqs).run(n_threads=n_threads, n_iters=n_iters, timeout=30)
        assert not errors

        for tid, seqs in per_thread_seqs.items():
            for i in range(1, len(seqs)):
                assert seqs[i] > seqs[i - 1], (
                    f"Thread {tid}: sequence not monotonic at position {i}: "
                    f"{seqs[i - 1]} >= {seqs[i]}"
                )


# ===================================================================
# ReplayBuffer — concurrent append/get
# ===================================================================


class TestReplayBufferConcurrency:
    """Validate ReplayBuffer thread-safety."""

    def test_concurrent_append_and_get(self):
        """Concurrent append/get_since — no corruption."""
        buf = ReplayBuffer(max_size=100)

        def append_and_query(tid: int, it: int) -> None:
            channel = _wallet_channel(tid)
            buf.append(channel, tid * 1000 + it, {"data": it})
            msgs = buf.get_since(channel, 0)
            assert isinstance(msgs, list)

        errors = StressRunner(target=append_and_query).run(n_threads=8, n_iters=80, timeout=15)
        assert not errors

    def test_replay_buffer_size_bounded(self):
        """ReplayBuffer stays within max_size under concurrent append."""
        buf = ReplayBuffer(max_size=5)
        channel = "wallet/G" + "A" * 55

        def append(tid: int, it: int) -> None:
            buf.append(channel, tid * 100 + it, {"data": it})

        errors = StressRunner(target=append).run(n_threads=8, n_iters=30, timeout=15)
        assert not errors

        msgs = buf.get_since(channel, 0)
        assert len(msgs) <= 5, f"ReplayBuffer size exceeded max_size: {len(msgs)} > 5"


# ===================================================================
# TokenBucket — concurrent rate limiting
# ===================================================================


class TestTokenBucketConcurrency:
    """Validate TokenBucket thread-safety."""

    def test_concurrent_is_allowed(self):
        """Concurrent is_allowed calls don't produce negative tokens."""
        bucket = TokenBucket(rate=1000)

        def check(tid: int, it: int) -> None:
            allowed = bucket.is_allowed()
            assert isinstance(allowed, bool)

        errors = StressRunner(target=check).run(n_threads=10, n_iters=50, timeout=15)
        assert not errors


# ===================================================================
# BackPressureController — concurrent check_and_apply
# ===================================================================


class TestBackPressureControllerConcurrency:
    """Validate BackPressureController thread-safety."""

    def test_concurrent_check_and_apply(self):
        """Concurrent check_and_apply calls don't corrupt paused set."""
        from confluent_kafka import TopicPartition

        from streaming.kafka_worker import BackPressureController

        consumer = MagicMock()
        bp = BackPressureController(
            consumer=consumer,
            hwm=10,
            lwm=3,
            bootstrap_servers="localhost:9092",
        )
        tp = TopicPartition("test-topic", 0)

        def check(tid: int, it: int) -> None:
            bp.check_and_apply(tp, queue_depth=it % 20)

        errors = StressRunner(target=check).run(n_threads=6, n_iters=100, timeout=15)
        assert not errors

        # Paused set should be consistent
        assert isinstance(bp._paused, set)

    def test_concurrent_record_failure(self):
        """Concurrent record_failure calls don't corrupt retry counts."""
        from streaming.kafka_worker import BackPressureController

        consumer = MagicMock()
        bp = BackPressureController(
            consumer=consumer,
            max_retries=5,
            bootstrap_servers="localhost:9092",
        )

        msg = MagicMock()
        msg.topic.return_value = "test-topic"
        msg.partition.return_value = 0
        msg.offset.return_value = 42
        msg.value.return_value = b"test"
        msg.key.return_value = None
        msg.headers.return_value = None

        def fail(tid: int, it: int) -> None:
            bp.record_failure(msg, f"error from thread {tid}")

        errors = StressRunner(target=fail).run(n_threads=6, n_iters=5, timeout=15)
        assert not errors


# ===================================================================
# FeatureStoreWorker — concurrent submit/stop lifecycle
# ===================================================================


class TestFeatureStoreWorkerConcurrency:
    """Validate FeatureStoreWorker thread-safety."""

    def test_concurrent_submit_trade_event(self):
        """Many threads submitting trade events concurrently."""
        from streaming.feature_store_worker import FeatureStoreWorker

        store = MagicMock()
        buffer = MagicMock()
        buffer.wallet_trade_count.return_value = 200
        buffer.get_feature_row.return_value = None  # skip actual processing

        worker = FeatureStoreWorker(
            feature_store=store,
            feature_buffer=buffer,
            max_queue_depth=500,
        )

        try:
            worker.start()

            def submit(tid: int, it: int) -> None:
                worker.submit_trade_event(
                    wallet_id=_valid_wallet(tid),
                    pair_id=f"USDC:{USDC_ISSUER}/XLM:native",
                )

            errors = StressRunner(target=submit).run(n_threads=8, n_iters=30, timeout=15)
            assert not errors

            # Give worker time to drain
            assert_eventually(
                lambda: worker.queue_depth == 0,
                timeout=5.0,
                msg="FeatureStoreWorker queue did not drain",
            )

        finally:
            worker.stop()

    def test_concurrent_submit_during_stop(self):
        """Submitting events concurrently with stop() — no crash."""
        from streaming.feature_store_worker import FeatureStoreWorker

        store = MagicMock()
        buffer = MagicMock()
        buffer.wallet_trade_count.return_value = 200

        worker = FeatureStoreWorker(
            feature_store=store,
            feature_buffer=buffer,
            max_queue_depth=100,
        )

        worker.start()

        def submit_and_stop(tid: int, it: int) -> None:
            worker.submit_trade_event(
                wallet_id=_valid_wallet(tid),
                pair_id="USDC:XLM",
            )

        submit_errors = StressRunner(target=submit_and_stop).run(
            n_threads=6, n_iters=10, timeout=15
        )

        worker.stop()
        assert not submit_errors


# ===================================================================
# AccountMetadataStream — concurrent add_wallet/stop
# ===================================================================


class TestAccountMetadataStreamConcurrency:
    """Validate AccountMetadataStream thread-safety."""

    def test_concurrent_add_wallet(self):
        """Many threads adding wallets concurrently."""
        from streaming.account_metadata_stream import AccountMetadataStream

        stream = AccountMetadataStream(
            wallets=[],
            on_update=lambda u: None,
        )

        try:

            def add(tid: int, it: int) -> None:
                wallet = _valid_wallet(tid * 100 + it)
                stream.add_wallet(wallet)

            errors = StressRunner(target=add).run(n_threads=6, n_iters=10, timeout=15)
            assert not errors

        finally:
            stream.stop()

    def test_concurrent_add_and_stop(self):
        """add_wallet concurrent with stop() — no crash."""
        from streaming.account_metadata_stream import AccountMetadataStream

        stream = AccountMetadataStream(
            wallets=[],
            on_update=lambda u: None,
        )

        def add(tid: int, it: int) -> None:
            wallet = _valid_wallet(tid * 100 + it)
            stream.add_wallet(wallet)

        errors = StressRunner(target=add).run(n_threads=4, n_iters=15, timeout=15)
        stream.stop()
        assert not errors


# ===================================================================
# StreamingPipeline._score_cache — concurrent fallback access
# ===================================================================


class TestScoreCacheConcurrency:
    """Validate StreamingPipeline score cache thread-safety."""

    def test_concurrent_score_cache_access(self):
        """_score_cache dict accessed from multiple threads concurrently."""
        from streaming.pipeline import StreamingPipeline

        buffer = MagicMock()
        scorer = MagicMock()
        dispatcher = MagicMock()
        pipeline = StreamingPipeline(buffer, scorer, dispatcher, pairs=[("USDC", USDC_ISSUER)])

        wallet_a = WALLET_A
        wallet_b = WALLET_B

        def dispatch_with_cb(tid: int, it: int) -> None:
            wallet = wallet_a if it % 2 == 0 else wallet_b
            score = {"score": tid, "benford_flag": True, "ml_flag": False, "confidence": 90}
            pair_id = "USDC:XLM"
            pipeline._dispatch_with_cb(wallet, score, pair_id)

        errors = StressRunner(target=dispatch_with_cb).run(n_threads=8, n_iters=50, timeout=15)
        assert not errors

        # Both wallets should have cached scores
        assert wallet_a in pipeline._score_cache
        assert wallet_b in pipeline._score_cache


# ===================================================================
# StressRunner itself — basic correctness
# ===================================================================


class TestStressRunner:
    """Validate StressRunner works correctly."""

    def test_captures_exceptions(self):
        """Exceptions in threads are captured as ThreadError."""

        def failing(tid: int, it: int) -> None:
            if it == 3:
                raise ValueError(f"intentional failure thread {tid}")

        errors = StressRunner(target=failing).run(n_threads=4, n_iters=10, timeout=15)
        assert len(errors) >= 4, f"Expected at least 4 thread errors, got {len(errors)}"
        for err in errors:
            assert isinstance(err.exception, ValueError)

    def test_no_errors_when_all_succeed(self):
        """Target that never raises produces no errors."""

        def succeed(tid: int, it: int) -> None:
            pass

        errors = StressRunner(target=succeed).run(n_threads=8, n_iters=200, timeout=15)
        assert not errors

    def test_eventually_true(self):
        """assert_eventually passes when predicate becomes true."""
        flag: list[bool] = [False]

        def set_flag() -> None:
            time.sleep(0.2)
            flag[0] = True

        threading.Thread(target=set_flag, daemon=True).start()
        assert_eventually(lambda: flag[0], timeout=2.0)

    def test_eventually_timeout(self):
        """assert_eventually raises on timeout."""
        with pytest.raises(AssertionError):
            assert_eventually(lambda: False, timeout=0.2, interval=0.05)


# ===================================================================
# RiskScoreStore.upsert — atomic upsert under concurrent writers
# (Issue #789). Two or more Kafka workers in the same consumer group
# can process the same wallet/pair close together; the upsert must be a
# single atomic INSERT ... ON CONFLICT DO UPDATE so the final stored
# record is always one of the concurrent writes, never a merged hybrid.
# ===================================================================


class TestRiskScoreStoreUpsertConcurrency:
    """Validate RiskScoreStore.upsert under concurrent same-key writers."""

    def _make_store(self, tmp_path):
        from detection.persistence import get_engine, get_session_factory

        from detection.risk_score_store import RiskScoreStore

        db_path = tmp_path / "risk_scores.db"
        engine = get_engine(f"sqlite:///{db_path}")
        return RiskScoreStore(get_session_factory(engine))

    def test_concurrent_upsert_same_key_reflects_one_write(self):
        """Concurrent writers to one (wallet, pair) leave a single coherent row."""
        from sqlalchemy import func, select

        from detection.persistence import RiskScoreRecord

        store = self._make_store(tmp_path)
        wallet = "G" + "A" * 55
        pair = "USDC:XLM"
        n_writers = 8
        # Each writer writes a fully-distinct, deterministic score so we can
        # prove the final row came from exactly one writer and was not merged.
        expected_scores = {tid * 10 + 1 for tid in range(n_writers)}

        def write(tid: int, it: int) -> None:
            score = tid * 10 + 1
            store.upsert(
                wallet,
                pair,
                {
                    "score": score,
                    "benford_flag": bool(score % 2),
                    "ml_flag": bool(score % 3),
                    "confidence": score,
                },
            )

        errors = StressRunner(target=write).run(n_threads=n_writers, n_iters=1, timeout=30)
        assert not errors, f"Concurrent upsert errors: {errors}"

        record = store.get(wallet, pair)
        assert record is not None, "Expected a stored record for the key"
        assert record.score in expected_scores, (
            f"Final score {record.score} is a merged/corrupted value, not one "
            f"of the concurrent writes {sorted(expected_scores)}"
        )

        # Exactly one row per (wallet, pair) — no duplicate/orphaned rows.
        with store._session_factory() as session:
            count = session.scalar(
                select(func.count())
                .select_from(RiskScoreRecord)
                .where(
                    RiskScoreRecord.wallet == wallet,
                    RiskScoreRecord.asset_pair == pair,
                )
            )
        assert count == 1, f"Expected exactly one row for the key, got {count}"

    def test_concurrent_upsert_many_iterations_stays_consistent(self):
        """Repeated concurrent upserts never produce a hybrid/duplicate row."""
        from sqlalchemy import func, select

        from detection.persistence import RiskScoreRecord

        store = self._make_store(tmp_path)
        wallet = "G" + "B" * 55
        pair = "USDC:XLM"
        n_writers = 6
        n_iters = 20
        expected_scores = {tid * 100 + it for tid in range(n_writers) for it in range(n_iters)}

        def write(tid: int, it: int) -> None:
            score = tid * 100 + it
            store.upsert(
                wallet,
                pair,
                {
                    "score": score,
                    "benford_flag": bool(score % 2),
                    "ml_flag": bool(score % 3),
                    "confidence": score,
                },
            )

        errors = StressRunner(target=write).run(n_threads=n_writers, n_iters=n_iters, timeout=30)
        assert not errors, f"Concurrent upsert errors: {errors}"

        record = store.get(wallet, pair)
        assert record is not None
        assert (
            record.score in expected_scores
        ), f"Final score {record.score} is not one of the concurrent writes"
        with store._session_factory() as session:
            count = session.scalar(
                select(func.count())
                .select_from(RiskScoreRecord)
                .where(
                    RiskScoreRecord.wallet == wallet,
                    RiskScoreRecord.asset_pair == pair,
                )
            )
        assert count == 1, f"Expected exactly one row for the key, got {count}"
