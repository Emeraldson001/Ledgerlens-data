"""Tests for streaming.feature_buffer.FeatureBuffer (Issue #12)."""

import datetime
import threading

import pandas as pd

from ingestion.data_models import Asset, Trade
from streaming.feature_buffer import FeatureBuffer

USDC_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"


def _make_trade(
    base_account: str = WALLET_A,
    counter_account: str = WALLET_B,
    base_amount: float = 100.0,
    trade_id: str = "t1",
) -> Trade:
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


# ---------------------------------------------------------------------------
# 1. update() adds trade to both accounts
# ---------------------------------------------------------------------------


def test_update_adds_to_both_accounts():
    buf = FeatureBuffer()
    buf.update(_make_trade())
    wallets = buf.all_wallets()
    assert WALLET_A in wallets
    assert WALLET_B in wallets


# ---------------------------------------------------------------------------
# 2. Evicts oldest trade when capacity is reached
# ---------------------------------------------------------------------------


def test_evicts_oldest_trade_at_capacity():
    max_trades = 5
    buf = FeatureBuffer(max_trades=max_trades)

    # Add max_trades + 1 trades; each trade has a unique amount we can track
    for i in range(max_trades + 1):
        trade = _make_trade(
            base_account=WALLET_A,
            counter_account=WALLET_B,
            base_amount=float(i + 1),
            trade_id=f"t{i}",
        )
        buf.update(trade)

    # Count should be capped at max_trades
    assert buf.wallet_trade_count(WALLET_A) == max_trades

    # The oldest trade (amount=1.0) should have been evicted; the second
    # one added (amount=2.0) is now the oldest in the deque.
    with buf._wallet_locks[WALLET_A]:
        amounts = [r["amount"] for r in buf._buffers[WALLET_A]]
    assert 1.0 not in amounts, "Oldest trade was not evicted"
    assert amounts[0] == 2.0, "Second trade should now be the oldest entry"


# ---------------------------------------------------------------------------
# 3. get_feature_row returns None for unknown wallet
# ---------------------------------------------------------------------------


def test_get_feature_row_returns_none_for_unknown_wallet():
    buf = FeatureBuffer()
    result = buf.get_feature_row("GUNKNOWN_WALLET_XYZ")
    assert result is None


# ---------------------------------------------------------------------------
# 4. get_feature_row returns pd.Series with expected feature columns
# ---------------------------------------------------------------------------


def test_get_feature_row_returns_series_with_expected_columns():
    buf = FeatureBuffer()
    # Add enough trades for Benford windows to be populated
    for i in range(25):
        buf.update(
            _make_trade(
                base_account=WALLET_A,
                counter_account=WALLET_B,
                base_amount=float(i + 1) * 10.5,
                trade_id=f"t{i}",
            )
        )

    result = buf.get_feature_row(WALLET_A)

    assert isinstance(result, pd.Series)
    assert "benford_chi_square_1h" in result.index
    assert "counterparty_concentration_ratio" in result.index


# ---------------------------------------------------------------------------
# 5. Concurrent writes to the same wallet cause no corruption
# ---------------------------------------------------------------------------


def test_concurrent_writes_no_corruption():
    max_trades = 500
    buf = FeatureBuffer(max_trades=max_trades)
    n_threads = 10
    trades_per_thread = 100
    errors: list[Exception] = []

    def writer(thread_idx: int) -> None:
        try:
            for i in range(trades_per_thread):
                buf.update(
                    _make_trade(
                        base_account=WALLET_A,
                        counter_account=WALLET_B,
                        base_amount=float(thread_idx * 1000 + i),
                        trade_id=f"t{thread_idx}-{i}",
                    )
                )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(idx,)) for idx in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Exceptions raised in writer threads: {errors}"
    # Buffer is capped; count must be ≤ max_trades
    assert buf.wallet_trade_count(WALLET_A) <= max_trades


# ---------------------------------------------------------------------------
# 6. Redelivery of the same trade_id is idempotent (Issue #670, invariant 1)
# ---------------------------------------------------------------------------


def test_redelivered_trade_id_does_not_double_count():
    """A trade_id already applied to a wallet must be a no-op on replay/redelivery."""
    buf = FeatureBuffer()
    trade = _make_trade(trade_id="dup-1")

    buf.update(trade)
    buf.update(trade)  # redelivery of the identical message
    buf.update(trade)  # a second, later redelivery

    assert buf.wallet_trade_count(WALLET_A) == 1
    assert buf.wallet_trade_count(WALLET_B) == 1


def test_redelivered_trade_id_does_not_double_count_benford_sketch():
    """Duplicate application must not skew Benford digit counts either."""
    buf = FeatureBuffer()
    trade = _make_trade(trade_id="dup-2", base_amount=123.0)

    buf.update(trade)
    metrics_once = buf._benford_sketches[WALLET_A][
        next(iter(buf._benford_sketches[WALLET_A]))
    ].to_metrics()

    buf.update(trade)
    buf.update(trade)
    metrics_after_replay = buf._benford_sketches[WALLET_A][
        next(iter(buf._benford_sketches[WALLET_A]))
    ].to_metrics()

    assert metrics_once.sample_size == metrics_after_replay.sample_size


def test_different_trade_ids_still_both_recorded():
    buf = FeatureBuffer()
    buf.update(_make_trade(trade_id="a"))
    buf.update(_make_trade(trade_id="b"))
    assert buf.wallet_trade_count(WALLET_A) == 2


def test_seen_trade_id_index_evicted_in_lockstep_with_deque():
    """Once a trade falls out of the capped deque, its trade_id becomes
    reusable — replaying an old trade_id after it has scrolled out of the
    window must be recorded again (it's outside the current retention
    window, not a live duplicate)."""
    max_trades = 3
    buf = FeatureBuffer(max_trades=max_trades)

    for i in range(max_trades):
        buf.update(_make_trade(base_amount=float(i), trade_id=f"t{i}"))
    assert buf.wallet_trade_count(WALLET_A) == max_trades

    # Push t0 out of the window.
    buf.update(_make_trade(base_amount=99.0, trade_id="t-new"))
    assert buf.wallet_trade_count(WALLET_A) == max_trades
    with buf._wallet_locks[WALLET_A]:
        assert "t0" not in buf._seen_trade_ids[WALLET_A]

    # t0 is no longer tracked as "seen" — replaying it re-adds it (evicting t1).
    buf.update(_make_trade(base_amount=0.0, trade_id="t0"))
    with buf._wallet_locks[WALLET_A]:
        amounts = [r["amount"] for r in buf._buffers[WALLET_A]]
    assert 0.0 in amounts
