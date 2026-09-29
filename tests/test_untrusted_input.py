"""Unit tests for the untrusted-ledger-input validation contract.

`ingestion/untrusted_input.py` is the shared boundary every Horizon/AMM
loader routes untrusted external records through before they become
`Trade` / `OrderBookEvent` / `AccountActivity` objects the rest of the
pipeline trusts. These tests exercise the validators directly; loader-level
"one bad record doesn't crash the page" behavior is covered in
`test_orderbook.py`, `test_amm_loader.py`, and `test_per_pair_pipeline.py`.
"""

import math
from datetime import UTC, datetime, timedelta

import pytest

from ingestion.data_models import AccountActivity, Asset, OrderBookEvent, Trade
from ingestion.untrusted_input import (
    MAX_STRING_FIELD_LENGTH,
    UntrustedInputError,
    safe_ratio,
    validate_account_activity,
    validate_orderbook_event,
    validate_trade,
)

VALID_BASE = "GCGPQMCLRXCUPCL3AVMYUUQML2WVC7A5M6HO5RKYSU4CIA7O7SI4VKWE"
VALID_COUNTER = "GB2HHLFDCBSBDAMU2QRDU4AJV63WQE2DWT7MBZWZRQDFYUXJXIPPUG7M"
VALID_ISSUER = "GA5ZSEJYB37JRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


def _trade(**overrides) -> Trade:
    fields = {
        "trade_id": "t1",
        "ledger_close_time": datetime(2024, 1, 1, tzinfo=UTC),
        "base_account": VALID_BASE,
        "counter_account": VALID_COUNTER,
        "base_asset": Asset(code="USDC", issuer=VALID_ISSUER),
        "counter_asset": Asset(code="XLM"),
        "base_amount": 100.0,
        "counter_amount": 50.0,
        "price": 2.0,
    }
    fields.update(overrides)
    return Trade(**fields)


# ---------------------------------------------------------------------------
# safe_ratio
# ---------------------------------------------------------------------------


def test_safe_ratio_normal_division():
    assert safe_ratio(6, 3) == 2.0


def test_safe_ratio_zero_denominator_returns_default():
    assert safe_ratio(1, 0) == 0.0
    assert safe_ratio(1, 0, default=-1.0) == -1.0


def test_safe_ratio_non_numeric_returns_default():
    assert safe_ratio("nan-ish", "oops") == 0.0
    assert safe_ratio(None, 5) == 0.0


def test_safe_ratio_rejects_nan_and_inf_results():
    assert safe_ratio(float("nan"), 1) == 0.0
    assert safe_ratio(float("inf"), 1) == 0.0


# ---------------------------------------------------------------------------
# validate_trade
# ---------------------------------------------------------------------------


def test_validate_trade_accepts_well_formed_trade():
    trade = _trade()
    assert validate_trade(trade, source="test") is trade


def test_validate_trade_rejects_malformed_account_id():
    trade = _trade(base_account="NOT-A-REAL-ACCOUNT")
    with pytest.raises(UntrustedInputError) as exc_info:
        validate_trade(trade, source="test")
    assert exc_info.value.field == "base_account"
    assert exc_info.value.source == "test"


def test_validate_trade_rejects_negative_amount():
    trade = _trade(base_amount=-5.0)
    with pytest.raises(UntrustedInputError, match="base_amount"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_zero_amount():
    trade = _trade(base_amount=0.0)
    with pytest.raises(UntrustedInputError, match="base_amount"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_nan_amount():
    trade = _trade(counter_amount=math.nan)
    with pytest.raises(UntrustedInputError, match="counter_amount"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_inf_price():
    trade = _trade(price=math.inf)
    with pytest.raises(UntrustedInputError, match="price"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_oversized_asset_code():
    trade = _trade(base_asset=Asset(code="X" * 13))
    with pytest.raises(UntrustedInputError, match="base_asset.code"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_asset_code_with_bad_characters():
    trade = _trade(counter_asset=Asset(code="US$D"))
    with pytest.raises(UntrustedInputError, match="counter_asset.code"):
        validate_trade(trade, source="test")


def test_validate_trade_accepts_xlm_native_code():
    trade = _trade(counter_asset=Asset(code="XLM"))
    validate_trade(trade, source="test")


def test_validate_trade_rejects_timestamp_before_stellar_genesis():
    trade = _trade(ledger_close_time=datetime(2010, 1, 1, tzinfo=UTC))
    with pytest.raises(UntrustedInputError, match="ledger_close_time"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_timestamp_far_in_future():
    future = datetime.now(UTC) + timedelta(days=1)
    trade = _trade(ledger_close_time=future)
    with pytest.raises(UntrustedInputError, match="ledger_close_time"):
        validate_trade(trade, source="test")


def test_validate_trade_rejects_oversized_trade_id():
    trade = _trade(trade_id="x" * (MAX_STRING_FIELD_LENGTH + 1))
    with pytest.raises(UntrustedInputError, match="trade_id"):
        validate_trade(trade, source="test")


# ---------------------------------------------------------------------------
# validate_orderbook_event
# ---------------------------------------------------------------------------


def _orderbook_event(**overrides) -> OrderBookEvent:
    fields = {
        "event_id": "e1",
        "account": VALID_BASE,
        "ledger_close_time": datetime(2024, 1, 1, tzinfo=UTC),
        "selling": Asset(code="USDC", issuer=VALID_ISSUER),
        "buying": Asset(code="XLM"),
        "amount": 10.0,
        "price": 1.5,
        "action": "created",
    }
    fields.update(overrides)
    return OrderBookEvent(**fields)


def test_validate_orderbook_event_accepts_well_formed_event():
    event = _orderbook_event()
    assert validate_orderbook_event(event, source="test") is event


def test_validate_orderbook_event_allows_zero_amount():
    """Cancellation events legitimately carry amount == 0."""
    event = _orderbook_event(amount=0.0, action="cancelled")
    validate_orderbook_event(event, source="test")


def test_validate_orderbook_event_rejects_bad_account():
    event = _orderbook_event(account="short")
    with pytest.raises(UntrustedInputError, match="account"):
        validate_orderbook_event(event, source="test")


def test_validate_orderbook_event_rejects_unknown_action():
    event = _orderbook_event(action="deleted")
    with pytest.raises(UntrustedInputError, match="action"):
        validate_orderbook_event(event, source="test")


# ---------------------------------------------------------------------------
# validate_account_activity
# ---------------------------------------------------------------------------


def test_validate_account_activity_accepts_well_formed_activity():
    activity = AccountActivity(
        account_id=VALID_BASE,
        account_created_at=datetime(2024, 1, 1, tzinfo=UTC),
        funding_account=VALID_COUNTER,
    )
    assert validate_account_activity(activity, source="test") is activity


def test_validate_account_activity_allows_none_funding_account():
    activity = AccountActivity(
        account_id=VALID_BASE,
        account_created_at=datetime(2024, 1, 1, tzinfo=UTC),
        funding_account=None,
    )
    validate_account_activity(activity, source="test")


def test_validate_account_activity_rejects_bad_funding_account():
    activity = AccountActivity(
        account_id=VALID_BASE,
        account_created_at=datetime(2024, 1, 1, tzinfo=UTC),
        funding_account="bogus",
    )
    with pytest.raises(UntrustedInputError, match="funding_account"):
        validate_account_activity(activity, source="test")


def test_untrusted_input_error_message_includes_source():
    trade = _trade(base_account="bogus")
    with pytest.raises(UntrustedInputError) as exc_info:
        validate_trade(trade, source="my_source")
    assert "my_source" in str(exc_info.value)
