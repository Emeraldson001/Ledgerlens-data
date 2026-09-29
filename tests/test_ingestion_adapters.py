"""Tests for ingestion.adapters (normalized blockchain event contract)."""

from datetime import UTC, datetime

import pytest

from ingestion.adapters.base import (
    AdapterValidationError,
    ChainAdapter,
    EventType,
    NormalizedAsset,
    NormalizedEvent,
)
from ingestion.adapters.evm_adapter import EvmAdapter
from ingestion.adapters.registry import (
    AdapterNotRegisteredError,
    AdapterRegistry,
)
from ingestion.adapters.stellar_adapter import StellarAdapter
from ingestion.data_models import AccountActivity, Asset, OrderBookEvent, Trade

WALLET_A = "GAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAWHF"
WALLET_B = "GBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBWHF"


def _trade() -> Trade:
    return Trade(
        trade_id="t1",
        ledger_close_time=datetime(2026, 1, 1, tzinfo=UTC),
        base_account=WALLET_A,
        counter_account=WALLET_B,
        base_asset=Asset(code="XLM", issuer=None),
        counter_asset=Asset(code="USDC", issuer="GISSUER"),
        base_amount=100.0,
        counter_amount=12.5,
        price=0.125,
    )


# ---------------------------------------------------------------------------
# NormalizedEvent contract
# ---------------------------------------------------------------------------


def test_normalized_asset_id_for_native():
    asset = NormalizedAsset(symbol="XLM", native=True)
    assert asset.asset_id() == "native:XLM"


def test_normalized_asset_id_for_issued():
    asset = NormalizedAsset(symbol="USDC", issuer="GISSUER")
    assert asset.asset_id() == "USDC:GISSUER"


def test_normalized_event_chain_lowercased():
    event = NormalizedEvent(
        event_id="e1",
        chain="STELLAR",
        event_type=EventType.TRADE,
        occurred_at=datetime.now(UTC),
        account=WALLET_A,
        asset=NormalizedAsset(symbol="XLM", native=True),
        amount=1.0,
    )
    assert event.chain == "stellar"


def test_normalized_event_rejects_negative_amount():
    with pytest.raises(ValueError):
        NormalizedEvent(
            event_id="e1",
            chain="stellar",
            event_type=EventType.TRADE,
            occurred_at=datetime.now(UTC),
            account=WALLET_A,
            asset=NormalizedAsset(symbol="XLM", native=True),
            amount=-1.0,
        )


def test_dedup_key_combines_chain_and_event_id():
    event = NormalizedEvent(
        event_id="e1",
        chain="stellar",
        event_type=EventType.TRADE,
        occurred_at=datetime.now(UTC),
        account=WALLET_A,
        asset=NormalizedAsset(symbol="XLM", native=True),
        amount=1.0,
    )
    assert event.dedup_key() == "stellar:e1"


# ---------------------------------------------------------------------------
# StellarAdapter
# ---------------------------------------------------------------------------


def test_stellar_adapter_normalizes_trade():
    adapter = StellarAdapter()
    event = adapter.normalize(_trade())

    assert event.chain == "stellar"
    assert event.event_type == EventType.TRADE
    assert event.account == WALLET_A
    assert event.counterparty == WALLET_B
    assert event.amount == 100.0
    assert event.counter_amount == 12.5
    assert event.asset.native is True
    assert event.counter_asset.issuer == "GISSUER"


def test_stellar_adapter_normalizes_order_book_event():
    adapter = StellarAdapter()
    raw = OrderBookEvent(
        event_id="ob1",
        account=WALLET_A,
        ledger_close_time=datetime(2026, 1, 1, tzinfo=UTC),
        selling=Asset(code="XLM", issuer=None),
        buying=Asset(code="USDC", issuer="GISSUER"),
        amount=50.0,
        price=0.1,
        action="created",
    )
    event = adapter.normalize(raw)
    assert event.event_type == EventType.ORDER_PLACED
    assert event.amount == 50.0


def test_stellar_adapter_normalizes_account_activity():
    adapter = StellarAdapter()
    raw = AccountActivity(
        account_id=WALLET_A,
        account_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        funding_account=WALLET_B,
    )
    event = adapter.normalize(raw)
    assert event.event_type == EventType.ACCOUNT_CREATED
    assert event.account == WALLET_A
    assert event.counterparty == WALLET_B


def test_stellar_adapter_rejects_unsupported_type():
    adapter = StellarAdapter()
    with pytest.raises(AdapterValidationError):
        adapter.normalize({"not": "a stellar model"})


def test_stellar_adapter_unknown_order_action_raises():
    adapter = StellarAdapter()
    raw = OrderBookEvent(
        event_id="ob1",
        account=WALLET_A,
        ledger_close_time=datetime(2026, 1, 1, tzinfo=UTC),
        selling=Asset(code="XLM", issuer=None),
        buying=Asset(code="USDC", issuer="GISSUER"),
        amount=50.0,
        price=0.1,
        action="unheard_of_action",
    )
    with pytest.raises(AdapterValidationError):
        adapter.normalize(raw)


# ---------------------------------------------------------------------------
# EvmAdapter
# ---------------------------------------------------------------------------


def _evm_transfer(**overrides) -> dict:
    base = {
        "tx_hash": "0xabc",
        "log_index": 3,
        "from": "0x1111",
        "to": "0x2222",
        "token_address": "0xa0b8",
        "token_symbol": "USDC",
        "value": 1_000_000,
        "decimals": 6,
        "block_time": 1732900000,
    }
    base.update(overrides)
    return base


def test_evm_adapter_normalizes_transfer():
    adapter = EvmAdapter()
    event = adapter.normalize(_evm_transfer())

    assert event.chain == "ethereum"
    assert event.event_type == EventType.TRANSFER
    assert event.account == "0x1111"
    assert event.counterparty == "0x2222"
    assert event.amount == pytest.approx(1.0)
    assert event.asset.symbol == "USDC"


def test_evm_adapter_defaults_decimals_to_18():
    adapter = EvmAdapter()
    raw = _evm_transfer(value=10**18)
    del raw["decimals"]
    event = adapter.normalize(raw)
    assert event.amount == pytest.approx(1.0)


def test_evm_adapter_missing_field_raises():
    adapter = EvmAdapter()
    raw = _evm_transfer()
    del raw["tx_hash"]
    with pytest.raises(AdapterValidationError):
        adapter.normalize(raw)


def test_evm_adapter_rejects_non_dict():
    adapter = EvmAdapter()
    with pytest.raises(AdapterValidationError):
        adapter.normalize("not a dict")


# ---------------------------------------------------------------------------
# normalize_batch — partial failure handling
# ---------------------------------------------------------------------------


def test_normalize_batch_collects_errors_without_aborting():
    adapter = StellarAdapter()
    good = _trade()
    bad = {"not": "valid"}

    normalized, errors = adapter.normalize_batch([good, bad, good])

    assert len(normalized) == 2
    assert len(errors) == 1
    assert isinstance(errors[0], AdapterValidationError)


# ---------------------------------------------------------------------------
# AdapterRegistry
# ---------------------------------------------------------------------------


class _FakeAdapter(ChainAdapter):
    chain = "fakechain"

    def normalize(self, raw_event):
        return NormalizedEvent(
            event_id="fake1",
            chain=self.chain,
            event_type=EventType.TRANSFER,
            occurred_at=datetime.now(UTC),
            account="acct",
            asset=NormalizedAsset(symbol="FAKE", native=True),
            amount=1.0,
        )


def test_registry_register_and_get():
    registry = AdapterRegistry()
    adapter = _FakeAdapter()
    registry.register(adapter)
    assert registry.get("fakechain") is adapter
    assert registry.get("FAKECHAIN") is adapter  # case-insensitive


def test_registry_missing_chain_raises():
    registry = AdapterRegistry()
    with pytest.raises(AdapterNotRegisteredError):
        registry.get("nonexistent")


def test_registry_normalize_convenience_method():
    registry = AdapterRegistry()
    registry.register(_FakeAdapter())
    event = registry.normalize("fakechain", raw_event=None)
    assert event.chain == "fakechain"


def test_registry_chains_lists_registered():
    registry = AdapterRegistry()
    registry.register(_FakeAdapter())
    registry.register(StellarAdapter())
    assert registry.chains() == ["fakechain", "stellar"]


def test_default_registry_has_stellar_and_ethereum():
    from ingestion.adapters.registry import default_registry

    assert "stellar" in default_registry
    assert "ethereum" in default_registry
