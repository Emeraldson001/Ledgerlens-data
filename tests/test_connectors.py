"""Tests for the data connector plugin boundary (ingestion/connectors/).

Covers:
  - the registry's success path (register/get/create/list)
  - failure/diagnostic paths (bad subclass, missing metadata, duplicate id,
    unknown id, missing required config)
  - that a third-party-style plugin can be added without touching any
    in-tree ingestion module (the actual "plugin-ready boundary" claim)
  - out-of-tree discovery via the `ledgerlens.connectors` entry point group
  - that all four existing Horizon loaders are reachable as connectors and
    that the adapters enforce their own required kwargs
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel

import ingestion.connectors as connectors_pkg
from ingestion.connectors.base import (
    ConnectorConfigError,
    ConnectorError,
    ConnectorMetadata,
    ConnectorNotFoundError,
    DataConnector,
    DuplicateConnectorError,
)
from ingestion.connectors.registry import PLUGIN_ENTRY_POINT_GROUP, ConnectorRegistry
from ingestion.data_models import AccountActivity, Asset, OrderBookEvent, Trade

registry = connectors_pkg.registry


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _Widget(BaseModel):
    widget_id: str
    value: float


def _make_connector_cls(connector_id: str, *, required_env=(), record_type=_Widget):
    """Build a minimal, fully-formed DataConnector subclass for tests."""

    class _Connector(DataConnector[record_type]):
        metadata = ConnectorMetadata(
            connector_id=connector_id,
            record_type=record_type,
            source="test-fixture",
            description="Test-only connector",
            required_env=required_env,
        )

        def load(self, **kwargs):
            yield record_type(widget_id="w1", value=1.0) if record_type is _Widget else None

    _Connector.__name__ = f"Connector_{connector_id.replace('-', '_')}"
    return _Connector


@pytest.fixture
def clean_registry():
    """A fresh, empty registry — isolated from the global one used by builtins."""
    return ConnectorRegistry()


@pytest.fixture
def registered_id():
    """Register a throwaway connector on the *global* registry and clean it up.

    Used by tests that need to exercise `ingestion.connectors.registry`
    (the real singleton) rather than a scratch instance, e.g. to check
    interaction with already-registered builtins.
    """
    ids: list[str] = []

    def _register(connector_id: str, **kwargs):
        cls = _make_connector_cls(connector_id, **kwargs)
        registry.register(cls)
        ids.append(connector_id)
        return cls

    yield _register

    for connector_id in ids:
        registry.unregister(connector_id)


# ---------------------------------------------------------------------------
# Registration — success path
# ---------------------------------------------------------------------------


def test_register_and_get_roundtrip(clean_registry):
    cls = _make_connector_cls("widget-source")
    clean_registry.register(cls)

    assert clean_registry.get("widget-source") is cls
    ids = [m.connector_id for m in clean_registry.list_metadata()]
    assert ids == ["widget-source"]


def test_register_is_decorator_friendly(clean_registry):
    from ingestion.connectors.registry import register_connector

    # register_connector always targets the *global* registry by design;
    # verify it returns the class unchanged so it composes as a decorator.
    cls = _make_connector_cls("decorator-check")
    returned = register_connector(cls)
    assert returned is cls
    registry.unregister("decorator-check")


def test_create_instantiates_and_validates(clean_registry):
    cls = _make_connector_cls("widget-source")
    clean_registry.register(cls)

    instance = clean_registry.create("widget-source")
    assert isinstance(instance, cls)

    records = list(instance.load())
    assert records == [_Widget(widget_id="w1", value=1.0)]


def test_reregistering_same_class_is_idempotent(clean_registry):
    cls = _make_connector_cls("widget-source")
    clean_registry.register(cls)
    clean_registry.register(cls)  # same class, no id collision -> no raise
    assert clean_registry.get("widget-source") is cls


# ---------------------------------------------------------------------------
# Registration — failure / diagnostics
# ---------------------------------------------------------------------------


def test_register_rejects_non_connector_class(clean_registry):
    class NotAConnector:
        pass

    with pytest.raises(TypeError, match="must be a subclass of DataConnector"):
        clean_registry.register(NotAConnector)


def test_register_rejects_abstract_subclass(clean_registry):
    class IncompleteConnector(DataConnector[_Widget]):
        metadata = ConnectorMetadata(connector_id="incomplete", record_type=_Widget, source="test")
        # load() intentionally not implemented

    with pytest.raises(TypeError, match="unimplemented abstract methods"):
        clean_registry.register(IncompleteConnector)


def test_register_rejects_missing_metadata(clean_registry):
    class NoMetadataConnector(DataConnector[_Widget]):
        def load(self, **kwargs):
            yield _Widget(widget_id="w1", value=1.0)

    with pytest.raises(TypeError, match="must define a class-level"):
        clean_registry.register(NoMetadataConnector)


def test_duplicate_connector_id_raises(clean_registry):
    cls_a = _make_connector_cls("shared-id")
    cls_b = _make_connector_cls("shared-id")
    clean_registry.register(cls_a)

    with pytest.raises(DuplicateConnectorError, match="shared-id"):
        clean_registry.register(cls_b)

    # replace=True is the explicit escape hatch
    clean_registry.register(cls_b, replace=True)
    assert clean_registry.get("shared-id") is cls_b


def test_get_unknown_connector_lists_available(clean_registry):
    clean_registry.register(_make_connector_cls("known-one"))

    with pytest.raises(ConnectorNotFoundError) as exc_info:
        clean_registry.get("does-not-exist")

    message = str(exc_info.value)
    assert "does-not-exist" in message
    assert "known-one" in message


def test_get_unknown_connector_on_empty_registry_says_none(clean_registry):
    with pytest.raises(ConnectorNotFoundError, match=r"\(none registered\)"):
        clean_registry.get("anything")


def test_missing_required_config_raises_actionable_error(clean_registry, monkeypatch):
    monkeypatch.delenv("FAKE_API_TOKEN", raising=False)
    cls = _make_connector_cls("needs-token", required_env=("FAKE_API_TOKEN",))
    clean_registry.register(cls)

    with pytest.raises(ConnectorConfigError, match="FAKE_API_TOKEN"):
        clean_registry.create("needs-token")


def test_required_config_present_allows_create(clean_registry, monkeypatch):
    monkeypatch.setenv("FAKE_API_TOKEN", "secret")
    cls = _make_connector_cls("needs-token", required_env=("FAKE_API_TOKEN",))
    clean_registry.register(cls)

    instance = clean_registry.create("needs-token")
    assert instance.health_check().ok is True


def test_health_check_never_raises_on_bad_config(clean_registry, monkeypatch):
    monkeypatch.delenv("FAKE_API_TOKEN", raising=False)
    cls = _make_connector_cls("needs-token", required_env=("FAKE_API_TOKEN",))
    clean_registry.register(cls)

    instance = cls()
    health = instance.health_check()
    assert health.ok is False
    assert "FAKE_API_TOKEN" in health.detail


# ---------------------------------------------------------------------------
# Plugin extensibility — the core "plugin-ready boundary" claim
# ---------------------------------------------------------------------------


def test_third_party_style_plugin_needs_no_ingestion_source_change(registered_id):
    """Simulate an out-of-tree contributor adding a brand-new source.

    The point of this test: nothing in `ingestion/` besides importing the
    connector package was touched to add "widget-source" below — it's
    defined entirely in this test module, yet becomes reachable through the
    same `registry.create(...)` / `.load()` / `.to_dataframe()` path as the
    built-in Horizon connectors.
    """
    registered_id("widget-source")

    connector = registry.create("widget-source")
    records = list(connector.load())
    assert records == [_Widget(widget_id="w1", value=1.0)]

    df = connector.to_dataframe(records)
    assert list(df.columns) == ["widget_id", "value"]
    assert df.iloc[0]["widget_id"] == "w1"


def test_default_to_dataframe_uses_model_dump(clean_registry):
    cls = _make_connector_cls("widget-source")
    clean_registry.register(cls)
    instance = clean_registry.create("widget-source")

    df = instance.to_dataframe(
        [_Widget(widget_id="a", value=2.0), _Widget(widget_id="b", value=3.0)]
    )
    assert isinstance(df, pd.DataFrame)
    assert df["widget_id"].tolist() == ["a", "b"]


# ---------------------------------------------------------------------------
# Out-of-tree discovery via entry points
# ---------------------------------------------------------------------------


def _fake_entry_point(name: str, connector_cls):
    ep = MagicMock()
    ep.name = name
    ep.value = f"fake.module:{connector_cls.__name__}"
    ep.load.return_value = connector_cls
    return ep


def test_discover_plugins_registers_entry_point_connectors(clean_registry):
    plugin_cls = _make_connector_cls("entry-point-plugin")
    fake_eps = [_fake_entry_point("entry-point-plugin", plugin_cls)]

    with patch("ingestion.connectors.registry.importlib_metadata.entry_points") as mock_eps:
        mock_eps.return_value = fake_eps
        loaded = clean_registry.discover_plugins()

    mock_eps.assert_called_once_with(group=PLUGIN_ENTRY_POINT_GROUP)
    assert loaded == ["entry-point-plugin"]
    assert clean_registry.get("entry-point-plugin") is plugin_cls


def test_discover_plugins_skips_broken_entry_point_without_raising(clean_registry):
    good_cls = _make_connector_cls("good-plugin")
    broken_ep = MagicMock()
    broken_ep.name = "broken-plugin"
    broken_ep.load.side_effect = ImportError("no such module")

    with patch("ingestion.connectors.registry.importlib_metadata.entry_points") as mock_eps:
        mock_eps.return_value = [broken_ep, _fake_entry_point("good-plugin", good_cls)]
        loaded = clean_registry.discover_plugins()

    assert loaded == ["good-plugin"]
    assert clean_registry.get("good-plugin") is good_cls
    with pytest.raises(ConnectorNotFoundError):
        clean_registry.get("broken-plugin")


def test_get_triggers_lazy_plugin_discovery_once(clean_registry):
    plugin_cls = _make_connector_cls("lazy-plugin")
    fake_eps = [_fake_entry_point("lazy-plugin", plugin_cls)]

    with patch("ingestion.connectors.registry.importlib_metadata.entry_points") as mock_eps:
        mock_eps.return_value = fake_eps

        clean_registry.get("lazy-plugin")  # triggers discovery
        with pytest.raises(ConnectorNotFoundError):
            clean_registry.get("still-not-there")  # second call must not re-discover

    assert mock_eps.call_count == 1


# ---------------------------------------------------------------------------
# Built-in connectors are registered and adapt the existing loaders
# ---------------------------------------------------------------------------


def test_all_builtin_connectors_are_registered():
    ids = {m.connector_id for m in registry.list_metadata()}
    assert {
        "stellar-sdex-trades",
        "stellar-amm-pool-trades",
        "stellar-orderbook-events",
        "stellar-account-activity",
    }.issubset(ids)


def test_sdex_connector_iterates_configured_pairs(monkeypatch):
    import config as config_module
    from ingestion.connectors.builtin import SdexTradeConnector

    issuer = "GA5XMC56L2KINLCAYMRPXVPBDWQMX2WWRUEJZNW77WKBTRTJRLPGHZ6I"
    monkeypatch.setattr(
        config_module.config,
        "WATCHED_ASSET_PAIRS",
        [("USDC", issuer), ("native", "native")],
    )

    trade = Trade(
        trade_id="t1",
        ledger_close_time=datetime(2024, 1, 1, tzinfo=UTC),
        base_account="GBASE",
        counter_account="GCOUNTER",
        base_asset=Asset(code="USDC", issuer=issuer),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=10.0,
        counter_amount=20.0,
        price=2.0,
    )

    with patch("ingestion.connectors.builtin.load_trades", return_value=iter([trade])) as mock_load:
        connector = SdexTradeConnector()
        records = list(connector.load())

    assert records == [trade]
    # native/native pair is skipped (asset == xlm), so exactly one call
    mock_load.assert_called_once()
    df = connector.to_dataframe(records)
    assert df.iloc[0]["base_asset"] == f"USDC:{issuer}"


def test_amm_connector_requires_pool_ids(monkeypatch):
    import config as config_module
    from ingestion.connectors.builtin import AmmPoolTradeConnector

    monkeypatch.setattr(config_module.config, "WATCHED_AMM_POOLS", [])
    connector = AmmPoolTradeConnector()

    with pytest.raises(ConnectorError, match="pool_ids"):
        list(
            connector.load(
                since=datetime(2024, 1, 1, tzinfo=UTC), until=datetime(2024, 1, 2, tzinfo=UTC)
            )
        )


def test_amm_connector_delegates_to_iter_amm_pool_trades():
    from ingestion.connectors.builtin import AmmPoolTradeConnector

    trade = Trade(
        trade_id="t1",
        ledger_close_time=datetime(2024, 1, 1, tzinfo=UTC),
        base_account="GBASE",
        counter_account="GCOUNTER",
        base_asset=Asset(code="USDC", issuer="GISSUER"),
        counter_asset=Asset(code="XLM", issuer=None),
        base_amount=10.0,
        counter_amount=20.0,
        price=2.0,
    )
    connector = AmmPoolTradeConnector()
    since, until = datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)

    with patch(
        "ingestion.connectors.builtin.iter_amm_pool_trades", return_value=iter([trade])
    ) as mock_iter:
        records = list(connector.load(since=since, until=until, pool_ids=["a" * 64]))

    mock_iter.assert_called_once_with("a" * 64, since, until)
    assert records == [trade]


def test_orderbook_connector_requires_account_ids():
    from ingestion.connectors.builtin import OrderBookConnector

    with pytest.raises(ConnectorError, match="account_ids"):
        list(OrderBookConnector().load())


def test_orderbook_connector_delegates_per_account():
    from ingestion.connectors.builtin import OrderBookConnector

    event = OrderBookEvent(
        event_id="e1",
        account="GACC",
        ledger_close_time=datetime(2024, 1, 1, tzinfo=UTC),
        selling=Asset(code="XLM", issuer=None),
        buying=Asset(code="USDC", issuer="GISSUER"),
        amount=5.0,
        price=1.0,
        action="created",
    )

    with patch(
        "ingestion.connectors.builtin.load_orderbook_events", return_value=iter([event])
    ) as mock_load:
        records = list(OrderBookConnector().load(account_ids=["GACC"]))

    mock_load.assert_called_once_with("GACC")
    assert records == [event]


def test_account_activity_connector_requires_account_ids():
    from ingestion.connectors.builtin import AccountActivityConnector

    with pytest.raises(ConnectorError, match="account_ids"):
        list(AccountActivityConnector().load())


def test_account_activity_connector_delegates_to_batch_loader():
    from ingestion.connectors.builtin import AccountActivityConnector

    activity = AccountActivity(
        account_id="GACC",
        account_created_at=datetime(2024, 1, 1, tzinfo=UTC),
        funding_account="GFUNDER",
    )

    with patch(
        "ingestion.connectors.builtin.load_accounts_activity", return_value=[activity]
    ) as mock_load:
        records = list(AccountActivityConnector().load(account_ids=["GACC"]))

    mock_load.assert_called_once_with(["GACC"])
    assert records == [activity]
