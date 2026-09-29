import pytest

from integrations.adapter_base import (
    AdapterAuthError,
    AdapterRegistry,
    AdapterResponse,
    AdapterUnavailableError,
)
from integrations.example_static_adapter import StaticLookupAdapter


def test_fetch_returns_uniform_response_envelope():
    adapter = StaticLookupAdapter("test_provider", table={"XLM": {"name": "Stellar Lumens"}})
    response = adapter.fetch({"key": "XLM"})
    assert isinstance(response, AdapterResponse)
    assert response.source == "test_provider"
    assert response.data == {"name": "Stellar Lumens"}
    assert response.degraded is False
    assert response.latency_ms >= 0


def test_fetch_missing_key_raises_typed_error():
    adapter = StaticLookupAdapter("test_provider", table={})
    with pytest.raises(AdapterAuthError):
        adapter.fetch({})


def test_fetch_wrong_api_key_raises_auth_error():
    adapter = StaticLookupAdapter("test_provider", table={"XLM": {}}, api_key="secret")
    with pytest.raises(AdapterAuthError):
        adapter.fetch({"key": "XLM", "api_key": "wrong"})


def test_health_check_reflects_manual_toggle():
    adapter = StaticLookupAdapter("test_provider", table={})
    assert adapter.health_check() is True
    adapter.set_healthy(False)
    assert adapter.health_check() is False


def test_registry_fetch_uses_first_healthy_adapter():
    primary = StaticLookupAdapter("primary", table={"XLM": "primary-data"})
    backup = StaticLookupAdapter("backup", table={"XLM": "backup-data"})

    registry = AdapterRegistry()
    registry.register("asset_metadata", primary)
    registry.register("asset_metadata", backup)

    response = registry.fetch("asset_metadata", {"key": "XLM"})
    assert response.data == "primary-data"
    assert response.source == "primary"


def test_registry_falls_back_when_primary_unhealthy():
    primary = StaticLookupAdapter("primary", table={"XLM": "primary-data"})
    primary.set_healthy(False)
    backup = StaticLookupAdapter("backup", table={"XLM": "backup-data"})

    registry = AdapterRegistry()
    registry.register("asset_metadata", primary)
    registry.register("asset_metadata", backup)

    response = registry.fetch("asset_metadata", {"key": "XLM"})
    assert response.source == "backup"


def test_registry_falls_back_when_primary_fetch_raises():
    primary = StaticLookupAdapter(
        "primary", table={}
    )  # will raise AdapterAuthError (missing key not in table)
    backup = StaticLookupAdapter("backup", table={"XLM": "backup-data"})

    registry = AdapterRegistry()
    registry.register("asset_metadata", primary)
    registry.register("asset_metadata", backup)

    response = registry.fetch("asset_metadata", {"key": "XLM"})
    assert response.source == "backup"


def test_registry_raises_unavailable_when_all_adapters_fail():
    primary = StaticLookupAdapter("primary", table={})
    backup = StaticLookupAdapter("backup", table={})

    registry = AdapterRegistry()
    registry.register("asset_metadata", primary)
    registry.register("asset_metadata", backup)

    with pytest.raises(AdapterUnavailableError) as excinfo:
        registry.fetch("asset_metadata", {"key": "XLM"})
    assert excinfo.value.attempted == ["primary", "backup"]


def test_registry_raises_unavailable_for_unknown_capability():
    registry = AdapterRegistry()
    with pytest.raises(AdapterUnavailableError):
        registry.fetch("nonexistent_capability", {})


def test_registry_get_returns_registered_adapters():
    registry = AdapterRegistry()
    adapter = StaticLookupAdapter("primary", table={})
    registry.register("asset_metadata", adapter)
    assert registry.get("asset_metadata") == [adapter]
    assert registry.get("other") == []
