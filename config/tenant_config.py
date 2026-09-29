"""Tenant configuration for multi-tenant namespace isolation."""

from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class TenantConfig:
    risk_threshold: int
    benford_min_sample: int
    alert_channels: list[str]
    asset_pair_whitelist: list[str]
    threshold_strategy: str = "static"
    threshold_config: dict[str, Any] = field(default_factory=dict)


class TenantNotFoundError(Exception):
    pass


_tenant_configs: dict[str, TenantConfig] = {}
_allowed_tenant_ids: set[str] = set()


def load_tenants_config(path: str = "config/tenants.yaml") -> None:
    global _tenant_configs, _allowed_tenant_ids
    with open(path) as f:
        data = yaml.safe_load(f)
    _tenant_configs = {
        tid: TenantConfig(
            risk_threshold=cfg["risk_threshold"],
            benford_min_sample=cfg["benford_min_sample"],
            alert_channels=cfg["alert_channels"],
            asset_pair_whitelist=cfg["asset_pair_whitelist"],
            threshold_strategy=cfg.get("threshold_strategy", "static"),
            threshold_config=cfg.get("threshold_config", {}),
        )
        for tid, cfg in data.get("tenants", {}).items()
    }
    _allowed_tenant_ids = set(_tenant_configs.keys())


def get_tenant_config(tenant_id: str) -> TenantConfig:
    if tenant_id not in _allowed_tenant_ids:
        raise TenantNotFoundError(f"Unknown tenant ID: {tenant_id}")
    return _tenant_configs[tenant_id]


def build_threshold_strategy(tenant_id: str) -> Any:
    """Build a ThresholdStrategy instance for the given tenant.

    Returns the appropriate strategy based on the tenant's
    ``threshold_strategy`` and ``threshold_config`` settings.
    """
    from importlib import import_module

    build_strategy = import_module("detection.threshold_strategy").build_strategy

    tc = get_tenant_config(tenant_id)
    kwargs: dict[str, Any] = dict(tc.threshold_config)
    if tc.threshold_strategy == "static" and "threshold" not in kwargs:
        kwargs["threshold"] = tc.risk_threshold / 100.0
    return build_strategy(tc.threshold_strategy, **kwargs)


class TenantContext:
    def __init__(self, tenant_id: str):
        if tenant_id not in _allowed_tenant_ids:
            raise TenantNotFoundError(f"Unknown tenant ID: {tenant_id}")
        self.tenant_id = tenant_id
        self.config = _tenant_configs[tenant_id]

    def redis_key(self, key: str) -> str:
        return f"{self.tenant_id}:{key}"

    def prometheus_labels(self, labels: dict[str, Any]) -> dict[str, Any]:
        return {"tenant": self.tenant_id, **labels}
