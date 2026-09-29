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
    rate_limit: "RateLimitConfig | None" = None
    cost_quota: "CostQuotaConfig | None" = None


@dataclass
class RateLimitConfig:
    """Per-tenant rate limit settings.

    ``rate`` is the sustained token refill rate in requests per second and
    ``burst`` is the maximum bucket capacity (i.e. the largest burst of
    requests allowed at once). Both are per tenant so that one tenant's
    high-volume usage cannot degrade availability for other tenants.
    """

    rate: float
    burst: int


@dataclass
class CostQuotaConfig:
    """Per-tenant budget for expensive, cost-accounted endpoints.

    ``budget`` is the maximum accumulated request cost allowed within a
    ``window_seconds`` sliding window. Each expensive endpoint declares a
    per-request cost estimate (see ``api/app.py``); the API rejects requests
    that would push a tenant over its budget with a retry-after hint instead
    of queuing indefinitely or degrading other tenants.
    """

    budget: float
    window_seconds: int = 60


class TenantNotFoundError(Exception):
    pass


_tenant_configs: dict[str, TenantConfig] = {}
_allowed_tenant_ids: set[str] = set()


def _parse_rate_limit(cfg: dict[str, Any]) -> RateLimitConfig | None:
    """Parse an optional per-tenant ``rate_limit`` block.

    Accepts either a nested mapping::

        rate_limit:
          rate: 50
          burst: 100

    or a shorthand string such as ``"50/s"`` / ``"50"`` (burst defaults to
    the rate). Returns ``None`` when no override is configured, in which case
    the API falls back to its default per-tenant limit.
    """
    raw = cfg.get("rate_limit")
    if raw is None:
        return None
    if isinstance(raw, dict):
        rate = float(raw["rate"])
        burst = int(raw.get("burst", rate))
        return RateLimitConfig(rate=rate, burst=burst)
    if isinstance(raw, str):
        rate = float(raw.rstrip("/s"))
        return RateLimitConfig(rate=rate, burst=int(rate))
    rate = float(raw)
    return RateLimitConfig(rate=rate, burst=int(rate))


def _parse_cost_quota(cfg: dict[str, Any]) -> CostQuotaConfig | None:
    """Parse an optional per-tenant ``cost_quota`` block.

    Accepts either a nested mapping::

        cost_quota:
          budget: 100
          window_seconds: 60

    or a shorthand number such as ``100`` (window defaults to 60s). Returns
    ``None`` when no override is configured, in which case the API falls back
    to its default per-tenant budget.
    """
    raw = cfg.get("cost_quota")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return CostQuotaConfig(
            budget=float(raw["budget"]),
            window_seconds=int(raw.get("window_seconds", 60)),
        )
    return CostQuotaConfig(budget=float(raw))


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
            rate_limit=_parse_rate_limit(cfg),
            cost_quota=_parse_cost_quota(cfg),
        )
        for tid, cfg in data.get("tenants", {}).items()
    }
    _allowed_tenant_ids = set(_tenant_configs.keys())


def get_tenant_config(tenant_id: str) -> TenantConfig:
    if tenant_id not in _allowed_tenant_ids:
        raise TenantNotFoundError(f"Unknown tenant ID: {tenant_id}")
    return _tenant_configs[tenant_id]


def get_tenant_rate_limit(tenant_id: str) -> RateLimitConfig | None:
    """Return the configured per-tenant rate limit override, if any."""
    return get_tenant_config(tenant_id).rate_limit


def get_tenant_cost_quota(tenant_id: str) -> CostQuotaConfig | None:
    """Return the configured per-tenant cost quota override, if any."""
    return get_tenant_config(tenant_id).cost_quota


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
