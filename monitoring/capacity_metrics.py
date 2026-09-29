"""Prometheus metrics for capacity planning (Issue #242).

Registers three gauges scraped by Prometheus on the internal
/metrics endpoint (not exposed externally):

  - ledgerlens_cpu_usage_ratio    labelled by component
  - ledgerlens_memory_usage_bytes (process-wide)
  - ledgerlens_trades_per_second  labelled by asset_pair

Import this module to register the metrics; call the update helpers
periodically from the main process loop.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PROM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metric definitions — registered once on import
# ---------------------------------------------------------------------------

CPU_USAGE_RATIO: Gauge | None = None
MEMORY_USAGE_BYTES: Gauge | None = None
TRADES_PER_SECOND: Gauge | None = None
E2E_LATENCY_SECONDS: Histogram | None = None

if _PROM_AVAILABLE:
    try:
        from prometheus_client import REGISTRY

        # Guard against duplicate registration in test re-imports.
        if "ledgerlens_cpu_usage_ratio" not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
            CPU_USAGE_RATIO = Gauge(
                "ledgerlens_cpu_usage_ratio",
                "CPU usage ratio (0.0–1.0) per component",
                ["component"],
            )
        else:
            CPU_USAGE_RATIO = REGISTRY._names_to_collectors["ledgerlens_cpu_usage_ratio"]  # type: ignore[attr-defined]

        if "ledgerlens_memory_usage_bytes" not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
            MEMORY_USAGE_BYTES = Gauge(
                "ledgerlens_memory_usage_bytes",
                "Process resident-set-size memory usage in bytes",
            )
        else:
            MEMORY_USAGE_BYTES = REGISTRY._names_to_collectors["ledgerlens_memory_usage_bytes"]  # type: ignore[attr-defined]

        if "ledgerlens_trades_per_second" not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
            TRADES_PER_SECOND = Gauge(
                "ledgerlens_trades_per_second",
                "Observed trade-event ingestion rate (events/s) per asset pair",
                ["asset_pair"],
            )
        else:
            TRADES_PER_SECOND = REGISTRY._names_to_collectors["ledgerlens_trades_per_second"]  # type: ignore[attr-defined]

        if "ledgerlens_e2e_latency_seconds" not in REGISTRY._names_to_collectors:  # type: ignore[attr-defined]
            E2E_LATENCY_SECONDS = Histogram(
                "ledgerlens_e2e_latency_seconds",
                "End-to-end latency from ingestion to consumer decision",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            )
        else:
            E2E_LATENCY_SECONDS = REGISTRY._names_to_collectors["ledgerlens_e2e_latency_seconds"]  # type: ignore[attr-defined]

    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to register capacity metrics: %s", exc)


# ---------------------------------------------------------------------------
# Update helpers
# ---------------------------------------------------------------------------


def set_cpu_usage(component: str, ratio: float) -> None:
    """Record CPU usage ratio (0.0–1.0) for *component*.

    Components: ``benford``, ``feature``, ``inference``, ``ingestion``.
    """
    if CPU_USAGE_RATIO is not None:
        CPU_USAGE_RATIO.labels(component=component).set(ratio)


def set_memory_usage(bytes_used: int) -> None:
    """Record process memory (RSS) in bytes."""
    if MEMORY_USAGE_BYTES is not None:
        MEMORY_USAGE_BYTES.set(bytes_used)


def set_trades_per_second(asset_pair: str, rate: float) -> None:
    """Record trade-event ingestion rate for *asset_pair*."""
    if TRADES_PER_SECOND is not None:
        TRADES_PER_SECOND.labels(asset_pair=asset_pair).set(rate)


def record_e2e_latency(latency_seconds: float) -> None:
    """Record end-to-end latency observation."""
    if E2E_LATENCY_SECONDS is not None:
        E2E_LATENCY_SECONDS.observe(latency_seconds)
