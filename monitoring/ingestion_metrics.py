"""Prometheus emitters for ingestion throughput, latency, and failures.

The module exposes a process-wide emitter for production call sites and an
``IngestionMetricsEmitter`` class that accepts a custom registry for isolated
consumers. Metric labels deliberately contain only bounded values: source,
pipeline stage, and exception class.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from prometheus_client import REGISTRY, CollectorRegistry, Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:  # pragma: no cover - optional observability dependency
    REGISTRY = None
    CollectorRegistry = Any  # type: ignore[misc,assignment]
    _PROM_AVAILABLE = False


class IngestionMetricsEmitter:
    """Emit low-cardinality metrics for any ingestion source and stage."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self._registry = registry or REGISTRY
        self.records = None
        self.failures = None
        self.throughput = None
        self.duration = None
        self.last_success = None
        if not _PROM_AVAILABLE:
            return

        self.records = self._metric(
            Counter,
            "ledgerlens_ingestion_records_total",
            "Total records successfully emitted by ingestion",
            ["source", "stage"],
        )
        self.failures = self._metric(
            Counter,
            "ledgerlens_ingestion_failures_total",
            "Total ingestion failures grouped by exception type",
            ["source", "stage", "error_type"],
        )
        self.throughput = self._metric(
            Gauge,
            "ledgerlens_ingestion_throughput_records_per_second",
            "Most recently observed ingestion batch throughput",
            ["source", "stage"],
        )
        self.duration = self._metric(
            Histogram,
            "ledgerlens_ingestion_duration_seconds",
            "Ingestion operation duration in seconds",
            ["source", "stage"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
        )
        self.last_success = self._metric(
            Gauge,
            "ledgerlens_ingestion_last_success_timestamp_seconds",
            "Unix timestamp of the last successful ingestion operation",
            ["source", "stage"],
        )

    def _metric(self, metric_type, name: str, description: str, labels: list[str], **kwargs):
        """Create a collector, reusing one already present in the same registry."""
        existing = getattr(self._registry, "_names_to_collectors", {}).get(name)
        if existing is not None:
            return existing
        return metric_type(name, description, labels, registry=self._registry, **kwargs)

    def emit_success(
        self,
        source: str,
        *,
        stage: str = "fetch",
        record_count: int = 1,
        duration_seconds: float | None = None,
    ) -> None:
        """Record a successful operation and its observed batch throughput."""
        count = max(0, int(record_count))
        if self.records is not None:
            self.records.labels(source=source, stage=stage).inc(count)
        if self.last_success is not None:
            self.last_success.labels(source=source, stage=stage).set(time.time())
        if duration_seconds is not None:
            duration = max(0.0, float(duration_seconds))
            if self.duration is not None:
                self.duration.labels(source=source, stage=stage).observe(duration)
            if self.throughput is not None:
                rate = count / duration if duration > 0 else float(count)
                self.throughput.labels(source=source, stage=stage).set(rate)

    def emit_failure(
        self,
        source: str,
        error: BaseException | type[BaseException] | str,
        *,
        stage: str = "fetch",
        duration_seconds: float | None = None,
    ) -> None:
        """Record a failed operation without exposing exception messages as labels."""
        if isinstance(error, str):
            error_type = error
        elif isinstance(error, type):
            error_type = error.__name__
        else:
            error_type = type(error).__name__
        if self.failures is not None:
            self.failures.labels(source=source, stage=stage, error_type=error_type).inc()
        if duration_seconds is not None and self.duration is not None:
            self.duration.labels(source=source, stage=stage).observe(
                max(0.0, float(duration_seconds))
            )


INGESTION_METRICS = IngestionMetricsEmitter()


def emit_ingestion_success(
    source: str,
    *,
    stage: str = "fetch",
    record_count: int = 1,
    duration_seconds: float | None = None,
) -> None:
    """Emit success through the process-wide ingestion metrics collector."""
    INGESTION_METRICS.emit_success(
        source,
        stage=stage,
        record_count=record_count,
        duration_seconds=duration_seconds,
    )


def emit_ingestion_failure(
    source: str,
    error: BaseException | type[BaseException] | str,
    *,
    stage: str = "fetch",
    duration_seconds: float | None = None,
) -> None:
    """Emit failure through the process-wide ingestion metrics collector."""
    INGESTION_METRICS.emit_failure(
        source,
        error,
        stage=stage,
        duration_seconds=duration_seconds,
    )
