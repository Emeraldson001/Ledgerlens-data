"""Health check monitoring system for long-running worker processes (#524).

Provides a unified, thread-safe health monitoring framework for streaming workers,
ingestion pipelines, web servers, and background tasks.

Components:
    - ``HealthStatus``: Standard health states (HEALTHY, DEGRADED, UNHEALTHY, STOPPED, UNKNOWN).
    - ``ComponentHealth``: Structured report for an individual worker/process.
    - ``WorkerHealthMonitor``: Active monitor that tracks heartbeats, error rates, and custom metrics.
    - ``HealthRegistry``: Global, thread-safe registry aggregating status across all long-running processes.

Usage:
    monitor = WorkerHealthMonitor("kafka_worker", stale_threshold_seconds=30)
    registry = get_health_registry()
    registry.register(monitor)

    # In worker poll/event loop:
    monitor.record_heartbeat(details={"messages_processed": 1024})

    # In API endpoint / health probe:
    status, report = registry.get_overall_status()
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)


class HealthStatus(enum.StrEnum):
    """Standard health state classification for worker components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


class ComponentHealth:
    """Structured health snapshot for a single monitored component.

    Attributes
    ----------
    name : str
        Human-readable component or worker name.
    status : HealthStatus
        Evaluated health state.
    last_heartbeat : float | None
        Epoch timestamp of the most recent heartbeat.
    heartbeat_age_seconds : float | None
        Elapsed time in seconds since the last heartbeat.
    details : dict[str, Any]
        Additional context-specific health metrics and state flags.
    error_message : str | None
        Description of any active error or degraded state, if applicable.
    """

    def __init__(
        self,
        name: str,
        status: HealthStatus = HealthStatus.UNKNOWN,
        last_heartbeat: float | None = None,
        heartbeat_age_seconds: float | None = None,
        details: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.last_heartbeat = last_heartbeat
        self.heartbeat_age_seconds = heartbeat_age_seconds
        self.details = details or {}
        self.error_message = error_message

    def to_dict(self) -> dict[str, Any]:
        """Serialize health report to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "heartbeat_age_seconds": (
                round(self.heartbeat_age_seconds, 2)
                if self.heartbeat_age_seconds is not None
                else None
            ),
            "details": self.details,
            "error_message": self.error_message,
        }

    def __repr__(self) -> str:
        return f"<ComponentHealth name={self.name!r} status={self.status.value!r}>"


class HealthCheckable(Protocol):
    """Protocol for components that expose custom health check capability."""

    def get_health(self) -> ComponentHealth:
        """Return a snapshot of component health."""
        ...


class WorkerHealthMonitor:
    """Thread-safe health monitor for long-running worker processes.

    Tracks worker heartbeats, checks heartbeat freshness against a configured
    stale threshold, and allows custom health validation callbacks.

    Parameters
    ----------
    name : str
        Unique identifier for the worker being monitored.
    stale_threshold_seconds : float | None
        Maximum allowed duration without a heartbeat before marking UNHEALTHY.
        Defaults to ``config.WORKER_HEALTH_STALE_THRESHOLD_SECONDS``.
    health_check_fn : Callable[[], tuple[HealthStatus, str | None]] | None
        Optional callback invoked during health evaluation to incorporate custom state.
    """

    def __init__(
        self,
        name: str,
        stale_threshold_seconds: float | None = None,
        health_check_fn: Callable[[], tuple[HealthStatus, str | None]] | None = None,
    ) -> None:
        self.name = name
        self.stale_threshold_seconds = (
            stale_threshold_seconds
            if stale_threshold_seconds is not None
            else config.WORKER_HEALTH_STALE_THRESHOLD_SECONDS
        )
        self._health_check_fn = health_check_fn
        self._lock = threading.Lock()
        self._last_heartbeat: float | None = None
        self._details: dict[str, Any] = {}
        self._stopped: bool = False
        self._last_error: str | None = None

    def record_heartbeat(
        self,
        details: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Register a heartbeat from the worker loop.

        Parameters
        ----------
        details : dict[str, Any] | None
            Optional dictionary of operational metrics (e.g. queue size, processed count).
        error_message : str | None
            Optional error description if an error occurred during execution.
        """
        with self._lock:
            self._last_heartbeat = time.time()
            if details:
                self._details.update(details)
            if error_message is not None:
                self._last_error = error_message
            elif error_message == "":
                self._last_error = None

    def mark_stopped(self, reason: str = "Worker shutdown cleanly") -> None:
        """Mark the worker as intentionally stopped."""
        with self._lock:
            self._stopped = True
            self._last_error = reason

    def update_details(self, key_or_dict: str | dict[str, Any], value: Any = None) -> None:
        """Update metrics or details without updating the heartbeat timestamp."""
        with self._lock:
            if isinstance(key_or_dict, dict):
                self._details.update(key_or_dict)
            else:
                self._details[key_or_dict] = value

    def get_health(self) -> ComponentHealth:
        """Evaluate and return the current health snapshot of the worker."""
        now = time.time()
        with self._lock:
            stopped = self._stopped
            last_hb = self._last_heartbeat
            details = dict(self._details)
            error_msg = self._last_error

        if stopped:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.STOPPED,
                last_heartbeat=last_hb,
                heartbeat_age_seconds=(now - last_hb) if last_hb else None,
                details=details,
                error_message=error_msg or "Worker stopped",
            )

        if last_hb is None:
            return ComponentHealth(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                last_heartbeat=None,
                heartbeat_age_seconds=None,
                details=details,
                error_message="No heartbeat recorded yet",
            )

        age = now - last_hb
        if age > self.stale_threshold_seconds * 2:
            status = HealthStatus.UNHEALTHY
            status_msg = f"Heartbeat severely stale ({age:.1f}s > {self.stale_threshold_seconds * 2:.1f}s threshold)"
        elif age > self.stale_threshold_seconds:
            status = HealthStatus.DEGRADED
            status_msg = (
                f"Heartbeat stale ({age:.1f}s > {self.stale_threshold_seconds:.1f}s threshold)"
            )
        else:
            status = HealthStatus.HEALTHY
            status_msg = None

        if error_msg and status == HealthStatus.HEALTHY:
            status = HealthStatus.DEGRADED

        # Invoke custom health check callback if provided
        if self._health_check_fn:
            try:
                fn_status, fn_msg = self._health_check_fn()
                if fn_status == HealthStatus.UNHEALTHY or (
                    fn_status == HealthStatus.DEGRADED and status == HealthStatus.HEALTHY
                ):
                    status = fn_status
                if fn_msg:
                    status_msg = f"{status_msg}; {fn_msg}" if status_msg else fn_msg
            except Exception as exc:
                logger.warning("Custom health check callback failed for %s: %s", self.name, exc)
                status = HealthStatus.DEGRADED
                status_msg = f"Health callback error: {exc}"

        return ComponentHealth(
            name=self.name,
            status=status,
            last_heartbeat=last_hb,
            heartbeat_age_seconds=age,
            details=details,
            error_message=status_msg or error_msg,
        )


class HealthRegistry:
    """Thread-safe central registry for worker health monitoring across the repo."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._components: dict[str, HealthCheckable | WorkerHealthMonitor] = {}

    def register(
        self,
        component: HealthCheckable | WorkerHealthMonitor | str,
        monitor: HealthCheckable | WorkerHealthMonitor | None = None,
    ) -> WorkerHealthMonitor | HealthCheckable:
        """Register a component or monitor with the central registry."""
        with self._lock:
            if isinstance(component, str):
                if monitor is None:
                    monitor = WorkerHealthMonitor(component)
                name = component
                target = monitor
            else:
                target = component
                name = getattr(component, "name", component.__class__.__name__)
            self._components[name] = target
            logger.debug("Registered worker health monitor: %s", name)
            return target

    def unregister(self, name: str) -> None:
        """Remove a component from the health registry."""
        with self._lock:
            self._components.pop(name, None)

    def get_health(self, name: str) -> ComponentHealth | None:
        """Retrieve health report for a specific component."""
        with self._lock:
            target = self._components.get(name)
        if target is None:
            return None
        return target.get_health()

    def get_all_health(self) -> dict[str, ComponentHealth]:
        """Return health snapshots for all registered components."""
        with self._lock:
            items = list(self._components.items())

        reports = {}
        for name, comp in items:
            try:
                reports[name] = comp.get_health()
            except Exception as exc:
                logger.error("Failed to fetch health for %s: %s", name, exc)
                reports[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    error_message=f"Health probe exception: {exc}",
                )
        return reports

    def get_overall_status(self) -> tuple[HealthStatus, dict[str, Any]]:
        """Calculate aggregate system health status and return detailed report."""
        reports = self.get_all_health()
        if not reports:
            return HealthStatus.HEALTHY, {"overall": "healthy", "components": {}}

        component_dicts = {}
        statuses = []
        for name, rep in reports.items():
            component_dicts[name] = rep.to_dict()
            statuses.append(rep.status)

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall = HealthStatus.UNHEALTHY
        elif any(s == HealthStatus.DEGRADED for s in statuses):
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.STOPPED for s in statuses):
            overall = HealthStatus.STOPPED
        else:
            overall = HealthStatus.HEALTHY

        summary = {
            "overall": overall.value,
            "components": component_dicts,
            "total_monitored": len(reports),
        }
        return overall, summary


_global_registry = HealthRegistry()


def get_health_registry() -> HealthRegistry:
    """Return the global worker health registry singleton."""
    return _global_registry
