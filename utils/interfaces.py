from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ServiceHealth:
    status: str
    details: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)


@runtime_checkable
class HealthCheckable(Protocol):
    def healthcheck(self) -> ServiceHealth: ...


@runtime_checkable
class MetricsProvider(Protocol):
    def get_metrics(self) -> dict[str, Any]: ...


@runtime_checkable
class ConfigurableService(Protocol):
    def configure(self, config: dict[str, Any]) -> None: ...


@runtime_checkable
class SecretSanitizer(Protocol):
    def sanitize(self, text: str) -> str: ...


class ServiceRegistry:
    def __init__(self) -> None:
        self._services: dict[str, Any] = {}

    def register(self, name: str, service: Any) -> None:
        self._services[name] = service

    def get(self, name: str) -> Any | None:
        return self._services.get(name)

    def check_all_health(self) -> dict[str, ServiceHealth]:
        results: dict[str, ServiceHealth] = {}
        for name, service in self._services.items():
            if isinstance(service, HealthCheckable):
                results[name] = service.healthcheck()
            else:
                results[name] = ServiceHealth(
                    status="UNKNOWN",
                    details={"error": "Service does not implement HealthCheckable"},
                )
        return results

    def collect_all_metrics(self) -> dict[str, dict[str, Any]]:
        metrics: dict[str, dict[str, Any]] = {}
        for name, service in self._services.items():
            if isinstance(service, MetricsProvider):
                metrics[name] = service.get_metrics()
        return metrics


default_registry = ServiceRegistry()
