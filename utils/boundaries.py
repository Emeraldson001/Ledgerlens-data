"""Typed service boundaries for the shared ``utils`` package.

This module gives every cross-cutting utility (retry, circuit breaking,
tracing, field encryption, structured logging) a formal ``typing.Protocol``
contract, plus a small typed registry (:class:`ServiceRegistry`) that
consumers use to resolve an implementation without importing the concrete
module directly.

Why this exists
----------------
``utils`` is imported by nearly every package in this repo (``detection``,
``ingestion``, ``api``, ``streaming``, ...). Without an explicit contract,
a change to a function's signature in ``utils/retry.py`` or
``utils/circuit_breaker.py`` can silently break a dozen call sites with no
type-checker signal, because callers usually just do
``from utils.retry import retry_with_backoff`` and duck-type against it.

Defining ``Protocol`` classes here gives us:

* A single, documented source of truth for what a "retry policy",
  "circuit breaker", "tracer", "field encryptor", and "structured logger"
  must provide.
* Static verification via mypy (``Protocol`` + ``runtime_checkable``)
  without forcing concrete classes to inherit from anything.
* A runtime conformance check (``validate_service_boundaries``) that CI can
  run to catch drift between the contract and the implementation before it
  reaches consumers -- see ``scripts/check_service_boundaries.py``.

Adding a new shared utility
----------------------------
1. Define a ``Protocol`` for its public surface below.
2. Register the concrete implementation in ``DEFAULT_BINDINGS``.
3. Add a case to ``validate_service_boundaries`` if the protocol needs
   more than a structural (duck-typed) check.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class ServiceBoundaryError(Exception):
    """Raised when a service cannot be resolved or fails its contract check.

    The message always includes the port name, the reason, and a pointer to
    where to look, so failures are actionable instead of a bare KeyError.
    """


@runtime_checkable
class RetryPolicyPort(Protocol):
    """Contract for retry/backoff decorators such as ``utils.retry.retry_with_backoff``."""

    def __call__(self, *args: Any, **kwargs: Any) -> Callable[..., Callable[..., Any]]: ...


@runtime_checkable
class CircuitBreakerPort(Protocol):
    """Contract satisfied by ``utils.circuit_breaker.CircuitBreaker``."""

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T: ...

    @property
    def state(self) -> Any: ...


@runtime_checkable
class TracerFactoryPort(Protocol):
    """Contract satisfied by ``utils.tracing.get_tracer``."""

    def __call__(self, name: str) -> Any: ...


@runtime_checkable
class FieldEncryptionPort(Protocol):
    """Contract satisfied by the ``encrypt_field`` / ``decrypt_field`` pair in
    ``utils.field_encryption``."""

    def encrypt(self, plaintext: str) -> bytes: ...

    def decrypt(self, blob: bytes) -> str: ...


@runtime_checkable
class StructuredLoggerFactoryPort(Protocol):
    """Contract satisfied by ``utils.logging.get_logger``."""

    def __call__(self, name: str) -> Any: ...


@dataclass
class _Binding:
    factory: Callable[[], Any]
    protocol: type
    description: str


@dataclass
class ServiceRegistry:
    """A minimal typed registry mapping a Protocol (the "port") to a factory.

    This is intentionally not a full dependency-injection framework. It
    exists so that a caller can do::

        from utils.boundaries import registry, CircuitBreakerPort

        breaker: CircuitBreakerPort = registry.resolve(CircuitBreakerPort)

    instead of importing ``utils.circuit_breaker`` directly, which keeps the
    caller coupled to the *contract* rather than the implementation module.
    Swapping an implementation (e.g. for a test double) only requires a new
    call to :meth:`register`.
    """

    _bindings: dict[type, _Binding] = field(default_factory=dict)

    def register(
        self,
        protocol: type,
        factory: Callable[[], Any],
        *,
        description: str = "",
    ) -> None:
        """Bind a factory to a Protocol type.

        ``factory`` is called lazily on each :meth:`resolve` so registries
        can bind cheap factories (e.g. ``lambda: get_logger(__name__)``)
        without constructing anything at import time.
        """
        self._bindings[protocol] = _Binding(
            factory=factory, protocol=protocol, description=description
        )

    def resolve(self, protocol: type[T]) -> T:
        """Return an instance satisfying ``protocol``.

        Raises :class:`ServiceBoundaryError` with an actionable message if
        no binding was registered, or if the resolved object does not
        structurally satisfy the protocol.
        """
        binding = self._bindings.get(protocol)
        if binding is None:
            known = ", ".join(p.__name__ for p in self._bindings) or "<none registered>"
            raise ServiceBoundaryError(
                f"No binding registered for {protocol.__name__!r}. "
                f"Known bindings: {known}. "
                f"Register one via utils.boundaries.registry.register({protocol.__name__}, factory)."
            )
        instance = binding.factory()
        if isinstance(protocol, type) and hasattr(protocol, "_is_runtime_protocol"):
            if not isinstance(instance, protocol):
                raise ServiceBoundaryError(
                    f"Binding for {protocol.__name__!r} produced {type(instance).__name__!r}, "
                    f"which does not satisfy the protocol's required methods/attributes. "
                    f"Check the factory registered in utils/boundaries.py DEFAULT_BINDINGS."
                )
        return instance

    def bound_protocols(self) -> list[str]:
        return sorted(p.__name__ for p in self._bindings)


def _default_circuit_breaker_factory() -> Any:
    from utils.circuit_breaker import CircuitBreaker

    return CircuitBreaker(name="default")


def _default_field_encryption_adapter() -> Any:
    from utils import field_encryption

    class _FieldEncryptionAdapter:
        def encrypt(self, plaintext: str) -> bytes:
            return field_encryption.encrypt_field(plaintext)

        def decrypt(self, blob: bytes) -> str:
            return field_encryption.decrypt_field(blob)

    return _FieldEncryptionAdapter()


def _default_tracer_factory_adapter() -> Any:
    from utils.tracing import get_tracer

    return get_tracer


def _default_logger_factory_adapter() -> Any:
    from utils.logging import get_logger

    return get_logger


def _default_retry_policy_adapter() -> Any:
    from utils.retry import retry_with_backoff

    return retry_with_backoff


registry = ServiceRegistry()
registry.register(
    CircuitBreakerPort,
    _default_circuit_breaker_factory,
    description="Default CircuitBreaker(name='default') instance",
)
registry.register(
    FieldEncryptionPort,
    _default_field_encryption_adapter,
    description="Adapter over utils.field_encryption.encrypt_field/decrypt_field",
)
registry.register(
    TracerFactoryPort,
    _default_tracer_factory_adapter,
    description="utils.tracing.get_tracer",
)
registry.register(
    StructuredLoggerFactoryPort,
    _default_logger_factory_adapter,
    description="utils.logging.get_logger",
)
registry.register(
    RetryPolicyPort,
    _default_retry_policy_adapter,
    description="utils.retry.retry_with_backoff",
)


PROTOCOLS: dict[str, type] = {
    "CircuitBreakerPort": CircuitBreakerPort,
    "FieldEncryptionPort": FieldEncryptionPort,
    "TracerFactoryPort": TracerFactoryPort,
    "StructuredLoggerFactoryPort": StructuredLoggerFactoryPort,
    "RetryPolicyPort": RetryPolicyPort,
}


def validate_service_boundaries(target: ServiceRegistry | None = None) -> list[str]:
    """Validate that every registered binding structurally satisfies its Protocol.

    Returns a list of human-readable diagnostic strings (empty when
    everything conforms). Never raises for a single bad binding -- it
    collects every failure so a CI run reports all drift in one pass
    instead of failing fast on the first port.
    """
    target = target or registry
    diagnostics: list[str] = []
    for name, protocol in PROTOCOLS.items():
        binding = target._bindings.get(protocol)
        if binding is None:
            diagnostics.append(f"[{name}] no binding registered")
            continue
        try:
            instance = binding.factory()
        except Exception as exc:  # pragma: no cover - defensive diagnostic path
            diagnostics.append(f"[{name}] factory raised {type(exc).__name__}: {exc}")
            continue
        if not isinstance(instance, protocol):
            missing = [
                m for m in dir(protocol) if not m.startswith("_") and not hasattr(instance, m)
            ]
            diagnostics.append(
                f"[{name}] {type(instance).__name__!r} does not satisfy protocol"
                f"{' (missing: ' + ', '.join(missing) + ')' if missing else ''} "
                f"-- see utils/boundaries.py:{name}"
            )
    return diagnostics


def describe_bindings(target: ServiceRegistry | None = None) -> str:
    """Human-readable summary of the current registry, for diagnostics/CLI output."""
    target = target or registry
    lines = []
    for name, protocol in PROTOCOLS.items():
        binding = target._bindings.get(protocol)
        status = "bound" if binding else "MISSING"
        desc = binding.description if binding else ""
        lines.append(f"  {name:<28} [{status}] {desc}")
    return "\n".join(lines)


__all__ = [
    "ServiceBoundaryError",
    "RetryPolicyPort",
    "CircuitBreakerPort",
    "TracerFactoryPort",
    "FieldEncryptionPort",
    "StructuredLoggerFactoryPort",
    "ServiceRegistry",
    "registry",
    "PROTOCOLS",
    "validate_service_boundaries",
    "describe_bindings",
]
