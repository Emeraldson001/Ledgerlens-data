"""Tests for the typed service-boundary contracts in utils/boundaries.py."""

from __future__ import annotations

import pytest

from utils.boundaries import (
    PROTOCOLS,
    CircuitBreakerPort,
    FieldEncryptionPort,
    RetryPolicyPort,
    ServiceBoundaryError,
    ServiceRegistry,
    StructuredLoggerFactoryPort,
    TracerFactoryPort,
    describe_bindings,
    registry,
    validate_service_boundaries,
)


class TestDefaultRegistry:
    def test_default_bindings_conform_to_their_protocols(self):
        diagnostics = validate_service_boundaries()
        assert diagnostics == [], f"unexpected drift: {diagnostics}"

    def test_all_declared_protocols_have_a_default_binding(self):
        bound = set(registry.bound_protocols())
        expected = {name for name in PROTOCOLS}
        assert bound == expected

    def test_resolve_circuit_breaker_returns_conforming_instance(self):
        breaker = registry.resolve(CircuitBreakerPort)
        assert hasattr(breaker, "call")
        assert callable(breaker.call)
        assert breaker.state is not None

    def test_resolve_field_encryption_round_trips(self):
        adapter = registry.resolve(FieldEncryptionPort)
        blob = adapter.encrypt("hello-wallet-id")
        assert adapter.decrypt(blob) == "hello-wallet-id"

    def test_resolve_tracer_factory_returns_callable(self):
        tracer_factory = registry.resolve(TracerFactoryPort)
        assert callable(tracer_factory)

    def test_resolve_logger_factory_returns_callable(self):
        logger_factory = registry.resolve(StructuredLoggerFactoryPort)
        logger = logger_factory("test_service_boundaries")
        assert hasattr(logger, "info")

    def test_resolve_retry_policy_returns_callable_decorator_factory(self):
        retry_factory = registry.resolve(RetryPolicyPort)
        assert callable(retry_factory)

    def test_describe_bindings_lists_every_protocol(self):
        summary = describe_bindings()
        for name in PROTOCOLS:
            assert name in summary


class TestServiceRegistry:
    def test_resolve_unregistered_protocol_raises_actionable_error(self):
        empty = ServiceRegistry()
        with pytest.raises(ServiceBoundaryError) as excinfo:
            empty.resolve(CircuitBreakerPort)
        message = str(excinfo.value)
        assert "CircuitBreakerPort" in message
        assert "register" in message.lower()

    def test_register_and_resolve_custom_binding(self):
        custom = ServiceRegistry()

        class _FakeCircuitBreaker:
            state = "closed"

            def call(self, func, *args, **kwargs):
                return func(*args, **kwargs)

        custom.register(CircuitBreakerPort, _FakeCircuitBreaker)
        resolved = custom.resolve(CircuitBreakerPort)
        assert resolved.call(lambda x: x + 1, 41) == 42

    def test_register_overrides_existing_binding(self):
        custom = ServiceRegistry()
        custom.register(TracerFactoryPort, lambda: (lambda name: f"tracer:{name}"))
        custom.register(TracerFactoryPort, lambda: (lambda name: f"other-tracer:{name}"))
        factory = custom.resolve(TracerFactoryPort)
        assert factory("x") == "other-tracer:x"

    def test_validate_service_boundaries_reports_missing_binding_by_name(self):
        empty = ServiceRegistry()
        diagnostics = validate_service_boundaries(empty)
        assert len(diagnostics) == len(PROTOCOLS)
        assert all("no binding registered" in d for d in diagnostics)

    def test_validate_service_boundaries_flags_non_conforming_instance(self):
        custom = ServiceRegistry()
        for name, protocol in PROTOCOLS.items():
            if name == "CircuitBreakerPort":
                custom.register(protocol, lambda: object())
            else:
                custom.register(protocol, registry._bindings[protocol].factory)
        diagnostics = validate_service_boundaries(custom)
        assert len(diagnostics) == 1
        assert "CircuitBreakerPort" in diagnostics[0]
