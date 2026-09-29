"""Tests for utils.circuit_breaker — three-state circuit breaker.

Covers the half-open recovery path (issue #680):
- OPEN → HALF_OPEN after the cooldown elapses (fake clock, no real sleeps)
- HALF_OPEN → CLOSED after enough successful trial calls
- HALF_OPEN → OPEN again when a trial call fails
"""

from __future__ import annotations

import pytest

from utils.circuit_breaker import CircuitBreaker, CircuitOpenError, CircuitState


class FakeClock:
    """Controllable monotonic clock — advance time manually in tests."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _open_breaker(clock: FakeClock) -> CircuitBreaker:
    """Return a breaker driven into OPEN via consecutive failures."""
    breaker = CircuitBreaker(
        name="test-component",
        failure_threshold=2,
        timeout_seconds=5,
        success_threshold=1,
        _clock=clock,
    )

    def fail() -> None:
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(fail)

    assert breaker.state == CircuitState.OPEN
    return breaker


def test_open_transitions_to_half_open_then_closed_on_success():
    """OPEN → HALF_OPEN after the timeout elapses → CLOSED on a successful trial call."""
    clock = FakeClock()
    breaker = _open_breaker(clock)

    # Timeout has not elapsed: calls are still rejected with CircuitOpenError.
    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not run")

    # Elapse the cooldown; the next call is the half-open recovery probe.
    clock.advance(5)
    assert breaker.call(lambda: "recovered") == "recovered"
    assert breaker.state == CircuitState.CLOSED


def test_half_open_reopens_when_the_trial_call_fails():
    """HALF_OPEN → OPEN again when the recovery probe fails."""
    clock = FakeClock()
    breaker = _open_breaker(clock)

    clock.advance(5)  # Cooldown elapsed → next call is the half-open probe.

    def fail() -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        breaker.call(fail)

    assert breaker.state == CircuitState.OPEN
