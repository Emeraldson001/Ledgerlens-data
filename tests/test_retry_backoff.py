"""Tests for `ingestion.retry_backoff` — the resilient retry/backoff layer."""

from __future__ import annotations

import random

import pytest

from ingestion.retry_backoff import (
    NETWORK_RETRY_POLICY,
    AttemptHistory,
    RetryConfigurationError,
    RetryExhaustedError,
    RetryPolicy,
    call_with_retry,
    retry_with_backoff,
)


def _no_sleep(_seconds: float) -> None:
    pass


class TestRetryPolicyValidation:
    def test_defaults_are_valid(self):
        RetryPolicy()  # should not raise

    def test_rejects_zero_max_attempts(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(max_attempts=0)

    def test_rejects_non_positive_base_delay(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(base_delay_seconds=0)

    def test_rejects_max_delay_below_base_delay(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(base_delay_seconds=10, max_delay_seconds=5)

    def test_rejects_multiplier_not_above_one(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(multiplier=1.0)

    def test_rejects_unknown_jitter_mode(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(jitter="chaotic")

    def test_rejects_empty_retryable_exceptions(self):
        with pytest.raises(RetryConfigurationError):
            RetryPolicy(retryable_exceptions=())


class TestDelayForAttempt:
    def test_first_attempt_has_no_delay(self):
        policy = RetryPolicy()
        assert policy.delay_for_attempt(1) == 0.0

    def test_none_jitter_is_deterministic_exponential(self):
        policy = RetryPolicy(
            base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=100, jitter="none"
        )
        assert policy.delay_for_attempt(2) == 1.0
        assert policy.delay_for_attempt(3) == 2.0
        assert policy.delay_for_attempt(4) == 4.0

    def test_delay_is_capped_at_max_delay(self):
        policy = RetryPolicy(
            base_delay_seconds=1.0, multiplier=10.0, max_delay_seconds=5.0, jitter="none"
        )
        assert policy.delay_for_attempt(5) == 5.0

    def test_full_jitter_stays_within_bounds(self):
        policy = RetryPolicy(
            base_delay_seconds=1.0, multiplier=2.0, max_delay_seconds=100, jitter="full"
        )
        rng = random.Random(42)
        for attempt in range(2, 6):
            delay = policy.delay_for_attempt(attempt, rng=rng)
            raw_cap = min(1.0 * (2.0 ** (attempt - 2)), 100)
            assert 0 <= delay <= raw_cap

    def test_equal_jitter_stays_within_bounds(self):
        policy = RetryPolicy(
            base_delay_seconds=2.0, multiplier=2.0, max_delay_seconds=100, jitter="equal"
        )
        rng = random.Random(1)
        delay = policy.delay_for_attempt(2, rng=rng)
        assert 1.0 <= delay <= 2.0  # half..raw


class TestCallWithRetry:
    def test_succeeds_on_first_attempt_without_retrying(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result = call_with_retry(fn, policy=RetryPolicy(max_attempts=3), sleep=_no_sleep)
        assert result == "ok"
        assert len(calls) == 1

    def test_retries_transient_failures_then_succeeds(self):
        attempts = {"count": 0}

        def flaky():
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise OSError("transient")
            return "recovered"

        policy = RetryPolicy(
            max_attempts=5, base_delay_seconds=0.01, jitter="none", retryable_exceptions=(IOError,)
        )
        result = call_with_retry(flaky, policy=policy, sleep=_no_sleep)
        assert result == "recovered"
        assert attempts["count"] == 3

    def test_raises_retry_exhausted_after_max_attempts(self):
        def always_fails():
            raise OSError("permanent")

        policy = RetryPolicy(
            max_attempts=3, base_delay_seconds=0.01, retryable_exceptions=(IOError,)
        )
        with pytest.raises(RetryExhaustedError) as exc:
            call_with_retry(always_fails, policy=policy, sleep=_no_sleep)
        assert exc.value.history.attempt_count == 3
        assert isinstance(exc.value.last_exception, IOError)
        assert isinstance(exc.value.__cause__, IOError)

    def test_non_retryable_exception_propagates_immediately(self):
        calls = {"count": 0}

        def fn():
            calls["count"] += 1
            raise ValueError("not retryable")

        policy = RetryPolicy(max_attempts=5, retryable_exceptions=(IOError,))
        with pytest.raises(ValueError):
            call_with_retry(fn, policy=policy, sleep=_no_sleep)
        assert calls["count"] == 1  # no retry consumed

    def test_on_attempt_callback_invoked_per_attempt(self):
        records = []

        def flaky():
            if len(records) < 2:
                raise OSError("retry me")
            return "done"

        policy = RetryPolicy(
            max_attempts=5, base_delay_seconds=0.01, retryable_exceptions=(IOError,)
        )
        call_with_retry(
            flaky, policy=policy, sleep=_no_sleep, on_attempt=lambda r: records.append(r)
        )
        assert len(records) == 3
        assert records[-1].succeeded is True


class TestRetryWithBackoffDecorator:
    def test_wraps_function_and_preserves_args(self):
        attempts = {"count": 0}

        @retry_with_backoff(
            RetryPolicy(max_attempts=3, base_delay_seconds=0.01, retryable_exceptions=(IOError,)),
            sleep=_no_sleep,
        )
        def fetch(cursor: str) -> str:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise OSError("timeout")
            return f"page-{cursor}"

        assert fetch("abc") == "page-abc"
        assert attempts["count"] == 2

    def test_preserves_function_metadata(self):
        @retry_with_backoff()
        def my_job():
            """docstring"""
            return 1

        assert my_job.__name__ == "my_job"
        assert my_job.__doc__ == "docstring"


def test_network_retry_policy_only_retries_io_errors():
    assert OSError in NETWORK_RETRY_POLICY.retryable_exceptions
    assert ValueError not in NETWORK_RETRY_POLICY.retryable_exceptions


def test_attempt_history_render_includes_each_attempt():
    history = AttemptHistory()
    policy = RetryPolicy(max_attempts=2, base_delay_seconds=0.01, retryable_exceptions=(IOError,))

    def fails_once():
        if not history.records:
            raise OSError("first failure")
        return "ok"

    call_with_retry(fails_once, policy=policy, sleep=_no_sleep, on_attempt=history.add)
    rendered = history.render()
    assert "attempt 1" in rendered
    assert "attempt 2" in rendered


class TestBackoffStateDoesNotLeakBetweenRequests:
    """After a call that retried several times before succeeding, the *next*
    independent call must begin at the base delay — not at an inflated delay
    carried over from the previous call's retries.

    This verifies that retry/backoff state is scoped per-call, not shared
    across calls (which would make a historical backfill progressively slower
    even when Horizon is healthy after a transient outage).
    """

    def test_second_call_starts_at_base_delay(self):
        """First call retries twice before succeeding; second call must use
        the base delay on its first retry — not the inflated delay from
        call 1's second retry."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            multiplier=2.0,
            jitter="none",
            retryable_exceptions=(IOError,),
        )
        delays_recorded: list[float] = []

        def capturing_sleep(seconds: float) -> None:
            delays_recorded.append(seconds)

        # --- Call 1: fail twice, succeed on attempt 3 ---
        call1_attempts = {"count": 0}

        def call1():
            call1_attempts["count"] += 1
            if call1_attempts["count"] < 3:
                raise OSError("transient")
            return "call1_ok"

        result1 = call_with_retry(call1, policy=policy, sleep=capturing_sleep)
        assert result1 == "call1_ok"
        assert call1_attempts["count"] == 3
        # Delays before attempt 2 and 3 should be base and base*multiplier
        assert delays_recorded == [1.0, 2.0]

        # --- Call 2: fail once, succeed on attempt 2 ---
        delays_recorded.clear()
        call2_attempts = {"count": 0}

        def call2():
            call2_attempts["count"] += 1
            if call2_attempts["count"] < 2:
                raise OSError("transient")
            return "call2_ok"

        result2 = call_with_retry(call2, policy=policy, sleep=capturing_sleep)
        assert result2 == "call2_ok"
        assert call2_attempts["count"] == 2
        # The ONLY delay in call 2 must be the base delay (1.0 s), NOT the
        # inflated 4.0 s or 8.0 s that would result from carrying over call 1's
        # backoff state.
        assert delays_recorded == [1.0], (
            f"Expected second independent call to start at base delay 1.0 s, "
            f"but got {delays_recorded}. Backoff state is leaking between calls."
        )

    def test_decorator_form_no_state_leak(self):
        """The @retry_with_backoff decorator must also reset per decorated-function
        call, not accumulate state across invocations of the same decorated function."""
        policy = RetryPolicy(
            max_attempts=5,
            base_delay_seconds=1.0,
            max_delay_seconds=60.0,
            multiplier=2.0,
            jitter="none",
            retryable_exceptions=(IOError,),
        )
        delays_recorded: list[float] = []

        def _sleep(s: float) -> None:
            delays_recorded.append(s)

        invocation = {"n": 0, "call_attempt": 0}

        @retry_with_backoff(policy, sleep=_sleep)
        def fetch_page():
            invocation["n"] += 1
            invocation["call_attempt"] += 1
            # First invocation: fail twice then succeed
            if invocation["n"] == 1 and invocation["call_attempt"] < 3:
                raise OSError("transient")
            return f"page-{invocation['n']}"

        # First outer call — retries twice inside
        invocation["call_attempt"] = 0
        r1 = fetch_page()
        assert r1 == "page-3"  # n increments each attempt

        # Reset attempt counter for the second outer call
        invocation["call_attempt"] = 0
        delays_recorded.clear()

        # Second outer call — succeeds immediately (no retry needed because
        # invocation["n"] is now 4, so the condition `n==1` is false)
        r2 = fetch_page()
        assert r2 == "page-4"
        # No delays at all since second call succeeds first attempt
        assert delays_recorded == [], (
            f"Second call should have had zero delays but recorded {delays_recorded}. "
            "State leaked from the first call's retry loop."
        )
