"""Resilient retry and backoff layer for ingestion jobs.

Ingestion modules (`horizon_fetcher`, `historical_loader`, `kafka_producer`,
...) each talk to a flaky external dependency (Horizon, Kafka, S3). This
module provides one reusable, typed retry layer for wrapping those calls,
instead of each loader hand-rolling its own `while` loop and `time.sleep`.

Design:
    - `RetryPolicy` is a typed, immutable contract: max attempts, delay
      bounds, backoff multiplier, jitter mode, and which exceptions are
      retryable.
    - `call_with_retry` / `@retry_with_backoff` execute a callable under a
      policy, logging each attempt and returning a full `AttemptHistory` so
      a failure explains exactly how many attempts were made, how long each
      delay was, and what each attempt raised.
    - `RetryExhaustedError` wraps the last exception *and* the attempt
      history, so a caller (or an on-call engineer reading a log line) does
      not have to reconstruct the retry timeline from scattered log lines.

API::

    policy = RetryPolicy(max_attempts=5, base_delay_seconds=0.5, retryable_exceptions=(IOError,))

    @retry_with_backoff(policy)
    def fetch_page(cursor):
        return horizon_client.get(cursor)

    # or, for one-off calls:
    result = call_with_retry(lambda: horizon_client.get(cursor), policy=policy)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfigurationError(ValueError):
    """Raised when a `RetryPolicy` is constructed with invalid parameters."""


@dataclass(frozen=True)
class AttemptRecord:
    """Diagnostic record of a single retry attempt."""

    attempt_number: int
    delay_before_seconds: float
    exception: BaseException | None = None
    succeeded: bool = False

    def __str__(self) -> str:
        if self.succeeded:
            return f"attempt {self.attempt_number}: succeeded"
        return (
            f"attempt {self.attempt_number}: raised "
            f"{type(self.exception).__name__}: {self.exception} "
            f"(waited {self.delay_before_seconds:.3f}s before this attempt)"
        )


@dataclass
class AttemptHistory:
    """Full record of every attempt made for one `call_with_retry` invocation."""

    records: list[AttemptRecord] = field(default_factory=list)

    def add(self, record: AttemptRecord) -> None:
        self.records.append(record)

    @property
    def attempt_count(self) -> int:
        return len(self.records)

    @property
    def total_delay_seconds(self) -> float:
        return sum(r.delay_before_seconds for r in self.records)

    def render(self) -> str:
        return "\n".join(f"  - {r}" for r in self.records)


class RetryExhaustedError(Exception):
    """Raised when every attempt allowed by a `RetryPolicy` has failed.

    Wraps the final exception (via `__cause__`, standard chaining) and the
    full `AttemptHistory` so the failure log names every attempt made, its
    delay, and what it raised -- not just the last one.
    """

    def __init__(self, history: AttemptHistory, last_exception: BaseException):
        self.history = history
        self.last_exception = last_exception
        message = (
            f"retry exhausted after {history.attempt_count} attempt(s), "
            f"total delay {history.total_delay_seconds:.3f}s. "
            f"Attempt log:\n{history.render()}"
        )
        super().__init__(message)
        self.__cause__ = last_exception


@dataclass(frozen=True)
class RetryPolicy:
    """Typed contract describing how a retryable call should be retried.

    Args:
        max_attempts: Total attempts allowed, including the first (must be >= 1).
        base_delay_seconds: Delay before the second attempt (must be > 0).
        max_delay_seconds: Upper bound on any single delay.
        multiplier: Exponential growth factor applied per subsequent attempt.
        jitter: "full" (random in [0, delay)), "equal" (delay/2 + random in
            [0, delay/2)), or "none" (no randomization).
        retryable_exceptions: Exception types that trigger a retry; any other
            exception propagates immediately without consuming an attempt.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    multiplier: float = 2.0
    jitter: str = "full"
    retryable_exceptions: tuple[type[BaseException], ...] = (Exception,)

    _VALID_JITTER = ("full", "equal", "none")

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise RetryConfigurationError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.base_delay_seconds <= 0:
            raise RetryConfigurationError(
                f"base_delay_seconds must be > 0, got {self.base_delay_seconds}"
            )
        if self.max_delay_seconds < self.base_delay_seconds:
            raise RetryConfigurationError(
                "max_delay_seconds must be >= base_delay_seconds "
                f"({self.max_delay_seconds} < {self.base_delay_seconds})"
            )
        if self.multiplier <= 1.0:
            raise RetryConfigurationError(f"multiplier must be > 1.0, got {self.multiplier}")
        if self.jitter not in self._VALID_JITTER:
            raise RetryConfigurationError(
                f"jitter must be one of {self._VALID_JITTER}, got {self.jitter!r}"
            )
        if not self.retryable_exceptions:
            raise RetryConfigurationError("retryable_exceptions must be non-empty")

    def delay_for_attempt(self, attempt_number: int, rng: random.Random | None = None) -> float:
        """Returns the delay in seconds before the given attempt (1-indexed).

        Attempt 1 always has zero delay. Attempt N (N > 1) has an exponential
        delay based on attempt N-1, capped at `max_delay_seconds`, then
        randomized according to `jitter`.
        """
        if attempt_number <= 1:
            return 0.0
        rng = rng or random
        raw = min(
            self.base_delay_seconds * (self.multiplier ** (attempt_number - 2)),
            self.max_delay_seconds,
        )
        if self.jitter == "none":
            return raw
        if self.jitter == "equal":
            half = raw / 2
            return half + rng.uniform(0, half)
        return rng.uniform(0, raw)  # "full"


def call_with_retry(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
    on_attempt: Callable[[AttemptRecord], None] | None = None,
) -> T:
    """Executes `fn` under `policy`, retrying on `policy.retryable_exceptions`.

    Args:
        fn: Zero-argument callable to execute. Wrap job-specific args with a
            lambda or `functools.partial`.
        policy: Retry policy; defaults to `RetryPolicy()`.
        sleep: Injectable sleep function, for deterministic tests.
        rng: Injectable `random.Random` instance, for deterministic tests.
        on_attempt: Optional callback invoked with each `AttemptRecord`,
            useful for metrics/logging integration beyond the default logger.

    Returns:
        The return value of the first successful call.

    Raises:
        RetryExhaustedError: every attempt raised a retryable exception.
        The original exception: `fn` raised something not in
            `policy.retryable_exceptions` -- fails fast, no retry consumed.
    """
    policy = policy or RetryPolicy()
    history = AttemptHistory()

    for attempt_number in range(1, policy.max_attempts + 1):
        delay = policy.delay_for_attempt(attempt_number, rng=rng)
        if delay > 0:
            sleep(delay)
        try:
            result = fn()
        except policy.retryable_exceptions as exc:
            record = AttemptRecord(
                attempt_number=attempt_number, delay_before_seconds=delay, exception=exc
            )
            history.add(record)
            if on_attempt:
                on_attempt(record)
            logger.warning("retry attempt failed: %s", record)
            if attempt_number == policy.max_attempts:
                raise RetryExhaustedError(history, exc) from exc
            continue
        else:
            record = AttemptRecord(
                attempt_number=attempt_number, delay_before_seconds=delay, succeeded=True
            )
            history.add(record)
            if on_attempt:
                on_attempt(record)
            return result

    raise AssertionError("unreachable: loop always returns or raises")  # pragma: no cover


def retry_with_backoff(
    policy: RetryPolicy | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of `call_with_retry` for wrapping ingestion job functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(
                lambda: func(*args, **kwargs), policy=policy, sleep=sleep, rng=rng
            )

        return wrapper

    return decorator


# Pre-built policy for network-bound ingestion calls (Horizon, S3, Kafka):
# transient I/O errors and timeouts are retryable; everything else fails fast.
NETWORK_RETRY_POLICY = RetryPolicy(
    max_attempts=5,
    base_delay_seconds=0.5,
    max_delay_seconds=30.0,
    multiplier=2.0,
    jitter="full",
    retryable_exceptions=(OSError, TimeoutError, ConnectionError),
)
