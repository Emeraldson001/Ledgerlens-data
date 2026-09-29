"""Retry/backoff decorator for flaky network calls (Horizon API)."""

import time
from collections.abc import Callable
from functools import wraps
from typing import Concatenate, ParamSpec, TypeVar

from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
P = ParamSpec("P")


def retry_with_backoff(
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[Concatenate[P], T]], Callable[Concatenate[P], T]]:
    """Retry a callable with exponential backoff on the given exceptions.

    Preserves the wrapped function's signature via ParamSpec, ensuring
    IDE autocomplete and type checking work correctly for wrapped functions.

    Args:
        max_attempts: Total number of attempts (including the first) before
            the last exception is re-raised.
        base_delay_seconds: Delay before the first retry; doubled (by
            `backoff_factor`) on each subsequent attempt.
        backoff_factor: Multiplier applied to the delay after each retry.
        exceptions: Exception types that should trigger a retry.

    Returns:
        A decorator that wraps a callable and retries it with exponential backoff.
    """

    def decorator(func: Callable[Concatenate[P], T]) -> Callable[Concatenate[P], T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            delay = base_delay_seconds
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                        func.__name__,
                        attempt,
                        max_attempts,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            raise AssertionError("unreachable")  # pragma: no cover

        return wrapper

    return decorator
