"""Rate-limited, retried dispatch for Horizon REST calls.

Every discrete Horizon REST request in the ingestion layer should go
through `fetch()` so the global per-worker rate limit and the 429 backoff
policy are applied consistently, instead of each call site reimplementing
its own throttling.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

from config import config
from ingestion.exceptions import SourceUnavailableError
from ingestion.rate_limiter import TokenBucketLimiter
from monitoring.ingestion_metrics import emit_ingestion_failure, emit_ingestion_success
from utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

_limiter: TokenBucketLimiter | None = None


def get_limiter() -> TokenBucketLimiter:
    """Process-wide limiter instance, lazily created from config."""
    global _limiter
    if _limiter is None:
        _limiter = TokenBucketLimiter(
            capacity=config.HORIZON_MAX_RPS, refill_rate_per_sec=config.HORIZON_MAX_RPS
        )
    return _limiter


class HorizonRateLimitExceeded(SourceUnavailableError, RuntimeError):
    """Raised when HORIZON_MAX_RETRIES consecutive 429s are exhausted.

    Retains the ``RuntimeError`` base so existing callers catching it keep
    working; ``SourceUnavailableError`` places it in the ingestion taxonomy.
    """


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) if response is not None else None


def fetch(
    call: Callable[[], T],
    limiter: TokenBucketLimiter | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    jitter_fn: Callable[[float, float], float] = random.uniform,
) -> T:
    """Run `call()` after acquiring a rate-limit token, retrying on 429.

    On a 429 response, backs off exponentially (1s, 2s, 4s, ... capped at
    60s) with +/-20% jitter and retries up to `config.HORIZON_MAX_RETRIES`
    times. Any other exception -- including other 4xx statuses such as 403
    -- propagates immediately without a retry.
    """
    limiter = limiter or get_limiter()
    base_delay = 1.0
    max_delay = 60.0

    for attempt in range(1, config.HORIZON_MAX_RETRIES + 1):
        limiter.acquire()
        started_at = time.perf_counter()
        try:
            result = call()
        except Exception as exc:  # noqa: BLE001
            # Broad catch justified: any exception from Horizon needs to be examined
            # to determine if it's a 429 (retryable rate limit) or another error
            # (propagate immediately). Logging and metrics emission happen for all.
            emit_ingestion_failure(
                "horizon",
                exc,
                stage="fetch",
                duration_seconds=time.perf_counter() - started_at,
            )
            if _status_code(exc) != 429:
                raise
            if attempt == config.HORIZON_MAX_RETRIES:
                raise HorizonRateLimitExceeded(
                    f"Horizon returned 429 on all {attempt} attempts",
                    source="horizon_fetcher.fetch",
                    reason=str(exc),
                ) from exc

            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            jitter = delay * 0.2
            delay = max(0.0, delay + jitter_fn(-jitter, jitter))
            logger.warning(
                "Horizon 429 (attempt %d/%d) — retrying in %.2fs",
                attempt,
                config.HORIZON_MAX_RETRIES,
                delay,
            )
            sleep_fn(delay)
        else:
            emit_ingestion_success(
                "horizon",
                stage="fetch",
                record_count=_record_count(result),
                duration_seconds=time.perf_counter() - started_at,
            )
            return result

    raise AssertionError("unreachable")  # pragma: no cover


def _record_count(result: object) -> int:
    """Best-effort count for a Horizon page without coupling to its schema."""
    if not isinstance(result, dict):
        return 1
    embedded = result.get("_embedded")
    if not isinstance(embedded, dict):
        return 1
    records = embedded.get("records")
    return len(records) if isinstance(records, list) else 1
