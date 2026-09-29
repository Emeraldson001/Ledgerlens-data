"""Correlation context for structured logs across pipeline stages.

Provides thread-local storage for correlation IDs and pipeline stage metadata,
enabling operators to trace a trade event as it flows through ingestion →
detection → scoring → alerting.

Usage::

    from utils.correlation import correlation_context, current_correlation_id

    # Set a correlation ID for the current thread (typically at pipeline entry)
    with correlation_context("trade-abc-123", stage="ingestion", pair_id="USDC:.../XLM:native"):
        logger.info("Processing trade")  # log record gets correlation_id, stage, pair_id

    # Or use the decorator style for pipeline stages
    @correlation_context.wrap(stage="detection")
    def score_wallet(wallet, features):
        logger.info("Scoring wallet")  # inherits correlation_id from caller

    # Access the current correlation ID from anywhere in the call stack
    cid = current_correlation_id()  # -> "trade-abc-123" or None

The correlation fields are automatically injected into every log record when
the ``CorrelationFilter`` is installed on the root logger (done by
``utils.logging._configure()``).
"""

from __future__ import annotations

import contextvars
import functools
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ── Correlation context vars ─────────────────────────────────────────────────

_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)
_correlation_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_stage", default=None
)
_correlation_pair_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_pair_id", default=None
)
_correlation_wallet: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_wallet", default=None
)
_correlation_extra: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "correlation_extra", default=None
)

# ── Pipeline stage constants ─────────────────────────────────────────────────


class PipelineStage:
    """Canonical names for the LedgerLens detection pipeline stages."""

    INGESTION = "ingestion"
    DETECTION = "detection"
    STREAMING = "streaming"
    SCORING = "scoring"
    ALERTING = "alerting"
    MONITORING = "monitoring"
    PERSISTENCE = "persistence"
    ONCHAIN = "onchain"

    ALL = (INGESTION, DETECTION, STREAMING, SCORING, ALERTING, MONITORING, PERSISTENCE, ONCHAIN)


# ── Correlation data class ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CorrelationContext:
    """Immutable snapshot of the current correlation context.

    Created by :func:`snapshot` to capture a consistent view of all
    correlation fields at a point in time.
    """

    correlation_id: str | None = None
    stage: str | None = None
    pair_id: str | None = None
    wallet: str | None = None
    extra: dict[str, Any] | None = None


# ── Public API ──────────────────────────────────────────────────────────────


def current_correlation_id() -> str | None:
    """Return the correlation ID for the current context, or ``None``."""
    return _correlation_id.get()


def current_stage() -> str | None:
    """Return the pipeline stage for the current context, or ``None``."""
    return _correlation_stage.get()


def current_pair_id() -> str | None:
    """Return the asset pair ID for the current context, or ``None``."""
    return _correlation_pair_id.get()


def current_wallet() -> str | None:
    """Return the wallet address for the current context, or ``None``."""
    return _correlation_wallet.get()


def current_extra() -> dict[str, Any] | None:
    """Return extra correlation fields for the current context, or ``None``."""
    return _correlation_extra.get()


def snapshot() -> CorrelationContext:
    """Capture a consistent snapshot of all correlation fields."""
    return CorrelationContext(
        correlation_id=current_correlation_id(),
        stage=current_stage(),
        pair_id=current_pair_id(),
        wallet=current_wallet(),
        extra=current_extra(),
    )


def generate_correlation_id() -> str:
    """Generate a new unique correlation ID.

    Uses a compact UUID4 format (32 hex chars, no dashes) suitable for
    embedding in log lines without excessive noise.
    """
    return uuid.uuid4().hex


def get_correlation_fields() -> dict[str, Any]:
    """Return all current correlation fields as a dict.

    Useful for injecting into structured log records.  Only non-None
    values are included.

    Returns:
        Dict with keys ``correlation_id``, ``stage``, ``pair_id``,
        ``wallet``, and any entries from ``extra``.
    """
    fields: dict[str, Any] = {}
    cid = current_correlation_id()
    if cid is not None:
        fields["correlation_id"] = cid
    stage = current_stage()
    if stage is not None:
        fields["stage"] = stage
    pair_id = current_pair_id()
    if pair_id is not None:
        fields["pair_id"] = pair_id
    wallet = current_wallet()
    if wallet is not None:
        fields["wallet"] = wallet
    extra = current_extra()
    if extra:
        fields.update(extra)
    return fields


# ── Context manager ─────────────────────────────────────────────────────────


class correlation_context:
    """Context manager / decorator that sets correlation fields.

    Supports two usage patterns:

    **Context manager** — enter/exit with automatic cleanup::

        with correlation_context("abc-123", stage="ingestion", pair_id="USDC:..."):
            logger.info("Processing")  # has correlation fields

        logger.info("After")  # correlation fields cleared

    **Decorator** — wrap a function so all log calls inside inherit the
    context::

        @correlation_context.wrap(stage="scoring")
        def score_wallet(wallet, features):
            logger.info("Scoring")  # has stage="scoring"
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        *,
        stage: str | None = None,
        pair_id: str | None = None,
        wallet: str | None = None,
        extra: dict[str, Any] | None = None,
        auto_generate: bool = True,
    ) -> None:
        if correlation_id is None and auto_generate:
            correlation_id = generate_correlation_id()
        self._cid = correlation_id
        self._stage = stage
        self._pair_id = pair_id
        self._wallet = wallet
        self._extra = extra
        self._token_cid: contextvars.Token | None = None
        self._token_stage: contextvars.Token | None = None
        self._token_pair: contextvars.Token | None = None
        self._token_wallet: contextvars.Token | None = None
        self._token_extra: contextvars.Token | None = None

    def __enter__(self) -> CorrelationContext:
        self._token_cid = _correlation_id.set(self._cid)
        self._token_stage = _correlation_stage.set(self._stage)
        self._token_pair = _correlation_pair_id.set(self._pair_id)
        self._token_wallet = _correlation_wallet.set(self._wallet)
        self._token_extra = _correlation_extra.set(self._extra)
        return snapshot()

    def __exit__(self, *exc: Any) -> None:
        if self._token_cid is not None:
            _correlation_id.reset(self._token_cid)
        if self._token_stage is not None:
            _correlation_stage.reset(self._token_stage)
        if self._token_pair is not None:
            _correlation_pair_id.reset(self._token_pair)
        if self._token_wallet is not None:
            _correlation_wallet.reset(self._token_wallet)
        if self._token_extra is not None:
            _correlation_extra.reset(self._token_extra)

    @classmethod
    def wrap(
        cls,
        correlation_id: str | None = None,
        *,
        stage: str | None = None,
        pair_id: str | None = None,
        wallet: str | None = None,
        extra: dict[str, Any] | None = None,
        auto_generate: bool = True,
    ) -> Callable:
        """Decorator factory that wraps a function in a correlation context.

        Usage::

            @correlation_context.wrap(stage="detection")
            def detect(wallet):
                logger.info("Running detection")  # inherits context
        """

        def decorator(fn: Callable) -> Callable:
            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                with cls(
                    correlation_id,
                    stage=stage,
                    pair_id=pair_id,
                    wallet=wallet,
                    extra=extra,
                    auto_generate=auto_generate,
                ):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator


# ── Log filter for structured logging integration ────────────────────────────


class CorrelationFilter:
    """Logging ``Filter`` that injects correlation fields into log records.

    Install on the root logger (or a specific handler) to make correlation
    fields available to log formatters::

        import logging
        from utils.correlation import CorrelationFilter

        root = logging.getLogger()
        root.addFilter(CorrelationFilter())

    After installation, every log record will have these attributes:
        - ``correlation_id`` (str or None)
        - ``stage`` (str or None)
        - ``pair_id`` (str or None)
        - ``wallet`` (str or None)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        fields = get_correlation_fields()
        record.correlation_id = fields.get("correlation_id")  # type: ignore[attr-defined]
        record.stage = fields.get("stage")  # type: ignore[attr-defined]
        record.pair_id = fields.get("pair_id")  # type: ignore[attr-defined]
        record.wallet = fields.get("wallet")  # type: ignore[attr-defined]
        return True


# ── Thread-safe propagation helpers ─────────────────────────────────────────


def propagate_to_thread(
    target_fn: Callable, correlation_id: str | None = None, **ctx: Any
) -> Callable:
    """Wrap a thread target to inherit the caller's correlation context.

    When spawning a background thread that should carry the same correlation
    ID as the parent::

        import threading
        from utils.correlation import propagate_to_thread

        def worker():
            logger.info("Background work")  # has same correlation_id

        t = threading.Thread(target=propagate_to_thread(worker, stage="streaming"))
        t.start()
    """
    snap = snapshot()
    effective_id = correlation_id or snap.correlation_id or generate_correlation_id()

    @functools.wraps(target_fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with correlation_context(
            effective_id,
            stage=ctx.get("stage", snap.stage),
            pair_id=ctx.get("pair_id", snap.pair_id),
            wallet=ctx.get("wallet", snap.wallet),
            extra=ctx.get("extra", snap.extra),
        ):
            return target_fn(*args, **kwargs)

    return wrapper
