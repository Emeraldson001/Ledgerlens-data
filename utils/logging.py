"""Structured logging and correlation setup shared across the pipeline.

Usage:
    from utils.logging import get_logger

    logger = get_logger(__name__)
    logger.info("Loaded %d trades", len(trades_df))

Correlation IDs
---------------
When the ``CorrelationFilter`` is active (installed automatically), every log
record carries ``correlation_id``, ``stage``, ``pair_id``, and ``wallet``
attributes.  In JSON mode these are included in the output automatically.
In text mode they are appended as ``[correlation_id=<id> stage=<stage>]``.

Secret redaction
-----------------
All formatters route their final output through ``sanitize_text`` so secrets
(passwords, API keys, tokens) never reach log sinks.
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any

from config import config
from utils.secrets import sanitize_text

_CONFIGURED = False
_CORRELATION_ID_HEADER = "x-correlation-id"
_PIPELINE_STAGE_HEADER = "x-pipeline-stage"
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_pipeline_stage: ContextVar[str | None] = ContextVar("pipeline_stage", default=None)
_context_fields: ContextVar[dict[str, Any] | None] = ContextVar("log_context_fields", default=None)


def new_correlation_id() -> str:
    """Return a globally unique, log-friendly correlation identifier."""
    return uuid.uuid4().hex


def get_log_context() -> dict[str, Any]:
    """Return a copy of the context currently attached to log records."""
    context = dict(_context_fields.get() or {})
    correlation_id = _correlation_id.get()
    pipeline_stage = _pipeline_stage.get()
    if correlation_id is not None:
        context["correlation_id"] = correlation_id
    if pipeline_stage is not None:
        context["pipeline_stage"] = pipeline_stage
    return context


@contextmanager
def log_context(
    *,
    correlation_id: str | None = None,
    pipeline_stage: str | None = None,
    **fields: Any,
) -> Iterator[str]:
    """Bind correlation metadata for all logs emitted inside the context.

    Nested contexts inherit unspecified values and restore their parent values
    on exit. The yielded value is the active correlation ID; when none is
    supplied or inherited, a new ID is generated.
    """
    active_correlation_id = correlation_id or _correlation_id.get() or new_correlation_id()
    correlation_token: Token[str | None] = _correlation_id.set(active_correlation_id)
    stage_token: Token[str | None] | None = None
    if pipeline_stage is not None:
        stage_token = _pipeline_stage.set(pipeline_stage)
    fields_token = _context_fields.set({**(_context_fields.get() or {}), **fields})
    try:
        yield active_correlation_id
    finally:
        _context_fields.reset(fields_token)
        if stage_token is not None:
            _pipeline_stage.reset(stage_token)
        _correlation_id.reset(correlation_token)


def correlation_headers(correlation_id: str | None = None) -> list[tuple[str, bytes]]:
    """Encode the active correlation context as transport-safe headers."""
    cid = correlation_id or _correlation_id.get() or new_correlation_id()
    headers = [(_CORRELATION_ID_HEADER, cid.encode("utf-8"))]
    stage = _pipeline_stage.get()
    if stage is not None:
        headers.append((_PIPELINE_STAGE_HEADER, stage.encode("utf-8")))
    return headers


def correlation_id_from_headers(
    headers: Mapping[str, str | bytes] | list[tuple[str, str | bytes]] | None,
) -> str | None:
    """Read a correlation ID from HTTP- or Kafka-style headers."""
    if not headers:
        return None
    items = headers.items() if isinstance(headers, Mapping) else headers
    for name, value in items:
        if name.lower() != _CORRELATION_ID_HEADER:
            continue
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return str(value)
    return None


class _CorrelationFilter(logging.Filter):
    """Inject the current correlation fields into every emitted log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in get_log_context().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


# ── Correlation-enriched JSON formatter ──────────────────────────────────────

_CORRELATION_FIELDS = ("correlation_id", "stage", "pair_id", "wallet")


class SecretsRedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        return sanitize_text(original)


class CorrelationJsonFormatter:
    """JSON formatter that includes correlation fields when present.

    Falls back to the standard ``pythonjsonlogger`` if the filter has not
    been installed (i.e. the record lacks correlation attributes).
    """

    def __init__(self, base_formatter: Any) -> None:
        self._base = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        for field_name in _CORRELATION_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                setattr(record, field_name, value)
        return sanitize_text(self._base.format(record))


class CorrelationTextFormatter(logging.Formatter):
    """Text formatter that appends correlation fields to the log line."""

    _CORR_FMT = "[correlation_id={cid} stage={stage}]"

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
    ) -> None:
        super().__init__(fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        cid = getattr(record, "correlation_id", None)
        stage = getattr(record, "stage", None)
        if cid or stage:
            parts = []
            if cid:
                parts.append(f"correlation_id={cid}")
            if stage:
                parts.append(f"stage={stage}")
            pair_id = getattr(record, "pair_id", None)
            if pair_id:
                parts.append(f"pair_id={pair_id}")
            wallet = getattr(record, "wallet", None)
            if wallet:
                # Truncate wallet for readability in text mode
                short = wallet[:8] + "..." if len(wallet) > 11 else wallet
                parts.append(f"wallet={short}")
            base = f"{base} [{' '.join(parts)}]"
        return sanitize_text(base)


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Install the correlation filter on the root logger so every handler
    # receives correlation-enriched records.
    from utils.correlation import CorrelationFilter

    corr_filter = CorrelationFilter()
    root_logger.addFilter(corr_filter)

    # Logs must stay off stdout so callers can pipe a script's stdout
    # (e.g. JSON results) without status/info noise mixed in.
    handler = logging.StreamHandler(sys.stderr)
    handler.addFilter(_CorrelationFilter())

    if config.LOG_FORMAT == "json":
        try:
            from pythonjsonlogger import jsonlogger

            base_formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
            formatter = CorrelationJsonFormatter(base_formatter)
        except ImportError:
            formatter = CorrelationTextFormatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
    else:
        formatter = CorrelationTextFormatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )

    handler.setFormatter(formatter)
    handler.addFilter(corr_filter)

    # Remove existing handlers to avoid duplicates
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    root_logger.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for `name` (typically `__name__`)."""
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """Override the root logger's verbosity, e.g. from a CLI --log-level flag."""
    _configure()
    logging.getLogger().setLevel(level.upper())


def setup_logger(name: str = "ledgerlens", level: int = logging.INFO) -> logging.Logger:
    """Standalone logger with secret redaction, independent of the root config."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = SecretsRedactingFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
