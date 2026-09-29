"""Traceable error taxonomy for data processing failures.

Historically, failures across ingestion, feature engineering, model
inference, streaming, and storage all surfaced as bare ``Exception`` /
``ValueError`` / ``RuntimeError`` instances with ad-hoc messages. That makes
triage slow: on-call has to read a stack trace and guess which subsystem
failed, whether it's retryable, and what to check first.

This module defines a small exception hierarchy rooted at
``LedgerLensError`` where every raised error carries:

- a namespaced ``code`` (e.g. ``"ING-001"``) that is stable and greppable
  across logs, dashboards, and runbooks;
- a ``category`` identifying the failing subsystem;
- structured ``context`` (arbitrary key/value diagnostic data — a wallet
  ID, a file path, a row index — whatever narrows down *where* to look);
- an optional ``remediation`` hint for the next human who sees it;
- a ``retryable`` flag so callers/callers' retry decorators can make an
  informed decision instead of blanket-retrying everything;
- the original triggering exception, chained via ``raise ... from cause``
  and also stored on ``.cause`` for structured serialisation.

Usage:
    from utils.errors import IngestionError, wrap_errors

    raise IngestionError(
        "002",
        "trade record missing required field 'amount'",
        context={"source_file": path, "row": row_index},
        remediation="Check the upstream Horizon export for schema drift.",
    )

    # Or convert arbitrary exceptions raised inside a block:
    with wrap_errors(TransformError, "001", context={"pair": pair}):
        compute_features(df)
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any


class ErrorCategory(StrEnum):
    INGESTION = "ingestion"
    VALIDATION = "validation"
    TRANSFORM = "transform"
    MODEL = "model"
    STORAGE = "storage"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    STREAMING = "streaming"


class LedgerLensError(Exception):
    """Base class for all taxonomy errors. Do not raise directly — use a subclass."""

    category: ErrorCategory = ErrorCategory.VALIDATION
    code_prefix: str = "GEN"
    default_retryable: bool = False

    def __init__(
        self,
        code_suffix: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        remediation: str | None = None,
        retryable: bool | None = None,
        cause: BaseException | None = None,
    ):
        self.code = f"{self.code_prefix}-{code_suffix}"
        self.message = message
        self.context = context or {}
        self.remediation = remediation
        self.retryable = self.default_retryable if retryable is None else retryable
        self.cause = cause
        super().__init__(self._render())
        if cause is not None:
            self.__cause__ = cause

    def _render(self) -> str:
        parts = [f"[{self.code}] {self.message}"]
        if self.context:
            ctx = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
            parts.append(f"(context: {ctx})")
        if self.remediation:
            parts.append(f"— {self.remediation}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Structured representation suitable for JSON logs or API error bodies."""
        return {
            "code": self.code,
            "category": self.category.value,
            "message": self.message,
            "context": self.context,
            "remediation": self.remediation,
            "retryable": self.retryable,
            "cause": repr(self.cause) if self.cause else None,
        }


class IngestionError(LedgerLensError):
    """Failures reading/parsing source data (Horizon, Kafka, files)."""

    category = ErrorCategory.INGESTION
    code_prefix = "ING"
    default_retryable = True


class ValidationError(LedgerLensError):
    """Data that was read successfully but fails schema/semantic validation."""

    category = ErrorCategory.VALIDATION
    code_prefix = "VAL"
    default_retryable = False


class TransformError(LedgerLensError):
    """Failures during feature engineering / data transformation."""

    category = ErrorCategory.TRANSFORM
    code_prefix = "XFM"
    default_retryable = False


class ModelError(LedgerLensError):
    """Failures during model training or inference."""

    category = ErrorCategory.MODEL
    code_prefix = "MDL"
    default_retryable = False


class StorageError(LedgerLensError):
    """Failures reading/writing durable storage (DB, Redis, filesystem)."""

    category = ErrorCategory.STORAGE
    code_prefix = "STO"
    default_retryable = True


class ConfigurationError(LedgerLensError):
    """Invalid or missing configuration detected at startup or runtime."""

    category = ErrorCategory.CONFIGURATION
    code_prefix = "CFG"
    default_retryable = False


class ExternalServiceError(LedgerLensError):
    """Failures calling out to an external service (Horizon API, Soroban RPC, etc)."""

    category = ErrorCategory.EXTERNAL_SERVICE
    code_prefix = "EXT"
    default_retryable = True


class StreamingError(LedgerLensError):
    """Failures in the streaming pipeline (Kafka consumer/producer, WS server)."""

    category = ErrorCategory.STREAMING
    code_prefix = "STR"
    default_retryable = True


@contextmanager
def wrap_errors(
    error_cls: type[LedgerLensError],
    code_suffix: str,
    *,
    context: dict[str, Any] | None = None,
    remediation: str | None = None,
    retryable: bool | None = None,
    exclude: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Convert any exception raised in the block into ``error_cls``.

    The original exception is preserved as both ``.cause`` on the new error
    and via Python's native exception chaining (``raise ... from``), so
    ``traceback.print_exc()`` still shows the true root cause underneath the
    taxonomy wrapper.

    Exceptions already of type ``error_cls`` (or listed in ``exclude``) pass
    through unmodified, so nested ``wrap_errors`` blocks don't double-wrap.
    """
    try:
        yield
    except error_cls:
        raise
    except exclude:
        raise
    except Exception as exc:
        raise error_cls(
            code_suffix,
            str(exc) or exc.__class__.__name__,
            context=context,
            remediation=remediation,
            retryable=retryable,
            cause=exc,
        ) from exc


def format_diagnostic(exc: BaseException) -> str:
    """Human-readable multi-line diagnostic, walking the full cause chain.

    Intended for on-call runbooks / incident channels where a single
    greppable code plus a readable root-cause chain is more useful than a
    raw Python traceback.
    """
    lines: list[str] = []
    current: BaseException | None = exc
    depth = 0
    while current is not None:
        indent = "  " * depth
        if isinstance(current, LedgerLensError):
            lines.append(f"{indent}[{current.code}] ({current.category.value}) {current.message}")
            if current.context:
                for k, v in current.context.items():
                    lines.append(f"{indent}    {k}: {v!r}")
            if current.remediation:
                lines.append(f"{indent}    remediation: {current.remediation}")
        else:
            lines.append(f"{indent}{current.__class__.__name__}: {current}")
        current = current.__cause__
        depth += 1
    return "\n".join(lines)
