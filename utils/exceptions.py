"""Base exception type shared across LedgerLens packages.

``LedgerLensError`` distinguishes errors raised deliberately by LedgerLens code
from stdlib/third-party exceptions that pass through unchanged. It carries an
optional ``context`` mapping so failures arrive at a log line or a dead-letter
queue with the structured detail needed to triage them.

This module intentionally has no repo-internal imports, so any package
(``ingestion/``, ``detection/``, ``reporting/``, ...) can adopt the base without
creating a dependency edge onto another domain package.

Usage:
    from utils.exceptions import LedgerLensError

    raise LedgerLensError("thing failed", context={"source": "loader"})

Domain packages should subclass this rather than raising it directly — see
``ingestion/exceptions.py`` for the ingestion taxonomy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class LedgerLensError(Exception):
    """Base class for errors raised by LedgerLens code.

    Args:
        message: Human-readable description; becomes ``str(exc)``.
        context: Optional structured detail for logging and triage. Copied on
            construction so later mutation of the caller's mapping cannot
            change the recorded context.
    """

    def __init__(self, message: str, *, context: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context) if context else {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({str(self)!r}, context={self.context!r})"
