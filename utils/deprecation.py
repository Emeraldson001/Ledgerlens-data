"""Structured deprecation decorator + registry for public modules (Issue #511).

Prior to this, deprecations in the codebase (e.g.
`detection/wallet_graph.py::funding_source_similarity`) were expressed by
hand: a manual `warnings.warn(..., DeprecationWarning)` call plus a
hand-written `.. deprecated::` docstring note, with no machine-checkable
record of *when* the symbol is allowed to actually be removed. That pattern
still works and is not required to change, but new deprecations should
prefer the `@deprecated` decorator below so `scripts/check_deprecation_policy.py`
can enforce the policy (removal version declared, not already past due) in
CI instead of relying on manual review.

Usage::

    from utils.deprecation import deprecated

    @deprecated(reason="Use GNNEncoder embeddings instead.", removal_version="0.4.0")
    def funding_source_similarity(wallet, graph):
        ...
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import TypeVar

T = TypeVar("T", bound=Callable)


@dataclass(frozen=True)
class DeprecationRecord:
    """Structured metadata attached to a deprecated symbol."""

    name: str
    module: str
    reason: str
    removal_version: str
    replacement: str | None = None


#: Every symbol decorated with `@deprecated` in this process, in decoration order.
_REGISTRY: list[DeprecationRecord] = []


def get_registered_deprecations() -> list[DeprecationRecord]:
    """Return all deprecations registered so far in this process (for introspection/testing)."""
    return list(_REGISTRY)


def deprecated(
    *, reason: str, removal_version: str, replacement: str | None = None
) -> Callable[[T], T]:
    """Mark a public function or class as deprecated.

    Emits a `DeprecationWarning` on every call, appends a structured
    ``.. deprecated:: <removal_version>`` note to the wrapped callable's
    docstring, records a :class:`DeprecationRecord` in the module registry,
    and stores the same metadata on `wrapped.__deprecated__` for programmatic
    inspection.

    Args:
        reason: Why the symbol is deprecated (shown in the warning message).
        removal_version: The `ledgerlens-data` version (matching the
            `pyproject.toml` ``[project] version`` scheme, e.g. ``"0.4.0"``)
            in which the symbol is planned to be removed.
        replacement: Optional name of the symbol to use instead.

    Raises:
        ValueError: If `reason` or `removal_version` is empty — a deprecation
            without either is not actionable for callers.
    """
    if not reason:
        raise ValueError("deprecated() requires a non-empty 'reason'.")
    if not removal_version:
        raise ValueError("deprecated() requires a non-empty 'removal_version'.")

    def decorator(func: T) -> T:
        record = DeprecationRecord(
            name=func.__qualname__,
            module=func.__module__,
            reason=reason,
            removal_version=removal_version,
            replacement=replacement,
        )
        _REGISTRY.append(record)

        message = (
            f"{func.__qualname__} is deprecated and will be removed in "
            f"{removal_version}: {reason}"
        )
        if replacement:
            message += f" Use {replacement} instead."

        @wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        doc_note = f"\n\n.. deprecated:: {removal_version}\n    {reason}"
        if replacement:
            doc_note += f" Use ``{replacement}`` instead."
        wrapper.__doc__ = (func.__doc__ or "") + doc_note
        wrapper.__deprecated__ = record
        return wrapper  # type: ignore[return-value]

    return decorator
