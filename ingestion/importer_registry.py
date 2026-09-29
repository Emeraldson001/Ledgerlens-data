"""Importer registry with registration-time capability conformance checks.

An importer declares `capabilities` (a set of names from `CAPABILITY_CONTRACTS`).
At `register()` each declared capability is exercised against a minimal
contract test; if any fails, registration is rejected with
`ImporterConformanceError` so broken importers fail fast rather than at
first use in production. See docs/importers.md.
"""

from collections.abc import Callable

import pandas as pd


class ImporterConformanceError(Exception):
    """Raised when an importer does not implement a declared capability."""


def _check_load_dataframe(importer) -> None:
    df = importer.load_dataframe(limit=1)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"load_dataframe returned {type(df).__name__}, expected DataFrame")


def _check_stream(importer) -> None:
    it = importer.stream(limit=1)
    if not hasattr(it, "__iter__"):
        raise TypeError("stream() must return an iterable")


def _check_incremental(importer) -> None:
    if not callable(getattr(importer, "load_since", None)):
        raise TypeError("missing callable load_since(cursor)")
    importer.load_since(None)


CAPABILITY_CONTRACTS: dict[str, Callable] = {
    "load_dataframe": _check_load_dataframe,
    "stream": _check_stream,
    "incremental": _check_incremental,
}

_REGISTRY: dict[str, object] = {}


def check_conformance(importer) -> None:
    name = getattr(importer, "name", type(importer).__name__)
    declared = set(getattr(importer, "capabilities", set()))
    if not declared:
        raise ImporterConformanceError(f"{name}: declares no capabilities")
    unknown = declared - CAPABILITY_CONTRACTS.keys()
    if unknown:
        raise ImporterConformanceError(f"{name}: unknown capabilities {sorted(unknown)}")
    for cap in sorted(declared):
        try:
            CAPABILITY_CONTRACTS[cap](importer)
        except Exception as exc:  # noqa: BLE001 - report any contract failure
            raise ImporterConformanceError(
                f"{name}: declared capability '{cap}' failed contract check: {exc}"
            ) from exc


def register(importer, name: str | None = None) -> None:
    name = name or getattr(importer, "name", type(importer).__name__)
    check_conformance(importer)
    _REGISTRY[name] = importer


def get(name: str):
    return _REGISTRY[name]


def registered() -> dict[str, object]:
    return dict(_REGISTRY)


def verify_all() -> list[str]:
    """Re-run conformance on every registered importer; return failures."""
    failures = []
    for imp in _REGISTRY.values():
        try:
            check_conformance(imp)
        except ImporterConformanceError as exc:
            failures.append(str(exc))
    return failures
