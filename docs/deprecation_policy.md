# Deprecation Policy for Public Modules

This document describes the structured deprecation mechanism and the policy
check that enforces it, added under Issue #511.

## Background

Before this, deprecations were expressed by hand — see
`detection/wallet_graph.py::funding_source_similarity` /
`network_centrality` — using a manual `warnings.warn(..., DeprecationWarning)`
call plus a hand-written `.. deprecated::` Sphinx docstring note. That pattern
still works and existing deprecations were **not** changed to use the new
mechanism, but it gives no machine-checkable record of when a deprecated
symbol is actually allowed to be removed.

## Structured deprecation: `utils.deprecation.deprecated`

New deprecations should use the `@deprecated` decorator:

```python
from utils.deprecation import deprecated

@deprecated(reason="Use GNNEncoder embeddings instead.", removal_version="0.4.0")
def funding_source_similarity(wallet, graph):
    ...
```

This:

- emits a `DeprecationWarning` on every call with a consistent message,
- appends a structured `.. deprecated:: 0.4.0` note to the docstring,
- records a `DeprecationRecord` (`name`, `module`, `reason`, `removal_version`,
  `replacement`) in an in-process registry retrievable via
  `utils.deprecation.get_registered_deprecations()`,
- stores the same metadata on `func.__deprecated__` for programmatic use.

`removal_version` must match the `[project] version` scheme in
`pyproject.toml` (e.g. `"0.4.0"`).

## Policy check: `scripts/check_deprecation_policy.py`

Statically scans (via `ast`, without importing the packages) every public
top-level package — any repo-root directory with an `__init__.py`, excluding
`tests/` and `scripts/` — for:

1. **Past-due deprecations** — a `@deprecated(removal_version=...)` whose
   version is `<=` the current `pyproject.toml` version. It should have been
   removed, not just deprecated.
2. **Unstructured deprecations** — a function that calls
   `warnings.warn(..., DeprecationWarning)` directly must document a
   `.. deprecated::` note in its docstring (as the existing
   `wallet_graph.py` functions already do), so a reviewer has the same
   minimum information the structured decorator provides automatically.

```bash
python -m scripts.check_deprecation_policy
# or
make check-deprecations
```

Exits `0` when no violations are found, `1` otherwise, printing
`file:line: symbol: message` for each violation.
