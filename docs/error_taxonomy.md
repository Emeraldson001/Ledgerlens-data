# Error Taxonomy

## Overview

Failures across ingestion, feature engineering, model inference, streaming,
and storage previously surfaced as bare `ValueError` / `RuntimeError`
instances with ad-hoc messages, forcing on-call to read a stack trace and
guess which subsystem failed and whether retrying makes sense.

`utils/errors.py` defines a small hierarchy rooted at `LedgerLensError`.
Every raised error carries a namespaced `code`, a `category`, structured
`context`, an optional `remediation` hint, a `retryable` flag, and the
original triggering exception (chained via `raise ... from cause`).

| Subclass | Code prefix | Category | Default retryable |
|---|---|---|---|
| `IngestionError` | `ING` | ingestion | yes |
| `ValidationError` | `VAL` | validation | no |
| `TransformError` | `XFM` | transform | no |
| `ModelError` | `MDL` | model | no |
| `StorageError` | `STO` | storage | yes |
| `ConfigurationError` | `CFG` | configuration | no |
| `ExternalServiceError` | `EXT` | external_service | yes |
| `StreamingError` | `STR` | streaming | yes |

## Contract

- `code` is `{PREFIX}-{suffix}` (e.g. `ING-002`) — stable, greppable across
  logs, dashboards, and runbooks. Choose a suffix per distinct failure mode
  within a module and keep it stable once used in a runbook.
- `to_dict()` returns a JSON-serialisable structure for structured logging
  or API error bodies.
- `wrap_errors(ErrorCls, suffix, context=..., remediation=...)` converts any
  exception raised inside the `with` block into the taxonomy, preserving
  the original as `.cause` and via native `__cause__` chaining. Already-typed
  exceptions and anything in `exclude=` pass through unmodified so nested
  `wrap_errors` blocks don't double-wrap.
- `format_diagnostic(exc)` renders the full cause chain (all taxonomy
  wrappers plus the root non-taxonomy exception) as a readable multi-line
  string for incident channels/runbooks.

## Usage

```python
from utils.errors import IngestionError, TransformError, wrap_errors

raise IngestionError(
    "002",
    "trade record missing required field 'amount'",
    context={"source_file": path, "row": row_index},
    remediation="Check the upstream Horizon export for schema drift.",
)

with wrap_errors(TransformError, "001", context={"pair": pair}):
    compute_features(df)
```

```python
from utils.errors import format_diagnostic

try:
    run_pipeline()
except LedgerLensError as exc:
    logger.error(format_diagnostic(exc))
    if exc.retryable:
        schedule_retry()
```

## Validation

```
pytest tests/test_error_taxonomy.py -v
```

Covers: code/category assignment, per-category retryable defaults and
overrides, message rendering, `to_dict()` structure, `wrap_errors` chaining
and pass-through behavior, `exclude=`, and `format_diagnostic` walking a
multi-level cause chain.

## Design tradeoffs / follow-ups

- This module is additive: existing code paths that raise bare
  `ValueError`/`RuntimeError` are unaffected, and nothing in the codebase
  was migrated to the new hierarchy as part of this change to avoid a
  broad, risky refactor across many modules. Adoption is expected
  incrementally, module by module, starting with ingestion and streaming
  where retryability matters most.
- Codes are assigned by the caller at the raise site rather than from a
  central registry; a follow-up could add a lint/test that scans for
  duplicate `(prefix, suffix)` pairs used with different messages, to keep
  codes meaningful as adoption grows.
