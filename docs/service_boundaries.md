# Typed Service Boundaries for Shared Utilities

## Problem

`utils/` (`circuit_breaker.py`, `retry.py`, `tracing.py`, `field_encryption.py`,
`logging.py`) is imported directly by almost every package in this repo
(`detection`, `ingestion`, `api`, `streaming`, `training`, ...). There was no
formal contract describing what these utilities guarantee, so a signature
change in one of them could silently break call sites elsewhere with no
static or CI signal.

## Design

`utils/boundaries.py` introduces:

- **Typed `Protocol` contracts** (`CircuitBreakerPort`, `RetryPolicyPort`,
  `TracerFactoryPort`, `FieldEncryptionPort`, `StructuredLoggerFactoryPort`)
  describing the public surface each utility must provide. These are
  `@runtime_checkable`, so they support both static (mypy) and runtime
  (`isinstance`) checks.
- **`ServiceRegistry`**, a minimal typed registry mapping a Protocol to a
  factory. Consumers resolve a port (`registry.resolve(CircuitBreakerPort)`)
  instead of importing the concrete module, which decouples call sites from
  implementation details and makes swapping in test doubles a one-line change.
- **`validate_service_boundaries()`**, which exercises every default binding
  and reports drift with actionable diagnostics (which port, which
  attribute is missing, where to fix it), rather than a bare `AttributeError`
  surfacing deep in an unrelated module.

## Validation

```
python scripts/check_service_boundaries.py   # CI entry point
make check-boundaries                        # same, via Makefile
pytest tests/test_service_boundaries.py -q   # unit tests
```

The CI workflow (`.github/workflows/ci.yml`) runs the check on every push and
pull request, before the main test suite.

## Tradeoffs / follow-up

- This intentionally does **not** force existing concrete classes
  (`CircuitBreaker`, etc.) to inherit from the protocols -- `Protocol` gives
  us structural typing, so existing code is unaffected and the migration to
  using `registry.resolve(...)` at call sites can happen incrementally.
- The registry is process-global (`utils.boundaries.registry`) for
  simplicity; a future iteration could scope it per-request if we need
  per-tenant overrides (see `config/tenant_config.py`).
- Only the five utilities with the widest fan-out are covered today. Adding
  a new shared utility should follow the pattern documented at the top of
  `utils/boundaries.py`.
