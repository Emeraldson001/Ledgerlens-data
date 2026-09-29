# Public API Compatibility Checks

## Problem

Seven packages (`analysis`, `detection`, `ingestion`, `privacy`,
`reporting`, `streaming`, `utils`) declare an explicit `__all__` in their
`__init__.py` -- that's the intended public contract for anything doing
`from detection import ...`, whether the caller is another package in this
repo, a notebook, `run_pipeline.py`, or an external consumer that depends on
`ledgerlens-data` as a library. Nothing previously stopped an edit to one of
those exported functions' signatures from silently breaking every caller.

## Design

- **`scripts/check_api_compatibility.py`** statically extracts the public
  surface of each package with `ast`: for functions, the parameter list
  (names, annotations, defaults, `*args`/`**kwargs`); for classes, the
  `__init__` signature plus every public (non-underscore) method's
  signature. It resolves both symbols defined directly in `__init__.py` and
  symbols re-exported from a submodule (`from .core import Widget` /
  `from detection.benford_engine import compute_benford_metrics`), matching
  both import styles already used across the repo's `__init__.py` files.
- **`tests/fixtures/api_baseline.json`** is the checked-in snapshot the
  current tree is compared against. It was generated with
  `python scripts/check_api_compatibility.py --update-baseline`.
- Comparisons only flag **removals** and **signature changes** as failures;
  adding a new exported symbol is never a break and is not reported.

## Validation

```
python scripts/check_api_compatibility.py                  # compare against baseline
python scripts/check_api_compatibility.py --package utils  # scope to one package
make check-api-compat                                       # same, via Makefile
pytest tests/test_api_compatibility.py -q                   # unit tests
```

To accept an intentional public API change:

```
python scripts/check_api_compatibility.py --update-baseline
git diff tests/fixtures/api_baseline.json   # review exactly what changed
```

The CI workflow runs the check on every push/PR, and
`tests/test_api_compatibility.py::TestRealBaselineFixture` re-runs the same
comparison as a pytest assertion so `pytest -q` alone also catches drift.

## Tradeoffs / follow-up

- Extraction is static (`ast`-only): it does not execute the target modules,
  so it stays fast and has no dependency on the heavy ML/streaming
  libraries some of these packages pull in at import time. The tradeoff is
  it can't resolve dynamically-constructed exports (e.g. `registry`, a
  module-level instance in `utils/__init__.py`, is recorded as
  `"kind": "unresolved"` rather than crashing -- consistent unresolved
  status on both sides of a comparison is not itself a failure).
- Only packages with an explicit `__all__` are covered. If a package
  intends its `__init__.py` exports to be a stable contract, add it to
  `PUBLIC_PACKAGES` in `scripts/check_api_compatibility.py`.
- Signature changes are all treated as breaking today (including e.g. only
  adding an optional keyword argument). A future iteration could classify
  changes as additive/breaking automatically; for now the diff is surfaced
  for human review via `--update-baseline`.
