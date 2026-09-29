# Source Package Integrity Checks

`utils/package_integrity.py` (Issue #540) runs a fast, dependency-free,
no-import structural sweep of the source tree before tests execute, and
reports every issue with a clear diagnosis pointing at the exact file.

## Why

A merge on `config.py` once silently dropped several attributes that call
sites elsewhere in the codebase still referenced (see the "Restored config
attributes" note near the bottom of `config.py`). That failure mode — a bad
merge or half-applied edit leaving a source tree that *looks* fine but is
structurally broken — doesn't show up as one clear test failure. It shows
up as a scatter of unrelated `AttributeError`/`ImportError` failures across
the suite, which is expensive to root-cause.

## What it checks

For every configured source package (`utils/package_integrity.py::DEFAULT_SOURCE_PACKAGES`):

* **Missing `__init__.py`** — a package directory (or nested directory)
  containing `.py` files but no `__init__.py`.
* **Unresolved merge conflict markers** — `<<<<<<<`, `=======`, `>>>>>>>`
  left in a committed file after a bad merge/rebase.
* **Syntax errors** — every `.py` file must parse with `ast.parse`.
* **Empty non-`__init__` modules** — a zero-byte `.py` file usually
  indicates a truncated merge or a botched save.

It never imports project code — purely filesystem + `ast` based — so it
needs no dependencies installed and has no side effects.

## When it runs

* Automatically, once, before any test collects — wired in via
  `pytest_sessionstart` in `tests/conftest.py`. A broken tree aborts the
  whole session immediately with a single readable report instead of
  cascading into unrelated collection/import errors.
* Standalone: `make check-integrity` or
  `python scripts/check_package_integrity.py`.
* In CI: a dedicated step in `.github/workflows/ci.yml` runs it
  independently of pytest, so the failure is visible even if someone later
  changes how the test suite is invoked.

## Local validation commands

```bash
make check-integrity
pytest tests/test_package_integrity.py -v
```
