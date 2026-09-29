# Module Dependency Rules

## Problem

Ledgerlens-data has ~15 top-level Python packages (`utils`, `config`,
`detection`, `ingestion`, `api`, `scripts`, `streaming`, ...) with no
machine-checked rule preventing accidental coupling -- e.g. a low-level
package like `utils` growing a dependency on `detection`, or a domain
package reaching into the HTTP API layer. That kind of drift is invisible
until someone tries to reuse a "shared" module in a lighter-weight context
and discovers it silently pulls in half the repo.

## Design

- **`config/module_boundaries.yml`** declares a three-layer architecture
  (`foundation` -> `domain` -> `entrypoint`) and which packages belong to
  each layer, plus an escape hatch (`forbidden_imports`) for one-off
  pairwise rules that don't fit the layer model (e.g. `streaming` must not
  import `api`, so streaming workers stay deployable independently of the
  HTTP process).
- **`scripts/check_module_dependencies.py`** statically parses every `.py`
  file in each declared package with `ast` (no execution, so it has no
  runtime/dependency requirements beyond PyYAML, which is already in
  `requirements.txt`), resolves imports to local top-level packages, and
  flags any import that reaches into a strictly higher layer or an
  explicitly forbidden package. Violations are reported as
  `path:lineno: message`, so they open directly in an editor or as a CI
  annotation.

## Validation

```
python scripts/check_module_dependencies.py                 # whole repo
python scripts/check_module_dependencies.py --package utils # scope to one package
make check-deps                                              # same, via Makefile
pytest tests/test_module_dependency_rules.py -q              # unit tests
```

Unit tests build synthetic package trees under `tmp_path` (via the
`fake_repo` fixture) so they exercise every rule type (layering violation,
same-layer import, higher-importing-lower, forbidden pair, non-local /
relative imports being ignored) without depending on the real repo's
current import graph -- plus a smoke test that loads the real
`config/module_boundaries.yml` and confirms the checker runs cleanly
against it. The CI workflow runs the real check on every push/PR.

## Tradeoffs / follow-up

- The layer model is coarse-grained (3 layers) by design: it catches the
  dependency inversions that matter most (foundation/entrypoint leakage)
  without requiring a full per-package dependency matrix that would need
  constant upkeep as packages are added.
- `forbidden_imports` currently has one seeded rule (`streaming -> api`).
  Add more pairwise rules here as specific decoupling requirements come up,
  rather than growing the layer count.
- Packages not yet assigned a layer (`tests`, `notebooks`, `data`,
  `templates`, `models`) are intentionally excluded; add them to a layer in
  `config/module_boundaries.yml` if they start being imported from
  production code paths.
