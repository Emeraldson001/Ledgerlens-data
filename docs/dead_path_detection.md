# Dead-Path Detection Reports

`analysis/dead_path_detector.py` (Issue #547) statically scans the source
tree for modules that look retired: no inbound Python import anywhere in
the source packages, scripts, or tests; no `if __name__ == "__main__":`
entry-point guard; and no textual reference from `Makefile`, CI workflows,
or docs. It never deletes or modifies anything — it only reports.

## Running it

```bash
make dead-path-report                                    # text report to stdout
python scripts/detect_dead_paths.py --format markdown \
    --output reports/dead_paths.md                        # markdown report to a file
python scripts/detect_dead_paths.py --format json \
    --output reports/dead_paths.json                      # machine-readable report
python scripts/detect_dead_paths.py --strict               # exit 1 if any candidates found
```

`reports/` is gitignored — generated reports are never committed.

## How a candidate is determined

A module is a candidate only if **all** of the following hold:

1. Zero Python `import`/`from ... import` references anywhere under the
   scanned source packages, `scripts/`, or `tests/` (including relative
   imports, resolved against the importing file's own package).
2. It has no `__main__` guard (i.e. it isn't a CLI entry point).
3. Its dotted module name or file path doesn't appear in `Makefile`,
   `.github/workflows/*.yml`, `docs/*.md`, `README.md`, or
   `pyproject.toml`.
4. It isn't listed in `analysis/dead_path_ignorelist.yaml`.

Every module scanned — not just candidates — is included in the JSON
report with its reference counts and every signal checked, so "why was/
wasn't this flagged" always has a concrete, inspectable answer.

## Ignorelist YAML schema

`analysis/dead_path_ignorelist.yaml` excludes known-intentional dead paths
from the report. The schema is intentionally minimal:

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `ignored` | mapping | yes | Top-level map of exclusions. |
| `<dotted.module.path>` | mapping key | yes | Exact dotted module name of the excluded module. Keys are matched exactly — globs and regex are **not** supported. |
| `<reason>` | string | yes | Free-text justification for the exclusion. A reason is required for every entry so no module is ignored silently. |

Comments (`#`) are allowed anywhere in the file (including inline, after a
value). Example entry:

```yaml
ignored:
  monitoring.capacity_metrics: >-
    Loaded only via importlib.import_module("monitoring.capacity_metrics")
    in tests/test_capacity_metrics.py (to exercise Prometheus metric
    registration on import), which the static import scanner cannot see.
```

## Known limitation: dynamic imports

This is a static, import-*statement* based analysis. It cannot see
`importlib.import_module(dynamic_string)` calls, string-based plugin
registration, or reflection. `monitoring.capacity_metrics` is a real
example: it's loaded only via `importlib.import_module("monitoring.capacity_metrics")`
in `tests/test_capacity_metrics.py`, so the scanner sees zero static
references. It's documented in `analysis/dead_path_ignorelist.yaml` rather
than silently mis-flagged.

If the scanner flags a module you know is reachable only dynamically, add
it to the ignorelist with a one-line reason — don't just ignore the
finding. If you're deleting a module the scanner flagged, still verify
with `git log`/`grep` first; this tool narrows the search, it doesn't
replace judgment.

## Local validation commands

```bash
pytest tests/test_dead_path_detector.py -v
```
