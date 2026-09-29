# Importer conformance requirements

Importers are registered via `ingestion.importer_registry.register()`. Each
importer must define:

- `name: str` — unique registry key.
- `capabilities: set[str]` — non-empty, drawn from `CAPABILITY_CONTRACTS`.

At registration every declared capability is exercised by a minimal contract
test; any failure raises `ImporterConformanceError` and the importer is not
registered.

| Capability       | Contract |
|------------------|----------|
| `load_dataframe` | `load_dataframe(limit=1)` returns a `pandas.DataFrame` |
| `stream`         | `stream(limit=1)` returns an iterable |
| `incremental`    | `load_since(None)` is callable and does not raise |

Contract checks must be cheap and side-effect free (e.g. return an empty
frame when no source is configured).

CI: `.github/workflows/importer-conformance.yml` runs
`scripts/check_importer_conformance.py` on any PR touching importer files.
