# Model Artifact Backward Compatibility

This document describes the backward compatibility contract for stored model
artifacts (`model_metadata.json` + `metrics.json`, written by
`detection.model_training.save_training_artifacts`) and the validation layer
that enforces it, added under Issue #510.

## Why this exists

`ModelInferenceEngine` (`detection/model_inference.py`) trusts the
`feature_schema_hash`/`feature_columns` recorded in `model_metadata.json` to
detect feature drift at scoring time. Archived versions
(`scripts/list_model_versions.py`, `models/archive/<version>/`) are compared
by downstream tooling and dashboards. None of that is useful if a newly
trained artifact can silently replace an older one with:

- feature columns the running inference code still depends on removed,
- a `ledgerlens_version` that regresses instead of advancing, or
- recorded metrics (e.g. `auc_roc`) that regress well beyond noise.

This mirrors the compatibility contract LedgerLens already enforces for the
Avro trade schema — see [`data/schema_evolution.md`](../data/schema_evolution.md)
— applied to trained-model artifacts instead of message schemas.

## Compatibility rules

| Change | Result |
|---|---|
| Required metadata field (`feature_schema_hash`, `feature_columns`, `ledgerlens_version`) missing from the candidate | ❌ Breaking |
| Feature column present in the baseline but removed from the candidate | ❌ Breaking |
| Feature column added in the candidate | ⚠️ Warning (informational) |
| `ledgerlens_version` regresses (candidate < baseline) | ❌ Breaking |
| Per-model metric (default `auc_roc`) regresses beyond `--max-metric-drop` (default `0.02`) | ❌ Breaking |
| Per-model metric regresses within the allowed budget | ⚠️ Warning |

## Usage

```bash
# Compare the current models/ directory against an archived baseline version
python -m scripts.check_artifact_compatibility \
    --baseline-dir models/archive/2026-06-01 \
    --candidate-dir models

# Or via the Makefile (defaults to the most recently archived version)
make validate-artifacts
```

If no baseline artifact exists yet (e.g. `models/archive/` is empty or the
requested version was never archived), the check is skipped and exits `0` —
there is nothing to be backward compatible with.

The underlying reusable API is
[`detection.artifact_compatibility.check_backward_compatibility`](../detection/artifact_compatibility.py),
which returns a `CompatibilityReport` (`report.compatible`, `report.breaking`,
`report.warnings`) for programmatic use, e.g. from
`scripts/retrain_if_drifted.py` before promoting a shadow model to production.

## Exit codes (`scripts/check_artifact_compatibility.py`)

- `0` — compatible, or no baseline to compare against.
- `1` — one or more breaking incompatibilities found.
- `2` — the candidate directory is missing required artifact files.
