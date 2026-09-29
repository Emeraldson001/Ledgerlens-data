# Evaluation Workflows for Detection Quality and Drift

## Overview

LedgerLens already has two independent evaluation primitives:

- `evaluation/backtest.py` -- replays a labelled dataset through the
  detection pipeline and reports precision/recall/F1/ROC-AUC/average
  precision, plus a per-asset-pair breakdown.
- `detection/drift_monitor.py` -- computes Population Stability Index
  (PSI) per feature against a reference distribution and flags features
  that have drifted.

Neither, on its own, answers the operational question that actually gates
a model promotion: *did quality regress, or has the feature distribution
drifted meaningfully, or both* -- against explicit, named thresholds, with
one report and a single pass/fail outcome. Getting that answer previously
meant running both tools separately and manually cross-referencing two
JSON files.

`evaluation/quality_drift_workflow.py` adds that as an orchestration layer
on top of both **unmodified** existing primitives:

- **`run_evaluation_workflow()`** -- runs `evaluation.backtest.run_backtest`
  against a "current" dataset, builds a reference feature distribution from
  a "reference" dataset (quantile-binned, so skewed features like trade
  amount don't collapse into one bin), computes PSI via
  `detection.drift_monitor.DriftMonitor`, evaluates both against
  caller-supplied gates, and writes one `evaluation_report.json`.
- **`QualityGate`** -- a named threshold on a backtest metric (`roc_auc`,
  `precision`, `recall`, `f1`, `average_precision`), with min and/or max
  bounds.
- **`DriftGate`** -- a max-PSI threshold, either global or scoped to a
  single feature.
- **`GateFailure`** -- one actionable diagnostic per failed gate: which
  gate, expected vs. actual value, and a human-readable explanation (e.g.
  *"feature 'liquidity_depth' has PSI=0.41, exceeding drift gate of 0.25 —
  distribution has shifted meaningfully from the reference window"*).
- **`EvaluationResult`** -- `passed: bool` plus both underlying reports and
  the full failure list; serialized to `evaluation_report.json`.

A CLI entry point (`python -m evaluation.quality_drift_workflow
reference.parquet current.parquet out_dir/`) exits non-zero on gate
failure, so it can be dropped directly into a CI promotion-gate step.

## Design tradeoffs

- **Orchestration only -- no metric logic duplicated.** Quality scoring
  stays entirely in `run_backtest`; drift computation stays entirely in
  `DriftMonitor`/`compute_psi`. This module only builds the reference
  distribution shape `DriftMonitor` expects and evaluates gates against
  both reports' output. Fixing a bug in PSI math or backtest metrics
  happens in one place, not two.
- **Quantile (equal-frequency) binning for the reference distribution**,
  not equal-width. LedgerLens features (trade amounts, wallet-graph
  degree) are typically heavy-tailed; equal-width bins would put nearly
  all reference mass in one bin, making PSI nearly blind to drift within
  it. The outermost bin edges are extended to +/-inf so out-of-range
  values in the current window still land in a bin instead of being
  silently dropped from the PSI comparison.
- **The report is written even when gates fail.** A failing CI run still
  needs `evaluation_report.json` on disk to inspect -- `passed=False` is
  not a reason to withhold the artifact.
- **Default gates are conservative, not opinionated about your model.**
  `DEFAULT_QUALITY_GATES`/`DEFAULT_DRIFT_GATES` provide a reasonable
  starting point (`roc_auc >= 0.6`, PSI within the same moderate-drift
  threshold `detection/drift_monitor.py` already uses), but any real
  promotion gate should pass explicit gates tuned to that model.

## Usage

```python
from evaluation.quality_drift_workflow import (
    DriftGate, QualityGate, run_evaluation_workflow,
)

result = run_evaluation_workflow(
    reference_dataset_path="data/training_snapshot.parquet",
    current_dataset_path="data/last_7_days.parquet",
    model_config={},  # uses the live RiskScorer; pass predict_fn to override
    output_dir="reports/eval_2026_07_29",
    quality_gates=(QualityGate(metric="roc_auc", min_value=0.75),),
    drift_gates=(DriftGate(max_psi=0.25),),
)

if not result.passed:
    for failure in result.failures:
        print(failure.message)
```

CLI:

```
python -m evaluation.quality_drift_workflow \
    data/training_snapshot.parquet data/last_7_days.parquet reports/eval_out \
    --min-roc-auc 0.75 --max-psi 0.25
```

## Validation

```
pytest tests/test_evaluation_workflow.py -v
```

Covers: `QualityGate`/`DriftGate` construction validation and pass/fail
evaluation in isolation (below-min, above-max, missing metric,
feature-scoped drift gate); `_build_reference_distribution` bin-edge/
proportion shape and its insufficient-sample skip path;
`_numeric_feature_columns` filtering; and `run_evaluation_workflow` end
to end against synthetic Parquet fixtures for three scenarios — a passing
run with matched distributions, a quality-gate failure (poor predictor),
and a drift-gate failure (shifted feature distribution) — plus confirming
`evaluation_report.json` is written in both the passing and failing case.

## Follow-up work

- Wire this workflow into a scheduled CI job (e.g. nightly against the
  last N days of production-scored data) rather than only manual/PR-time
  invocation.
- Extend `QualityGate` to support per-asset-pair thresholds using the
  `per_asset_pair` breakdown `run_backtest` already produces, for chains
  or pairs with materially different base rates.
