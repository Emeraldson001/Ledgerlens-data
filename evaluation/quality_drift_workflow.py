"""Evaluation workflow combining detection-quality backtesting and feature drift.

``evaluation/backtest.py`` answers "how good are the model's predictions on
a labelled dataset". ``detection/drift_monitor.py`` answers "how much has
the feature distribution shifted from a reference". Both exist, but there
was no single, reusable workflow that runs both, evaluates the results
against explicit pass/fail gates, and produces one combined report -- so
"did this candidate model regress on quality *or* is it scoring on
drifted-away-from-training-data features" required manually running two
separate tools and eyeballing two separate JSON files.

This module adds that as a small orchestration layer on top of the
existing, unmodified primitives:

* :func:`run_evaluation_workflow` -- runs `evaluation.backtest.run_backtest`
  against a "current" labelled dataset, builds a reference feature
  distribution from a "reference" dataset (e.g. the training set, or a
  known-good prior production window) and computes PSI drift via
  `detection.drift_monitor.DriftMonitor`, evaluates both against
  caller-supplied gates, and writes one combined
  `evaluation_report.json`.
* :class:`QualityGate` / :class:`DriftGate` -- declarative pass/fail
  thresholds for backtest metrics and per-feature PSI, respectively.
* :class:`GateFailure` -- a single actionable diagnostic: which gate
  failed, the threshold, the actual value, and a human-readable message,
  so a failing CI run points directly at what regressed instead of
  requiring a manual diff of two JSON reports.
* :class:`EvaluationResult` -- the combined outcome (`passed: bool`,
  quality report, drift report, and the list of `GateFailure`s).

This is intended to be run in CI (or manually before promoting a model)
as a single gate: nonzero exit / `passed=False` blocks promotion.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from detection.drift_monitor import PSI_MODERATE_DRIFT_THRESHOLD, DriftMonitor
from evaluation.backtest import run_backtest
from utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_N_BINS = 10

#: Metrics produced by evaluation.backtest.run_backtest that quality gates
#: may reference.
_QUALITY_METRIC_NAMES = {
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
}


@dataclass(frozen=True)
class QualityGate:
    """A pass/fail threshold on a single backtest quality metric.

    At least one of ``min_value``/``max_value`` must be set. Both may be
    set to define an acceptable band (rare, but supported -- e.g. bounding
    recall from above to catch a threshold miscalibration that flags
    everything).
    """

    metric: str
    min_value: float | None = None
    max_value: float | None = None

    def __post_init__(self) -> None:
        if self.metric not in _QUALITY_METRIC_NAMES:
            raise ValueError(
                f"unknown quality metric {self.metric!r}; expected one of "
                f"{sorted(_QUALITY_METRIC_NAMES)}"
            )
        if self.min_value is None and self.max_value is None:
            raise ValueError(f"QualityGate for {self.metric!r} needs min_value and/or max_value")

    def evaluate(self, report: dict[str, Any]) -> GateFailure | None:
        actual = report.get(self.metric)
        if actual is None:
            return GateFailure(
                gate_name=f"quality:{self.metric}",
                expected=self._expected_desc(),
                actual="missing",
                message=f"backtest report has no {self.metric!r} metric",
            )
        if self.min_value is not None and actual < self.min_value:
            return GateFailure(
                gate_name=f"quality:{self.metric}",
                expected=self._expected_desc(),
                actual=actual,
                message=f"{self.metric}={actual:.4f} is below minimum {self.min_value:.4f}",
            )
        if self.max_value is not None and actual > self.max_value:
            return GateFailure(
                gate_name=f"quality:{self.metric}",
                expected=self._expected_desc(),
                actual=actual,
                message=f"{self.metric}={actual:.4f} is above maximum {self.max_value:.4f}",
            )
        return None

    def _expected_desc(self) -> str:
        parts = []
        if self.min_value is not None:
            parts.append(f">= {self.min_value}")
        if self.max_value is not None:
            parts.append(f"<= {self.max_value}")
        return " and ".join(parts)


@dataclass(frozen=True)
class DriftGate:
    """A pass/fail PSI threshold, applied to one feature or all features.

    ``feature=None`` (default) applies ``max_psi`` to every feature the
    drift report checked; ``feature="some_column"`` restricts the gate to
    that single feature (for features known to be more drift-sensitive,
    e.g. a live liquidity metric vs. a slowly-changing account-age feature).
    """

    max_psi: float = PSI_MODERATE_DRIFT_THRESHOLD
    feature: str | None = None

    def evaluate(self, drift_report: dict[str, Any]) -> list[GateFailure]:
        failures = []
        for entry in drift_report.get("features", []):
            if self.feature is not None and entry["feature"] != self.feature:
                continue
            if entry["psi"] > self.max_psi:
                failures.append(
                    GateFailure(
                        gate_name=f"drift:{entry['feature']}",
                        expected=f"psi <= {self.max_psi}",
                        actual=entry["psi"],
                        message=(
                            f"feature {entry['feature']!r} has PSI={entry['psi']:.4f}, "
                            f"exceeding drift gate of {self.max_psi} — distribution has "
                            "shifted meaningfully from the reference window"
                        ),
                    )
                )
        return failures


@dataclass
class GateFailure:
    """One actionable diagnostic: what failed, expected vs. actual, why."""

    gate_name: str
    expected: str
    actual: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate_name,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
        }


@dataclass
class EvaluationResult:
    """Combined outcome of a quality + drift evaluation run."""

    passed: bool
    quality_report: dict[str, Any]
    drift_report: dict[str, Any]
    failures: list[GateFailure] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "quality_report": self.quality_report,
            "drift_report": self.drift_report,
            "failures": [f.to_dict() for f in self.failures],
        }


#: A reasonable default gate set for CI: require decent discrimination and
#: forbid severe drift on any checked feature. Callers evaluating a
#: specific model should pass explicit gates instead of relying on these.
DEFAULT_QUALITY_GATES = (QualityGate(metric="roc_auc", min_value=0.6),)
DEFAULT_DRIFT_GATES = (DriftGate(max_psi=PSI_MODERATE_DRIFT_THRESHOLD),)


def _numeric_feature_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        col for col in df.columns if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]


def _build_reference_distribution(
    reference_df: pd.DataFrame, feature_columns: list[str], n_bins: int = DEFAULT_N_BINS
) -> dict[str, dict[str, Any]]:
    """Build the reference-distribution shape `DriftMonitor` expects.

    Bin edges are quantile-based (equal-frequency bins over the reference
    data) rather than equal-width, so a skewed feature (e.g. trade amount)
    doesn't collapse most of the reference mass into a single bin, which
    would make PSI insensitive to drift within that bin.
    """
    reference_distribution: dict[str, dict[str, Any]] = {}
    for col in feature_columns:
        values = reference_df[col].dropna().to_numpy(dtype=float)
        if len(values) < n_bins:
            logger.warning(
                "skipping drift feature with insufficient reference samples",
                extra={"feature": col, "n_samples": len(values), "n_bins": n_bins},
            )
            continue

        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(values, quantiles))
        if len(bin_edges) < 2:
            continue
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        bin_indices = np.digitize(values, bins=bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)
        counts = np.bincount(bin_indices, minlength=len(bin_edges) - 1)
        expected_proportions = (counts / counts.sum()).tolist()

        reference_distribution[col] = {
            "bin_edges": bin_edges.tolist(),
            "expected_proportions": expected_proportions,
        }
    return reference_distribution


def run_evaluation_workflow(
    reference_dataset_path: str,
    current_dataset_path: str,
    model_config: dict[str, Any],
    output_dir: str,
    quality_gates: tuple[QualityGate, ...] = DEFAULT_QUALITY_GATES,
    drift_gates: tuple[DriftGate, ...] = DEFAULT_DRIFT_GATES,
    threshold: float | None = None,
    feature_columns: list[str] | None = None,
    n_bins: int = DEFAULT_N_BINS,
) -> EvaluationResult:
    """Run a combined detection-quality + drift evaluation and gate on it.

    Args:
        reference_dataset_path: labelled Parquet dataset representing the
            "known good" distribution (typically the training set or a
            prior production window) that drift is measured against.
        current_dataset_path: labelled Parquet dataset to backtest and to
            compare against the reference distribution for drift. Passed
            straight through to `evaluation.backtest.run_backtest`.
        model_config: forwarded to `run_backtest` (see its docstring);
            may include a `predict_fn` override for offline replay.
        output_dir: directory for `backtest_report.json`, `pr_curve.png`
            (written by `run_backtest`) and this workflow's own
            `evaluation_report.json`.
        quality_gates: thresholds evaluated against the backtest report.
            Defaults to requiring `roc_auc >= 0.6`.
        drift_gates: PSI thresholds evaluated against the drift report.
            Defaults to `PSI_MODERATE_DRIFT_THRESHOLD` on every feature.
        threshold: forwarded to `run_backtest` (risk-probability cutoff).
        feature_columns: which columns to check for drift. Defaults to
            every numeric column shared by both datasets except `label`.
        n_bins: number of quantile bins used to build the reference
            distribution for PSI (see `_build_reference_distribution`).

    Returns:
        `EvaluationResult` with `passed=True` iff every gate passed.
        `evaluation_report.json` is written to `output_dir` regardless of
        pass/fail, so a failing CI run still leaves a report to inspect.
    """
    os.makedirs(output_dir, exist_ok=True)

    quality_report = run_backtest(current_dataset_path, model_config, output_dir, threshold)

    reference_df = pd.read_parquet(reference_dataset_path)
    current_df = pd.read_parquet(current_dataset_path)

    if feature_columns is None:
        feature_columns = _numeric_feature_columns(reference_df, exclude={"label"})
        feature_columns = [c for c in feature_columns if c in current_df.columns]

    reference_distribution = _build_reference_distribution(reference_df, feature_columns, n_bins)
    drift_monitor = DriftMonitor(reference_distribution)
    drift_report = drift_monitor.compute(current_df[list(reference_distribution)]).to_dict()

    failures: list[GateFailure] = []
    for gate in quality_gates:
        failure = gate.evaluate(quality_report)
        if failure is not None:
            failures.append(failure)
    for gate in drift_gates:
        failures.extend(gate.evaluate(drift_report))

    result = EvaluationResult(
        passed=not failures,
        quality_report=quality_report,
        drift_report=drift_report,
        failures=failures,
    )

    with open(os.path.join(output_dir, "evaluation_report.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2, sort_keys=True)

    if not result.passed:
        logger.warning(
            "evaluation workflow gate failures",
            extra={"n_failures": len(failures), "failures": [f.message for f in failures]},
        )

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a combined detection-quality and drift evaluation workflow"
    )
    parser.add_argument(
        "reference_dataset_path", help="Parquet dataset defining the reference distribution"
    )
    parser.add_argument(
        "current_dataset_path", help="Labelled Parquet dataset to backtest and check for drift"
    )
    parser.add_argument("output_dir", help="Directory to write evaluation_report.json into")
    parser.add_argument(
        "--threshold", type=float, default=None, help="Risk-probability cutoff in [0, 1]"
    )
    parser.add_argument(
        "--min-roc-auc", type=float, default=0.6, help="Minimum acceptable ROC-AUC (default 0.6)"
    )
    parser.add_argument(
        "--max-psi",
        type=float,
        default=PSI_MODERATE_DRIFT_THRESHOLD,
        help="Maximum acceptable per-feature PSI (default matches detection.drift_monitor)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_evaluation_workflow(
        args.reference_dataset_path,
        args.current_dataset_path,
        {},
        args.output_dir,
        quality_gates=(QualityGate(metric="roc_auc", min_value=args.min_roc_auc),),
        drift_gates=(DriftGate(max_psi=args.max_psi),),
        threshold=args.threshold,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
