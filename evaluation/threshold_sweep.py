from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from utils.logging import get_logger

logger = get_logger(__name__)

__all__ = ["SweepPoint", "ImpactReport", "ThresholdSweep", "run_sweep"]


@dataclass
class SweepPoint:
    """Dataclass representing metrics at a specific threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    alert_count: int
    false_positive_rate: float
    false_negative_rate: float


@dataclass
class ImpactReport:
    """Dataclass representing the impact of changing from one threshold to another."""

    current_threshold: float
    proposed_threshold: float
    precision_delta: float
    recall_delta: float
    f1_delta: float
    alert_count_delta: int
    current_metrics: SweepPoint
    proposed_metrics: SweepPoint


class ThresholdSweep:
    """Threshold sweep diagnostics engine."""

    def __init__(
        self, y_true: np.ndarray, y_score: np.ndarray, grid: list[float] | None = None
    ) -> None:
        """Initialize ThresholdSweep.

        Args:
            y_true: True binary labels (0 or 1).
            y_score: Predicted probabilities or scores.
            grid: List of thresholds to evaluate. Defaults to 0.01 to 0.99.
        """
        if y_true.size == 0 or y_score.size == 0:
            raise ValueError("Input arrays must not be empty.")
        if y_true.shape != y_score.shape:
            raise ValueError("Input arrays must have the same shape.")

        unique_labels = np.unique(y_true)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise ValueError("y_true must contain only binary labels (0 and 1).")

        self.y_true = np.array(y_true, dtype=int)
        self.y_score = np.array(y_score, dtype=float)
        self.grid = grid if grid is not None else [i / 100 for i in range(1, 100)]
        self._results: list[SweepPoint] | None = None

    def _compute_metrics(self, threshold: float) -> SweepPoint:
        y_pred = (self.y_score >= threshold).astype(int)
        precision = float(precision_score(self.y_true, y_pred, zero_division=0))
        recall = float(recall_score(self.y_true, y_pred, zero_division=0))
        f1 = float(f1_score(self.y_true, y_pred, zero_division=0))
        alert_count = int(np.sum(y_pred))

        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(self.y_true)) > 1:
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        else:
            if self.y_true[0] == 1:
                tp = int(np.sum(y_pred == 1))
                fn = int(np.sum(y_pred == 0))
            else:
                tn = int(np.sum(y_pred == 0))
                fp = int(np.sum(y_pred == 1))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        return SweepPoint(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            alert_count=alert_count,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
        )

    def sweep(self) -> list[SweepPoint]:
        """Run the sweep over all thresholds in the grid.

        Returns:
            A list of SweepPoint objects containing metrics for each threshold.
        """
        if self._results is not None:
            return self._results

        logger.info(f"Running threshold sweep over {len(self.grid)} grid points.")
        self._results = [self._compute_metrics(t) for t in self.grid]
        return self._results

    def find_optimal_threshold(
        self, metric: str = "f1", recall_floor: float | None = None
    ) -> SweepPoint:
        """Find the optimal threshold based on a given metric, optionally constrained by recall.

        Args:
            metric: The metric to maximize ('f1', 'precision', 'recall').
            recall_floor: Minimum acceptable recall value.

        Returns:
            The SweepPoint that maximizes the given metric.
        """
        valid_metrics = {"f1", "precision", "recall"}
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric '{metric}'. Must be one of {valid_metrics}.")

        points = self.sweep()

        if recall_floor is not None:
            constrained_points = [p for p in points if p.recall >= recall_floor]
            if not constrained_points:
                logger.warning(
                    f"No threshold satisfies recall_floor >= {recall_floor}. Falling back to highest recall."
                )
                return max(points, key=lambda p: p.recall)
            points = constrained_points

        return max(points, key=lambda p: getattr(p, metric))

    def impact_report(self, current_threshold: float, proposed_threshold: float) -> ImpactReport:
        """Compute the impact of changing from current_threshold to proposed_threshold.

        Args:
            current_threshold: The baseline threshold.
            proposed_threshold: The new threshold to evaluate.

        Returns:
            An ImpactReport containing metrics and deltas.
        """
        current_metrics = self._compute_metrics(current_threshold)
        proposed_metrics = self._compute_metrics(proposed_threshold)

        return ImpactReport(
            current_threshold=current_threshold,
            proposed_threshold=proposed_threshold,
            precision_delta=proposed_metrics.precision - current_metrics.precision,
            recall_delta=proposed_metrics.recall - current_metrics.recall,
            f1_delta=proposed_metrics.f1 - current_metrics.f1,
            alert_count_delta=proposed_metrics.alert_count - current_metrics.alert_count,
            current_metrics=current_metrics,
            proposed_metrics=proposed_metrics,
        )

    def export_sweep_json(self, path: str) -> None:
        """Export sweep results and optimal thresholds to a JSON file.

        Args:
            path: The file path to save the JSON output.
        """
        points = self.sweep()
        optimal_f1 = self.find_optimal_threshold(metric="f1")
        optimal_f1_recall_85 = self.find_optimal_threshold(metric="f1", recall_floor=0.85)

        data = {
            "sweep": [asdict(p) for p in points],
            "optimal_f1": asdict(optimal_f1),
            "optimal_f1_recall_85": asdict(optimal_f1_recall_85),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported sweep results to {path}")


def run_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: list[float] | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    """Convenience function to run a threshold sweep, find optimal points, and optionally export.

    Args:
        y_true: True binary labels (0 or 1).
        y_score: Predicted probabilities or scores.
        grid: List of thresholds to evaluate.
        output_path: Optional path to export JSON results.

    Returns:
        Dictionary containing sweep points, optimal f1 point, and constrained optimal f1 point.
    """
    sweep_engine = ThresholdSweep(y_true, y_score, grid=grid)
    sweep_points = sweep_engine.sweep()
    optimal = sweep_engine.find_optimal_threshold(metric="f1")
    optimal_constrained = sweep_engine.find_optimal_threshold(metric="f1", recall_floor=0.85)

    if output_path:
        sweep_engine.export_sweep_json(output_path)

    return {
        "sweep_points": sweep_points,
        "optimal": optimal,
        "optimal_constrained": optimal_constrained,
    }
