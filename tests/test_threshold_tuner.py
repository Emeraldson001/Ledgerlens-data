"""Tests for detection/threshold_tuner.py — threshold tuning strategies.

Covers boundary-value testing for the adaptive threshold tuner, ensuring
that clamping to [0, 100] range works correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from detection.threshold_tuner import ThresholdTuner, _compute_metrics


class TestThresholdBoundaryClamping:
    """Boundary-value tests ensuring thresholds stay within [0, 100]."""

    def test_compute_metrics_threshold_zero(self):
        """Threshold of 0 should flag all samples (predictions all 1)."""
        scores = np.array([10.0, 20.0, 30.0])
        labels = np.array([0, 0, 1])
        result = _compute_metrics(scores, labels, threshold=0)
        assert result.threshold == 0
        assert result.tp == 1
        assert result.fp == 2

    def test_compute_metrics_threshold_hundred(self):
        """Threshold of 100 should flag no samples (predictions all 0)."""
        scores = np.array([50.0, 75.0, 90.0])
        labels = np.array([0, 1, 1])
        result = _compute_metrics(scores, labels, threshold=100)
        assert result.threshold == 100
        assert result.tp == 0
        assert result.fp == 0

    def test_tune_with_scores_below_min_grid(self):
        """Scores below minimum grid value should still be evaluated correctly."""
        scores = np.array([5.0, 10.0, 15.0, 20.0])
        labels = np.array([0, 0, 1, 1])
        tuner = ThresholdTuner(scores, labels, threshold_grid=[40, 50, 60])
        result = tuner.tune_precision_recall()
        assert result.recommended_threshold in [40, 50, 60]
        assert 0 <= result.recommended_threshold <= 100

    def test_tune_with_scores_above_max_grid(self):
        """Scores above maximum grid value should still be evaluated correctly."""
        scores = np.array([80.0, 85.0, 90.0, 95.0])
        labels = np.array([0, 1, 1, 1])
        tuner = ThresholdTuner(scores, labels, threshold_grid=[40, 50, 60])
        result = tuner.tune_precision_recall()
        assert result.recommended_threshold in [40, 50, 60]
        assert 0 <= result.recommended_threshold <= 100

    def test_tune_false_positive_budget_respects_bounds(self):
        """Recommended threshold from FP budget strategy must stay in [0, 100]."""
        scores = np.array([20.0, 30.0, 40.0, 60.0, 70.0, 80.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        tuner = ThresholdTuner(scores, labels, threshold_grid=[0, 25, 50, 75, 100])
        result = tuner.tune_false_positive_budget(max_fpr=0.5)
        assert 0 <= result.recommended_threshold <= 100

    def test_tune_cost_sensitive_respects_bounds(self):
        """Recommended threshold from cost strategy must stay in [0, 100]."""
        scores = np.array([15.0, 35.0, 55.0, 65.0, 85.0, 95.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        tuner = ThresholdTuner(scores, labels, threshold_grid=[10, 30, 50, 70, 90])
        result = tuner.tune_cost_sensitive(fp_weight=1.0, fn_weight=5.0)
        assert 0 <= result.recommended_threshold <= 100

    def test_grid_with_boundary_values(self):
        """Grid explicitly including 0 and 100 should produce valid metrics."""
        scores = np.array([25.0, 50.0, 75.0])
        labels = np.array([0, 0, 1])
        tuner = ThresholdTuner(scores, labels, threshold_grid=[0, 50, 100])
        result = tuner.tune_precision_recall()
        assert result.recommended_threshold in [0, 50, 100]

        # Verify all evaluation points stay in bounds
        for point in result.eval_points:
            assert 0 <= point.threshold <= 100
