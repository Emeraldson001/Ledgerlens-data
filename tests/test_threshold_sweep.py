"""Tests for evaluation/threshold_sweep.py — threshold sweep diagnostics engine.

Acceptance criteria covered:
  - ThresholdSweep computes correct metrics across a grid of thresholds.
  - Optimal threshold finder respects metric choice and recall constraints.
  - Impact reports compute correct deltas between thresholds.
  - JSON export conforms to expected schema.
  - Edge cases: all-positive, all-negative, empty, mismatched inputs.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation.threshold_sweep import (
    ImpactReport,
    SweepPoint,
    ThresholdSweep,
    run_sweep,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

Y_TRUE = np.array([0] * 50 + [1] * 50)
Y_SCORE = np.array([0.10 + 0.008 * i for i in range(50)] + [0.51 + 0.008 * i for i in range(50)])


# ---------------------------------------------------------------------------
# Basic sweep
# ---------------------------------------------------------------------------


def test_sweep_perfect_separator_finds_exact_threshold():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    optimal = sweep.find_optimal_threshold(metric="f1")
    assert abs(optimal.threshold - 0.5) < 0.05


def test_sweep_returns_correct_point_count():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    points = sweep.sweep()
    assert len(points) == 99  # default grid 0.01..0.99


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_sweep_all_positive_edge_case():
    y_true = np.ones(100, dtype=int)
    y_score = np.linspace(0, 1, 100)
    sweep = ThresholdSweep(y_true, y_score)
    points = sweep.sweep()
    assert len(points) > 0
    # All labels are positive, so low thresholds should have recall == 1.0
    assert points[0].recall == pytest.approx(1.0)


def test_sweep_all_negative_edge_case():
    y_true = np.zeros(100, dtype=int)
    y_score = np.linspace(0, 1, 100)
    sweep = ThresholdSweep(y_true, y_score)
    points = sweep.sweep()
    assert len(points) > 0


def test_sweep_empty_raises():
    with pytest.raises(ValueError, match="not be empty"):
        ThresholdSweep(np.array([]), np.array([]))


def test_sweep_length_mismatch_raises():
    with pytest.raises(ValueError, match="same shape"):
        ThresholdSweep(np.array([1, 0]), np.array([0.5]))


# ---------------------------------------------------------------------------
# Impact report
# ---------------------------------------------------------------------------


def test_impact_report_deltas_correct():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    report = sweep.impact_report(0.3, 0.7)
    assert isinstance(report, ImpactReport)
    assert report.precision_delta == pytest.approx(
        report.proposed_metrics.precision - report.current_metrics.precision
    )
    assert report.recall_delta == pytest.approx(
        report.proposed_metrics.recall - report.current_metrics.recall
    )
    assert report.f1_delta == pytest.approx(report.proposed_metrics.f1 - report.current_metrics.f1)
    assert report.alert_count_delta == (
        report.proposed_metrics.alert_count - report.current_metrics.alert_count
    )


# ---------------------------------------------------------------------------
# Optimal threshold with constraints
# ---------------------------------------------------------------------------


def test_optimal_with_recall_constraint():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    optimal = sweep.find_optimal_threshold(metric="f1", recall_floor=0.85)
    assert optimal.recall >= 0.85


def test_optimal_recall_floor_fallback():
    """When no threshold meets an extreme recall floor, fall back to highest recall."""
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    optimal = sweep.find_optimal_threshold(metric="f1", recall_floor=1.01)
    # Should still return a valid point (the one with highest recall)
    assert isinstance(optimal, SweepPoint)


def test_find_optimal_unknown_metric_raises():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    with pytest.raises(ValueError, match="Unknown metric"):
        sweep.find_optimal_threshold(metric="unknown")


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def test_export_json_schema(tmp_path):
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    sweep.sweep()
    out_file = str(tmp_path / "sweep.json")
    sweep.export_sweep_json(out_file)

    with open(out_file) as f:
        data = json.load(f)

    assert "sweep" in data
    assert "optimal_f1" in data
    assert "optimal_f1_recall_85" in data
    assert len(data["sweep"]) == 99


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def test_run_sweep_convenience():
    result = run_sweep(Y_TRUE, Y_SCORE)
    assert "sweep_points" in result
    assert "optimal" in result
    assert "optimal_constrained" in result
    assert isinstance(result["optimal"], SweepPoint)


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


def test_sweep_points_have_valid_ranges():
    sweep = ThresholdSweep(Y_TRUE, Y_SCORE)
    points = sweep.sweep()

    for pt in points:
        assert 0 <= pt.precision <= 1
        assert 0 <= pt.recall <= 1
        assert 0 <= pt.f1 <= 1
        assert 0 <= pt.false_positive_rate <= 1
        assert 0 <= pt.false_negative_rate <= 1
        assert pt.alert_count >= 0
