"""Tests for detection/threshold_strategy.py — pluggable threshold strategies.

Acceptance criteria covered:
  - StaticStrategy returns configured or config-derived threshold.
  - StatisticalStrategy finds F1-optimal threshold and respects recall floor.
  - AdaptiveStrategy delegates to ThresholdAgent.
  - build_strategy factory resolves valid names and rejects unknown ones.
  - Property: alert_count is monotonically non-increasing as threshold increases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from detection.threshold_strategy import (
    AdaptiveStrategy,
    StaticStrategy,
    StatisticalStrategy,
    build_strategy,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# Clean separator at ~0.5: negatives in [0.10, 0.49], positives in [0.51, 0.90]
Y_TRUE = np.array([0] * 50 + [1] * 50)
Y_SCORE = np.array([0.10 + 0.008 * i for i in range(50)] + [0.51 + 0.008 * i for i in range(50)])


# ---------------------------------------------------------------------------
# StaticStrategy
# ---------------------------------------------------------------------------


def test_static_strategy_returns_configured_value():
    strategy = StaticStrategy(threshold=0.7)
    result = strategy.select_threshold(Y_SCORE)
    assert result.threshold == 0.7
    assert result.strategy_name == "static"


def test_static_strategy_default_reads_config(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "RISK_SCORE_FLAG_THRESHOLD", 65)
    strategy = StaticStrategy()
    result = strategy.select_threshold(Y_SCORE)
    assert result.threshold == pytest.approx(0.65)


def test_static_strategy_diagnostics_single_point():
    strategy = StaticStrategy(threshold=0.6)
    diag = strategy.diagnostics(Y_SCORE, Y_TRUE)
    assert len(diag.sweep_curve) == 1
    assert diag.sweep_curve[0]["threshold"] == 0.6


# ---------------------------------------------------------------------------
# StatisticalStrategy
# ---------------------------------------------------------------------------


def test_statistical_strategy_finds_optimal_f1():
    strategy = StatisticalStrategy(recall_floor=None)
    result = strategy.select_threshold(Y_SCORE, labels=Y_TRUE)
    assert abs(result.threshold - 0.5) < 0.05


def test_statistical_strategy_respects_recall_floor():
    strategy = StatisticalStrategy(recall_floor=0.9)
    result = strategy.select_threshold(Y_SCORE, labels=Y_TRUE)
    preds = (Y_SCORE >= result.threshold).astype(int)
    recall = preds[Y_TRUE == 1].sum() / Y_TRUE.sum()
    assert recall >= 0.9


def test_statistical_strategy_fallback_when_no_recall_floor_met():
    """With an impossibly high recall_floor, fall back to highest-recall threshold."""
    strategy = StatisticalStrategy(recall_floor=0.999)
    result = strategy.select_threshold(Y_SCORE, labels=Y_TRUE)
    assert result.threshold is not None
    assert isinstance(result.threshold, float)


# ---------------------------------------------------------------------------
# AdaptiveStrategy
# ---------------------------------------------------------------------------


def test_adaptive_strategy_wraps_threshold_agent():
    mock_agent = MagicMock()
    mock_agent.select_threshold.return_value = 65  # 0-100 scale
    strategy = AdaptiveStrategy(agent=mock_agent)
    result = strategy.select_threshold(np.array([0.5]))
    assert result.threshold == pytest.approx(0.65)
    mock_agent.select_threshold.assert_called_once()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_contain_sweep_curve():
    strategy = StatisticalStrategy()  # default 99-point grid
    diag = strategy.diagnostics(Y_SCORE, Y_TRUE)
    assert len(diag.sweep_curve) == 99


def test_diagnostics_precision_recall_at_threshold():
    strategy = StatisticalStrategy()
    diag = strategy.diagnostics(Y_SCORE, Y_TRUE)
    assert 0 <= diag.precision_at_threshold <= 1
    assert 0 <= diag.recall_at_threshold <= 1
    assert 0 <= diag.f1_at_threshold <= 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_strategy_static():
    strategy = build_strategy("static", threshold=0.6)
    assert isinstance(strategy, StaticStrategy)


def test_build_strategy_statistical():
    strategy = build_strategy("statistical", recall_floor=0.8)
    assert isinstance(strategy, StatisticalStrategy)


def test_build_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy_name"):
        build_strategy("unknown")


# ---------------------------------------------------------------------------
# Property-based: alert count monotonically non-increasing with threshold
# ---------------------------------------------------------------------------


@given(
    data=st.lists(
        st.tuples(st.sampled_from([0, 1]), st.floats(0.01, 0.99)),
        min_size=20,
        max_size=200,
    )
)
@settings(max_examples=30)
def test_sweep_monotonic_alert_count(data):
    """Alert count in sweep curve must be non-increasing as threshold rises."""
    y_true = np.array([d[0] for d in data])
    y_score = np.array([d[1] for d in data])

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return  # skip degenerate cases

    strategy = StatisticalStrategy(recall_floor=None)
    diag = strategy.diagnostics(y_score, y_true)
    curve = sorted(diag.sweep_curve, key=lambda pt: pt["threshold"])

    for i in range(1, len(curve)):
        assert curve[i]["alert_count"] <= curve[i - 1]["alert_count"]
