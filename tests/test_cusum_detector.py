"""Tests for monitoring/cusum_detector.py (CUSUMDetector).

Covers:
- Alarm triggers when statistic exceeds threshold
- Statistics reset after alarm so subsequent normal observations do not re-trigger
- acknowledge() clears alarm and statistics
- is_alarm reflects current state
"""

import pytest

from monitoring.cusum_detector import CUSUMDetector


def _make_detector(**kwargs):
    defaults = dict(
        metric_name="test_metric",
        target_mean=0.0,
        allowable_slack=0.5,
        decision_threshold=5.0,
    )
    defaults.update(kwargs)
    return CUSUMDetector(**defaults)


# ---------------------------------------------------------------------------
# Alarm triggers correctly
# ---------------------------------------------------------------------------


def test_alarm_triggers_on_sustained_shift():
    detector = _make_detector()
    assert not detector.is_alarm

    for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
        detector.update(value)

    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# Statistics reset after alarm — no continuous re-alarm (issue #785)
# ---------------------------------------------------------------------------


def test_no_continuous_alarm_after_reset():
    detector = _make_detector()

    trigger_values = [10.0] * 10
    for value in trigger_values:
        detector.update(value)

    assert detector.is_alarm is True

    baseline_values = [0.0] * 10
    alarm_count = 0
    for value in baseline_values:
        if detector.update(value):
            alarm_count += 1

    assert alarm_count == 0, (
        f"Expected 0 alarms after reset, got {alarm_count}. "
        "CUSUM statistics were not reset correctly after the initial alarm."
    )
    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# acknowledge() clears alarm and statistics
# ---------------------------------------------------------------------------


def test_acknowledge_clears_alarm():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    assert detector.is_alarm is True

    detector.acknowledge()
    assert detector.is_alarm is False
    assert detector.s_high == 0.0
    assert detector.s_low == 0.0


def test_acknowledge_allows_future_alarms():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    detector.acknowledge()

    for value in [10.0] * 10:
        detector.update(value)

    assert detector.is_alarm is True


# ---------------------------------------------------------------------------
# is_alarm reflects state transitions
# ---------------------------------------------------------------------------


def test_is_alarm_false_before_alarm():
    detector = _make_detector()
    assert detector.is_alarm is False


def test_update_returns_false_when_already_alarming():
    detector = _make_detector()
    for value in [10.0] * 10:
        detector.update(value)
    assert detector.is_alarm is True

    result = detector.update(20.0)
    assert result is False
