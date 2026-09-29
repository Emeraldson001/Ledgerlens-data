"""Tests for monitoring/drift_detector.py — MMD-based covariance shift detection."""

import time

import numpy as np
import pytest

from monitoring.drift_detector import CovarianceShiftDetector, check_drift_monitor_health

RNG = np.random.default_rng(42)


@pytest.fixture
def detector():
    return CovarianceShiftDetector(threshold=0.05)


def test_identical_windows_mmd_near_zero(detector):
    """Identical reference and test windows must produce MMD near zero."""
    X = RNG.normal(0, 1, (200, 3))
    report = detector.detect(X, X, feature_names=["a", "b", "c"])
    for val in report.mmd_per_feature.values():
        assert abs(val) < 0.05
    assert not report.drift_detected


def test_shifted_gaussian_triggers_drift(detector):
    """Test window drawn from a shifted Gaussian must produce MMD above threshold."""
    ref = RNG.normal(0, 1, (200, 2))
    cur = RNG.normal(5, 1, (200, 2))
    report = detector.detect(ref, cur, feature_names=["x", "y"])
    for val in report.mmd_per_feature.values():
        assert val > detector.threshold
    assert report.drift_detected


def test_near_zero_variance_feature_excluded(detector):
    """Near-zero-variance feature in reference window must be excluded from report."""
    ref = np.hstack([np.ones((100, 1)), RNG.normal(0, 1, (100, 1))])
    cur = np.hstack([np.ones((100, 1)), RNG.normal(0, 1, (100, 1))])
    report = detector.detect(ref, cur, feature_names=["constant", "normal"])
    assert "constant" not in report.mmd_per_feature
    assert "normal" in report.mmd_per_feature


# ---------------------------------------------------------------------------
# Grand 2 (issue #671, Task F) — drift-monitor heartbeat / health
# ---------------------------------------------------------------------------


class TestDriftMonitorHealth:
    def test_health_is_stale_before_any_successful_check(self, detector):
        assert detector.health.is_stale(max_age_seconds=3600) is True

    def test_successful_detect_records_heartbeat(self, detector):
        X = RNG.normal(0, 1, (50, 2))
        detector.detect(X, X, feature_names=["a", "b"])
        assert detector.health.last_success_at is not None
        assert detector.health.is_stale(max_age_seconds=3600) is False
        assert detector.health.total_successes == 1
        assert detector.health.consecutive_failures == 0

    def test_heartbeat_stale_beyond_max_age(self, detector):
        X = RNG.normal(0, 1, (50, 2))
        detector.detect(X, X, feature_names=["a", "b"])
        # Simulate the heartbeat aging past the SLA without a new call.
        assert detector.health.is_stale(max_age_seconds=0, now=time.time() + 1) is True

    def test_deliberately_broken_detect_raises_and_records_failure(self, detector, monkeypatch):
        """Acceptance criterion: deliberately breaking detect() must trigger
        a distinct 'drift-check failed' signal (here: the health tracker
        flips to stale/failed) rather than silently doing nothing."""

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated drift-detector failure")

        monkeypatch.setattr("monitoring.drift_detector._mmd", _boom)

        X = RNG.normal(0, 1, (50, 2))
        with pytest.raises(RuntimeError, match="simulated drift-detector failure"):
            detector.detect(X, X, feature_names=["a", "b"])

        assert detector.health.total_failures == 1
        assert detector.health.consecutive_failures == 1
        assert detector.health.last_failure_reason == "simulated drift-detector failure"
        # Never having succeeded, the monitor must be reported unhealthy.
        status = check_drift_monitor_health(detector, max_age_seconds=3600)
        assert status["stale"] is True
        assert status["healthy"] is False

    def test_health_recovers_after_failure_then_success(self, detector, monkeypatch):
        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr("monitoring.drift_detector._mmd", _boom)
        X = RNG.normal(0, 1, (50, 2))
        with pytest.raises(RuntimeError):
            detector.detect(X, X, feature_names=["a", "b"])
        assert detector.health.is_stale(max_age_seconds=3600) is True

        monkeypatch.undo()
        detector.detect(X, X, feature_names=["a", "b"])
        assert detector.health.is_stale(max_age_seconds=3600) is False
        assert detector.health.consecutive_failures == 0

    def test_check_drift_monitor_health_uses_config_default_max_age(self, detector, monkeypatch):
        from config import config

        monkeypatch.setattr(config, "DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS", 3600)
        X = RNG.normal(0, 1, (50, 2))
        detector.detect(X, X, feature_names=["a", "b"])
        status = check_drift_monitor_health(detector)
        assert status["healthy"] is True
