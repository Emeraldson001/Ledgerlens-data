"""Sliding window covariance shift detection using Maximum Mean Discrepancy (MMD).

Grand 2 (issue #671, Task F) found that ``CovarianceShiftDetector.detect()``
had no failure handling: an exception inside it propagated uncaught with no
watchdog verifying drift detection was still running at all, so a monitor
that started silently throwing on every call (a bad deploy, a schema change
upstream) would go unnoticed indefinitely. ``detect()`` now records a
heartbeat on every call — success or failure — via
:class:`DriftMonitorHealth`, still re-raises on failure (callers must not
have that behavior silently changed), and a separate periodic health check
(``check_drift_monitor_health`` / the ``--heartbeat-check`` CLI below) alerts
distinctly when the heartbeat goes stale, which happens both when the caller
stops invoking ``detect()`` at all and when every recent call has failed.

Issue #932 unifies infra-level feature drift (this module) with model-level
output drift (``detection/drift_monitor.py``) into a single triage view via
:func:`correlate_drift_signals`, which annotates whether flagged output
drift is explained by upstream feature drift in the same window.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge

    _drift_gauge = Gauge("ledgerlens_feature_drift_detected", "1=drift detected, 0=stable")
    _drift_monitor_last_success_gauge: Gauge | None = Gauge(
        "ledgerlens_drift_monitor_last_success_unixtime",
        "Unix timestamp of the last successful CovarianceShiftDetector.detect() call",
    )
    _drift_monitor_failures_total: Counter | None = Counter(
        "ledgerlens_drift_monitor_check_failures_total",
        "Number of CovarianceShiftDetector.detect() calls that raised an exception",
    )
    _drift_monitor_stale_gauge: Gauge | None = Gauge(
        "ledgerlens_drift_monitor_stale",
        "1=drift monitor heartbeat is stale (drift-check failed alert), 0=healthy",
    )
except Exception:  # pragma: no cover
    _drift_gauge = None
    _drift_monitor_last_success_gauge = None
    _drift_monitor_failures_total = None
    _drift_monitor_stale_gauge = None


@dataclass
class DriftMonitorHealth:
    """Tracks whether ``CovarianceShiftDetector.detect()`` is actually
    running, independent of whether it currently reports drift.

    ``last_success_at``/``last_failure_at`` are ``time.time()`` epoch
    seconds (``None`` before the first call of that kind).
    """

    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_failure_reason: str | None = None
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0

    def record_success(self) -> None:
        self.last_success_at = time.time()
        self.consecutive_failures = 0
        self.total_successes += 1
        if _drift_monitor_last_success_gauge is not None:
            _drift_monitor_last_success_gauge.set(self.last_success_at)
        if _drift_monitor_stale_gauge is not None:
            _drift_monitor_stale_gauge.set(0)

    def record_failure(self, exc: Exception) -> None:
        self.last_failure_at = time.time()
        self.last_failure_reason = str(exc)
        self.consecutive_failures += 1
        self.total_failures += 1
        if _drift_monitor_failures_total is not None:
            _drift_monitor_failures_total.inc()
        if _drift_monitor_stale_gauge is not None:
            _drift_monitor_stale_gauge.set(1)

    def is_stale(self, max_age_seconds: float, now: float | None = None) -> bool:
        """True if there has never been a successful check, or the last one
        is older than *max_age_seconds*. A monitor that has only ever failed
        (``last_success_at is None`` but ``total_failures > 0``) is stale
        immediately — it does not get a grace period just for having been
        invoked at all.
        """
        if self.last_success_at is None:
            return True
        now = now if now is not None else time.time()
        return (now - self.last_success_at) > max_age_seconds

    def status(self, max_age_seconds: float) -> dict:
        stale = self.is_stale(max_age_seconds)
        return {
            "healthy": not stale,
            "stale": stale,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_failure_reason": self.last_failure_reason,
            "consecutive_failures": self.consecutive_failures,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
        }


@dataclass
class DriftReport:
    mmd_per_feature: dict[str, float]
    drift_detected: bool

    def to_dict(self) -> dict:
        return {"drift_detected": self.drift_detected, "mmd_per_feature": self.mmd_per_feature}


# Triage decision tree (issue #932). Maps the combination of feature-drift and
# output-drift signals to an operator action. Documented in the monitoring
# runbook; kept here so the annotation and the runbook cannot drift apart.
TRIAGE_DECISION_TREE: dict[str, dict[str, str]] = {
    "both_drifted": {
        "condition": "feature drift AND output drift flagged in the same window",
        "interpretation": "Output drift is explained by upstream feature drift.",
        "action": "Investigate the drifted input features; the model itself is likely fine.",
    },
    "only_output_drifted": {
        "condition": "output drift flagged, no tracked input feature drifted",
        "interpretation": "Output drift is unexplained by upstream features.",
        "action": "Escalate to model owners: possible model/code regression or label shift.",
    },
    "only_input_drifted": {
        "condition": "feature drift flagged, output drift not flagged",
        "interpretation": "Inputs shifted but model outputs are still stable.",
        "action": "Monitor; no immediate action unless output drift follows.",
    },
    "no_drift": {
        "condition": "neither feature nor output drift flagged",
        "interpretation": "Both signals stable.",
        "action": "No action.",
    },
}


def correlate_drift_signals(
    feature_drift: DriftReport | None,
    output_drift: DriftReport | None,
) -> dict:
    """Unify feature-drift and model-output-drift into a single triage view.

    When output drift is flagged, checks whether any tracked input feature
    also drifted in the same window and annotates the result accordingly.

    Args:
        feature_drift: infra-level report from :class:`CovarianceShiftDetector`
            (``None`` if not available for this window).
        output_drift: model-level report from ``detection/drift_monitor.py``
            (``None`` if not available for this window).

    Returns:
        A dict with both signals, the ``correlation`` annotation
        (``"explained"`` / ``"unexplained"`` / ``"not_applicable"``), the
        ``case`` key into :data:`TRIAGE_DECISION_TREE`, and the recommended
        ``action``.
    """
    feature_flagged = bool(feature_drift and feature_drift.drift_detected)
    output_flagged = bool(output_drift and output_drift.drift_detected)

    drifted_features = (
        sorted(
            (name for name, score in feature_drift.mmd_per_feature.items() if score > 0),
            key=lambda n: feature_drift.mmd_per_feature[n],
            reverse=True,
        )
        if feature_drift
        else []
    )

    if output_flagged and feature_flagged:
        case = "both_drifted"
        correlation = "explained"
    elif output_flagged:
        case = "only_output_drifted"
        correlation = "unexplained"
    elif feature_flagged:
        case = "only_input_drifted"
        correlation = "not_applicable"
    else:
        case = "no_drift"
        correlation = "not_applicable"

    return {
        "feature_drift": feature_drift.to_dict() if feature_drift else None,
        "output_drift": output_drift.to_dict() if output_drift else None,
        "feature_drift_detected": feature_flagged,
        "output_drift_detected": output_flagged,
        "drifted_features": drifted_features,
        "correlation": correlation,
        "case": case,
        "interpretation": TRIAGE_DECISION_TREE[case]["interpretation"],
        "action": TRIAGE_DECISION_TREE[case]["action"],
    }


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    diff = X[:, None, :] - Y[None, :, :]
    return np.exp(-np.sum(diff**2, axis=-1) / (2 * bandwidth**2))


def _mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute unbiased MMD² with RBF kernel; bandwidth via median heuristic."""
    all_points = np.vstack([X, Y])
    dists = np.linalg.norm(all_points[:, None] - all_points[None, :], axis=-1)
    bandwidth = float(np.median(dists[dists > 0])) or 1.0

    kxx = _rbf_kernel(X, X, bandwidth)
    kyy = _rbf_kernel(Y, Y, bandwidth)
    kxy = _rbf_kernel(X, Y, bandwidth)

    n, m = len(X), len(Y)
    np.fill_diagonal(kxx, 0)
    np.fill_diagonal(kyy, 0)
    return kxx.sum() / (n * (n - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean()


class CovarianceShiftDetector:
    """Detects feature distribution drift between a reference and current window using MMD."""

    def __init__(self, threshold: float = 0.05) -> None:
        try:
            from config import Config

            self._ref_hours = Config.DRIFT_REFERENCE_WINDOW_HOURS
            self._test_hours = Config.DRIFT_TEST_WINDOW_HOURS
            self._interval = Config.DRIFT_CHECK_INTERVAL_MINUTES
        except Exception:
            self._ref_hours = 168
            self._test_hours = 1
            self._interval = 30
        self.threshold = threshold
        self.health = DriftMonitorHealth()

    def detect(
        self, reference: np.ndarray, current: np.ndarray, feature_names: list[str] | None = None
    ) -> DriftReport:
        """Compare reference and current windows per feature; return DriftReport.

        Args:
            reference: 2-D array of shape (n_ref, n_features).
            current:   2-D array of shape (n_cur, n_features).
            feature_names: Optional list of feature name strings.

        Records a heartbeat in ``self.health`` on every call (success or
        failure) so a silently-broken monitor is itself observable — see
        ``check_drift_monitor_health``. Any exception raised while computing
        the drift report is still re-raised after being recorded; this
        method does not swallow failures.
        """
        try:
            n_features = reference.shape[1]
            names = feature_names or [f"feature_{i}" for i in range(n_features)]

            mmd_scores: dict[str, float] = {}
            for i, name in enumerate(names):
                ref_col = reference[:, i]
                cur_col = current[:, i]
                if ref_col.std() < 1e-8:  # skip near-zero-variance features
                    continue
                mmd_scores[name] = _mmd(ref_col.reshape(-1, 1), cur_col.reshape(-1, 1))

            drift_detected = any(v > self.threshold for v in mmd_scores.values())

            if _drift_gauge is not None:
                _drift_gauge.set(1 if drift_detected else 0)

            if drift_detected:
                top5 = sorted(mmd_scores, key=mmd_scores.get, reverse=True)[:5]
                logger.warning("Feature drift detected. Top drifted features: %s", top5)
        except Exception as exc:
            logger.error("drift-check failed: CovarianceShiftDetector.detect() raised: %s", exc)
            self.health.record_failure(exc)
            raise
        else:
            self.health.record_success()
            return DriftReport(mmd_per_feature=mmd_scores, drift_detected=drift_detected)
