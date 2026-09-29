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


def check_drift_monitor_health(
    detector: CovarianceShiftDetector,
    max_age_seconds: float | None = None,
) -> dict:
    """Evaluate *detector*'s heartbeat and log a distinct "drift-check failed"
    alert if it is stale (never succeeded, or last succeeded too long ago).

    Intended to be called periodically (e.g. a Kubernetes CronJob or the
    ``python -m monitoring.drift_detector --heartbeat-check`` CLI below) from
    a process that shares the long-lived ``detector`` instance with whatever
    calls ``detect()`` in the live pipeline. Returns the same dict as
    ``DriftMonitorHealth.status()``.
    """
    from config import config

    max_age = (
        max_age_seconds
        if max_age_seconds is not None
        else config.DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS
    )
    status = detector.health.status(max_age)
    if status["stale"]:
        logger.critical(
            "drift-check failed: drift monitor heartbeat is stale (last success=%s, "
            "consecutive_failures=%d, max_age=%ss) — feature drift detection may not be "
            "running; treat as if drift monitoring is DOWN until resolved",
            status["last_success_at"],
            status["consecutive_failures"],
            max_age,
        )
    return status


def _cli() -> int:  # pragma: no cover — thin operational wrapper
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Drift monitor heartbeat/health check")
    parser.add_argument("--heartbeat-check", action="store_true", default=False)
    parser.add_argument("--max-age-seconds", type=float, default=None)
    args = parser.parse_args()

    if not args.heartbeat_check:
        parser.print_help()
        return 1

    # A standalone CLI invocation has no long-lived detector to inspect; this
    # reports "stale" by construction, documenting the *shape* of the check
    # for operators wiring it into a real scheduler that shares the
    # in-process detector with the live pipeline (see docstring above).
    detector = CovarianceShiftDetector()
    status = check_drift_monitor_health(detector, max_age_seconds=args.max_age_seconds)
    print(json.dumps(status, indent=2))
    return 1 if status["stale"] else 0


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(_cli())
