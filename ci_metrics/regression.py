"""
ci_metrics/regression.py — Regression detector for CI metric trends.

The :class:`RegressionDetector` compares the latest :class:`CIRunRecord`
against a rolling baseline (mean of the previous N records for the same
branch) and emits :class:`RegressionAlert` objects for metrics that have
degraded beyond configurable thresholds.

Threshold logic:
  - ``warning_pct``  (default 5%):  relative degradation that triggers a WARNING.
  - ``critical_pct`` (default 15%): relative degradation that triggers a CRITICAL.
  - "Degradation" for ``higher_is_better`` metrics = value dropped below baseline.
  - "Degradation" for ``lower_is_better`` metrics = value rose above baseline.

Usage::

    store = MetricsStore()
    detector = RegressionDetector(store, baseline_window=10)
    alerts = detector.check(latest_record)
    for alert in alerts:
        print(alert.message)
"""

from __future__ import annotations

import logging
from pathlib import Path

from ci_metrics.contracts import CIRunRecord, MetricSnapshot, RegressionAlert
from ci_metrics.store import MetricsStore

logger = logging.getLogger(__name__)

DEFAULT_WARNING_PCT: float = 5.0
DEFAULT_CRITICAL_PCT: float = 15.0
DEFAULT_BASELINE_WINDOW: int = 10


class RegressionDetector:
    """Detect metric regressions by comparing a run against its rolling baseline.

    Args:
        store: :class:`MetricsStore` used to retrieve historical records.
        baseline_window: Number of prior runs to average for the baseline.
        warning_pct: Relative degradation (%) that triggers a WARNING alert.
        critical_pct: Relative degradation (%) that triggers a CRITICAL alert.
    """

    def __init__(
        self,
        store: MetricsStore,
        *,
        baseline_window: int = DEFAULT_BASELINE_WINDOW,
        warning_pct: float = DEFAULT_WARNING_PCT,
        critical_pct: float = DEFAULT_CRITICAL_PCT,
    ) -> None:
        self.store = store
        self.baseline_window = baseline_window
        self.warning_pct = warning_pct
        self.critical_pct = critical_pct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, latest: CIRunRecord) -> list[RegressionAlert]:
        """Compare *latest* against the rolling baseline.

        Returns a (possibly empty) list of :class:`RegressionAlert` objects.
        An empty list means no regressions were detected.
        """
        # Fetch historical records for the same branch, excluding the latest run
        history = [r for r in self.store.for_branch(latest.branch) if r.run_id != latest.run_id][
            -self.baseline_window :
        ]

        if not history:
            logger.info(
                "No baseline history for branch '%s'; skipping regression check.",
                latest.branch,
            )
            return []

        # Build a map: metric_name -> list of historical values
        historical_values: dict[str, list[float]] = {}
        for record in history:
            for m in record.metrics:
                historical_values.setdefault(m.name, []).append(m.value)

        alerts: list[RegressionAlert] = []
        for metric in latest.metrics:
            hist = historical_values.get(metric.name)
            if not hist:
                continue  # no prior data for this metric on this branch
            baseline = sum(hist) / len(hist)
            alert = self._evaluate(metric, baseline)
            if alert is not None:
                alerts.append(alert)
                logger.warning("[%s] %s", alert.severity.upper(), alert.message)

        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate(self, metric: MetricSnapshot, baseline: float) -> RegressionAlert | None:
        """Return an alert if *metric* has regressed relative to *baseline*."""
        if baseline == 0.0:
            return None  # can't compute percentage change

        if metric.higher_is_better:
            # Regression = current < baseline
            delta_pct = ((metric.value - baseline) / abs(baseline)) * 100.0
        else:
            # Regression = current > baseline  (e.g. runtime, error rate)
            delta_pct = ((baseline - metric.value) / abs(baseline)) * 100.0

        # delta_pct < 0 means regression
        if delta_pct >= -self.warning_pct:
            return None  # within tolerance

        severity = "critical" if delta_pct <= -self.critical_pct else "warning"
        direction = "dropped" if metric.higher_is_better else "increased"
        message = (
            f"Metric '{metric.name}' has {direction} by "
            f"{abs(delta_pct):.1f}% "
            f"(latest={metric.value:.4f}, baseline={baseline:.4f}) "
            f"[{severity.upper()}]"
        )
        return RegressionAlert(
            metric_name=metric.name,
            latest_value=metric.value,
            baseline_value=round(baseline, 6),
            delta_pct=round(delta_pct, 2),
            severity=severity,  # type: ignore[arg-type]
            message=message,
        )


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def record_run(
    record: CIRunRecord,
    store_path: Path | str | None = None,
    *,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
    warning_pct: float = DEFAULT_WARNING_PCT,
    critical_pct: float = DEFAULT_CRITICAL_PCT,
    fail_on_critical: bool = False,
) -> list[RegressionAlert]:
    """Append *record* to the store and return any regression alerts.

    This is the single entry-point used by CI post-test hooks.

    Args:
        record: The :class:`CIRunRecord` to persist.
        store_path: Override the default store path.
        baseline_window: Rolling window for baseline computation.
        warning_pct: Degradation % that triggers a WARNING.
        critical_pct: Degradation % that triggers a CRITICAL.
        fail_on_critical: If ``True``, raise ``RuntimeError`` when any
            CRITICAL alert is emitted.  Use this as a hard CI gate.

    Returns:
        List of :class:`RegressionAlert` objects (empty = no regressions).

    Raises:
        RuntimeError: If *fail_on_critical* is ``True`` and a CRITICAL alert
            was emitted — so CI can fail the step with a clear diagnostic.
    """
    from ci_metrics.store import DEFAULT_STORE_PATH

    path = Path(store_path) if store_path is not None else DEFAULT_STORE_PATH
    store = MetricsStore(path)
    store.append(record)

    detector = RegressionDetector(
        store,
        baseline_window=baseline_window,
        warning_pct=warning_pct,
        critical_pct=critical_pct,
    )
    alerts = detector.check(record)

    critical = [a for a in alerts if a.severity == "critical"]
    if fail_on_critical and critical:
        msgs = "\n".join(a.message for a in critical)
        raise RuntimeError(
            f"CI regression detected — {len(critical)} CRITICAL metric(s) degraded:\n{msgs}"
        )

    return alerts
