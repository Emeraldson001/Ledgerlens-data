"""
ci_metrics/ — CI regression trend monitoring hooks.

Records test/benchmark metrics from each CI run into a persistent JSON store
and exposes trend analysis to detect regressions before they ship.

Public surface::

    from ci_metrics import MetricsStore, CIRunRecord, RegressionDetector, record_run

Key concepts:

  CIRunRecord   Typed dataclass capturing all metrics for one CI run.
  MetricsStore  Append-only JSON-line store; thread-safe via file locking.
  RegressionDetector  Compares latest run against a rolling baseline and flags
                      metrics that have degraded beyond configurable thresholds.
  record_run    One-call helper used by CI scripts to append a record and
                immediately check for regressions.

See ci_metrics/store.py and ci_metrics/regression.py for implementation details.
"""

from ci_metrics.contracts import CIRunRecord, MetricSnapshot, RegressionAlert
from ci_metrics.regression import RegressionDetector, record_run
from ci_metrics.store import MetricsStore

__all__ = [
    "CIRunRecord",
    "MetricSnapshot",
    "MetricsStore",
    "RegressionAlert",
    "RegressionDetector",
    "record_run",
]
