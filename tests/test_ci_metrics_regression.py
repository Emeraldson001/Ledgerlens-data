"""Regression-detection tests for ci_metrics/regression.py (Issue #800)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ci_metrics import CIRunRecord, MetricSnapshot, record_run
from ci_metrics.contracts import RegressionAlert
from ci_metrics.regression import DEFAULT_CRITICAL_PCT, DEFAULT_WARNING_PCT, RegressionDetector
from ci_metrics.store import MetricsStore


def _make_store(tmp_path: Path, n_baseline: int = 5) -> tuple[MetricsStore, CIRunRecord]:
    store_path = tmp_path / "history.jsonl"
    branch = "test-branch"
    for i in range(n_baseline):
        rec = CIRunRecord(
            run_id=f"baseline-{i}",
            commit_sha="abc123",
            branch=branch,
            timestamp_utc="2024-01-01T00:00:00Z",
            metrics=[MetricSnapshot(name="test_pass_rate", value=1.0)],
        )
        record_run(rec, store_path=store_path)
    store = MetricsStore(store_path)
    return store, branch


class TestRegressionFlagsDegradedMetric:
    """A metric that dropped several points below baseline should be flagged."""

    def test_critical_regression_higher_is_better(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, branch = _make_store(Path(tmp), n_baseline=5)
            latest = CIRunRecord(
                run_id="latest-degraded",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="test_pass_rate", value=0.80)],
            )
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.severity == "critical"
            assert alert.metric_name == "test_pass_rate"
            assert alert.latest_value == pytest.approx(0.80)
            assert alert.delta_pct < -DEFAULT_CRITICAL_PCT

    def test_warning_regression_within_critical_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "history.jsonl"
            branch = "test-branch-warn"
            for i in range(5):
                rec = CIRunRecord(
                    run_id=f"baseline-warn-{i}",
                    commit_sha="abc123",
                    branch=branch,
                    timestamp_utc="2024-01-01T00:00:00Z",
                    metrics=[MetricSnapshot(name="coverage", value=90.0)],
                )
                record_run(rec, store_path=store_path)
            # ~8.9% drop — triggers WARNING (5%) but not CRITICAL (15%)
            latest = CIRunRecord(
                run_id="latest-warn",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="coverage", value=82.0)],
            )
            store = MetricsStore(store_path)
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.severity == "warning"
            assert alert.metric_name == "coverage"


class TestNoFalsePositives:
    """Metrics that improved or stayed within tolerance must not be flagged."""

    def test_within_tolerance_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "history.jsonl"
            branch = "test-branch-ok"
            for i in range(5):
                rec = CIRunRecord(
                    run_id=f"baseline-ok-{i}",
                    commit_sha="abc123",
                    branch=branch,
                    timestamp_utc="2024-01-01T00:00:00Z",
                    metrics=[MetricSnapshot(name="runtime_s", value=10.0, higher_is_better=False)],
                )
                record_run(rec, store_path=store_path)
            # 4% increase is within the 5% warning threshold
            latest = CIRunRecord(
                run_id="latest-ok",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="runtime_s", value=10.4, higher_is_better=False)],
            )
            store = MetricsStore(store_path)
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert alerts == []

    def test_improved_metric_not_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store, branch = _make_store(Path(tmp), n_baseline=5)
            latest = CIRunRecord(
                run_id="latest-improved",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="test_pass_rate", value=1.0)],
            )
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert all(a.severity != "critical" for a in alerts)

    def test_regressed_lower_is_better_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "history.jsonl"
            branch = "test-branch-lower"
            for i in range(5):
                rec = CIRunRecord(
                    run_id=f"baseline-lower-{i}",
                    commit_sha="abc123",
                    branch=branch,
                    timestamp_utc="2024-01-01T00:00:00Z",
                    metrics=[MetricSnapshot(name="mutation_score", value=85.0, higher_is_better=False)],
                )
                record_run(rec, store_path=store_path)
            # 23.5% increase for a lower-is-better metric is a regression
            latest = CIRunRecord(
                run_id="latest-lower",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="mutation_score", value=105.0, higher_is_better=False)],
            )
            store = MetricsStore(store_path)
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert len(alerts) == 1
            alert = alerts[0]
            assert alert.severity == "critical"

    def test_zero_baseline_skips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "history.jsonl"
            branch = "test-branch-zero"
            for i in range(3):
                rec = CIRunRecord(
                    run_id=f"baseline-zero-{i}",
                    commit_sha="abc123",
                    branch=branch,
                    timestamp_utc="2024-01-01T00:00:00Z",
                    metrics=[MetricSnapshot(name="new_metric", value=0.0)],
                )
                record_run(rec, store_path=store_path)
            latest = CIRunRecord(
                run_id="latest-zero",
                commit_sha="def456",
                branch=branch,
                timestamp_utc="2024-01-02T00:00:00Z",
                metrics=[MetricSnapshot(name="new_metric", value=1.0)],
            )
            store = MetricsStore(store_path)
            store.append(latest)
            detector = RegressionDetector(store)
            alerts = detector.check(latest)
            assert alerts == []
