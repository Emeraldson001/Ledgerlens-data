"""Tests for evaluation/run_comparator.py — Issue #534.

Covers:
* RunMetrics.load from JSON file
* RunMetrics.load missing file raises FileNotFoundError
* RunMetrics.load non-JSON raises ValueError
* _flatten_metrics: nested dicts, non-numeric skipped
* ModelRunComparator.compare_paths: basic diff
* Diff status classification: improved / regressed / unchanged / added / removed
* Regression detection against tolerance
* compare_all across multiple runs
* list_runs
* ComparisonReport.summary text output
* ComparisonReport.save JSON output
* ComparisonReport.has_regressions
* Higher-is-better vs lower-is-better metric direction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation.run_comparator import (
    ComparisonReport,
    ModelRunComparator,
    RunMetrics,
    _flatten_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_metrics(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)
    return path


def _run_dir(root: Path, run_id: str, metrics: dict[str, Any]) -> Path:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    _write_metrics(d / "metrics.json", metrics)
    return d


# ---------------------------------------------------------------------------
# _flatten_metrics
# ---------------------------------------------------------------------------


class TestFlattenMetrics:
    def test_flat_dict(self):
        result = _flatten_metrics({"auc": 0.9, "f1": 0.8})
        assert result == {"auc": 0.9, "f1": 0.8}

    def test_nested_dict(self):
        result = _flatten_metrics({"models": {"rf": {"auc": 0.88}, "xgb": {"auc": 0.91}}})
        assert "models.rf.auc" in result
        assert "models.xgb.auc" in result
        assert result["models.rf.auc"] == 0.88

    def test_non_numeric_skipped(self):
        result = _flatten_metrics({"name": "run_1", "auc": 0.9, "flag": True})
        assert "name" not in result
        assert "flag" not in result
        assert result["auc"] == 0.9

    def test_int_values_included(self):
        result = _flatten_metrics({"n_samples": 500})
        assert result["n_samples"] == 500.0

    def test_empty_dict(self):
        assert _flatten_metrics({}) == {}

    def test_deeply_nested(self):
        result = _flatten_metrics({"a": {"b": {"c": {"d": 1.0}}}})
        assert "a.b.c.d" in result


# ---------------------------------------------------------------------------
# RunMetrics
# ---------------------------------------------------------------------------


class TestRunMetrics:
    def test_load_flat_metrics(self, tmp_path):
        path = _write_metrics(tmp_path / "metrics.json", {"auc": 0.9, "f1": 0.85})
        rm = RunMetrics.load(path)
        assert rm.metrics["auc"] == 0.9
        assert rm.metrics["f1"] == 0.85

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RunMetrics.load(tmp_path / "missing.json")

    def test_load_non_json_raises(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not json {{")
        with pytest.raises(ValueError):
            RunMetrics.load(bad)

    def test_load_non_object_raises(self, tmp_path):
        path = tmp_path / "array.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="JSON object"):
            RunMetrics.load(path)

    def test_run_id_inferred_from_parent(self, tmp_path):
        run_dir = tmp_path / "run_20240601"
        run_dir.mkdir()
        path = _write_metrics(run_dir / "metrics.json", {"auc": 0.9})
        rm = RunMetrics.load(path)
        assert rm.run_id == "run_20240601"

    def test_explicit_run_id(self, tmp_path):
        path = _write_metrics(tmp_path / "metrics.json", {"auc": 0.9})
        rm = RunMetrics.load(path, run_id="custom_id")
        assert rm.run_id == "custom_id"

    def test_nested_metrics_flattened(self, tmp_path):
        data = {"random_forest": {"auc": 0.88}, "xgboost": {"auc": 0.91}}
        path = _write_metrics(tmp_path / "metrics.json", data)
        rm = RunMetrics.load(path)
        assert "random_forest.auc" in rm.metrics
        assert "xgboost.auc" in rm.metrics

    def test_to_dict(self, tmp_path):
        path = _write_metrics(tmp_path / "metrics.json", {"auc": 0.9})
        rm = RunMetrics.load(path)
        d = rm.to_dict()
        assert "run_id" in d
        assert "metrics" in d


# ---------------------------------------------------------------------------
# ModelRunComparator — compare_paths
# ---------------------------------------------------------------------------


class TestModelRunComparatorPaths:
    def test_basic_compare(self, tmp_path):
        baseline = _write_metrics(tmp_path / "baseline.json", {"auc": 0.85, "f1": 0.80})
        candidate = _write_metrics(tmp_path / "candidate.json", {"auc": 0.87, "f1": 0.82})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        assert len(report.diffs) == 2
        assert report.baseline_run_id is not None
        assert report.candidate_run_id is not None

    def test_improved_metric_detected(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.85})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.90})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        auc_diff = next(d for d in report.diffs if d.metric == "auc")
        assert auc_diff.status == "improved"
        assert auc_diff.delta == pytest.approx(0.05)

    def test_regressed_metric_detected(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.85})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        auc_diff = next(d for d in report.diffs if d.metric == "auc")
        assert auc_diff.status == "regressed"

    def test_unchanged_metric(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"f1": 0.80})
        candidate = _write_metrics(tmp_path / "c.json", {"f1": 0.80})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        f1_diff = next(d for d in report.diffs if d.metric == "f1")
        assert f1_diff.status == "unchanged"
        assert f1_diff.delta == 0.0

    def test_added_metric(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.85})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.85, "pr_auc": 0.80})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        added = [d for d in report.diffs if d.status == "added"]
        assert any(d.metric == "pr_auc" for d in added)

    def test_removed_metric(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.85, "old_metric": 0.5})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.85})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        removed = [d for d in report.diffs if d.status == "removed"]
        assert any(d.metric == "old_metric" for d in removed)

    def test_delta_pct_computed(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.80})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.88})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        auc_diff = next(d for d in report.diffs if d.metric == "auc")
        assert auc_diff.delta_pct == pytest.approx(10.0, rel=0.01)

    def test_delta_pct_none_when_baseline_zero(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"loss": 0.0})
        candidate = _write_metrics(tmp_path / "c.json", {"loss": 0.1})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        loss_diff = next(d for d in report.diffs if d.metric == "loss")
        assert loss_diff.delta_pct is None

    def test_lower_is_better_loss_improved(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"loss": 0.5})
        candidate = _write_metrics(tmp_path / "c.json", {"loss": 0.3})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        loss_diff = next(d for d in report.diffs if d.metric == "loss")
        assert loss_diff.status == "improved"

    def test_lower_is_better_loss_regressed(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"loss": 0.3})
        candidate = _write_metrics(tmp_path / "c.json", {"loss": 0.5})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        loss_diff = next(d for d in report.diffs if d.metric == "loss")
        assert loss_diff.status == "regressed"


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------


class TestRegressionDetection:
    def test_regression_flagged_above_tolerance(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.85})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        assert len(report.regressions) == 1
        assert report.regressions[0].metric == "auc"
        assert report.has_regressions is True

    def test_regression_not_flagged_within_tolerance(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.895})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        assert len(report.regressions) == 0
        assert report.has_regressions is False

    def test_regression_tolerance_in_metadata(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.80})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.05)
        assert report.metadata["regression_tolerance"] == 0.05

    def test_regression_flag_has_correct_fields(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.80})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        flag = report.regressions[0]
        assert flag.baseline_value == pytest.approx(0.90)
        assert flag.candidate_value == pytest.approx(0.80)
        assert flag.delta == pytest.approx(-0.10)

    def test_multiple_regressions(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90, "f1": 0.85})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.80, "f1": 0.75})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        assert len(report.regressions) == 2


# ---------------------------------------------------------------------------
# Directory-based comparison
# ---------------------------------------------------------------------------


class TestModelRunComparatorDirectory:
    def test_compare_by_run_id(self, tmp_path):
        _run_dir(tmp_path, "run_A", {"auc": 0.85})
        _run_dir(tmp_path, "run_B", {"auc": 0.90})
        comp = ModelRunComparator(runs_dir=tmp_path)
        report = comp.compare("run_A", "run_B")
        auc_diff = next(d for d in report.diffs if d.metric == "auc")
        assert auc_diff.status == "improved"

    def test_compare_without_runs_dir_raises(self):
        comp = ModelRunComparator()
        with pytest.raises(RuntimeError, match="runs_dir not set"):
            comp.compare("run_A", "run_B")

    def test_list_runs(self, tmp_path):
        _run_dir(tmp_path, "run_A", {"auc": 0.85})
        _run_dir(tmp_path, "run_B", {"auc": 0.90})
        # Add a directory without metrics.json — should be excluded
        (tmp_path / "no_metrics").mkdir()
        comp = ModelRunComparator(runs_dir=tmp_path)
        runs = comp.list_runs()
        assert "run_A" in runs
        assert "run_B" in runs
        assert "no_metrics" not in runs

    def test_list_runs_empty_dir(self, tmp_path):
        comp = ModelRunComparator(runs_dir=tmp_path / "empty")
        assert comp.list_runs() == []

    def test_list_runs_no_runs_dir_raises(self):
        comp = ModelRunComparator()
        with pytest.raises(RuntimeError, match="runs_dir not set"):
            comp.list_runs()

    def test_compare_all_consecutive_pairs(self, tmp_path):
        _run_dir(tmp_path, "run_1", {"auc": 0.80})
        _run_dir(tmp_path, "run_2", {"auc": 0.83})
        _run_dir(tmp_path, "run_3", {"auc": 0.86})
        comp = ModelRunComparator(runs_dir=tmp_path)
        reports = comp.compare_all()
        assert len(reports) == 2
        assert reports[0].baseline_run_id == "run_1"
        assert reports[0].candidate_run_id == "run_2"
        assert reports[1].baseline_run_id == "run_2"
        assert reports[1].candidate_run_id == "run_3"

    def test_compare_all_single_run_returns_empty(self, tmp_path):
        _run_dir(tmp_path, "run_1", {"auc": 0.80})
        comp = ModelRunComparator(runs_dir=tmp_path)
        reports = comp.compare_all()
        assert reports == []


# ---------------------------------------------------------------------------
# ComparisonReport output
# ---------------------------------------------------------------------------


class TestComparisonReport:
    def _make_report(self, tmp_path: Path) -> ComparisonReport:
        baseline = _write_metrics(
            tmp_path / "b.json",
            {"auc": 0.90, "f1": 0.85, "loss": 0.20},
        )
        candidate = _write_metrics(
            tmp_path / "c.json",
            {"auc": 0.88, "f1": 0.87, "loss": 0.18, "pr_auc": 0.83},
        )
        comp = ModelRunComparator()
        return comp.compare_paths(baseline, candidate, regression_tolerance=0.01)

    def test_summary_contains_run_ids(self, tmp_path):
        report = self._make_report(tmp_path)
        s = report.summary()
        assert "baseline" in s.lower() or report.baseline_run_id in s
        assert "candidate" in s.lower() or report.candidate_run_id in s

    def test_summary_contains_regression_info(self, tmp_path):
        report = self._make_report(tmp_path)
        s = report.summary()
        assert "REGRESSION" in s or "regressions" in s.lower()

    def test_summary_lists_improvements(self, tmp_path):
        report = self._make_report(tmp_path)
        s = report.summary()
        assert "IMPROVEMENT" in s or "✓" in s

    def test_to_dict_structure(self, tmp_path):
        report = self._make_report(tmp_path)
        d = report.to_dict()
        assert "baseline_run_id" in d
        assert "candidate_run_id" in d
        assert "diffs" in d
        assert "regressions" in d
        assert "summary" in d
        assert "generated_at" in d

    def test_save_creates_json_file(self, tmp_path):
        report = self._make_report(tmp_path)
        out_path = tmp_path / "report.json"
        result = report.save(out_path)
        assert result == out_path
        assert out_path.exists()
        with out_path.open() as f:
            data = json.load(f)
        assert "diffs" in data

    def test_save_creates_parent_dirs(self, tmp_path):
        report = self._make_report(tmp_path)
        nested = tmp_path / "reports" / "2024" / "run_comparison.json"
        report.save(nested)
        assert nested.exists()

    def test_has_regressions_true(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.80})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        assert report.has_regressions is True

    def test_has_regressions_false(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.92})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, regression_tolerance=0.01)
        assert report.has_regressions is False

    def test_added_property(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.90, "new_metric": 0.5})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        assert len(report.added) == 1
        assert report.added[0].metric == "new_metric"

    def test_removed_property(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90, "old": 0.5})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.90})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate)
        assert len(report.removed) == 1
        assert report.removed[0].metric == "old"

    def test_metadata_stored_in_report(self, tmp_path):
        baseline = _write_metrics(tmp_path / "b.json", {"auc": 0.90})
        candidate = _write_metrics(tmp_path / "c.json", {"auc": 0.92})
        comp = ModelRunComparator()
        report = comp.compare_paths(baseline, candidate, metadata={"experiment": "issue-534"})
        assert report.metadata.get("experiment") == "issue-534"
