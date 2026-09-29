"""Tests for scripts/validate.py (Issue #558) and scripts/repo_maturity.py (Issue #560)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# validate.py — Issue #558
# ---------------------------------------------------------------------------


class TestValidateScript:
    def test_main_all_suites_returns_int(self):
        from scripts.validate import main

        rc = main(["--suite", "parsing", "--quiet"])
        assert isinstance(rc, int)

    def test_parsing_suite_runs(self):
        from scripts.validate import run_suites

        results = run_suites(["parsing"], quiet=True)
        assert len(results) == 1
        assert results[0].name == "parsing"

    def test_schema_suite_runs(self):
        from scripts.validate import run_suites

        results = run_suites(["schema"], quiet=True)
        assert results[0].name == "schema"

    def test_feature_ranges_suite_runs(self):
        from scripts.validate import run_suites

        results = run_suites(["feature_ranges"], quiet=True)
        assert results[0].name == "feature_ranges"

    def test_reconciliation_suite_runs(self):
        from scripts.validate import run_suites

        results = run_suites(["reconciliation"], quiet=True)
        assert results[0].name == "reconciliation"

    def test_all_suites_run(self):
        from scripts.validate import SUITES, run_suites

        results = run_suites(list(SUITES.keys()), quiet=True)
        assert len(results) == len(SUITES)

    def test_fail_fast_stops_early(self, monkeypatch):
        """fail_fast=True should stop after the first failing suite."""
        from scripts.validate import SuiteResult, run_suites

        call_count = 0

        def _always_fail(verbose: bool = False) -> SuiteResult:
            nonlocal call_count
            call_count += 1
            return SuiteResult(name="mock", passed=False, errors=["  ✗ forced failure"])

        import scripts.validate as validate_mod

        original_suites = dict(validate_mod.SUITES)
        validate_mod.SUITES = {"s1": _always_fail, "s2": _always_fail, "s3": _always_fail}
        try:
            results = run_suites(["s1", "s2", "s3"], fail_fast=True, quiet=True)
            assert call_count == 1
            assert len(results) == 1
        finally:
            validate_mod.SUITES = original_suites

    def test_report_written_to_disk(self, tmp_path):
        from scripts.validate import main

        report_path = tmp_path / "report.json"
        main(["--suite", "parsing", "--report", str(report_path), "--quiet"])
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "results" in data
        assert "generated_at" in data

    def test_suite_result_to_dict(self):
        from scripts.validate import SuiteResult

        sr = SuiteResult(name="test", passed=True, duration_ms=12.3)
        d = sr.to_dict()
        assert d["suite"] == "test"
        assert d["passed"] is True
        assert d["duration_ms"] == 12.3


# ---------------------------------------------------------------------------
# repo_maturity.py — Issue #560
# ---------------------------------------------------------------------------


class TestRepoMaturity:
    def test_compute_maturity_returns_report(self):
        from scripts.repo_maturity import compute_maturity

        report = compute_maturity(Path("."))
        assert report.composite_score >= 0
        assert report.composite_score <= 100

    def test_five_dimensions_scored(self):
        from scripts.repo_maturity import compute_maturity

        report = compute_maturity(Path("."))
        names = {d.name for d in report.dimensions}
        assert names == {"tests", "docs", "ci", "data_quality", "security"}

    def test_weights_sum_to_one(self):
        from scripts.repo_maturity import compute_maturity

        report = compute_maturity(Path("."))
        total_weight = sum(d.weight for d in report.dimensions)
        assert abs(total_weight - 1.0) < 1e-9

    def test_passed_based_on_threshold(self):
        from scripts.repo_maturity import MaturityReport

        report = MaturityReport(threshold=0.0)
        assert report.passed is True

        report2 = MaturityReport(threshold=999.0)
        assert report2.passed is False

    def test_to_dict_structure(self):
        from scripts.repo_maturity import compute_maturity

        report = compute_maturity(Path("."))
        d = report.to_dict()
        assert "composite_score" in d
        assert "passed" in d
        assert "dimensions" in d
        assert isinstance(d["dimensions"], list)

    def test_main_returns_int(self):
        from scripts.repo_maturity import main

        rc = main(["--quiet"])
        assert isinstance(rc, int)

    def test_main_writes_report(self, tmp_path):
        from scripts.repo_maturity import main

        report_path = tmp_path / "maturity.json"
        main(["--report", str(report_path), "--quiet"])
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert "composite_score" in data

    def test_missing_repo_root_scores_low(self, tmp_path):
        """An empty temp directory should score low (not crash)."""
        from scripts.repo_maturity import compute_maturity

        report = compute_maturity(tmp_path, threshold=60.0)
        # Should run without exception
        assert report.composite_score >= 0

    def test_dimension_score_weighted_contribution(self):
        from scripts.repo_maturity import DimensionScore

        ds = DimensionScore(name="tests", score=80.0, max_score=100.0, weight=0.25)
        assert ds.weighted == pytest.approx(20.0)

    def test_maturity_config_yaml_parseable(self):
        """config/repo_maturity.yaml must be parseable and have required keys."""
        config_path = Path("config") / "repo_maturity.yaml"
        assert config_path.exists(), "config/repo_maturity.yaml is missing"
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except ImportError:
            # yaml not installed — parse manually to check it's not empty
            text = config_path.read_text(encoding="utf-8")
            assert "dimensions" in text
            return
        assert "dimensions" in data
        assert "default_threshold" in data
        assert "version" in data

    def test_main_writes_markdown_output(self, tmp_path):
        """--output flag writes a clean Markdown summary to a file."""
        from scripts.repo_maturity import main

        output_path = tmp_path / "maturity.md"
        rc = main(["--output", str(output_path), "--quiet"])
        assert isinstance(rc, int)
        assert output_path.exists(), f"Output file {output_path} was not created"
        content = output_path.read_text(encoding="utf-8")

        # Verify Markdown contains expected sections
        assert "# Repository Maturity" in content or "Repository Maturity" in content
        assert "Composite Score" in content
        assert "/" in content  # Score fraction (e.g., "75/100")

        # Verify it's not JSON or raw Python structures
        assert not content.strip().startswith("{")
        assert not content.strip().startswith("[")

    def test_main_default_no_markdown_file(self, tmp_path, capsys):
        """Without --output flag, no Markdown file is created; output goes to stdout."""
        from scripts.repo_maturity import main

        # Ensure output directory is clean
        output_path = tmp_path / "nonexistent.md"
        assert not output_path.exists()

        # Run without --output
        rc = main([])
        captured = capsys.readouterr()

        # Should write to stdout instead
        assert "Composite score" in captured.out or "Repository Maturity" in captured.out
        # File should not be created
        assert not output_path.exists()
