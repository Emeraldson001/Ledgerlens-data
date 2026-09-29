"""Tests for scripts/quality_metrics.py (Issue #604).

Covers:
- Individual metric collectors (coverage, mutation, critical_pass, drift, cycle_time)
- compute_composite_score with custom weights
- QualityReport.to_dict and .summary
- build_badge_json color thresholds
- CLI main() exit codes and --skip-missing behavior
"""

from __future__ import annotations

import json
import sqlite3
import textwrap
from pathlib import Path
from unittest.mock import patch

from scripts.quality_metrics import (
    DEFAULT_WEIGHTS,
    MetricReading,
    QualityReport,
    build_badge_json,
    collect_critical_pass_rate,
    collect_cycle_time,
    collect_drift_rate,
    collect_mutation_score,
    collect_test_coverage,
    compute_composite_score,
    main,
    report_to_csv,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_cobertura(tmp_path: Path, line_rate: float) -> Path:
    xml = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <coverage line-rate="{line_rate}" branch-rate="0.5" version="7.0">
          <packages />
        </coverage>
    """)
    p = tmp_path / "coverage.xml"
    p.write_text(xml)
    return p


def _write_junit(tmp_path: Path, cases: list[dict]) -> Path:
    parts = ['<?xml version="1.0"?>\n<testsuites>\n  <testsuite name="pytest">']
    for c in cases:
        classname = c.get("classname", "tests.test_example")
        name = c.get("name", "test_x")
        if c.get("outcome", "passed") == "passed":
            parts.append(f'    <testcase classname="{classname}" name="{name}" />')
        else:
            msg = c.get("message", "fail")
            parts.append(
                f'    <testcase classname="{classname}" name="{name}">'
                f'<failure message="{msg}" /></testcase>'
            )
    parts.append("  </testsuite>\n</testsuites>")
    p = tmp_path / "junit.xml"
    p.write_text("\n".join(parts))
    return p


def _write_taxonomy(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "test_criticality.json"
    p.write_text(json.dumps({"version": 1, "entries": entries}))
    return p


def _write_mutmut_cache(tmp_path: Path, results: list[str]) -> Path:
    p = tmp_path / ".mutmut-cache"
    conn = sqlite3.connect(str(p))
    conn.execute(
        "CREATE TABLE mutant (id INTEGER PRIMARY KEY, line INTEGER, status TEXT, filename TEXT)"
    )
    for i, status in enumerate(results):
        conn.execute("INSERT INTO mutant VALUES (?, ?, ?, ?)", (i, i + 1, status, "test.py"))
    conn.commit()
    conn.close()
    return p


# ---------------------------------------------------------------------------
# collect_test_coverage
# ---------------------------------------------------------------------------


class TestCollectTestCoverage:
    def test_correct_percentage(self, tmp_path):
        p = _write_cobertura(tmp_path, 0.87)
        r = collect_test_coverage(p)
        assert abs(r.raw_value - 87.0) < 0.1
        assert abs(r.score - 87.0) < 0.1

    def test_missing_file_score_zero(self, tmp_path):
        r = collect_test_coverage(tmp_path / "missing.xml")
        assert r.score == 0.0
        assert r.raw_value is None

    def test_missing_file_skip_missing_gives_100(self, tmp_path):
        r = collect_test_coverage(tmp_path / "missing.xml", skip_missing=True)
        assert r.score == 100.0

    def test_full_coverage(self, tmp_path):
        p = _write_cobertura(tmp_path, 1.0)
        r = collect_test_coverage(p)
        assert abs(r.score - 100.0) < 0.1

    def test_zero_coverage(self, tmp_path):
        p = _write_cobertura(tmp_path, 0.0)
        r = collect_test_coverage(p)
        assert abs(r.score - 0.0) < 0.1

    def test_metric_name(self, tmp_path):
        p = _write_cobertura(tmp_path, 0.5)
        r = collect_test_coverage(p)
        assert r.name == "test_coverage"


# ---------------------------------------------------------------------------
# collect_mutation_score
# ---------------------------------------------------------------------------


class TestCollectMutationScore:
    def test_all_killed(self, tmp_path):
        p = _write_mutmut_cache(tmp_path, ["ok", "ok", "ok"])
        r = collect_mutation_score(p)
        assert abs(r.score - 100.0) < 0.1

    def test_all_survived(self, tmp_path):
        p = _write_mutmut_cache(tmp_path, ["survived", "survived"])
        r = collect_mutation_score(p)
        assert abs(r.score - 0.0) < 0.1

    def test_mixed(self, tmp_path):
        p = _write_mutmut_cache(tmp_path, ["ok", "ok", "survived"])
        r = collect_mutation_score(p)
        assert abs(r.score - 66.67) < 1.0

    def test_missing_cache_returns_zero(self, tmp_path):
        r = collect_mutation_score(tmp_path / ".mutmut-cache")
        assert r.score == 0.0

    def test_missing_cache_skip_missing(self, tmp_path):
        r = collect_mutation_score(tmp_path / ".mutmut-cache", skip_missing=True)
        assert r.score == 100.0

    def test_suspicious_and_timeout_count_as_killed(self, tmp_path):
        p = _write_mutmut_cache(tmp_path, ["ok", "suspicious", "timeout", "survived"])
        r = collect_mutation_score(p)
        # 3 killed, 1 survived → 75%
        assert abs(r.score - 75.0) < 0.1

    def test_metric_name(self, tmp_path):
        p = _write_mutmut_cache(tmp_path, ["ok"])
        r = collect_mutation_score(p)
        assert r.name == "mutation_score"


# ---------------------------------------------------------------------------
# collect_critical_pass_rate
# ---------------------------------------------------------------------------


class TestCollectCriticalPassRate:
    def test_all_critical_passed(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "passed"}],
        )
        r = collect_critical_pass_rate(junit_p, taxonomy_p)
        assert abs(r.score - 100.0) < 0.1

    def test_critical_test_failed(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "failed"}],
        )
        r = collect_critical_pass_rate(junit_p, taxonomy_p)
        assert r.score == 0.0

    def test_partial_critical_pass(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [
                {"classname": "tests.test_benford", "name": "test_a", "outcome": "passed"},
                {"classname": "tests.test_benford", "name": "test_b", "outcome": "failed"},
            ],
        )
        r = collect_critical_pass_rate(junit_p, taxonomy_p)
        assert abs(r.score - 50.0) < 0.1

    def test_no_critical_tests_in_taxonomy_returns_100(self, tmp_path):
        taxonomy_p = _write_taxonomy(tmp_path, [])
        junit_p = _write_junit(tmp_path, [])
        r = collect_critical_pass_rate(junit_p, taxonomy_p)
        assert r.score == 100.0

    def test_missing_junit_returns_zero(self, tmp_path):
        taxonomy_p = _write_taxonomy(tmp_path, [])
        r = collect_critical_pass_rate(tmp_path / "missing.xml", taxonomy_p)
        assert r.score == 0.0

    def test_missing_junit_skip_missing_returns_100(self, tmp_path):
        taxonomy_p = _write_taxonomy(tmp_path, [])
        r = collect_critical_pass_rate(tmp_path / "missing.xml", taxonomy_p, skip_missing=True)
        assert r.score == 100.0


# ---------------------------------------------------------------------------
# collect_drift_rate
# ---------------------------------------------------------------------------


class TestCollectDriftRate:
    def test_no_drift_score_100(self, tmp_path):
        p = tmp_path / "drift_report.json"
        p.write_text(json.dumps({"drift_fraction": 0.0}))
        r = collect_drift_rate(p)
        assert abs(r.score - 100.0) < 0.1

    def test_full_drift_score_0(self, tmp_path):
        p = tmp_path / "drift_report.json"
        p.write_text(json.dumps({"drift_fraction": 1.0}))
        r = collect_drift_rate(p)
        assert abs(r.score - 0.0) < 0.1

    def test_half_drift_score_50(self, tmp_path):
        p = tmp_path / "drift_report.json"
        p.write_text(json.dumps({"drift_fraction": 0.5}))
        r = collect_drift_rate(p)
        assert abs(r.score - 50.0) < 0.1

    def test_features_drifted_format(self, tmp_path):
        p = tmp_path / "drift_report.json"
        p.write_text(json.dumps({"features_checked": 10, "features_drifted": 2}))
        r = collect_drift_rate(p)
        assert abs(r.score - 80.0) < 0.1

    def test_missing_file_returns_100(self, tmp_path):
        r = collect_drift_rate(tmp_path / "no_drift.json")
        assert r.score == 100.0

    def test_metric_name(self, tmp_path):
        p = tmp_path / "drift_report.json"
        p.write_text(json.dumps({"drift_fraction": 0.0}))
        r = collect_drift_rate(p)
        assert r.name == "drift_rate"


# ---------------------------------------------------------------------------
# collect_cycle_time
# ---------------------------------------------------------------------------


class TestCollectCycleTime:
    def test_short_cycle_time_scores_100(self):
        # Simulate 24h between merges (< TARGET_CYCLE_TIME_HOURS=72)
        # Return 10 timestamps 24h apart in descending order
        import time as _time

        now = int(_time.time())
        timestamps = [str(now - i * 86400) for i in range(10)]  # 10 merges, 1 day apart

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "\n".join(timestamps) + "\n"
            r = collect_cycle_time()
        assert r.score == 100.0

    def test_very_long_cycle_time_scores_zero(self):
        import time as _time

        now = int(_time.time())
        # 20-day intervals far exceed WORST_CYCLE_TIME_HOURS=336
        timestamps = [str(now - i * 86400 * 20) for i in range(5)]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "\n".join(timestamps) + "\n"
            r = collect_cycle_time()
        assert r.score == 0.0

    def test_git_failure_returns_100_when_skip_missing(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("git not found")):
            r = collect_cycle_time(skip_missing=True)
        assert r.score == 100.0

    def test_fewer_than_2_commits_returns_100(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value.stdout = "1700000000\n"
            r = collect_cycle_time()
        assert r.score == 100.0


# ---------------------------------------------------------------------------
# compute_composite_score
# ---------------------------------------------------------------------------


class TestCompositeScore:
    def _reading(self, name: str, score: float) -> MetricReading:
        return MetricReading(name, score, score, "test")

    def test_equal_weights_averages_scores(self):
        readings = [
            self._reading("test_coverage", 80.0),
            self._reading("mutation_score", 60.0),
        ]
        weights = {"test_coverage": 1.0, "mutation_score": 1.0}
        composite = compute_composite_score(readings, weights)
        assert abs(composite - 70.0) < 0.1

    def test_empty_readings_returns_zero(self):
        assert compute_composite_score([]) == 0.0

    def test_unknown_metric_weight_zero(self):
        readings = [self._reading("mystery_metric", 100.0)]
        weights = {"test_coverage": 1.0}  # mystery_metric not in weights
        composite = compute_composite_score(readings, weights)
        # mystery_metric has weight 0 → composite = 0 / sum(used_weights)
        assert composite == 0.0

    def test_default_weights_used_when_none(self):
        readings = [MetricReading(k, 100.0, 100.0, "test") for k in DEFAULT_WEIGHTS]
        composite = compute_composite_score(readings, None)
        assert abs(composite - 100.0) < 0.1

    def test_custom_weight_overrides(self):
        readings = [
            self._reading("test_coverage", 100.0),
            self._reading("mutation_score", 0.0),
        ]
        weights = {"test_coverage": 0.9, "mutation_score": 0.1}
        composite = compute_composite_score(readings, weights)
        assert abs(composite - 90.0) < 0.1


# ---------------------------------------------------------------------------
# QualityReport
# ---------------------------------------------------------------------------


class TestQualityReport:
    def _report(self, composite: float, threshold: float) -> QualityReport:
        readings = [MetricReading("test_coverage", composite, composite, "test")]
        return QualityReport(
            composite_score=composite,
            passed_threshold=composite >= threshold,
            threshold=threshold,
            metrics=readings,
            weights=DEFAULT_WEIGHTS,
        )

    def test_to_dict_has_required_keys(self):
        d = self._report(80.0, 70.0).to_dict()
        assert "composite_score" in d
        assert "passed_threshold" in d
        assert "threshold" in d
        assert "metrics" in d

    def test_summary_pass_when_above_threshold(self):
        s = self._report(85.0, 70.0).summary()
        assert "PASS" in s

    def test_summary_fail_when_below_threshold(self):
        s = self._report(55.0, 70.0).summary()
        assert "FAIL" in s

    def test_to_dict_serializable(self):
        import json

        d = self._report(75.0, 70.0).to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# build_badge_json
# ---------------------------------------------------------------------------


class TestBuildBadgeJson:
    def _report(self, score: float) -> QualityReport:
        return QualityReport(score, score >= 70, 70, [], DEFAULT_WEIGHTS)

    def test_excellent_is_brightgreen(self):
        b = build_badge_json(self._report(95.0))
        assert b["color"] == "brightgreen"

    def test_good_is_green(self):
        b = build_badge_json(self._report(80.0))
        assert b["color"] == "green"

    def test_fair_is_yellow(self):
        b = build_badge_json(self._report(65.0))
        assert b["color"] == "yellow"

    def test_poor_is_red(self):
        b = build_badge_json(self._report(40.0))
        assert b["color"] == "red"

    def test_badge_has_schema_version(self):
        b = build_badge_json(self._report(80.0))
        assert b["schemaVersion"] == 1


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_exit_0_when_above_threshold(self, tmp_path):
        coverage_p = _write_cobertura(tmp_path, 0.9)
        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "quality_metrics.json"

        code = main(
            [
                "--coverage",
                str(coverage_p),
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--skip-missing",
                "--threshold",
                "70",
            ]
        )
        assert code == 0
        assert out_p.exists()

    def test_exit_1_when_below_threshold(self, tmp_path):
        # Provide 0% coverage → very low composite
        coverage_p = _write_cobertura(tmp_path, 0.0)
        junit_p = _write_junit(
            tmp_path, [{"classname": "tests.test_benford", "name": "t", "outcome": "failed"}]
        )
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        out_p = tmp_path / "quality_metrics.json"

        code = main(
            [
                "--coverage",
                str(coverage_p),
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--skip-missing",
                "--threshold",
                "99",  # unreachable threshold given 0% coverage
            ]
        )
        assert code == 1

    def test_output_json_written(self, tmp_path):
        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "metrics.json"
        main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--skip-missing",
            ]
        )
        assert out_p.exists()
        data = json.loads(out_p.read_text())
        assert "composite_score" in data

    def test_badge_written_with_flag(self, tmp_path):
        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        badge_p = tmp_path / "badge.json"
        out_p = tmp_path / "metrics.json"
        main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--badge",
                "--badge-output",
                str(badge_p),
                "--skip-missing",
            ]
        )
        assert badge_p.exists()
        data = json.loads(badge_p.read_text())
        assert data["schemaVersion"] == 1

    def test_invalid_weights_json_returns_2(self, tmp_path):
        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "metrics.json"
        code = main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--weights",
                "not-valid-json",
            ]
        )
        assert code == 2

    def test_custom_weights_override(self, tmp_path):
        coverage_p = _write_cobertura(tmp_path, 1.0)  # 100%
        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "metrics.json"
        main(
            [
                "--coverage",
                str(coverage_p),
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--skip-missing",
                "--weights",
                '{"test_coverage": 1.0}',
            ]
        )
        data = json.loads(out_p.read_text())
        # With 100% coverage and weight 1.0, composite should be high
        assert data["composite_score"] >= 90.0

    def test_output_format_csv_writes_csv_file(self, tmp_path):
        """Test that --output-format csv produces valid CSV output."""
        import csv

        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "metrics.csv"
        main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--output-format",
                "csv",
                "--skip-missing",
            ]
        )
        assert out_p.exists()

        # Parse the CSV and verify structure
        with open(out_p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) > 0, "CSV should have data rows"
        assert "metric_name" in rows[0]
        assert "score" in rows[0]

    def test_output_format_csv_contains_expected_columns(self, tmp_path):
        """Test that CSV output has the expected columns."""
        import csv

        junit_p = _write_junit(tmp_path, [])
        taxonomy_p = _write_taxonomy(tmp_path, [])
        out_p = tmp_path / "metrics.csv"
        main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--output",
                str(out_p),
                "--output-format",
                "csv",
                "--skip-missing",
            ]
        )

        with open(out_p) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check expected columns
        fieldnames = set(reader.fieldnames)
        expected = {"metric_name", "raw_value", "score", "source", "notes"}
        assert expected.issubset(fieldnames), f"Missing columns. Got {fieldnames}"

    def test_report_to_csv_function(self, tmp_path):
        """Test the report_to_csv function directly."""
        import csv
        import io

        metrics = [
            MetricReading("test_coverage", 85.5, 85.5, "coverage.xml"),
            MetricReading("mutation_score", None, 100.0, "cache", "skipped"),
        ]
        report = QualityReport(
            composite_score=90.0,
            passed_threshold=True,
            threshold=70.0,
            metrics=metrics,
            weights=DEFAULT_WEIGHTS,
        )

        csv_output = report_to_csv(report)

        # Parse the CSV string
        reader = csv.DictReader(io.StringIO(csv_output))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["metric_name"] == "test_coverage"
        assert rows[0]["score"] == "85.50"
        assert rows[1]["metric_name"] == "mutation_score"
        assert rows[1]["raw_value"] == ""  # None should be empty string
