"""Tests for scripts/release_gate.py (Issue #559).

Covers:
- TestCriticalityTaxonomy: pattern loading, criticality_for, reason_for
- parse_junit_xml: outcome parsing for passed/failed/error/skipped
- ReleaseGateEvaluator: CRITICAL/HIGH/LOW blocking logic, allow_high_failures flag
- CLI main(): exit codes 0/1/2
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.release_gate import (
    ReleaseGateEvaluator,
    TestCriticalityTaxonomy,
    TestResult,
    main,
    parse_junit_xml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_taxonomy(tmp_path: Path, entries: list[dict]) -> Path:
    data = {"version": 1, "entries": entries}
    p = tmp_path / "test_criticality.json"
    p.write_text(json.dumps(data))
    return p


def _write_junit(tmp_path: Path, cases: list[dict]) -> Path:
    """Write a minimal JUnit XML with the given test cases.

    Each case dict: {"classname": ..., "name": ..., "outcome": "passed"|"failed"|"skipped"}
    """
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<testsuites>\n  <testsuite name="pytest">']
    for c in cases:
        classname = c.get("classname", "tests.test_example")
        name = c.get("name", "test_dummy")
        outcome = c.get("outcome", "passed")
        message = c.get("message", "assertion failed")
        if outcome == "passed":
            parts.append(f'    <testcase classname="{classname}" name="{name}" />')
        elif outcome == "failed":
            parts.append(
                f'    <testcase classname="{classname}" name="{name}">'
                f'<failure message="{message}">stack trace here</failure>'
                f"</testcase>"
            )
        elif outcome == "error":
            parts.append(
                f'    <testcase classname="{classname}" name="{name}">'
                f'<error message="{message}">error trace</error>'
                f"</testcase>"
            )
        elif outcome == "skipped":
            parts.append(
                f'    <testcase classname="{classname}" name="{name}">'
                f'<skipped message="reason"/>'
                f"</testcase>"
            )
    parts.append("  </testsuite>\n</testsuites>")
    p = tmp_path / "junit.xml"
    p.write_text("\n".join(parts))
    return p


# ---------------------------------------------------------------------------
# TestCriticalityTaxonomy
# ---------------------------------------------------------------------------


class TestCriticalityTaxonomyClass:
    def test_loads_entries(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL", "reason": "core"}],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert len(t.entries) == 1

    def test_missing_file_returns_all_low(self, tmp_path):
        t = TestCriticalityTaxonomy(tmp_path / "nonexistent.json")
        assert t.criticality_for("tests/test_benford.py::test_foo") == "LOW"

    def test_file_pattern_matches_all_tests_in_file(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert t.criticality_for("tests/test_benford.py::test_foo") == "CRITICAL"
        assert t.criticality_for("tests/test_benford.py::test_bar") == "CRITICAL"

    def test_node_id_pattern_matches_exact_test(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py::test_specific", "criticality": "HIGH"}],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert t.criticality_for("tests/test_benford.py::test_specific") == "HIGH"
        assert t.criticality_for("tests/test_benford.py::test_other") == "LOW"

    def test_unmatched_pattern_returns_low(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert t.criticality_for("tests/test_completely_different.py::test_foo") == "LOW"

    def test_higher_criticality_wins_when_multiple_match(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [
                {"pattern": "tests/test_benford.py", "criticality": "HIGH"},
                {"pattern": "tests/test_benford.py::test_foo", "criticality": "CRITICAL"},
            ],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert t.criticality_for("tests/test_benford.py::test_foo") == "CRITICAL"

    def test_reason_for_returns_reason(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [
                {
                    "pattern": "tests/test_benford.py",
                    "criticality": "CRITICAL",
                    "reason": "core scoring",
                }
            ],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert "core scoring" in t.reason_for("tests/test_benford.py::test_foo")

    def test_reason_for_empty_when_no_match(self, tmp_path):
        tpath = _write_taxonomy(tmp_path, [])
        t = TestCriticalityTaxonomy(tpath)
        assert t.reason_for("tests/test_foo.py::bar") == ""

    def test_invalid_criticality_level_skipped(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [
                {"pattern": "tests/test_benford.py", "criticality": "INVALID"},
                {"pattern": "tests/test_other.py", "criticality": "CRITICAL"},
            ],
        )
        t = TestCriticalityTaxonomy(tpath)
        # Invalid entry is skipped; valid entry loads fine
        assert len(t.entries) == 1

    def test_glob_wildcard_in_pattern(self, tmp_path):
        tpath = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_model_*.py", "criticality": "HIGH"}],
        )
        t = TestCriticalityTaxonomy(tpath)
        assert t.criticality_for("tests/test_model_inference.py::test_foo") == "HIGH"
        assert t.criticality_for("tests/test_benford.py::test_foo") == "LOW"


# ---------------------------------------------------------------------------
# parse_junit_xml
# ---------------------------------------------------------------------------


class TestParseJunitXml:
    def test_parses_passed_tests(self, tmp_path):
        p = _write_junit(
            tmp_path, [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "passed"}]
        )
        results = parse_junit_xml(p)
        assert len(results) == 1
        assert results[0].outcome == "passed"

    def test_parses_failed_tests(self, tmp_path):
        p = _write_junit(
            tmp_path,
            [
                {
                    "classname": "tests.test_benford",
                    "name": "test_bar",
                    "outcome": "failed",
                    "message": "AssertionError",
                }
            ],
        )
        results = parse_junit_xml(p)
        assert results[0].outcome == "failed"
        assert "AssertionError" in results[0].message

    def test_parses_error_tests(self, tmp_path):
        p = _write_junit(
            tmp_path,
            [
                {
                    "classname": "tests.test_x",
                    "name": "test_y",
                    "outcome": "error",
                    "message": "ImportError",
                }
            ],
        )
        results = parse_junit_xml(p)
        assert results[0].outcome == "error"

    def test_parses_skipped_tests(self, tmp_path):
        p = _write_junit(
            tmp_path, [{"classname": "tests.test_x", "name": "test_skip", "outcome": "skipped"}]
        )
        results = parse_junit_xml(p)
        assert results[0].outcome == "skipped"

    def test_node_id_format(self, tmp_path):
        p = _write_junit(
            tmp_path, [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "passed"}]
        )
        results = parse_junit_xml(p)
        # node_id should be: "tests/test_benford.py::test_foo"
        assert results[0].node_id == "tests/test_benford.py::test_foo"

    def test_mixed_outcomes(self, tmp_path):
        cases = [
            {"classname": "tests.test_a", "name": "test_pass", "outcome": "passed"},
            {"classname": "tests.test_b", "name": "test_fail", "outcome": "failed"},
            {"classname": "tests.test_c", "name": "test_skip", "outcome": "skipped"},
        ]
        p = _write_junit(tmp_path, cases)
        results = parse_junit_xml(p)
        outcomes = {r.outcome for r in results}
        assert outcomes == {"passed", "failed", "skipped"}


# ---------------------------------------------------------------------------
# ReleaseGateEvaluator
# ---------------------------------------------------------------------------


class TestReleaseGateEvaluator:
    def _taxonomy(self, tmp_path: Path, entries: list[dict]) -> TestCriticalityTaxonomy:
        p = _write_taxonomy(tmp_path, entries)
        return TestCriticalityTaxonomy(p)

    def _result(self, file: str, name: str, outcome: str) -> TestResult:
        node_id = f"{file}::{name}"
        return TestResult(node_id=node_id, classname="", name=name, outcome=outcome)

    def test_all_passed_gate_passes(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        results = [self._result("tests/test_benford.py", "test_foo", "passed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is True
        assert gate.critical_failures == []

    def test_critical_failure_blocks_gate(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        results = [self._result("tests/test_benford.py", "test_foo", "failed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is False
        assert len(gate.critical_failures) == 1
        assert "CRITICAL" in gate.critical_failures[0].criticality

    def test_high_failure_blocks_gate_by_default(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_inference_shap.py", "criticality": "HIGH"}],
        )
        results = [self._result("tests/test_inference_shap.py", "test_x", "failed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is False
        assert len(gate.high_failures) == 1

    def test_high_failure_allowed_with_flag(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_inference_shap.py", "criticality": "HIGH"}],
        )
        results = [self._result("tests/test_inference_shap.py", "test_x", "failed")]
        gate = ReleaseGateEvaluator(taxonomy, allow_high_failures=True).evaluate(results)
        assert gate.passed is True
        assert len(gate.high_failures) == 1  # still reported, just doesn't block

    def test_low_failure_never_blocks(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_backtest.py", "criticality": "LOW"}],
        )
        results = [self._result("tests/test_backtest.py", "test_z", "failed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is True
        assert gate.critical_failures == []
        assert gate.high_failures == []

    def test_unmatched_failure_treated_as_low(self, tmp_path):
        taxonomy = self._taxonomy(tmp_path, [])
        results = [self._result("tests/test_unknown.py", "test_q", "failed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is True  # LOW never blocks

    def test_error_outcome_treated_as_failure(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        results = [self._result("tests/test_benford.py", "test_err", "error")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.passed is False

    def test_gate_result_summary_contains_failure_info(self, tmp_path):
        taxonomy = self._taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL", "reason": "core"}],
        )
        results = [self._result("tests/test_benford.py", "test_foo", "failed")]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        summary = gate.summary()
        assert "FAILED" in summary
        assert "CRITICAL" in summary

    def test_evaluated_tests_count(self, tmp_path):
        taxonomy = self._taxonomy(tmp_path, [])
        results = [
            self._result("tests/test_a.py", "t1", "passed"),
            self._result("tests/test_b.py", "t2", "failed"),
            self._result("tests/test_c.py", "t3", "skipped"),
        ]
        gate = ReleaseGateEvaluator(taxonomy).evaluate(results)
        assert gate.evaluated_tests == 3
        assert gate.total_failures == 1


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------


class TestCLIMain:
    def test_exit_0_all_pass(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "passed"}],
        )
        code = main(["--junit", str(junit_p), "--taxonomy", str(taxonomy_p)])
        assert code == 0

    def test_exit_1_critical_failure(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_benford.py", "criticality": "CRITICAL"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [{"classname": "tests.test_benford", "name": "test_foo", "outcome": "failed"}],
        )
        code = main(["--junit", str(junit_p), "--taxonomy", str(taxonomy_p)])
        assert code == 1

    def test_exit_2_missing_junit(self, tmp_path):
        taxonomy_p = _write_taxonomy(tmp_path, [])
        code = main(["--junit", str(tmp_path / "missing.xml"), "--taxonomy", str(taxonomy_p)])
        assert code == 2

    def test_output_json_written(self, tmp_path):
        taxonomy_p = _write_taxonomy(tmp_path, [])
        junit_p = _write_junit(tmp_path, [])
        out_p = tmp_path / "gate_result.json"
        main(["--junit", str(junit_p), "--taxonomy", str(taxonomy_p), "--output-json", str(out_p)])
        assert out_p.exists()
        data = json.loads(out_p.read_text())
        assert "passed" in data
        assert "critical_failures" in data

    def test_allow_high_failures_flag(self, tmp_path):
        taxonomy_p = _write_taxonomy(
            tmp_path,
            [{"pattern": "tests/test_inference_shap.py", "criticality": "HIGH"}],
        )
        junit_p = _write_junit(
            tmp_path,
            [{"classname": "tests.test_inference_shap", "name": "test_x", "outcome": "failed"}],
        )
        code = main(
            [
                "--junit",
                str(junit_p),
                "--taxonomy",
                str(taxonomy_p),
                "--allow-high-failures",
            ]
        )
        assert code == 0
