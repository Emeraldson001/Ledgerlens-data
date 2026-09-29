"""Tests for utils/diagnostics.py — repository health diagnostics framework.

Tests cover:
- DiagnosticResult creation and serialization
- DiagnosticReport aggregation and status determination
- DiagnosticRegistry registration and execution
- Check protocol conformance
- Category filtering and fail-fast mode
"""

from __future__ import annotations

import pytest

from utils.diagnostics import (
    CheckCategory,
    CheckStatus,
    DiagnosticRegistry,
    DiagnosticReport,
    DiagnosticResult,
)

# =============================================================================
# Test Fixtures
# =============================================================================


class PassingCheck:
    """Mock check that always passes."""

    name = "passing_check"
    category = CheckCategory.ENVIRONMENT

    def run(self) -> DiagnosticResult:
        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message="Check passed",
        )


class FailingCheck:
    """Mock check that always fails."""

    name = "failing_check"
    category = CheckCategory.CODE_HEALTH

    def run(self) -> DiagnosticResult:
        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.FAIL,
            message="Check failed",
            remediation="Fix the issue",
        )


class WarningCheck:
    """Mock check that produces a warning."""

    name = "warning_check"
    category = CheckCategory.DEPENDENCIES

    def run(self) -> DiagnosticResult:
        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.WARN,
            message="Minor issue detected",
        )


class ErrorCheck:
    """Mock check that raises an exception."""

    name = "error_check"
    category = CheckCategory.RUNTIME

    def run(self) -> DiagnosticResult:
        raise RuntimeError("Unexpected error during check")


# =============================================================================
# DiagnosticResult Tests
# =============================================================================


def test_diagnostic_result_creation():
    """DiagnosticResult can be created with required fields."""
    result = DiagnosticResult(
        check_name="test_check",
        category=CheckCategory.ENVIRONMENT,
        status=CheckStatus.PASS,
        message="All good",
    )

    assert result.check_name == "test_check"
    assert result.category == CheckCategory.ENVIRONMENT
    assert result.status == CheckStatus.PASS
    assert result.message == "All good"
    assert result.details == {}
    assert result.remediation is None


def test_diagnostic_result_with_details():
    """DiagnosticResult can include details and remediation."""
    result = DiagnosticResult(
        check_name="test_check",
        category=CheckCategory.DEPENDENCIES,
        status=CheckStatus.FAIL,
        message="Missing packages",
        details={"missing": ["pkg1", "pkg2"]},
        remediation="Run: pip install pkg1 pkg2",
    )

    assert result.details == {"missing": ["pkg1", "pkg2"]}
    assert result.remediation == "Run: pip install pkg1 pkg2"


def test_diagnostic_result_to_dict():
    """DiagnosticResult.to_dict() produces correct structure."""
    result = DiagnosticResult(
        check_name="test_check",
        category=CheckCategory.CODE_HEALTH,
        status=CheckStatus.WARN,
        message="Warning message",
        details={"count": 3},
        duration_ms=12.5,
    )

    result_dict = result.to_dict()

    assert result_dict["check_name"] == "test_check"
    assert result_dict["category"] == "code_health"
    assert result_dict["status"] == "warn"
    assert result_dict["message"] == "Warning message"
    assert result_dict["details"] == {"count": 3}
    assert result_dict["duration_ms"] == 12.5


def test_diagnostic_result_is_healthy_pass():
    """PASS status is considered healthy."""
    result = DiagnosticResult(
        check_name="test",
        category=CheckCategory.ENVIRONMENT,
        status=CheckStatus.PASS,
        message="OK",
    )
    assert result.is_healthy() is True


def test_diagnostic_result_is_healthy_skip():
    """SKIP status is considered healthy."""
    result = DiagnosticResult(
        check_name="test",
        category=CheckCategory.RUNTIME,
        status=CheckStatus.SKIP,
        message="Skipped",
    )
    assert result.is_healthy() is True


def test_diagnostic_result_is_healthy_fail():
    """FAIL status is not healthy."""
    result = DiagnosticResult(
        check_name="test",
        category=CheckCategory.CODE_HEALTH,
        status=CheckStatus.FAIL,
        message="Failed",
    )
    assert result.is_healthy() is False


def test_diagnostic_result_is_healthy_error():
    """ERROR status is not healthy."""
    result = DiagnosticResult(
        check_name="test",
        category=CheckCategory.DATA_ARTIFACTS,
        status=CheckStatus.ERROR,
        message="Error occurred",
    )
    assert result.is_healthy() is False


# =============================================================================
# DiagnosticReport Tests
# =============================================================================


def test_diagnostic_report_overall_status_all_pass():
    """Overall status is PASS when all checks pass."""
    results = [
        DiagnosticResult("check1", CheckCategory.ENVIRONMENT, CheckStatus.PASS, "OK"),
        DiagnosticResult("check2", CheckCategory.DEPENDENCIES, CheckStatus.PASS, "OK"),
    ]

    report = DiagnosticReport(
        results=results,
        overall_status=CheckStatus.PASS,
        categories_checked={CheckCategory.ENVIRONMENT, CheckCategory.DEPENDENCIES},
        total_duration_ms=100.0,
    )

    assert report.overall_status == CheckStatus.PASS
    assert report.is_healthy() is True


def test_diagnostic_report_overall_status_with_warnings():
    """Overall status is WARN when warnings present but no failures."""
    results = [
        DiagnosticResult("check1", CheckCategory.ENVIRONMENT, CheckStatus.PASS, "OK"),
        DiagnosticResult("check2", CheckCategory.DEPENDENCIES, CheckStatus.WARN, "Warning"),
    ]

    report = DiagnosticReport(
        results=results,
        overall_status=CheckStatus.WARN,
        categories_checked={CheckCategory.ENVIRONMENT, CheckCategory.DEPENDENCIES},
        total_duration_ms=100.0,
    )

    assert report.overall_status == CheckStatus.WARN
    assert report.is_healthy() is True  # Warnings are still healthy


def test_diagnostic_report_overall_status_with_failures():
    """Overall status is FAIL when any check fails."""
    results = [
        DiagnosticResult("check1", CheckCategory.ENVIRONMENT, CheckStatus.PASS, "OK"),
        DiagnosticResult("check2", CheckCategory.CODE_HEALTH, CheckStatus.FAIL, "Failed"),
    ]

    report = DiagnosticReport(
        results=results,
        overall_status=CheckStatus.FAIL,
        categories_checked={CheckCategory.ENVIRONMENT, CheckCategory.CODE_HEALTH},
        total_duration_ms=100.0,
    )

    assert report.overall_status == CheckStatus.FAIL
    assert report.is_healthy() is False


def test_diagnostic_report_to_dict():
    """DiagnosticReport.to_dict() includes all summary fields."""
    results = [
        DiagnosticResult("pass_check", CheckCategory.ENVIRONMENT, CheckStatus.PASS, "OK"),
        DiagnosticResult("warn_check", CheckCategory.DEPENDENCIES, CheckStatus.WARN, "Warning"),
        DiagnosticResult("fail_check", CheckCategory.CODE_HEALTH, CheckStatus.FAIL, "Failed"),
    ]

    report = DiagnosticReport(
        results=results,
        overall_status=CheckStatus.FAIL,
        categories_checked={
            CheckCategory.ENVIRONMENT,
            CheckCategory.DEPENDENCIES,
            CheckCategory.CODE_HEALTH,
        },
        total_duration_ms=150.5,
    )

    report_dict = report.to_dict()

    assert report_dict["overall_status"] == "fail"
    assert report_dict["total_checks"] == 3
    assert report_dict["pass_count"] == 1
    assert report_dict["warn_count"] == 1
    assert report_dict["fail_count"] == 1
    assert report_dict["error_count"] == 0
    assert report_dict["skip_count"] == 0
    assert len(report_dict["checks"]) == 3


def test_diagnostic_report_summary():
    """DiagnosticReport.summary() produces human-readable output."""
    results = [
        DiagnosticResult(
            "check1", CheckCategory.ENVIRONMENT, CheckStatus.FAIL, "Failed", remediation="Fix it"
        ),
    ]

    report = DiagnosticReport(
        results=results,
        overall_status=CheckStatus.FAIL,
        categories_checked={CheckCategory.ENVIRONMENT},
        total_duration_ms=50.0,
    )

    summary = report.summary()

    assert "Overall Status: FAIL" in summary
    assert "check1: Failed" in summary
    assert "Fix: Fix it" in summary


# =============================================================================
# DiagnosticRegistry Tests
# =============================================================================


def test_registry_register_check():
    """Checks can be registered with the registry."""
    registry = DiagnosticRegistry()
    check = PassingCheck()

    registry.register(check)

    retrieved = registry.get_check("passing_check")
    assert retrieved is not None
    assert retrieved.name == "passing_check"


def test_registry_register_duplicate_raises():
    """Registering a check with duplicate name raises ValueError."""
    registry = DiagnosticRegistry()
    check1 = PassingCheck()
    check2 = PassingCheck()

    registry.register(check1)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(check2)


def test_registry_get_nonexistent_check():
    """Getting a non-existent check returns None."""
    registry = DiagnosticRegistry()
    assert registry.get_check("nonexistent") is None


def test_registry_list_checks():
    """list_checks() returns all registered checks."""
    registry = DiagnosticRegistry()
    registry.register(PassingCheck())
    registry.register(FailingCheck())

    checks = registry.list_checks()

    assert len(checks) == 2
    assert ("failing_check", CheckCategory.CODE_HEALTH) in checks
    assert ("passing_check", CheckCategory.ENVIRONMENT) in checks


def test_registry_list_checks_filtered_by_category():
    """list_checks() can filter by category."""
    registry = DiagnosticRegistry()
    registry.register(PassingCheck())  # ENVIRONMENT
    registry.register(FailingCheck())  # CODE_HEALTH
    registry.register(WarningCheck())  # DEPENDENCIES

    env_checks = registry.list_checks(category=CheckCategory.ENVIRONMENT)

    assert len(env_checks) == 1
    assert env_checks[0][0] == "passing_check"


def test_registry_run_all_executes_all_checks():
    """run_all() executes all registered checks."""
    registry = DiagnosticRegistry()
    registry.register(PassingCheck())
    registry.register(FailingCheck())

    report = registry.run_all()

    assert len(report.results) == 2
    assert report.overall_status == CheckStatus.FAIL  # One failure


def test_registry_run_all_with_category_filter():
    """run_all() can filter by category."""
    registry = DiagnosticRegistry()
    registry.register(PassingCheck())  # ENVIRONMENT
    registry.register(FailingCheck())  # CODE_HEALTH

    report = registry.run_all(categories=[CheckCategory.ENVIRONMENT])

    assert len(report.results) == 1
    assert report.results[0].check_name == "passing_check"


def test_registry_run_all_captures_exceptions():
    """run_all() captures exceptions and reports as ERROR status."""
    registry = DiagnosticRegistry()
    registry.register(ErrorCheck())

    report = registry.run_all()

    assert len(report.results) == 1
    result = report.results[0]
    assert result.status == CheckStatus.ERROR
    assert "exception" in result.details


def test_registry_run_all_fail_fast():
    """run_all() stops on first failure when fail_fast=True."""
    registry = DiagnosticRegistry()
    registry.register(FailingCheck())
    registry.register(PassingCheck())

    report = registry.run_all(fail_fast=True)

    # Should stop after first check (the failing one)
    assert len(report.results) == 1
    assert report.results[0].check_name == "failing_check"


def test_registry_run_all_records_duration():
    """run_all() records execution duration for each check."""
    registry = DiagnosticRegistry()
    registry.register(PassingCheck())

    report = registry.run_all()

    assert report.total_duration_ms > 0
    assert report.results[0].duration_ms >= 0
