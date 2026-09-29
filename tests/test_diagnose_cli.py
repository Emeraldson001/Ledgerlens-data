"""Tests for scripts/diagnose.py — repository diagnostics CLI.

Tests cover:
- CLI argument parsing
- List mode
- JSON output mode
- Category filtering
- Fail-fast mode
- Exit code behavior
"""

from __future__ import annotations

import json
from unittest import mock

from scripts.diagnose import main
from utils.diagnostics import CheckCategory, CheckStatus, DiagnosticReport, DiagnosticResult


def _make_pass_report() -> DiagnosticReport:
    return DiagnosticReport(
        results=[DiagnosticResult("check1", CheckCategory.ENVIRONMENT, CheckStatus.PASS, "OK")],
        overall_status=CheckStatus.PASS,
        categories_checked={CheckCategory.ENVIRONMENT},
        total_duration_ms=50.0,
    )


def _make_fail_report() -> DiagnosticReport:
    return DiagnosticReport(
        results=[DiagnosticResult("check1", CheckCategory.CODE_HEALTH, CheckStatus.FAIL, "Failed")],
        overall_status=CheckStatus.FAIL,
        categories_checked={CheckCategory.CODE_HEALTH},
        total_duration_ms=50.0,
    )


def _make_warn_report() -> DiagnosticReport:
    return DiagnosticReport(
        results=[
            DiagnosticResult("check1", CheckCategory.DEPENDENCIES, CheckStatus.WARN, "Warning")
        ],
        overall_status=CheckStatus.WARN,
        categories_checked={CheckCategory.DEPENDENCIES},
        total_duration_ms=50.0,
    )


# =============================================================================
# CLI Tests
# =============================================================================


def test_main_list_mode():
    """--list flag lists available checks and exits 0."""
    with mock.patch("scripts.diagnose._print_check_list") as mock_list:
        exit_code = main(["--list"])

        assert exit_code == 0
        mock_list.assert_called_once()


def test_main_all_checks_pass():
    """Exit code 0 when all checks pass."""
    # Mock at the point of import in scripts.diagnose
    with mock.patch("scripts.diagnose.run_diagnostics", return_value=_make_pass_report()):
        exit_code = main([])

        assert exit_code == 0


def test_main_some_checks_fail():
    """Exit code 1 when some checks fail."""
    with mock.patch("scripts.diagnose.run_diagnostics", return_value=_make_fail_report()):
        exit_code = main([])

        assert exit_code == 1


def test_main_warnings_still_exit_0():
    """Warnings don't cause non-zero exit — warnings are still healthy."""
    with mock.patch("scripts.diagnose.run_diagnostics", return_value=_make_warn_report()):
        exit_code = main([])

        assert exit_code == 0


def test_main_json_output_mode(capsys):
    """--json flag outputs valid JSON."""
    with mock.patch("scripts.diagnose.run_diagnostics", return_value=_make_pass_report()):
        exit_code = main(["--json"])

        captured = capsys.readouterr()
        output_data = json.loads(captured.out)

        assert exit_code == 0
        assert output_data["overall_status"] == "pass"
        assert "checks" in output_data
        assert isinstance(output_data["checks"], list)


def test_main_json_output_contains_all_fields(capsys):
    """JSON output contains summary counters and check list."""
    with mock.patch("scripts.diagnose.run_diagnostics", return_value=_make_fail_report()):
        main(["--json"])

        captured = capsys.readouterr()
        data = json.loads(captured.out)

        assert "total_checks" in data
        assert "pass_count" in data
        assert "fail_count" in data
        assert "total_duration_ms" in data


def test_main_category_filtering():
    """--categories flag is forwarded to run_diagnostics."""
    with mock.patch(
        "scripts.diagnose.run_diagnostics", return_value=_make_pass_report()
    ) as mock_run:
        main(["--categories", "environment", "dependencies"])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["categories"] == ["environment", "dependencies"]


def test_main_fail_fast_mode():
    """--fail-fast flag is forwarded to run_diagnostics."""
    with mock.patch(
        "scripts.diagnose.run_diagnostics", return_value=_make_pass_report()
    ) as mock_run:
        main(["--fail-fast"])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["fail_fast"] is True


def test_main_handles_keyboard_interrupt(capsys):
    """Keyboard interrupt results in exit code 2."""
    with mock.patch("scripts.diagnose.run_diagnostics", side_effect=KeyboardInterrupt):
        exit_code = main([])

        assert exit_code == 2


def test_main_handles_runtime_error(capsys):
    """Unexpected exception results in exit code 2."""
    with mock.patch("scripts.diagnose.run_diagnostics", side_effect=RuntimeError("boom")):
        exit_code = main([])

        assert exit_code == 2


def test_main_handles_import_error_for_checks(capsys):
    """If diagnostics_checks can't be imported, exit code 2 with message."""
    import sys

    # Set the module sentinel to None — Python treats this as a failed import
    # and raises ImportError when `import utils.diagnostics_checks` is executed.
    original = sys.modules.get("utils.diagnostics_checks")
    sys.modules["utils.diagnostics_checks"] = None  # type: ignore[assignment]

    try:
        exit_code = main([])
    finally:
        # Always restore the original entry so subsequent tests aren't affected
        if original is None:
            sys.modules.pop("utils.diagnostics_checks", None)
        else:
            sys.modules["utils.diagnostics_checks"] = original

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Cannot import" in captured.err


def test_main_no_arguments_runs_all_categories():
    """Running with no arguments runs checks in all categories."""
    with mock.patch(
        "scripts.diagnose.run_diagnostics", return_value=_make_pass_report()
    ) as mock_run:
        main([])

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        # No category filter when no --categories specified
        assert call_kwargs["categories"] is None
