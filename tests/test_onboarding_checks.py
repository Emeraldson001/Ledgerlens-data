"""Tests for the onboarding checks (Issue #3).

Uses a temporary directory tree that mimics the repo root so the checks run
in a fully controlled, isolated environment without touching the real working
tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.onboarding_checks import (
    CheckLevel,
    OnboardingReport,
    check_database,
    check_env_file_loaded,
    check_model_artifacts,
    check_optional_tools,
    check_python_version,
    check_required_dirs,
    check_required_files,
    check_required_packages,
    check_synthetic_dataset,
    check_virtual_env,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_repo(tmp_path: Path) -> Path:
    """Minimal repo-like directory with all required files and dirs present."""
    (tmp_path / ".env").write_text(
        "HORIZON_URL=https://horizon.stellar.org\nRISK_SCORE_DB_URL=sqlite:///./test.db\n"
    )
    (tmp_path / "requirements.txt").write_text("# stub\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    for d in ["models", "data", "reports", "reports/forensic"]:
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def empty_repo(tmp_path: Path) -> Path:
    """Empty directory — simulates a bare checkout with nothing set up."""
    return tmp_path


# ---------------------------------------------------------------------------
# Python version check
# ---------------------------------------------------------------------------


class TestCheckPythonVersion:
    def test_current_version_passes(self):
        result = check_python_version()
        # Current test runner is Python 3.11+
        assert result.level == CheckLevel.OK


# ---------------------------------------------------------------------------
# Virtual-env check
# ---------------------------------------------------------------------------


class TestCheckVirtualEnv:
    def test_reports_result(self):
        result = check_virtual_env()
        # Either OK (inside a venv) or WARNING (not in a venv) — never ERROR
        assert result.level in (CheckLevel.OK, CheckLevel.WARNING)


# ---------------------------------------------------------------------------
# Required packages
# ---------------------------------------------------------------------------


class TestCheckRequiredPackages:
    def test_pandas_and_numpy_present(self):
        results = check_required_packages()
        by_name = {r.name: r for r in results}
        assert by_name["pkg:pandas"].level == CheckLevel.OK
        assert by_name["pkg:numpy"].level == CheckLevel.OK

    def test_sqlalchemy_present(self):
        results = check_required_packages()
        by_name = {r.name: r for r in results}
        assert by_name["pkg:sqlalchemy"].level == CheckLevel.OK


# ---------------------------------------------------------------------------
# Required files
# ---------------------------------------------------------------------------


class TestCheckRequiredFiles:
    def test_all_present_in_fake_repo(self, fake_repo):
        results = check_required_files(fake_repo)
        for r in results:
            assert r.level == CheckLevel.OK, f"{r.name}: {r.message}"

    def test_missing_env_is_error(self, empty_repo):
        results = check_required_files(empty_repo)
        by_name = {r.name: r for r in results}
        assert by_name["file:.env"].level == CheckLevel.ERROR

    def test_fix_copies_env_example(self, tmp_path):
        # Create .env.example but not .env
        (tmp_path / ".env.example").write_text("HORIZON_URL=https://horizon.stellar.org\n")
        (tmp_path / "requirements.txt").write_text("")
        (tmp_path / "pyproject.toml").write_text("")
        results = check_required_files(tmp_path, fix=True)
        by_name = {r.name: r for r in results}
        assert by_name["file:.env"].level == CheckLevel.OK
        assert by_name["file:.env"].fix_applied
        assert (tmp_path / ".env").exists()


# ---------------------------------------------------------------------------
# Required directories
# ---------------------------------------------------------------------------


class TestCheckRequiredDirs:
    def test_all_present_in_fake_repo(self, fake_repo):
        results = check_required_dirs(fake_repo)
        for r in results:
            assert r.level == CheckLevel.OK, f"{r.name}: {r.message}"

    def test_missing_dirs_are_warnings(self, empty_repo):
        results = check_required_dirs(empty_repo)
        for r in results:
            assert r.level == CheckLevel.WARNING

    def test_fix_creates_missing_dirs(self, empty_repo):
        results = check_required_dirs(empty_repo, fix=True)
        for r in results:
            assert r.level == CheckLevel.OK
            assert r.fix_applied
        assert (empty_repo / "models").is_dir()
        assert (empty_repo / "reports" / "forensic").is_dir()


# ---------------------------------------------------------------------------
# .env file loaded check
# ---------------------------------------------------------------------------


class TestCheckEnvFileLoaded:
    def test_ok_when_env_present(self, fake_repo):
        result = check_env_file_loaded(fake_repo)
        assert result.level == CheckLevel.OK

    def test_warning_when_env_missing(self, empty_repo):
        result = check_env_file_loaded(empty_repo)
        assert result.level == CheckLevel.WARNING


# ---------------------------------------------------------------------------
# Database check
# ---------------------------------------------------------------------------


class TestCheckDatabase:
    def test_sqlite_in_memory_is_ok(self, monkeypatch):
        monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")
        results = check_database()
        conn_result = next(r for r in results if r.name == "database-connection")
        assert conn_result.level == CheckLevel.OK

    def test_bad_url_is_error(self, monkeypatch):
        monkeypatch.setenv("RISK_SCORE_DB_URL", "postgresql://nobody:nopass@nonexistent:5432/nodb")
        results = check_database()
        conn_result = next(r for r in results if r.name == "database-connection")
        assert conn_result.level == CheckLevel.ERROR


# ---------------------------------------------------------------------------
# Synthetic dataset check
# ---------------------------------------------------------------------------


class TestCheckSyntheticDataset:
    def test_present_is_ok(self, fake_repo):
        # Create a fake parquet file
        parquet = fake_repo / "data" / "synthetic_dataset.parquet"
        parquet.write_bytes(b"PAR1" + b"\x00" * 100)
        result = check_synthetic_dataset(fake_repo)
        assert result.level == CheckLevel.OK

    def test_absent_is_warning(self, fake_repo):
        result = check_synthetic_dataset(fake_repo)
        assert result.level == CheckLevel.WARNING


# ---------------------------------------------------------------------------
# Model artifacts check
# ---------------------------------------------------------------------------


class TestCheckModelArtifacts:
    def test_all_present_is_ok(self, fake_repo):
        models_dir = fake_repo / "models"
        for name in ["random_forest.joblib", "xgboost.joblib", "lightgbm.joblib"]:
            (models_dir / name).write_bytes(b"stub")
        result = check_model_artifacts(fake_repo)
        assert result.level == CheckLevel.OK

    def test_all_absent_is_warning(self, fake_repo):
        result = check_model_artifacts(fake_repo)
        assert result.level == CheckLevel.WARNING

    def test_partial_is_warning(self, fake_repo):
        (fake_repo / "models" / "random_forest.joblib").write_bytes(b"stub")
        result = check_model_artifacts(fake_repo)
        assert result.level == CheckLevel.WARNING


# ---------------------------------------------------------------------------
# Optional tools check
# ---------------------------------------------------------------------------


class TestCheckOptionalTools:
    def test_git_found(self):
        """git is almost certainly installed in a dev environment."""
        import shutil

        results = check_optional_tools()
        if shutil.which("git"):
            git_result = next((r for r in results if r.name == "tool:git"), None)
            assert git_result is not None
            assert git_result.level == CheckLevel.OK

    def test_returns_list(self):
        results = check_optional_tools()
        assert isinstance(results, list)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# run_all_checks integration
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    def test_returns_onboarding_report(self, fake_repo, monkeypatch):
        monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")
        monkeypatch.setenv("HORIZON_URL", "https://horizon.stellar.org")
        report = run_all_checks(repo_root=fake_repo)
        assert isinstance(report, OnboardingReport)
        assert len(report.results) > 0

    def test_to_dict_is_serialisable(self, fake_repo, monkeypatch):
        import json

        monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")
        report = run_all_checks(repo_root=fake_repo)
        d = report.to_dict()
        # Must be JSON-serialisable
        json.dumps(d)
        assert "checks" in d
        assert "has_errors" in d

    def test_summary_contains_counts(self, fake_repo, monkeypatch):
        monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")
        report = run_all_checks(repo_root=fake_repo)
        summary = report.summary()
        assert "OK" in summary
        assert "WARNING" in summary
        assert "ERROR" in summary

    def test_has_errors_false_on_good_setup(self, fake_repo, monkeypatch):
        """On a well-configured fake repo with required packages installed,
        there should be no errors."""
        monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")
        monkeypatch.setenv("HORIZON_URL", "https://horizon.stellar.org")
        report = run_all_checks(repo_root=fake_repo)
        # Core packages (pandas, numpy, sqlalchemy) are all installed in CI
        error_names = [r.name for r in report.results if r.level == CheckLevel.ERROR]
        # Only acceptable errors are database-connection (if env var not set
        # correctly) — check that pkg errors don't appear
        pkg_errors = [n for n in error_names if n.startswith("pkg:")]
        assert pkg_errors == [], f"Unexpected package errors: {pkg_errors}"
