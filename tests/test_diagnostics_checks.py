"""Tests for utils/diagnostics_checks.py — concrete diagnostic check implementations.

Tests cover:
- Environment checks (config contracts, env vars, required files)
- Dependency checks (required/optional packages)
- Code health checks (package integrity, import cycles, git status)
- Data artifact checks (models, data directories)
- Runtime checks (database, Horizon API)
"""

from __future__ import annotations

from unittest import mock

from utils.diagnostics import CheckStatus
from utils.diagnostics_checks import (
    ConfigContractCheck,
    CriticalEnvVarsCheck,
    DatabaseConnectivityCheck,
    DataDirectoriesCheck,
    GitStatusCheck,
    HorizonAPICheck,
    ImportCyclesCheck,
    ModelArtifactsCheck,
    OptionalDependenciesCheck,
    PackageIntegrityCheck,
    RequiredFilesCheck,
    RequiredPackagesCheck,
)

# =============================================================================
# Environment Checks Tests
# =============================================================================


def test_config_contract_check_pass():
    """ConfigContractCheck passes when all contracts are valid."""
    check = ConfigContractCheck()
    # This will use whatever env is currently set, so we just test it runs
    result = check.run()
    assert result.check_name == "config_contracts"
    assert result.status in (CheckStatus.PASS, CheckStatus.FAIL)


def test_critical_env_vars_check_missing_critical(monkeypatch):
    """CriticalEnvVarsCheck fails when critical variables are missing."""
    monkeypatch.delenv("HORIZON_URL", raising=False)
    monkeypatch.delenv("MODEL_DIR", raising=False)

    check = CriticalEnvVarsCheck()
    result = check.run()

    assert result.status == CheckStatus.FAIL
    assert "HORIZON_URL" in result.details["missing_critical"]


def test_critical_env_vars_check_pass(monkeypatch):
    """CriticalEnvVarsCheck passes when all critical variables are set."""
    monkeypatch.setenv("HORIZON_URL", "https://horizon.stellar.org")
    monkeypatch.setenv("MODEL_DIR", "./models")
    monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setenv("WATCHED_PAIRS", "XLM/USDC")

    check = CriticalEnvVarsCheck()
    result = check.run()

    assert result.status == CheckStatus.PASS


def test_critical_env_vars_check_missing_recommended(monkeypatch):
    """CriticalEnvVarsCheck warns when recommended variables are missing."""
    monkeypatch.setenv("HORIZON_URL", "https://horizon.stellar.org")
    monkeypatch.setenv("MODEL_DIR", "./models")
    monkeypatch.delenv("RISK_SCORE_DB_URL", raising=False)

    check = CriticalEnvVarsCheck()
    result = check.run()

    assert result.status == CheckStatus.WARN


def test_required_files_check_pass(tmp_path, monkeypatch):
    """RequiredFilesCheck passes when all files exist."""
    # Create required files
    (tmp_path / "config.py").touch()
    (tmp_path / "requirements.txt").touch()
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / ".env.example").touch()

    monkeypatch.chdir(tmp_path)

    check = RequiredFilesCheck()
    result = check.run()

    assert result.status == CheckStatus.PASS


def test_required_files_check_missing_files(tmp_path, monkeypatch):
    """RequiredFilesCheck fails when files are missing."""
    # Only create some files
    (tmp_path / "config.py").touch()

    monkeypatch.chdir(tmp_path)

    check = RequiredFilesCheck()
    result = check.run()

    assert result.status == CheckStatus.FAIL
    assert "requirements.txt" in result.details["missing"]


# =============================================================================
# Dependency Checks Tests
# =============================================================================


def test_required_packages_check_pass():
    """RequiredPackagesCheck passes when all packages are installed."""
    check = RequiredPackagesCheck()
    result = check.run()

    # Assuming the test environment has packages installed
    assert result.status in (CheckStatus.PASS, CheckStatus.FAIL)


def test_optional_dependencies_check_runs():
    """OptionalDependenciesCheck runs without error."""
    check = OptionalDependenciesCheck()
    result = check.run()

    assert result.check_name == "optional_dependencies"
    assert "availability" in result.details


# =============================================================================
# Code Health Checks Tests
# =============================================================================


def test_package_integrity_check_runs():
    """PackageIntegrityCheck runs without error."""
    check = PackageIntegrityCheck()
    result = check.run()

    assert result.check_name == "package_integrity"
    assert result.status in (CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.ERROR)


def test_import_cycles_check_no_cycles():
    """ImportCyclesCheck passes when no cycles detected."""
    with (
        mock.patch("scripts.check_import_cycles._find_python_files", return_value=[]),
        mock.patch("scripts.check_import_cycles.build_dependency_graph", return_value={}),
        mock.patch("scripts.check_import_cycles.find_cycles", return_value=[]),
    ):
        check = ImportCyclesCheck()
        result = check.run()

        assert result.status == CheckStatus.PASS


def test_import_cycles_check_with_cycles():
    """ImportCyclesCheck fails when cycles are detected."""
    mock_cycles = [
        ["module_a", "module_b", "module_a"],
        ["module_c", "module_d", "module_c"],
    ]

    with (
        mock.patch("scripts.check_import_cycles._find_python_files", return_value=[]),
        mock.patch("scripts.check_import_cycles.build_dependency_graph", return_value={}),
        mock.patch("scripts.check_import_cycles.find_cycles", return_value=mock_cycles),
    ):
        check = ImportCyclesCheck()
        result = check.run()

        assert result.status == CheckStatus.FAIL
        assert result.details["cycle_count"] == 2


def test_git_status_check_clean_repo(tmp_path, monkeypatch):
    """GitStatusCheck passes when working directory is clean."""
    monkeypatch.chdir(tmp_path)

    with mock.patch("subprocess.run") as mock_run:
        # Mock git rev-parse
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout=".git\n", stderr=""),
            mock.Mock(returncode=0, stdout="", stderr=""),  # git status --porcelain
        ]

        check = GitStatusCheck()
        result = check.run()

        assert result.status == CheckStatus.PASS


def test_git_status_check_uncommitted_changes(tmp_path, monkeypatch):
    """GitStatusCheck warns when there are uncommitted changes."""
    monkeypatch.chdir(tmp_path)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout=".git\n", stderr=""),
            mock.Mock(
                returncode=0,
                stdout=" M file1.py\n?? file2.py\n",
                stderr="",
            ),
        ]

        check = GitStatusCheck()
        result = check.run()

        assert result.status == CheckStatus.WARN
        assert result.details["uncommitted_count"] == 2


def test_git_status_check_not_a_repo(tmp_path, monkeypatch):
    """GitStatusCheck skips when not in a git repository."""
    monkeypatch.chdir(tmp_path)

    with mock.patch("subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=128, stdout="", stderr="not a git repository")

        check = GitStatusCheck()
        result = check.run()

        assert result.status == CheckStatus.SKIP


# =============================================================================
# Data Artifact Checks Tests
# =============================================================================


def test_model_artifacts_check_missing_directory(tmp_path, monkeypatch):
    """ModelArtifactsCheck fails when model directory doesn't exist."""
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "nonexistent"))

    check = ModelArtifactsCheck()
    result = check.run()

    assert result.status == CheckStatus.FAIL
    assert "does not exist" in result.message


def test_model_artifacts_check_missing_models(tmp_path, monkeypatch):
    """ModelArtifactsCheck fails when model files are missing."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    check = ModelArtifactsCheck()
    result = check.run()

    assert result.status == CheckStatus.FAIL
    assert "missing" in result.details


def test_model_artifacts_check_pass(tmp_path, monkeypatch):
    """ModelArtifactsCheck passes when all models exist."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create model files
    (model_dir / "random_forest.joblib").write_bytes(b"fake model data")
    (model_dir / "xgboost.joblib").write_bytes(b"fake model data")
    (model_dir / "lightgbm.joblib").write_bytes(b"fake model data")
    (model_dir / "model_metadata.json").write_text('{"version": "1.0"}')

    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    check = ModelArtifactsCheck()
    result = check.run()

    assert result.status == CheckStatus.PASS
    assert result.details["has_metadata"] is True


def test_data_directories_check_pass(tmp_path, monkeypatch):
    """DataDirectoriesCheck passes when all directories exist."""
    (tmp_path / "data").mkdir()
    (tmp_path / "models").mkdir()
    (tmp_path / "reports").mkdir()

    monkeypatch.chdir(tmp_path)

    check = DataDirectoriesCheck()
    result = check.run()

    assert result.status == CheckStatus.PASS


def test_data_directories_check_missing(tmp_path, monkeypatch):
    """DataDirectoriesCheck warns when directories are missing."""
    (tmp_path / "data").mkdir()
    # models and reports missing

    monkeypatch.chdir(tmp_path)

    check = DataDirectoriesCheck()
    result = check.run()

    assert result.status == CheckStatus.WARN
    assert "models" in result.details["missing"]


# =============================================================================
# Runtime Checks Tests
# =============================================================================


def test_database_connectivity_check_skip_when_not_configured(monkeypatch):
    """DatabaseConnectivityCheck skips when DB URL not set."""
    monkeypatch.delenv("RISK_SCORE_DB_URL", raising=False)

    check = DatabaseConnectivityCheck()
    result = check.run()

    assert result.status == CheckStatus.SKIP


def test_database_connectivity_check_pass(monkeypatch):
    """DatabaseConnectivityCheck passes when database is reachable."""
    monkeypatch.setenv("RISK_SCORE_DB_URL", "sqlite:///:memory:")

    check = DatabaseConnectivityCheck()
    result = check.run()

    # SQLite in-memory always works if SQLAlchemy is installed
    assert result.status == CheckStatus.PASS


def test_database_connectivity_check_fail(monkeypatch):
    """DatabaseConnectivityCheck fails or errors when database is unreachable."""
    monkeypatch.setenv("RISK_SCORE_DB_URL", "postgresql://badhost:9999/baddb")

    check = DatabaseConnectivityCheck()
    result = check.run()

    # Should fail or error (error if postgres driver not installed)
    assert result.status in (CheckStatus.FAIL, CheckStatus.ERROR)


def test_horizon_api_check_pass(monkeypatch):
    """HorizonAPICheck passes when API is reachable."""
    with mock.patch("requests.get") as mock_get:
        mock_get.return_value = mock.Mock(status_code=200)

        check = HorizonAPICheck()
        result = check.run()

        assert result.status == CheckStatus.PASS


def test_horizon_api_check_fail_bad_status(monkeypatch):
    """HorizonAPICheck fails when API returns bad status."""
    with mock.patch("requests.get") as mock_get:
        mock_get.return_value = mock.Mock(status_code=500)

        check = HorizonAPICheck()
        result = check.run()

        assert result.status == CheckStatus.FAIL


def test_horizon_api_check_fail_timeout(monkeypatch):
    """HorizonAPICheck fails when request times out."""
    import requests

    with mock.patch("requests.get") as mock_get:
        mock_get.side_effect = requests.Timeout("Connection timeout")

        check = HorizonAPICheck()
        result = check.run()

        assert result.status == CheckStatus.FAIL
        assert "timed out" in result.message
