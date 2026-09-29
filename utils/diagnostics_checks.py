"""Concrete diagnostic check implementations.

This module contains all diagnostic checks for repository health. Checks are
organized by category and automatically registered with the global registry.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

from utils.diagnostics import (
    CheckCategory,
    CheckStatus,
    DiagnosticResult,
    register_check,
)

# =============================================================================
# ENVIRONMENT CHECKS
# =============================================================================


class ConfigContractCheck:
    """Check that all runtime mode config contracts can be validated."""

    name = "config_contracts"
    category = CheckCategory.ENVIRONMENT

    def run(self) -> DiagnosticResult:
        try:
            from config.contracts import RUNTIME_MODES, validate_mode

            errors = []
            for mode in sorted(RUNTIME_MODES):
                try:
                    validate_mode(mode)
                except (OSError, ValueError) as exc:
                    errors.append(f"{mode}: {str(exc).splitlines()[0]}")

            if not errors:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.PASS,
                    message=f"All {len(RUNTIME_MODES)} runtime mode contracts validated",
                    details={"modes": sorted(RUNTIME_MODES)},
                )

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(errors)} runtime mode(s) have config errors",
                details={"errors": errors},
                remediation="Set missing environment variables or create missing files",
            )

        except ImportError as exc:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.ERROR,
                message="Cannot import config.contracts",
                details={"exception": str(exc)},
            )


class CriticalEnvVarsCheck:
    """Check that critical environment variables are set."""

    name = "critical_env_vars"
    category = CheckCategory.ENVIRONMENT

    # Core variables needed for basic operation
    CRITICAL_VARS = [
        "HORIZON_URL",
        "MODEL_DIR",
    ]

    # Optional but commonly needed
    RECOMMENDED_VARS = [
        "RISK_SCORE_DB_URL",
        "WATCHED_PAIRS",
    ]

    def run(self) -> DiagnosticResult:
        missing_critical = [var for var in self.CRITICAL_VARS if not os.getenv(var)]
        missing_recommended = [var for var in self.RECOMMENDED_VARS if not os.getenv(var)]

        if missing_critical:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(missing_critical)} critical environment variable(s) missing",
                details={
                    "missing_critical": missing_critical,
                    "missing_recommended": missing_recommended,
                },
                remediation=f"Set: {', '.join(missing_critical)}",
            )

        if missing_recommended:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.WARN,
                message=f"{len(missing_recommended)} recommended variable(s) missing",
                details={"missing_recommended": missing_recommended},
                remediation=f"Consider setting: {', '.join(missing_recommended)}",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message="All critical environment variables are set",
            details={
                "critical_vars": self.CRITICAL_VARS,
                "recommended_vars": self.RECOMMENDED_VARS,
            },
        )


class RequiredFilesCheck:
    """Check that required configuration files exist."""

    name = "required_files"
    category = CheckCategory.ENVIRONMENT

    REQUIRED_FILES = [
        "config.py",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
    ]

    def run(self) -> DiagnosticResult:
        repo_root = Path.cwd()
        missing = []
        found = []

        for filename in self.REQUIRED_FILES:
            filepath = repo_root / filename
            if filepath.exists():
                found.append(filename)
            else:
                missing.append(filename)

        if missing:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(missing)} required file(s) missing",
                details={"missing": missing, "found": found},
                remediation="Ensure you're in the repository root",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message=f"All {len(self.REQUIRED_FILES)} required files present",
            details={"files": found},
        )


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================


class RequiredPackagesCheck:
    """Check that critical Python packages are installed."""

    name = "required_packages"
    category = CheckCategory.DEPENDENCIES

    REQUIRED_PACKAGES = [
        "pandas",
        "numpy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "sqlalchemy",
        "pydantic",
        "stellar_sdk",
    ]

    def run(self) -> DiagnosticResult:
        missing = []
        installed = {}

        for pkg in self.REQUIRED_PACKAGES:
            try:
                spec = importlib.util.find_spec(pkg)
                if spec is None:
                    missing.append(pkg)
                else:
                    # Try to get version
                    try:
                        mod = importlib.import_module(pkg)
                        version = getattr(mod, "__version__", "unknown")
                        installed[pkg] = version
                    except Exception:
                        installed[pkg] = "installed"
            except (ImportError, ModuleNotFoundError):
                missing.append(pkg)

        if missing:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(missing)} required package(s) missing",
                details={"missing": missing, "installed": installed},
                remediation="Run: make install or pip install -r requirements.txt",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message=f"All {len(self.REQUIRED_PACKAGES)} required packages installed",
            details={"packages": installed},
        )


class OptionalDependenciesCheck:
    """Check availability of optional feature dependencies."""

    name = "optional_dependencies"
    category = CheckCategory.DEPENDENCIES

    OPTIONAL_GROUPS = {
        "kafka": ["confluent_kafka", "fastavro"],
        "gnn": ["torch", "torch_geometric"],
        "rl": ["stable_baselines3", "gymnasium"],
    }

    def run(self) -> DiagnosticResult:
        availability = {}
        partially_available = []
        unavailable = []

        for group, packages in self.OPTIONAL_GROUPS.items():
            available_count = 0
            for pkg in packages:
                try:
                    spec = importlib.util.find_spec(pkg)
                    if spec is not None:
                        available_count += 1
                except (ImportError, ModuleNotFoundError):
                    pass

            if available_count == len(packages):
                availability[group] = "available"
            elif available_count == 0:
                availability[group] = "unavailable"
                unavailable.append(group)
            else:
                availability[group] = "partial"
                partially_available.append(group)

        if partially_available:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.WARN,
                message=f"{len(partially_available)} optional group(s) partially available",
                details={"availability": availability},
                remediation=f"Some {', '.join(partially_available)} packages missing",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message="Optional dependencies checked",
            details={"availability": availability},
        )


# =============================================================================
# CODE HEALTH CHECKS
# =============================================================================


class PackageIntegrityCheck:
    """Check Python package integrity (missing __init__.py, syntax errors)."""

    name = "package_integrity"
    category = CheckCategory.CODE_HEALTH

    def run(self) -> DiagnosticResult:
        try:
            from utils.package_integrity import (
                DEFAULT_SOURCE_PACKAGES,
                check_source_package_integrity,
            )

            report = check_source_package_integrity(root=".", packages=DEFAULT_SOURCE_PACKAGES)

            if report.ok:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.PASS,
                    message="All source packages are structurally sound",
                    details={
                        "packages_checked": len(DEFAULT_SOURCE_PACKAGES),
                    },
                )

            issues = str(report).split("\n")
            issue_summary = [line for line in issues if line.strip().startswith("-")]

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message="Package integrity issues found",
                details={"issues": issue_summary[:10]},  # Limit to first 10
                remediation="Run: python scripts/check_package_integrity.py",
            )

        except Exception as exc:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.ERROR,
                message="Cannot check package integrity",
                details={"exception": str(exc)},
            )


class ImportCyclesCheck:
    """Check for circular import dependencies."""

    name = "import_cycles"
    category = CheckCategory.CODE_HEALTH

    def run(self) -> DiagnosticResult:
        try:
            from importlib import import_module
            from pathlib import Path

            cycle_checker = import_module("scripts.check_import_cycles")
            _find_python_files = cycle_checker._find_python_files
            build_dependency_graph = cycle_checker.build_dependency_graph
            find_cycles = cycle_checker.find_cycles

            root = Path.cwd()
            packages = ["detection", "ingestion", "streaming"]

            files = _find_python_files(root, packages)
            graph = build_dependency_graph(files, root)
            cycles = find_cycles(graph, cross_package_only=False)

            if not cycles:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.PASS,
                    message="No import cycles detected",
                )

            cycle_descriptions = [" -> ".join(cycle[:3]) + " ..." for cycle in cycles[:5]]

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(cycles)} import cycle(s) detected",
                details={"cycle_count": len(cycles), "examples": cycle_descriptions},
                remediation="Run: make check-cycles",
            )

        except ImportError:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.SKIP,
                message="Import cycle checker not available",
            )
        except Exception as exc:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.ERROR,
                message="Cannot check import cycles",
                details={"exception": str(exc)},
            )


class GitStatusCheck:
    """Check git repository status."""

    name = "git_status"
    category = CheckCategory.CODE_HEALTH

    def run(self) -> DiagnosticResult:
        try:
            # Check if we're in a git repo
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if result.returncode != 0:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.SKIP,
                    message="Not a git repository",
                )

            # Check for uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=2,
            )

            uncommitted = result.stdout.strip().split("\n") if result.stdout.strip() else []
            uncommitted_count = len([line for line in uncommitted if line])

            if uncommitted_count > 0:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.WARN,
                    message=f"{uncommitted_count} uncommitted change(s)",
                    details={"uncommitted_count": uncommitted_count},
                    remediation="Commit or stash changes before deploying",
                )

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.PASS,
                message="Working directory is clean",
            )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.SKIP,
                message="Git not available",
            )
        except Exception as exc:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.ERROR,
                message="Cannot check git status",
                details={"exception": str(exc)},
            )


# =============================================================================
# DATA ARTIFACT CHECKS
# =============================================================================


class ModelArtifactsCheck:
    """Check that trained model artifacts exist."""

    name = "model_artifacts"
    category = CheckCategory.DATA_ARTIFACTS

    EXPECTED_MODELS = ["random_forest.joblib", "xgboost.joblib", "lightgbm.joblib"]

    def run(self) -> DiagnosticResult:
        model_dir = Path(os.getenv("MODEL_DIR", "./models"))

        if not model_dir.exists():
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message="Model directory does not exist",
                details={"model_dir": str(model_dir)},
                remediation=f"Create {model_dir} or run: python -m detection.model_training",
            )

        found = []
        missing = []

        for model_file in self.EXPECTED_MODELS:
            model_path = model_dir / model_file
            if model_path.exists():
                # Get file size
                size_mb = model_path.stat().st_size / (1024 * 1024)
                found.append({"name": model_file, "size_mb": round(size_mb, 2)})
            else:
                missing.append(model_file)

        # Check for model_metadata.json
        metadata_path = model_dir / "model_metadata.json"
        has_metadata = metadata_path.exists()

        if missing:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"{len(missing)} model artifact(s) missing",
                details={"missing": missing, "found": found, "has_metadata": has_metadata},
                remediation="Run: python -m detection.model_training",
            )

        if not has_metadata:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.WARN,
                message="Models exist but metadata is missing",
                details={"found": found, "has_metadata": False},
                remediation="Retrain models to generate metadata",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message=f"All {len(self.EXPECTED_MODELS)} model artifacts present",
            details={"models": found, "has_metadata": True},
        )


class DataDirectoriesCheck:
    """Check that expected data directories exist."""

    name = "data_directories"
    category = CheckCategory.DATA_ARTIFACTS

    EXPECTED_DIRS = [
        "data",
        "models",
        "reports",
    ]

    def run(self) -> DiagnosticResult:
        repo_root = Path.cwd()
        missing = []
        found = []

        for dirname in self.EXPECTED_DIRS:
            dirpath = repo_root / dirname
            if dirpath.exists() and dirpath.is_dir():
                found.append(dirname)
            else:
                missing.append(dirname)

        if missing:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.WARN,
                message=f"{len(missing)} expected director(ies) missing",
                details={"missing": missing, "found": found},
                remediation=f"Create: {', '.join(missing)}",
            )

        return DiagnosticResult(
            check_name=self.name,
            category=self.category,
            status=CheckStatus.PASS,
            message=f"All {len(self.EXPECTED_DIRS)} data directories exist",
            details={"directories": found},
        )


# =============================================================================
# RUNTIME CHECKS
# =============================================================================


class DatabaseConnectivityCheck:
    """Check database connectivity."""

    name = "database_connectivity"
    category = CheckCategory.RUNTIME

    def run(self) -> DiagnosticResult:
        db_url = os.getenv("RISK_SCORE_DB_URL")

        if not db_url:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.SKIP,
                message="RISK_SCORE_DB_URL not configured",
            )

        try:
            from sqlalchemy import create_engine, text

            # Sanitize URL for display
            from utils.secrets import sanitize_url

            display_url = sanitize_url(db_url)

            engine = create_engine(db_url, pool_pre_ping=True, connect_args={"timeout": 3})

            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                result.fetchone()

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.PASS,
                message="Database is reachable",
                details={"url": display_url, "component": "database"},
            )

        except ImportError:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.ERROR,
                message="SQLAlchemy not installed",
                remediation="Run: make install",
            )
        except Exception as exc:
            display_url = sanitize_url(db_url)
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"Database connectivity check failed for {display_url}",
                details={
                    "url": display_url,
                    "component": "database",
                    "error": str(exc)[:100],
                },
                remediation="Check database URL and network connectivity",
            )


class HorizonAPICheck:
    """Check Horizon API reachability."""

    name = "horizon_api"
    category = CheckCategory.RUNTIME

    def run(self) -> DiagnosticResult:
        horizon_url = os.getenv("HORIZON_URL", "https://horizon.stellar.org")

        try:
            import requests

            response = requests.get(
                horizon_url,
                timeout=5,
                headers={"Accept": "application/json"},
            )

            if response.status_code == 200:
                return DiagnosticResult(
                    check_name=self.name,
                    category=self.category,
                    status=CheckStatus.PASS,
                    message="Horizon API is reachable",
                    details={"url": horizon_url, "status_code": 200},
                )

            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message=f"Horizon API returned HTTP {response.status_code}",
                details={"url": horizon_url, "status_code": response.status_code},
                remediation="Check HORIZON_URL or network connectivity",
            )

        except ImportError:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.WARN,
                message="requests package not available, cannot check Horizon",
            )
        except requests.Timeout:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message="Horizon API request timed out",
                details={"url": horizon_url},
                remediation="Check network connectivity",
            )
        except Exception as exc:
            return DiagnosticResult(
                check_name=self.name,
                category=self.category,
                status=CheckStatus.FAIL,
                message="Cannot reach Horizon API",
                details={"url": horizon_url, "error": str(exc)[:100]},
                remediation="Check HORIZON_URL or network connectivity",
            )


# =============================================================================
# AUTO-REGISTER ALL CHECKS
# =============================================================================


def _register_all_checks() -> None:
    """Automatically register all check classes defined in this module."""
    checks = [
        # Environment
        ConfigContractCheck(),
        CriticalEnvVarsCheck(),
        RequiredFilesCheck(),
        # Dependencies
        RequiredPackagesCheck(),
        OptionalDependenciesCheck(),
        # Code Health
        PackageIntegrityCheck(),
        ImportCyclesCheck(),
        GitStatusCheck(),
        # Data Artifacts
        ModelArtifactsCheck(),
        DataDirectoriesCheck(),
        # Runtime
        DatabaseConnectivityCheck(),
        HorizonAPICheck(),
    ]

    for check in checks:
        register_check(check)


# Register all checks when this module is imported
_register_all_checks()
