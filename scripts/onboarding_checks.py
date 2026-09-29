"""LedgerLens first-time local setup onboarding checks.

Validates that a fresh checkout has everything it needs to run ``make test``
and ``python run_pipeline.py`` successfully.  Checks are grouped into:

- **Python** — version, virtual-environment, and pip-installed packages
- **Files** — presence of required data files, config files, and model dirs
- **Environment** — required / optional env vars from ``.env`` or the shell
- **Database** — DB URL reachability and schema migration state
- **Optional tools** — make, docker, pre-commit (advisory; not required)

Usage::

    python -m scripts.onboard            # run all checks, print results
    python -m scripts.onboard --fix      # attempt auto-fixes (copy .env.example, mkdir)
    python -m scripts.onboard --json     # machine-readable output
    python -m scripts.onboard --strict   # exit 1 on any WARNING (not just ERROR)
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CheckLevel(StrEnum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class CheckResult:
    name: str
    level: CheckLevel
    message: str
    fix_applied: bool = False
    fix_hint: str = ""

    @property
    def ok(self) -> bool:
        return self.level == CheckLevel.OK

    def __str__(self) -> str:
        icon = {"OK": "✓", "WARNING": "⚠", "ERROR": "✗"}[self.level.value]
        base = f"  [{icon}] {self.name}: {self.message}"
        if self.fix_hint and self.level != CheckLevel.OK:
            base += f"\n       Fix: {self.fix_hint}"
        if self.fix_applied:
            base += "  (auto-fixed)"
        return base


@dataclass
class OnboardingReport:
    results: list[CheckResult] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(r.level == CheckLevel.ERROR for r in self.results)

    @property
    def has_warnings(self) -> bool:
        return any(r.level == CheckLevel.WARNING for r in self.results)

    def summary(self) -> str:
        counts = {level: 0 for level in CheckLevel}
        for r in self.results:
            counts[r.level] += 1
        return (
            f"{counts[CheckLevel.OK]} OK  "
            f"{counts[CheckLevel.WARNING]} WARNING  "
            f"{counts[CheckLevel.ERROR]} ERROR"
        )

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "has_errors": self.has_errors,
            "has_warnings": self.has_warnings,
            "checks": [
                {
                    "name": r.name,
                    "level": r.level.value,
                    "message": r.message,
                    "fix_applied": r.fix_applied,
                    "fix_hint": r.fix_hint,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

# Minimum required Python version
_MIN_PYTHON = (3, 11)

# Packages that must be importable for the pipeline to work at all
_REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "sqlalchemy",
    "sklearn",
    "xgboost",
    "lightgbm",
    "pydantic",
    "dotenv",
]

# Packages that power optional features (missing → WARNING not ERROR)
_OPTIONAL_PACKAGES = [
    "shap",
    "torch",
    "torch_geometric",
    "community",  # python-louvain
    "networkx",
    "opacus",
    "stable_baselines3",
    "gymnasium",
]

# Files that must exist for a functional dev environment
_REQUIRED_FILES = [
    ".env",
    "requirements.txt",
    "pyproject.toml",
]

# Directories that should exist (created automatically if --fix)
_REQUIRED_DIRS = [
    "models",
    "data",
    "reports",
    "reports/forensic",
]

# Required env vars for the default pipeline mode
_REQUIRED_ENV_VARS: list[tuple[str, str]] = [
    ("HORIZON_URL", "Stellar Horizon API base URL"),
    ("RISK_SCORE_DB_URL", "SQLAlchemy DB URL for persisting risk scores"),
]

# Optional but common env vars — missing → WARNING
_OPTIONAL_ENV_VARS: list[tuple[str, str]] = [
    ("LEDGERLENS_CONTRACT_ID", "Soroban contract ID for on-chain submission"),
    ("LEDGERLENS_SUBMITTER_SECRET", "Secret key for Soroban contract calls"),
    ("MODEL_SIGNING_PRIVATE_KEY_PATH", "Ed25519 private key for signing model artifacts"),
    ("ALERT_WEBHOOK_URL", "Webhook URL for streaming alert delivery"),
]


def _ok(name: str, msg: str) -> CheckResult:
    return CheckResult(name=name, level=CheckLevel.OK, message=msg)


def _warn(name: str, msg: str, fix_hint: str = "") -> CheckResult:
    return CheckResult(name=name, level=CheckLevel.WARNING, message=msg, fix_hint=fix_hint)


def _error(name: str, msg: str, fix_hint: str = "") -> CheckResult:
    return CheckResult(name=name, level=CheckLevel.ERROR, message=msg, fix_hint=fix_hint)


# ── Python version ──────────────────────────────────────────────────────────


def check_python_version() -> CheckResult:
    v = sys.version_info[:2]
    if v >= _MIN_PYTHON:
        return _ok("python-version", f"Python {v[0]}.{v[1]} (>= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]})")
    return _error(
        "python-version",
        f"Python {v[0]}.{v[1]} detected — LedgerLens requires >= {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}",
        fix_hint=f"Install Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]}+ and re-create your virtual environment.",
    )


def check_virtual_env() -> CheckResult:
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.getenv("VIRTUAL_ENV") is not None
        or os.getenv("CONDA_DEFAULT_ENV") is not None
    )
    if in_venv:
        venv_path = os.getenv("VIRTUAL_ENV", sys.prefix)
        return _ok("virtual-env", f"Active virtual environment: {venv_path}")
    return _warn(
        "virtual-env",
        "No active virtual environment detected.",
        fix_hint="python -m venv .venv && source .venv/bin/activate",
    )


def check_required_packages() -> list[CheckResult]:
    results = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            results.append(_ok(f"pkg:{pkg}", f"{pkg} importable"))
        except ImportError:
            results.append(
                _error(
                    f"pkg:{pkg}",
                    f"Required package '{pkg}' not installed.",
                    fix_hint="make install  (or: pip install -r requirements.txt)",
                )
            )
    return results


def check_optional_packages() -> list[CheckResult]:
    results = []
    for pkg in _OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            results.append(_ok(f"pkg:{pkg}", f"{pkg} importable"))
        except ImportError:
            results.append(
                _warn(
                    f"pkg:{pkg}",
                    f"Optional package '{pkg}' not installed — some features will be disabled.",
                    fix_hint="pip install -r requirements.txt",
                )
            )
    return results


# ── Files and directories ───────────────────────────────────────────────────


def check_required_files(repo_root: Path, fix: bool = False) -> list[CheckResult]:
    results = []
    for fname in _REQUIRED_FILES:
        path = repo_root / fname
        if path.exists():
            results.append(_ok(f"file:{fname}", f"{fname} present"))
        else:
            if fname == ".env" and fix:
                example = repo_root / ".env.example"
                if example.exists():
                    shutil.copy(example, path)
                    r = _ok(f"file:{fname}", ".env created from .env.example (auto-fixed)")
                    r.fix_applied = True
                    results.append(r)
                    continue
            results.append(
                _error(
                    f"file:{fname}",
                    f"{fname} not found.",
                    fix_hint=(
                        "cp .env.example .env  # then edit as needed"
                        if fname == ".env"
                        else f"Ensure {fname} is present (it should be part of the repo)."
                    ),
                )
            )
    return results


def check_required_dirs(repo_root: Path, fix: bool = False) -> list[CheckResult]:
    results = []
    for dname in _REQUIRED_DIRS:
        path = repo_root / dname
        if path.is_dir():
            results.append(_ok(f"dir:{dname}", f"{dname}/ present"))
        else:
            if fix:
                path.mkdir(parents=True, exist_ok=True)
                r = _ok(f"dir:{dname}", f"{dname}/ created (auto-fixed)")
                r.fix_applied = True
                results.append(r)
            else:
                results.append(
                    _warn(
                        f"dir:{dname}",
                        f"Directory {dname}/ does not exist.",
                        fix_hint=f"mkdir -p {dname}",
                    )
                )
    return results


def check_env_file_loaded(repo_root: Path) -> CheckResult:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return _warn(
            "env-file",
            ".env file not found — environment variables must be set manually.",
            fix_hint="cp .env.example .env",
        )
    # Attempt to load it
    try:
        from dotenv import dotenv_values

        values = dotenv_values(env_path)
        return _ok("env-file", f".env loaded ({len(values)} variables)")
    except Exception as exc:
        return _warn("env-file", f".env found but could not be parsed: {exc}")


# ── Environment variables ───────────────────────────────────────────────────


def check_env_vars() -> list[CheckResult]:
    # Make sure .env is loaded before checking
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass

    results = []
    for var, description in _REQUIRED_ENV_VARS:
        val = os.getenv(var)
        if val:
            results.append(_ok(f"env:{var}", f"{var} set"))
        else:
            results.append(
                _warn(
                    f"env:{var}",
                    f"Required env var {var} not set ({description}).",
                    fix_hint=f"Add {var}=<value> to your .env file.",
                )
            )
    for var, description in _OPTIONAL_ENV_VARS:
        val = os.getenv(var)
        if val:
            results.append(_ok(f"env:{var}", f"{var} set"))
        else:
            results.append(
                _warn(
                    f"env:{var}",
                    f"Optional env var {var} not set ({description}).",
                    fix_hint=f"Add {var}=<value> to .env if you need this feature.",
                )
            )
    return results


# ── Database ─────────────────────────────────────────────────────────────────


def check_database() -> list[CheckResult]:
    results = []
    db_url = os.getenv("RISK_SCORE_DB_URL", "sqlite:///./risk_scores.db")
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(
            db_url, connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        results.append(_ok("database-connection", f"Database reachable: {db_url}"))
    except Exception as exc:
        results.append(
            _error(
                "database-connection",
                f"Cannot connect to RISK_SCORE_DB_URL ({db_url}): {exc}",
                fix_hint="Check RISK_SCORE_DB_URL in your .env file.",
            )
        )
        return results

    # Migration status
    try:
        from detection.persistence import get_engine
        from migrations import MigrationRunner

        runner = MigrationRunner(get_engine(db_url))
        status = runner.status()
        if status.is_up_to_date:
            results.append(
                _ok("database-migrations", f"All {len(status.applied)} migrations applied")
            )
        else:
            results.append(
                _warn(
                    "database-migrations",
                    f"{len(status.pending)} pending migration(s): {status.pending}",
                    fix_hint="python -m scripts.migrate",
                )
            )
    except Exception as exc:
        results.append(
            _warn(
                "database-migrations",
                f"Could not check migration status: {exc}",
                fix_hint="python -m scripts.migrate",
            )
        )

    return results


# ── Optional tools ──────────────────────────────────────────────────────────


def check_optional_tools() -> list[CheckResult]:
    tools = [
        ("make", "make test / make lint shortcuts"),
        ("docker", "containerised pipeline runs"),
        ("pre-commit", "automatic pre-commit hooks"),
        ("git", "version control"),
    ]
    results = []
    for tool, purpose in tools:
        if shutil.which(tool):
            results.append(_ok(f"tool:{tool}", f"{tool} available"))
        else:
            results.append(
                _warn(
                    f"tool:{tool}",
                    f"'{tool}' not found in PATH ({purpose}).",
                    fix_hint=f"Install {tool} for the best development experience.",
                )
            )
    return results


# ── Synthetic dataset ────────────────────────────────────────────────────────


def check_synthetic_dataset(repo_root: Path) -> CheckResult:
    path = repo_root / "data" / "synthetic_dataset.parquet"
    if path.exists():
        size_kb = path.stat().st_size // 1024
        return _ok("synthetic-dataset", f"data/synthetic_dataset.parquet present ({size_kb} KB)")
    return _warn(
        "synthetic-dataset",
        "data/synthetic_dataset.parquet not found — model training / CI will need it.",
        fix_hint=(
            "python -m scripts.generate_synthetic_dataset "
            "--output data/synthetic_dataset.parquet"
        ),
    )


# ── Model artifacts ──────────────────────────────────────────────────────────


def check_model_artifacts(repo_root: Path) -> CheckResult:
    models_dir = repo_root / "models"
    expected = ["random_forest.joblib", "xgboost.joblib", "lightgbm.joblib"]
    present = [f for f in expected if (models_dir / f).exists()]
    if len(present) == len(expected):
        return _ok("model-artifacts", f"All {len(expected)} model artifacts present in models/")
    if present:
        missing = [f for f in expected if f not in present]
        return _warn(
            "model-artifacts",
            f"Some model artifacts missing: {missing}",
            fix_hint=(
                "python -m scripts.generate_synthetic_dataset "
                "--output data/synthetic_dataset.parquet && "
                "python -m detection.model_training "
                "--data-path data/synthetic_dataset.parquet"
            ),
        )
    return _warn(
        "model-artifacts",
        "No trained model artifacts found in models/ — risk scoring will not work.",
        fix_hint=(
            "python -m scripts.generate_synthetic_dataset "
            "--output data/synthetic_dataset.parquet && "
            "python -m detection.model_training "
            "--data-path data/synthetic_dataset.parquet"
        ),
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_checks(
    repo_root: Path | None = None,
    *,
    fix: bool = False,
    include_optional_packages: bool = False,
) -> OnboardingReport:
    """Execute all onboarding checks and return an :class:`OnboardingReport`.

    Parameters
    ----------
    repo_root:
        Root of the LedgerLens repository.  Defaults to the directory
        two levels above this file (i.e. the repo root).
    fix:
        When True, attempt auto-fixes for trivial problems (copy .env,
        create missing directories).
    include_optional_packages:
        When True, also check optional package imports (adds many OK entries).
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    report = OnboardingReport()
    add = report.results.append
    extend = report.results.extend

    # Python
    add(check_python_version())
    add(check_virtual_env())
    extend(check_required_packages())
    if include_optional_packages:
        extend(check_optional_packages())

    # Files / dirs
    extend(check_required_files(repo_root, fix=fix))
    extend(check_required_dirs(repo_root, fix=fix))
    add(check_env_file_loaded(repo_root))

    # Environment
    extend(check_env_vars())

    # Database
    extend(check_database())

    # Data / models
    add(check_synthetic_dataset(repo_root))
    add(check_model_artifacts(repo_root))

    # Optional tools
    extend(check_optional_tools())

    return report
