"""Repository health diagnostics framework.

Provides a modular system for checking repository health across configuration,
dependencies, code quality, data artifacts, and runtime readiness. Each check
is self-contained and produces a structured result.

Usage::

    from utils.diagnostics import DiagnosticRegistry, run_diagnostics

    # Run all checks
    report = run_diagnostics()

    # Run specific categories
    report = run_diagnostics(categories=["environment", "dependencies"])

    # Check if repository is healthy
    if report.overall_status == "PASS":
        print("Repository is healthy!")

Architecture:
    - DiagnosticCheck: Protocol defining the interface each check must implement
    - DiagnosticResult: Structured result from a single check
    - DiagnosticReport: Aggregated results from all checks
    - DiagnosticRegistry: Central registry of all available checks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class CheckStatus(StrEnum):
    """Status of a diagnostic check."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"
    ERROR = "error"


class CheckCategory(StrEnum):
    """Category of diagnostic check."""

    ENVIRONMENT = "environment"
    DEPENDENCIES = "dependencies"
    CODE_HEALTH = "code_health"
    DATA_ARTIFACTS = "data_artifacts"
    RUNTIME = "runtime"


@dataclass(frozen=True)
class DiagnosticResult:
    """Result from a single diagnostic check.

    Attributes:
        check_name: Unique identifier for the check
        category: Category this check belongs to
        status: Pass/Warn/Fail/Skip/Error
        message: Human-readable summary
        details: Additional structured information
        remediation: Suggested fix for failures
        duration_ms: Execution time in milliseconds
    """

    check_name: str
    category: CheckCategory
    status: CheckStatus
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "check_name": self.check_name,
            "category": self.category.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "remediation": self.remediation,
            "duration_ms": round(self.duration_ms, 2),
        }

    def is_healthy(self) -> bool:
        """Check passes or was skipped."""
        return self.status in (CheckStatus.PASS, CheckStatus.SKIP)


class DiagnosticCheck(Protocol):
    """Protocol for diagnostic checks.

    Each check must implement:
    - name: Unique identifier
    - category: Which category it belongs to
    - run(): Execute the check and return a result
    """

    name: str
    category: CheckCategory

    def run(self) -> DiagnosticResult:
        """Execute the check and return a structured result."""
        ...


@dataclass
class DiagnosticReport:
    """Aggregated diagnostic report from all checks.

    Attributes:
        results: List of individual check results
        overall_status: Worst status across all checks
        categories_checked: Set of categories that were run
        total_duration_ms: Total execution time
    """

    results: list[DiagnosticResult]
    overall_status: CheckStatus
    categories_checked: set[CheckCategory]
    total_duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_status": self.overall_status.value,
            "total_checks": len(self.results),
            "pass_count": sum(1 for r in self.results if r.status == CheckStatus.PASS),
            "warn_count": sum(1 for r in self.results if r.status == CheckStatus.WARN),
            "fail_count": sum(1 for r in self.results if r.status == CheckStatus.FAIL),
            "error_count": sum(1 for r in self.results if r.status == CheckStatus.ERROR),
            "skip_count": sum(1 for r in self.results if r.status == CheckStatus.SKIP),
            "categories_checked": [c.value for c in sorted(self.categories_checked)],
            "total_duration_ms": round(self.total_duration_ms, 2),
            "checks": [r.to_dict() for r in self.results],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        lines.append(f"Overall Status: {self.overall_status.value.upper()}")
        lines.append(f"Total Checks: {len(self.results)}")

        by_status = {}
        for result in self.results:
            by_status.setdefault(result.status, []).append(result)

        for status in [CheckStatus.FAIL, CheckStatus.ERROR, CheckStatus.WARN]:
            if status in by_status:
                lines.append(f"  {status.value.upper()}: {len(by_status[status])}")
                for r in by_status[status]:
                    lines.append(f"    - {r.check_name}: {r.message}")
                    if r.remediation:
                        lines.append(f"      Fix: {r.remediation}")

        if CheckStatus.PASS in by_status:
            lines.append(f"  PASS: {len(by_status[CheckStatus.PASS])}")

        lines.append(f"\nExecution time: {self.total_duration_ms:.0f}ms")
        return "\n".join(lines)

    def is_healthy(self) -> bool:
        """Repository is healthy if no failures or errors."""
        return self.overall_status in (CheckStatus.PASS, CheckStatus.WARN, CheckStatus.SKIP)


class DiagnosticRegistry:
    """Central registry of all diagnostic checks."""

    def __init__(self) -> None:
        self._checks: dict[str, DiagnosticCheck] = {}

    def register(self, check: DiagnosticCheck) -> None:
        """Register a diagnostic check."""
        if check.name in self._checks:
            raise ValueError(f"Check '{check.name}' is already registered")
        self._checks[check.name] = check

    def get_check(self, name: str) -> DiagnosticCheck | None:
        """Get a check by name."""
        return self._checks.get(name)

    def list_checks(self, category: CheckCategory | None = None) -> list[tuple[str, CheckCategory]]:
        """List all registered checks, optionally filtered by category."""
        checks = [
            (name, check.category)
            for name, check in sorted(self._checks.items())
            if category is None or check.category == category
        ]
        return checks

    def run_all(
        self, categories: list[CheckCategory] | None = None, fail_fast: bool = False
    ) -> DiagnosticReport:
        """Run all checks, optionally filtered by category.

        Args:
            categories: If provided, only run checks in these categories
            fail_fast: Stop on first failure

        Returns:
            Aggregated diagnostic report
        """
        import time

        start_time = time.perf_counter()
        results: list[DiagnosticResult] = []
        categories_checked: set[CheckCategory] = set()

        checks_to_run = [
            (name, check)
            for name, check in sorted(self._checks.items())
            if categories is None or check.category in categories
        ]

        for name, check in checks_to_run:
            check_start = time.perf_counter()
            try:
                result = check.run()
                result = DiagnosticResult(
                    check_name=result.check_name,
                    category=result.category,
                    status=result.status,
                    message=result.message,
                    details=result.details,
                    remediation=result.remediation,
                    duration_ms=(time.perf_counter() - check_start) * 1000,
                )
                results.append(result)
                categories_checked.add(check.category)

                if fail_fast and result.status == CheckStatus.FAIL:
                    break

            except Exception as exc:
                results.append(
                    DiagnosticResult(
                        check_name=name,
                        category=check.category,
                        status=CheckStatus.ERROR,
                        message=f"Check raised exception: {type(exc).__name__}",
                        details={"exception": str(exc)},
                        duration_ms=(time.perf_counter() - check_start) * 1000,
                    )
                )
                categories_checked.add(check.category)

                if fail_fast:
                    break

        # Determine overall status
        statuses = [r.status for r in results]
        if CheckStatus.ERROR in statuses or CheckStatus.FAIL in statuses:
            overall_status = CheckStatus.FAIL
        elif CheckStatus.WARN in statuses:
            overall_status = CheckStatus.WARN
        elif CheckStatus.PASS in statuses:
            overall_status = CheckStatus.PASS
        else:
            overall_status = CheckStatus.SKIP

        total_duration_ms = (time.perf_counter() - start_time) * 1000

        return DiagnosticReport(
            results=results,
            overall_status=overall_status,
            categories_checked=categories_checked,
            total_duration_ms=total_duration_ms,
        )


# Global registry instance
_default_registry = DiagnosticRegistry()


def register_check(check: DiagnosticCheck) -> None:
    """Register a check with the default registry."""
    _default_registry.register(check)


def run_diagnostics(
    categories: list[str] | None = None, fail_fast: bool = False
) -> DiagnosticReport:
    """Run repository diagnostics.

    Args:
        categories: If provided, only run checks in these categories
        fail_fast: Stop on first failure

    Returns:
        Diagnostic report
    """
    category_enums = None
    if categories:
        category_enums = [CheckCategory(c) for c in categories]

    return _default_registry.run_all(categories=category_enums, fail_fast=fail_fast)


def list_available_checks(category: str | None = None) -> list[tuple[str, str]]:
    """List all available diagnostic checks.

    Args:
        category: If provided, filter to this category

    Returns:
        List of (check_name, category) tuples
    """
    cat_enum = CheckCategory(category) if category else None
    return [(name, cat.value) for name, cat in _default_registry.list_checks(category=cat_enum)]
