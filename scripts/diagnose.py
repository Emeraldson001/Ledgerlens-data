#!/usr/bin/env python3
"""Repository health diagnostics CLI.

Comprehensive health check system for the LedgerLens-data repository. Checks
configuration, dependencies, code quality, data artifacts, and runtime readiness.

Usage:
    # Run all checks
    python -m scripts.diagnose

    # Run specific categories
    python -m scripts.diagnose --categories environment dependencies

    # JSON output for CI
    python -m scripts.diagnose --json

    # List available checks
    python -m scripts.diagnose --list

    # Fail fast on first failure
    python -m scripts.diagnose --fail-fast

Exit codes:
    0  All checks passed (or only warnings)
    1  One or more checks failed
    2  Error during execution

Examples:
    # Check environment before deployment
    python -m scripts.diagnose --categories environment runtime

    # Quick code health check
    python -m scripts.diagnose --categories code_health

    # CI validation
    python -m scripts.diagnose --json --fail-fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure we can import from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.diagnostics import CheckCategory, list_available_checks, run_diagnostics
from utils.logging import get_logger

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LedgerLens repository health diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        choices=[c.value for c in CheckCategory],
        help="Run only checks in these categories (default: all)",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List available checks and exit",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed information for each check",
    )

    return parser


def _print_check_list(category: str | None = None) -> None:
    """Print available checks grouped by category."""
    from collections import defaultdict

    checks_by_category = defaultdict(list)
    for check_name, cat in list_available_checks(category=category):
        checks_by_category[cat].append(check_name)

    logger.info("Available diagnostic checks:")
    for cat in sorted(checks_by_category.keys()):
        logger.info(f"  {cat}:")
        for check_name in sorted(checks_by_category[cat]):
            logger.info(f"    - {check_name}")


def _print_verbose_report(report: dict[str, any]) -> None:
    """Print detailed report with all check information."""

    logger.info(f"\n{'='*70}")
    logger.info("Repository Health Diagnostics")
    logger.info(f"{'='*70}\n")

    logger.info(f"Overall Status: {report['overall_status'].upper()}")
    logger.info(f"Total Checks: {report['total_checks']}")
    logger.info(f"  PASS: {report['pass_count']}")
    logger.info(f"  WARN: {report['warn_count']}")
    logger.info(f"  FAIL: {report['fail_count']}")
    logger.info(f"  ERROR: {report['error_count']}")
    logger.info(f"  SKIP: {report['skip_count']}")
    logger.info(f"Categories: {', '.join(report['categories_checked'])}")
    logger.info(f"Execution time: {report['total_duration_ms']:.0f}ms\n")

    # Group by category
    from collections import defaultdict

    by_category = defaultdict(list)
    for check in report["checks"]:
        by_category[check["category"]].append(check)

    for category in sorted(by_category.keys()):
        logger.info(f"\n{category.upper()}")
        logger.info("-" * 70)

        checks = by_category[category]
        for check in sorted(checks, key=lambda x: x["check_name"]):
            status_symbol = {
                "pass": "✓",
                "warn": "⚠",
                "fail": "✗",
                "error": "!",
                "skip": "-",
            }.get(check["status"], "?")

            logger.info(f"\n  [{status_symbol}] {check['check_name']}")
            logger.info(f"      {check['message']}")

            if check.get("details"):
                logger.info(f"      Details: {json.dumps(check['details'], indent=10)[:200]}")

            if check.get("remediation"):
                logger.info(f"      Fix: {check['remediation']}")

    logger.info(f"\n{'='*70}\n")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # List checks and exit
    if args.list:
        _print_check_list()
        return 0

    # Import checks to register them
    try:
        import utils.diagnostics_checks  # noqa: F401
    except ImportError as exc:
        logger.error("Cannot import diagnostic checks: %s", exc)
        return 2

    # Run diagnostics
    try:
        report = run_diagnostics(categories=args.categories, fail_fast=args.fail_fast)

        report_dict = report.to_dict()

        if args.json:
            # Machine-readable JSON output for CI/tooling integration
            print(json.dumps(report_dict, indent=2))
        elif args.verbose:
            _print_verbose_report(report_dict)
        else:
            logger.info(report.summary())

        # Exit code based on overall status
        if report.is_healthy():
            return 0
        else:
            return 1

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 2
    except Exception as exc:
        logger.error("Error running diagnostics: %s", exc, exc_info=args.verbose)
        return 2


if __name__ == "__main__":
    sys.exit(main())
