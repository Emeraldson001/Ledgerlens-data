"""CLI entry point: first-time local setup onboarding checks.

Usage::

    python -m scripts.onboard               # run all checks (human-readable)
    python -m scripts.onboard --fix         # attempt auto-fixes
    python -m scripts.onboard --json        # machine-readable JSON output
    python -m scripts.onboard --strict      # exit 1 on WARNING, not just ERROR
    python -m scripts.onboard --all-pkgs    # also check optional package imports
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.onboarding_checks import CheckLevel, run_all_checks
from utils.logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LedgerLens first-time local setup onboarding checks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt auto-fixes (copy .env.example → .env, create missing directories)",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Output machine-readable JSON",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 on any WARNING (not just ERROR)",
    )
    parser.add_argument(
        "--all-pkgs",
        action="store_true",
        help="Also check optional package imports (torch, shap, …)",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Override the repository root directory",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root) if args.repo_root else None

    report = run_all_checks(
        repo_root=repo_root,
        fix=args.fix,
        include_optional_packages=args.all_pkgs,
    )

    if args.json_output:
        # Machine-readable JSON output for CI/tooling integration
        print(json.dumps(report.to_dict(), indent=2))
    else:
        sep = "─" * 62
        logger.info(f"\n{sep}")
        logger.info(" LedgerLens Onboarding Checks")
        logger.info(sep)
        for result in report.results:
            logger.info(result)
        logger.info(sep)
        logger.info(f" Summary: {report.summary()}")
        logger.info(sep + "\n")

        if report.has_errors or report.has_warnings:
            logger.info("Suggested next steps:")
            hints_shown: set[str] = set()
            for r in report.results:
                if r.level != CheckLevel.OK and r.fix_hint and r.fix_hint not in hints_shown:
                    hints_shown.add(r.fix_hint)
                    logger.info(f"  → {r.fix_hint}")

    # Exit code
    if report.has_errors:
        return 1
    if args.strict and report.has_warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
