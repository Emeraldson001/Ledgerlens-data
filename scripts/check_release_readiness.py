#!/usr/bin/env python3
"""
Release readiness checker for LedgerLens-data.

Validates that the repository is in a releasable state by checking:
  - CHANGELOG.md has content under [Unreleased]
  - All required documentation sections are present
  - pyproject.toml version is consistent
  - No TODOs / FIXMEs left in core modules
  - Model metadata is present and valid
  - Feature schema is documented

Usage:
    python scripts/check_release_readiness.py [--strict] [--output <path>]

Exit codes:
    0  All checks passed
    1  One or more checks failed
    2  Usage error
"""

import argparse
import json
import pathlib
import re
import sys
from datetime import UTC, datetime
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning"


def check_changelog(root: pathlib.Path) -> CheckResult:
    """CHANGELOG.md must have a non-empty [Unreleased] section."""
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        return CheckResult("CHANGELOG.md exists", False, "CHANGELOG.md not found")

    content = changelog.read_text()
    match = re.search(r"## \[Unreleased\](.*?)(?=## \[|\Z)", content, re.DOTALL)
    if not match:
        return CheckResult(
            "CHANGELOG.md unreleased section",
            False,
            "No ## [Unreleased] section found",
        )

    section = match.group(1).strip()
    if not section:
        return CheckResult(
            "CHANGELOG.md unreleased section",
            False,
            "[Unreleased] section is empty — document what changed",
        )

    return CheckResult(
        "CHANGELOG.md unreleased section",
        True,
        f"[Unreleased] has {len(section.splitlines())} lines of content",
    )


def check_version_consistency(root: pathlib.Path) -> CheckResult:
    """Version in pyproject.toml should match the version tag in config.py if present."""
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return CheckResult(
            "version consistency", False, "pyproject.toml not found", severity="warning"
        )

    pyproject_text = pyproject.read_text()
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject_text, re.MULTILINE)
    if not version_match:
        return CheckResult(
            "version consistency",
            False,
            "No version field in pyproject.toml",
            severity="warning",
        )

    version = version_match.group(1)
    return CheckResult(
        "version consistency",
        True,
        f"Version: {version}",
    )


def check_model_metadata(root: pathlib.Path) -> CheckResult:
    """models/metrics.json should exist and contain required fields."""
    metrics = root / "models" / "metrics.json"
    if not metrics.exists():
        return CheckResult(
            "model metadata",
            False,
            "models/metrics.json not found",
            severity="warning",
        )

    try:
        data = json.loads(metrics.read_text())
    except json.JSONDecodeError as exc:
        return CheckResult("model metadata", False, f"metrics.json is invalid JSON: {exc}")

    required = {"models"}
    missing = required - data.keys()
    if missing:
        return CheckResult(
            "model metadata",
            False,
            f"metrics.json missing fields: {missing}",
        )

    return CheckResult("model metadata", True, "models/metrics.json is valid")


def check_no_leftover_todos(root: pathlib.Path) -> CheckResult:
    """Core detection modules should not have TODO/FIXME/HACK markers before release."""
    core_dirs = [root / "detection", root / "ingestion", root / "streaming"]
    todo_pattern = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)
    found = []

    for d in core_dirs:
        if not d.exists():
            continue
        for py_file in d.rglob("*.py"):
            for lineno, line in enumerate(py_file.read_text().splitlines(), 1):
                if todo_pattern.search(line):
                    rel = py_file.relative_to(root)
                    found.append(f"{rel}:{lineno}: {line.strip()}")

    if found:
        return CheckResult(
            "no leftover TODOs",
            False,
            f"{len(found)} TODO/FIXME/HACK markers in core modules:\n"
            + "\n".join(f"  {f}" for f in found[:10])
            + ("\n  ..." if len(found) > 10 else ""),
            severity="warning",
        )
    return CheckResult("no leftover TODOs", True, "No TODO/FIXME markers in core modules")


def check_security_docs(root: pathlib.Path) -> CheckResult:
    """Security documentation must be present."""
    required = [
        root / "docs" / "security.md",
        root / "docs" / "security_threat_model.md",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        return CheckResult(
            "security documentation",
            False,
            f"Missing security docs: {missing}",
        )
    return CheckResult("security documentation", True, "All security docs present")


def check_contributing_guide(root: pathlib.Path) -> CheckResult:
    """CONTRIBUTING.md must exist."""
    contrib = root / "CONTRIBUTING.md"
    if not contrib.exists():
        return CheckResult("CONTRIBUTING.md", False, "CONTRIBUTING.md not found")
    return CheckResult("CONTRIBUTING.md", True, "CONTRIBUTING.md present")


def check_feature_schema_documented(root: pathlib.Path) -> CheckResult:
    """Feature dictionary documentation must exist."""
    feature_dict = root / "data" / "feature_dictionary.md"
    if not feature_dict.exists():
        return CheckResult(
            "feature schema documentation",
            False,
            "data/feature_dictionary.md not found",
            severity="warning",
        )
    return CheckResult("feature schema documentation", True, "Feature dictionary present")


def check_env_example(root: pathlib.Path) -> CheckResult:
    """.env.example must document all config variables."""
    env_example = root / ".env.example"
    if not env_example.exists():
        return CheckResult(".env.example", False, ".env.example not found")

    lines = env_example.read_text().splitlines()
    key_count = sum(1 for line in lines if line.strip() and not line.startswith("#"))
    if key_count < 5:
        return CheckResult(
            ".env.example",
            False,
            f".env.example has only {key_count} variables — may be incomplete",
            severity="warning",
        )
    return CheckResult(".env.example", True, f".env.example documents {key_count} variables")


def run_checks(root: pathlib.Path, strict: bool) -> list[CheckResult]:
    checkers = [
        check_changelog,
        check_version_consistency,
        check_model_metadata,
        check_no_leftover_todos,
        check_security_docs,
        check_contributing_guide,
        check_feature_schema_documented,
        check_env_example,
    ]
    return [checker(root) for checker in checkers]


def render_report(results: list[CheckResult], strict: bool) -> tuple[str, int]:
    """Returns (report_text, exit_code)."""
    passed = [r for r in results if r.passed]
    failed_errors = [r for r in results if not r.passed and r.severity == "error"]
    failed_warnings = [r for r in results if not r.passed and r.severity == "warning"]

    lines = [
        "# LedgerLens Release Readiness Report",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Summary",
        f"- ✅ Passed:  {len(passed)}",
        f"- ❌ Errors:  {len(failed_errors)}",
        f"- ⚠️  Warnings: {len(failed_warnings)}",
        "",
        "## Results",
    ]

    for r in results:
        icon = "✅" if r.passed else ("❌" if r.severity == "error" else "⚠️")
        lines.append(f"\n### {icon} {r.name}")
        lines.append(r.message)

    exit_code = 0
    if failed_errors:
        exit_code = 1
    elif failed_warnings and strict:
        exit_code = 1

    if exit_code == 0:
        lines += ["", "---", "**Repository is READY for release.** 🚀"]
    else:
        lines += ["", "---", "**Repository is NOT ready for release.** Fix the above issues first."]

    return "\n".join(lines), exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Check release readiness")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Write report to this path",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root (default: cwd)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    results = run_checks(root, args.strict)
    report, exit_code = render_report(results, args.strict)

    print(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"\nReport written to {args.output}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
