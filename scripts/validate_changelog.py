#!/usr/bin/env python3
"""
Changelog validator for LedgerLens-data.

Validates that CHANGELOG.md is correctly formatted (Keep a Changelog standard)
and that pull requests touching high-impact paths have corresponding changelog
entries.

Usage:
    python scripts/validate_changelog.py [--check-pr] [--base-ref <ref>]

    --check-pr     In CI: fail if changed files touch data/model paths but
                   CHANGELOG.md has no new [Unreleased] content in the diff.
    --base-ref     Git ref to diff against (default: origin/main).
    --strict       Also require changelog entries for non-critical path changes.

Exit codes:
    0  All checks passed
    1  Validation failed
    2  Usage error
"""

import argparse
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass, field

# ─── Changelog format rules ───────────────────────────────────────────────────

# Keep a Changelog section headers
SECTION_PATTERN = re.compile(
    r"^## \[(?P<version>[^\]]+)\](?:\s+-\s+(?P<date>\d{4}-\d{2}-\d{2}))?$",
    re.MULTILINE,
)
SUBSECTION_PATTERN = re.compile(
    r"^### (?P<kind>Added|Changed|Deprecated|Removed|Fixed|Security)$",
    re.MULTILINE,
)
VALID_KINDS = {"Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"}

# Paths that REQUIRE a changelog entry when modified
HIGH_IMPACT_PATHS = [
    # Feature schema changes
    r"detection/feature_engineering\.py",
    r"data/feature_dictionary\.md",
    r"data/feature_ranges\.json",
    # Model artifacts and training
    r"detection/model_training\.py",
    r"detection/model_inference\.py",
    r"detection/ensemble_calibrator\.py",
    r"detection/drift_monitor\.py",
    r"models/.*",
    r"training/.*",
    r"scripts/retrain_if_drifted\.py",
    # Data ingestion schema
    r"ingestion/data_models\.py",
    r"data/trade_avro_schema\.json",
    # Shared contracts / API surface
    r"detection/persistence\.py",
    r"detection/risk_score_store\.py",
    r"integrations/contract_client\.py",
    # Privacy / security primitives
    r"detection/differential_privacy\.py",
    r"detection/shap_explainer\.py",
    r"detection/privacy/.*",
    r"utils/field_encryption\.py",
    # Config — env variable additions/removals affect downstream
    r"config\.py",
    r"\.env\.example",
]


@dataclass
class ValidationResult:
    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class ChangelogReport:
    results: list[ValidationResult] = field(default_factory=list)

    def add(self, result: ValidationResult) -> None:
        self.results.append(result)

    @property
    def errors(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[ValidationResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def passed(self) -> list[ValidationResult]:
        return [r for r in self.results if r.passed]


# ─── Validators ───────────────────────────────────────────────────────────────


def validate_structure(content: str) -> list[ValidationResult]:
    """Validate overall CHANGELOG.md structure."""
    results = []

    # Must have an [Unreleased] section
    if "[Unreleased]" not in content:
        results.append(
            ValidationResult(
                "unreleased section",
                False,
                "CHANGELOG.md must contain an ## [Unreleased] section",
            )
        )
    else:
        results.append(
            ValidationResult("unreleased section", True, "## [Unreleased] section present")
        )

    # All version sections must have an ISO date (except Unreleased)
    for match in SECTION_PATTERN.finditer(content):
        version = match.group("version")
        date = match.group("date")
        if version == "Unreleased":
            continue
        if not date:
            results.append(
                ValidationResult(
                    f"version date [{version}]",
                    False,
                    f"## [{version}] is missing a release date (expected: YYYY-MM-DD)",
                )
            )
        else:
            results.append(
                ValidationResult(
                    f"version date [{version}]",
                    True,
                    f"## [{version}] dated {date}",
                )
            )

    # Subsection types must be valid Keep a Changelog kinds
    for match in SUBSECTION_PATTERN.finditer(content):
        kind = match.group("kind")
        if kind not in VALID_KINDS:
            results.append(
                ValidationResult(
                    f"subsection kind {kind}",
                    False,
                    f"### {kind} is not a valid Keep a Changelog subsection. "
                    f"Valid: {sorted(VALID_KINDS)}",
                )
            )

    return results


def validate_entry_format(content: str) -> list[ValidationResult]:
    """Validate that changelog entries follow expected formatting conventions."""
    results = []
    lines = content.splitlines()

    entry_pattern = re.compile(r"^- .+")
    blank_entry = re.compile(r"^-\s*$")
    issues = []

    in_unreleased = False

    for i, line in enumerate(lines, 1):
        if re.match(r"## \[Unreleased\]", line):
            in_unreleased = True
            continue
        elif re.match(r"## \[", line):
            in_unreleased = False
            continue

        if in_unreleased:
            if re.match(r"### ", line):
                line.strip()
            elif blank_entry.match(line):
                issues.append(f"  line {i}: empty changelog entry '- '")
            elif line.strip().startswith("-") and not entry_pattern.match(line):
                issues.append(f"  line {i}: malformed entry '{line.strip()}'")

    if issues:
        results.append(
            ValidationResult(
                "entry format",
                False,
                "Malformed changelog entries:\n" + "\n".join(issues),
                severity="warning",
            )
        )
    else:
        results.append(
            ValidationResult("entry format", True, "All changelog entries properly formatted")
        )

    return results


def validate_unreleased_not_empty_on_pr(content: str) -> list[ValidationResult]:
    """[Unreleased] must have at least one entry — failing empty or placeholder-only."""
    match = re.search(r"## \[Unreleased\](.*?)(?=## \[|\Z)", content, re.DOTALL)
    if not match:
        return [
            ValidationResult(
                "unreleased content",
                False,
                "No [Unreleased] section to check",
            )
        ]

    section = match.group(1)
    # Count non-empty, non-header, non-blank lines
    entries = [
        ln
        for ln in section.splitlines()
        if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith("<!--")
    ]
    if not entries:
        return [
            ValidationResult(
                "unreleased content",
                False,
                "[Unreleased] section is empty. "
                "Document what changed before submitting this PR.",
            )
        ]

    return [
        ValidationResult(
            "unreleased content",
            True,
            f"[Unreleased] has {len(entries)} entries",
        )
    ]


def get_changed_files(base_ref: str) -> list[str]:
    """Return list of files changed relative to base_ref."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def classify_high_impact_changes(changed_files: list[str]) -> list[str]:
    """Return changed files that match high-impact path patterns."""
    patterns = [re.compile(p) for p in HIGH_IMPACT_PATHS]
    return [f for f in changed_files if any(p.search(f) for p in patterns)]


def changelog_was_updated(base_ref: str) -> bool:
    """Check if CHANGELOG.md itself was modified in this diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base_ref, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return "CHANGELOG.md" in result.stdout
    except subprocess.CalledProcessError:
        return False


def validate_pr_coverage(base_ref: str, strict: bool) -> list[ValidationResult]:
    """Validate that high-impact changes are accompanied by a changelog entry."""
    results = []

    changed = get_changed_files(base_ref)
    if not changed:
        return [
            ValidationResult(
                "pr coverage",
                True,
                "No changed files detected (not in a git repo or no diff against base)",
                severity="warning",
            )
        ]

    high_impact = classify_high_impact_changes(changed)
    changelog_updated = changelog_was_updated(base_ref)

    if high_impact:
        if not changelog_updated:
            results.append(
                ValidationResult(
                    "pr coverage",
                    False,
                    f"This PR touches {len(high_impact)} high-impact path(s) but "
                    f"CHANGELOG.md was not updated.\n"
                    f"  Affected paths:\n"
                    + "\n".join(f"    - {f}" for f in high_impact[:10])
                    + ("\n    ..." if len(high_impact) > 10 else "")
                    + "\n\n  Add an entry under ## [Unreleased] describing the change.",
                )
            )
        else:
            results.append(
                ValidationResult(
                    "pr coverage",
                    True,
                    f"CHANGELOG.md updated alongside {len(high_impact)} high-impact path(s)",
                )
            )
    else:
        if strict and not changelog_updated:
            results.append(
                ValidationResult(
                    "pr coverage",
                    False,
                    "No changelog update found (--strict requires changelog for all PRs)",
                    severity="warning",
                )
            )
        else:
            results.append(
                ValidationResult(
                    "pr coverage",
                    True,
                    "No high-impact paths changed — changelog update not required",
                )
            )

    return results


# ─── Model-specific validations ───────────────────────────────────────────────


def validate_model_change_mentions_metrics(
    content: str, changed_files: list[str]
) -> list[ValidationResult]:
    """
    If model training/inference files changed, changelog should mention
    metrics (AUC, F1, precision, recall) or explicitly note 'no metric change'.
    """
    model_paths = re.compile(
        r"detection/(model_training|model_inference|ensemble_calibrator|drift_monitor)\.py"
    )
    model_changed = any(model_paths.search(f) for f in changed_files)
    if not model_changed:
        return []

    unreleased_match = re.search(r"## \[Unreleased\](.*?)(?=## \[|\Z)", content, re.DOTALL)
    section = unreleased_match.group(1) if unreleased_match else ""

    metrics_keywords = re.compile(
        r"\b(AUC|F1|precision|recall|ROC|accuracy|performance|metric|score|no metric change)\b",
        re.IGNORECASE,
    )
    if not metrics_keywords.search(section):
        return [
            ValidationResult(
                "model change metrics",
                False,
                "Model code changed but no performance metrics mentioned in [Unreleased]. "
                "Include AUC/F1 or note 'no metric change'.",
                severity="warning",
            )
        ]

    return [
        ValidationResult(
            "model change metrics",
            True,
            "Changelog mentions model metrics alongside model code change",
        )
    ]


def validate_data_schema_change_documented(
    content: str, changed_files: list[str]
) -> list[ValidationResult]:
    """
    If feature_engineering.py or data_models.py changed, changelog should
    mention 'feature', 'schema', 'column', or 'field'.
    """
    schema_paths = re.compile(
        r"(detection/feature_engineering\.py|ingestion/data_models\.py"
        r"|data/trade_avro_schema\.json|data/feature_dictionary\.md)"
    )
    schema_changed = any(schema_paths.search(f) for f in changed_files)
    if not schema_changed:
        return []

    unreleased_match = re.search(r"## \[Unreleased\](.*?)(?=## \[|\Z)", content, re.DOTALL)
    section = unreleased_match.group(1) if unreleased_match else ""

    schema_keywords = re.compile(
        r"\b(feature|schema|column|field|data model|avro|parquet|struct)\b",
        re.IGNORECASE,
    )
    if not schema_keywords.search(section):
        return [
            ValidationResult(
                "data schema documentation",
                False,
                "Data/feature schema files changed but [Unreleased] doesn't mention "
                "'feature', 'schema', 'column', or 'field'. "
                "Document the schema change and any downstream impact.",
                severity="warning",
            )
        ]

    return [
        ValidationResult(
            "data schema documentation",
            True,
            "Changelog documents data schema change",
        )
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CHANGELOG.md")
    parser.add_argument(
        "--check-pr",
        action="store_true",
        help="Check that high-impact changes have a changelog entry",
    )
    parser.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to diff against for PR checks (default: origin/main)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also require changelog entries for non-high-impact changes",
    )
    args = parser.parse_args()

    changelog_path = pathlib.Path("CHANGELOG.md")
    if not changelog_path.exists():
        print("❌ CHANGELOG.md not found")
        return 1

    content = changelog_path.read_text()
    report = ChangelogReport()

    # Always run format checks
    for r in validate_structure(content):
        report.add(r)
    for r in validate_entry_format(content):
        report.add(r)
    for r in validate_unreleased_not_empty_on_pr(content):
        report.add(r)

    # PR-specific checks
    if args.check_pr:
        for r in validate_pr_coverage(args.base_ref, args.strict):
            report.add(r)

        changed = get_changed_files(args.base_ref)
        for r in validate_model_change_mentions_metrics(content, changed):
            report.add(r)
        for r in validate_data_schema_change_documented(content, changed):
            report.add(r)

    # Print results
    print("CHANGELOG.md Validation Report")
    print("=" * 60)
    for r in report.results:
        icon = "✅" if r.passed else ("❌" if r.severity == "error" else "⚠️ ")
        print(f"{icon} {r.name}")
        if not r.passed:
            for line in r.message.splitlines():
                print(f"   {line}")

    print()
    print(
        f"Results: {len(report.passed)} passed, {len(report.errors)} errors, {len(report.warnings)} warnings"
    )

    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
