#!/usr/bin/env python3
"""Build a dependency risk report for LedgerLens core packages.

The default mode is deterministic and offline: it reads direct requirements from
``requirements.txt`` and pinned versions from ``requirements.lock``.  Use
``--osv`` to enrich the report with OSV.dev advisories; advisory lookup failures
are reported in the output and do not make the command fail.

Usage:
    python scripts/dependency_risk_report.py
    python scripts/dependency_risk_report.py --format markdown
    python scripts/dependency_risk_report.py --osv --output reports/dependency_risk.json

Exit codes:
    0  Report generated; no high/critical risks found.
    1  Report generated with high/critical risks.
    2  Input or execution error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REQUIREMENTS = REPO_ROOT / "requirements.txt"
DEFAULT_LOCKFILE = REPO_ROOT / "requirements.lock"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "dependency_risk.json"

# These packages are on the production scoring/ingestion path. Other direct
# dependencies remain visible when explicitly selected, but are not core risk.
CORE_PACKAGES = {
    "numpy",
    "pandas",
    "scikit-learn",
    "sqlalchemy",
    "pydantic",
    "stellar-sdk",
    "cryptography",
    "requests",
    "fastapi",
    "torch",
    "xgboost",
    "lightgbm",
}

RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class Advisory:
    identifier: str
    summary: str
    severity: list[str] = field(default_factory=list)
    url: str = ""


@dataclass
class PackageRisk:
    name: str
    required: str
    locked_version: str | None
    core: bool
    risk: str
    reasons: list[str] = field(default_factory=list)
    advisories: list[Advisory] = field(default_factory=list)


@dataclass
class DependencyRiskReport:
    generated_at: str
    requirements_file: str
    lockfile: str
    core_packages: list[str]
    osv_requested: bool
    advisory_errors: list[str]
    packages: list[PackageRisk]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def high_risk_count(self) -> int:
        return sum(RISK_ORDER[p.risk] >= RISK_ORDER["high"] for p in self.packages)

    def markdown(self) -> str:
        lines = [
            "# Dependency Risk Report",
            "",
            f"Generated: `{self.generated_at}`  ",
            f"Requirements: `{self.requirements_file}`  ",
            f"Lockfile: `{self.lockfile}`",
            "",
            "| Package | Required | Locked | Core | Risk | Reasons | Advisories |",
            "|---|---|---|---|---|---|---|",
        ]
        for package in self.packages:
            advisories = ", ".join(a.identifier for a in package.advisories) or "—"
            reasons = "; ".join(package.reasons) or "—"
            lines.append(
                f"| `{package.name}` | `{package.required}` | "
                f"`{package.locked_version or 'missing'}` | "
                f"{'yes' if package.core else 'no'} | **{package.risk}** | "
                f"{reasons} | {advisories} |"
            )
        lines.extend(["", f"High/critical risks: **{self.high_risk_count}**"])
        if self.advisory_errors:
            lines.extend(["", "## Advisory lookup warnings", ""])
            lines.extend(f"- {error}" for error in self.advisory_errors)
        lines.append("")
        return "\n".join(lines)


def _parse_requirements(path: Path) -> dict[str, Requirement]:
    requirements: dict[str, Requirement] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(("-", "git+", "http://", "https://")):
            continue
        line = line.split("#", 1)[0].strip()
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            requirements[canonicalize_name(requirement.name)] = requirement
    return requirements


def _parse_lockfile(path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    if not path.exists():
        return locked
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "==" not in raw or raw.lstrip().startswith("#"):
            continue
        name, version = raw.split("==", 1)
        locked[canonicalize_name(name)] = version
    return locked


def _query_osv(name: str, version: str, timeout: float = 5.0) -> list[Advisory]:
    payload = json.dumps(
        {"package": {"name": name, "ecosystem": "PyPI"}, "version": version}
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.osv.dev/v1/query",
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    advisories = []
    for item in data.get("vulns", []):
        severity = [entry.get("type", "") for entry in item.get("severity", [])]
        advisories.append(
            Advisory(
                identifier=item.get("id", "unknown"),
                summary=item.get("summary", "No summary provided"),
                severity=[value for value in severity if value],
                url=item.get("database_specific", {}).get("url", "")
                or f"https://osv.dev/vulnerability/{item.get('id', '')}",
            )
        )
    return advisories


def _risk_for(
    name: str,
    requirement: Requirement,
    locked_version: str | None,
    advisories: list[Advisory],
) -> PackageRisk:
    normalized = canonicalize_name(name)
    core = normalized in {canonicalize_name(p) for p in CORE_PACKAGES}
    reasons: list[str] = []
    if locked_version is None:
        reasons.append("missing from requirements.lock")
    elif not requirement.specifier.contains(locked_version, prereleases=True):
        reasons.append(f"locked version {locked_version} is outside {requirement.specifier}")
    if advisories:
        reasons.append(f"{len(advisories)} OSV advisory(ies)")

    if any(
        "CRITICAL" in advisory_severity.upper()
        for advisory in advisories
        for advisory_severity in advisory.severity
    ):
        risk = "critical"
    elif advisories or (core and locked_version is None):
        risk = "high"
    elif core or locked_version is None:
        risk = "medium"
    else:
        risk = "low"
    return PackageRisk(
        name=requirement.name,
        required=str(requirement.specifier) or "any",
        locked_version=locked_version,
        core=core,
        risk=risk,
        reasons=reasons,
        advisories=advisories,
    )


def _display_path(path: Path) -> str:
    """Return a stable repository-relative path when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_report(
    requirements_path: Path = DEFAULT_REQUIREMENTS,
    lockfile_path: Path = DEFAULT_LOCKFILE,
    *,
    core_only: bool = True,
    osv: bool = False,
) -> DependencyRiskReport:
    requirements = _parse_requirements(requirements_path)
    locked = _parse_lockfile(lockfile_path)
    advisory_errors: list[str] = []
    packages: list[PackageRisk] = []

    selected = {
        name: requirement
        for name, requirement in requirements.items()
        if not core_only or name in {canonicalize_name(p) for p in CORE_PACKAGES}
    }
    for name, requirement in sorted(selected.items()):
        version = locked.get(name)
        advisories: list[Advisory] = []
        if osv and version:
            try:
                advisories = _query_osv(name, version)
            except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                advisory_errors.append(f"{name}: OSV lookup failed ({exc})")
        packages.append(_risk_for(name, requirement, version, advisories))

    return DependencyRiskReport(
        generated_at=datetime.now(UTC).isoformat(),
        requirements_file=_display_path(requirements_path),
        lockfile=_display_path(lockfile_path),
        core_packages=sorted(CORE_PACKAGES),
        osv_requested=osv,
        advisory_errors=advisory_errors,
        packages=packages,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, default=DEFAULT_REQUIREMENTS)
    parser.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument(
        "--all", action="store_true", help="Report all direct packages, not only core packages"
    )
    parser.add_argument("--osv", action="store_true", help="Query OSV.dev for known advisories")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        report = build_report(
            args.requirements,
            args.lockfile,
            core_only=not args.all,
            osv=args.osv,
        )
        content = (
            report.markdown()
            if args.format == "markdown"
            else json.dumps(report.to_dict(), indent=2) + "\n"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        print(f"Dependency risk report written to {args.output}")
        return 1 if report.high_risk_count else 0
    except (OSError, ValueError) as exc:
        print(f"Dependency risk report failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
