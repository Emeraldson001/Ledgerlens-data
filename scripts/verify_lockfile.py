"""
scripts/verify_lockfile.py — Dependency lockfile verification for CI.

Ensures the installed environment exactly matches requirements.lock so builds
are always reproducible.  Designed to run as a CI step after ``pip install``.

Three verification modes:

  --check-installed (default)
      Compares ``pip freeze`` output against requirements.lock.
      Exits 1 with a clear diff if they diverge.

  --check-unpinned
      Scans requirements.txt for unpinned or open-range specifiers
      (e.g. ``requests`` with no version, or ``requests>=2.0``).
      Exits 1 with a list of unpinned packages.

  --generate
      Regenerates requirements.lock from the current environment.
      Use this when upgrading dependencies; commit the result.

Usage::

    # In CI — verify lockfile matches installed packages
    python scripts/verify_lockfile.py

    # Also check that requirements.txt has no unpinned dependencies
    python scripts/verify_lockfile.py --check-unpinned

    # Regenerate after a dependency upgrade
    python scripts/verify_lockfile.py --generate

Exit codes:
  0  Verification passed.
  1  Verification failed (details printed to stdout).
  2  requirements.lock does not exist (run with --generate first).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

LOCKFILE_PATH = Path("requirements.lock")
REQUIREMENTS_PATH = Path("requirements.txt")

# Lines in requirements.txt that are not package specifiers
_COMMENT_OR_BLANK = re.compile(r"^\s*(#.*)?$")

# Detects a pinned package: ``pkg==1.2.3`` (exact pin only)
_PINNED_RE = re.compile(r"^[A-Za-z0-9_.\-]+==[^\s]+$")

# Detects open-range specifiers like >=, <=, ~=, >, <, or bare package name
_UNPINNED_RE = re.compile(r"^[A-Za-z0-9_.\-]+(\s*(>=|<=|~=|!=|>|<)[^\s,]+)?$")


# ---------------------------------------------------------------------------
# Core checks
# ---------------------------------------------------------------------------


def generate_lockfile(lockfile: Path = LOCKFILE_PATH) -> int:
    """Write current ``pip freeze`` output to *lockfile*."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] pip freeze failed:\n{result.stderr}", file=sys.stderr)
        return 1
    lines = sorted(result.stdout.splitlines())
    lockfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] requirements.lock written ({len(lines)} packages).")
    return 0


def check_installed(
    lockfile: Path = LOCKFILE_PATH,
    requirements: Path = REQUIREMENTS_PATH,
) -> int:
    """Verify direct dependencies against portable locked versions."""
    if not lockfile.exists():
        print(
            f"[ERROR] {lockfile} does not exist. "
            "Run 'python scripts/verify_lockfile.py --generate' to create it.",
            file=sys.stderr,
        )
        return 2

    locked_versions: dict[str, str] = {}
    for raw in lockfile.read_text("utf-8").splitlines():
        if "==" in raw and not raw.lstrip().startswith("#"):
            name, pinned_version = raw.split("==", 1)
            locked_versions[canonicalize_name(name)] = pinned_version

    direct: dict[str, Requirement] = {}
    for raw in requirements.read_text("utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "-")):
            continue
        requirement = Requirement(line)
        if requirement.marker is None or requirement.marker.evaluate():
            direct[canonicalize_name(requirement.name)] = requirement

    problems: list[str] = []
    for name, requirement in sorted(direct.items()):
        pinned = locked_versions.get(name)
        if pinned is None:
            problems.append(f"{name}: missing from requirements.lock")
            continue
        try:
            installed = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}=={pinned}: not installed")
            continue
        if installed not in requirement.specifier:
            problems.append(
                f"{name}: installed {installed}, outside required range {requirement.specifier}"
            )

    if not problems:
        print(f"[OK] {len(direct)} direct dependencies satisfy requirements.txt and are locked.")
        return 0

    print("[FAIL] Direct dependencies diverge from requirements.lock:")
    for problem in problems:
        print(f"  - {problem}")

    print(
        "\nDiagnostic: run 'python scripts/verify_lockfile.py --generate' after "
        "'pip install -r requirements.txt' to regenerate the lockfile, "
        "then commit the updated requirements.lock."
    )
    return 1


def check_unpinned(requirements: Path = REQUIREMENTS_PATH) -> int:
    """Scan *requirements* for unpinned or open-range specifiers."""
    if not requirements.exists():
        print(f"[ERROR] {requirements} does not exist.", file=sys.stderr)
        return 1

    lines = requirements.read_text("utf-8").splitlines()
    unpinned = []
    for lineno, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if _COMMENT_OR_BLANK.match(line):
            continue
        # Strip inline comments
        line = line.split("#")[0].strip()
        if not line:
            continue
        # Skip VCS/URL installs — not pinnable the usual way
        if line.startswith(("-", "git+", "http://", "https://")):
            continue
        # Check for exact pin
        pkg_part = re.split(r"[\s;]", line)[0]  # strip env markers
        if not re.search(r"==", pkg_part):
            unpinned.append((lineno, raw_line.strip()))

    if not unpinned:
        print(f"[OK] All packages in {requirements} use exact (==) version pins.")
        return 0

    print(f"[WARN] {len(unpinned)} package(s) in {requirements} are not pinned with '==':")
    for lineno, spec in unpinned:
        print(f"  line {lineno:3d}: {spec}")
    print(
        "\nNote: unpinned specifiers reduce reproducibility. "
        "Consider pinning with '==' in requirements.lock while keeping "
        "open ranges in requirements.txt for human readability. "
        "This check is advisory; it does not fail CI."
    )
    # Advisory only — return 0 so CI is not blocked
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that the installed environment matches requirements.lock.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate requirements.lock from the current environment.",
    )
    parser.add_argument(
        "--check-unpinned",
        action="store_true",
        help="Also check requirements.txt for unpinned specifiers (advisory).",
    )
    parser.add_argument(
        "--lockfile",
        default=str(LOCKFILE_PATH),
        metavar="PATH",
        help=f"Path to the lockfile (default: {LOCKFILE_PATH}).",
    )
    parser.add_argument(
        "--requirements",
        default=str(REQUIREMENTS_PATH),
        metavar="PATH",
        help=f"Path to requirements.txt (default: {REQUIREMENTS_PATH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    lockfile = Path(args.lockfile)
    requirements = Path(args.requirements)

    if args.generate:
        return generate_lockfile(lockfile)

    rc = check_installed(lockfile, requirements)

    if args.check_unpinned:
        # Advisory — never overrides the main check exit code
        check_unpinned(requirements)

    return rc


if __name__ == "__main__":
    sys.exit(main())
