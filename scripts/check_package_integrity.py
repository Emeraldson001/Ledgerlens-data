#!/usr/bin/env python3
"""CLI entry point for source package integrity checks (Issue #540).

Runs `utils.package_integrity.check_source_package_integrity` and prints a
human-readable report. Exits non-zero on any finding so it can gate CI
independently of pytest (the same check also runs automatically at the
start of every `pytest` session via `tests/conftest.py`).

Usage:
    python scripts/check_package_integrity.py
    python scripts/check_package_integrity.py --packages detection,streaming

Exit codes:
    0  No integrity issues found.
    1  One or more integrity issues found — see printed report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.package_integrity import (  # noqa: E402
    DEFAULT_SOURCE_PACKAGES,
    check_source_package_integrity,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--packages",
        default=",".join(DEFAULT_SOURCE_PACKAGES),
        help="Comma-separated list of top-level source packages to check.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root to check packages under (default: current directory).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    packages = tuple(p.strip() for p in args.packages.split(",") if p.strip())
    report = check_source_package_integrity(root=args.root, packages=packages)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
