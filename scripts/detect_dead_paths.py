#!/usr/bin/env python3
"""CLI entry point for dead-path detection reports (Issue #547).

Sweeps the configured source packages for modules with no inbound Python
import, no `__main__` entry-point guard, and no reference from
Makefile/CI/docs, and reports them as candidates for retirement. Read-only —
never deletes or modifies source files.

Usage:
    python scripts/detect_dead_paths.py
    python scripts/detect_dead_paths.py --format markdown --output reports/dead_paths.md
    python scripts/detect_dead_paths.py --format json --output reports/dead_paths.json
    python scripts/detect_dead_paths.py --strict   # exit 1 if any candidates found

Exit codes:
    0  Report generated successfully (candidates may still be listed).
    1  --strict was passed and at least one candidate was found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.dead_path_detector import detect_dead_paths  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--format",
        choices=("text", "markdown", "json"),
        default="text",
        help="Output format (default: text).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the report to this path instead of stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any dead-path candidates are found.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = detect_dead_paths(root=args.root)

    if args.format == "markdown":
        rendered = report.render_markdown()
    elif args.format == "json":
        rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    else:
        rendered = report.render()

    if args.output:
        Path(args.output).write_text(rendered)
        print(f"Wrote {args.format} report to {args.output}")
    else:
        print(rendered)

    if args.strict and report.candidates:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
