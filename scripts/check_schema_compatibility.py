#!/usr/bin/env python3
"""Enforce Avro schema compatibility against the pull request's merge base.

Compares `data/trade_avro_schema.json` as it exists in the working tree against
the same file as it exists in the baseline ref (the branch the PR targets), and
fails when the change would break messages already in the Kafka topic.

The baseline is read from git rather than from a committed `.prev` file: a
baseline that the change under test can edit enforces nothing, because the
author would update both in the same commit.

Compatibility rules are implemented in `ingestion/avro_codec.py`
(`check_backward_compatibility` / `check_forward_compatibility`) and documented
in `data/schema_evolution.md`.

Usage:
    python scripts/check_schema_compatibility.py
    python scripts/check_schema_compatibility.py --baseline-ref origin/main
    python scripts/check_schema_compatibility.py --dry-run

Exit codes:
    0  Schema is compatible, unchanged, or newly added (no baseline).
    1  Schema change is incompatible — violations are printed.
    2  Baseline ref could not be resolved (misconfiguration, not a schema
       break) — kept distinct so CI infrastructure failures are not misread.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# Allow running as `python scripts/check_schema_compatibility.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.avro_codec import (  # noqa: E402
    check_backward_compatibility,
    check_forward_compatibility,
)

DEFAULT_SCHEMA_PATH = "data/trade_avro_schema.json"
DEFAULT_BASELINE_REF = "origin/main"

EXIT_OK = 0
EXIT_INCOMPATIBLE = 1
EXIT_BASELINE_UNRESOLVED = 2


class BaselineUnavailable(Exception):
    """The baseline ref exists but does not contain the schema file."""


class BaselineRefError(Exception):
    """The baseline ref itself could not be resolved."""


def default_baseline_ref() -> str:
    """Resolve the ref to compare against.

    In GitHub Actions on a pull request, ``GITHUB_BASE_REF`` names the target
    branch (e.g. ``main``); the workflow fetches it as ``origin/<branch>``.
    Falls back to ``origin/main`` for local runs.
    """
    base = os.environ.get("GITHUB_BASE_REF", "").strip()
    return f"origin/{base}" if base else DEFAULT_BASELINE_REF


def read_baseline_from_git(ref: str, schema_path: str) -> dict:
    """Return the schema as it exists at *ref*.

    Raises:
        BaselineUnavailable: The ref resolves but has no such file (new schema).
        BaselineRefError: The ref itself cannot be resolved.
    """
    try:
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise BaselineRefError(f"cannot resolve baseline ref {ref!r}") from exc

    result = subprocess.run(
        ["git", "show", f"{ref}:{schema_path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BaselineUnavailable(f"{schema_path} does not exist at {ref}")
    return json.loads(result.stdout)


def evaluate(baseline: dict, current: dict, mode: str = "both") -> tuple[bool, list[str]]:
    """Return ``(is_compatible, violations)`` for *baseline* -> *current*."""
    violations: list[str] = []

    if mode in ("both", "backward"):
        ok, errs = check_backward_compatibility(baseline, current)
        if not ok:
            violations.extend(f"[backward] {e}" for e in errs)

    if mode in ("both", "forward"):
        ok, errs = check_forward_compatibility(baseline, current)
        if not ok:
            violations.extend(f"[forward] {e}" for e in errs)

    return not violations, violations


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce Avro schema compatibility against the merge base.",
    )
    parser.add_argument(
        "--schema-path",
        default=DEFAULT_SCHEMA_PATH,
        help=f"Schema file to check (default: {DEFAULT_SCHEMA_PATH}).",
    )
    parser.add_argument(
        "--baseline-ref",
        default=None,
        help="Git ref to compare against (default: GITHUB_BASE_REF, else origin/main).",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "backward", "forward"),
        default="both",
        help="Which compatibility direction(s) to enforce (default: both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the decision but always exit 0.",
    )
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    baseline_reader: Callable[[str, str], dict] = read_baseline_from_git,
) -> int:
    """Entry point. *baseline_reader* is injectable so tests need no git."""
    args = _parse_args(argv)
    ref = args.baseline_ref or default_baseline_ref()

    current_path = Path(args.schema_path)
    if not current_path.is_file():
        print(f"Schema file not found: {args.schema_path}")
        return EXIT_OK if args.dry_run else EXIT_BASELINE_UNRESOLVED

    current = json.loads(current_path.read_text(encoding="utf-8"))

    try:
        baseline = baseline_reader(ref, args.schema_path)
    except BaselineUnavailable:
        print(f"No baseline for {args.schema_path} at {ref} — treating as a new schema.")
        return EXIT_OK
    except BaselineRefError as exc:
        print(f"Cannot resolve baseline: {exc}")
        print(
            "Hint: the workflow must check out with fetch-depth: 0 and fetch the "
            "base branch before running this check."
        )
        return EXIT_OK if args.dry_run else EXIT_BASELINE_UNRESOLVED

    if baseline == current:
        print(f"{args.schema_path} is unchanged against {ref}.")
        return EXIT_OK

    compatible, violations = evaluate(baseline, current, args.mode)

    if compatible:
        print(f"{args.schema_path} changed against {ref} and is compatible ({args.mode}).")
        return EXIT_OK

    print(f"Schema incompatibility against {ref} ({len(violations)} violation(s)):")
    for v in violations:
        print(f"  {v}")
    print()
    print("See data/schema_evolution.md for the migration procedure.")

    if args.dry_run:
        print("(dry run — not failing)")
        return EXIT_OK
    return EXIT_INCOMPATIBLE


if __name__ == "__main__":
    raise SystemExit(main())
