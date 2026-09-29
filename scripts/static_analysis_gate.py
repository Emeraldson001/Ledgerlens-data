"""
scripts/static_analysis_gate.py — Repository-wide static analysis gate.

Runs three checks in sequence and exits non-zero if any gate fails:

  1. mypy  — strict type checking on core modules (detection/, ingestion/,
             streaming/, ci_metrics/, benchmarks/)
  2. bandit — security linting (SAST) on the same modules; fails on HIGH
              severity findings
  3. radon cc — cyclomatic complexity; fails if any function exceeds the
                configured threshold (default: grade C / complexity > 10)

Each check prints a clear summary of findings so engineers know exactly what
to fix and where.  All three checks run even if earlier ones fail, so you see
the full picture on every invocation.

Usage::

    python scripts/static_analysis_gate.py           # default thresholds
    python scripts/static_analysis_gate.py --complexity-max 8
    python scripts/static_analysis_gate.py --skip-bandit  # CI without bandit

Exit codes:
  0  All gates passed.
  1  One or more gates failed (details printed to stdout/stderr).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Modules analysed by every gate
ANALYSIS_TARGETS: list[str] = [
    "detection",
    "ingestion",
    "streaming",
    "ci_metrics",
    "benchmarks",
    "utils",
    "config.py",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, label: str) -> tuple[int, str, str]:
    """Run *cmd*, capture output, and return (returncode, stdout, stderr)."""
    print(f"\n{'=' * 60}")
    print(f"[GATE] {label}")
    print(f"  cmd: {' '.join(cmd)}")
    print("=" * 60)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {label}\n")
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def run_mypy(targets: list[str]) -> int:
    """Run mypy with project settings from pyproject.toml."""
    existing = [t for t in targets if Path(t).exists()]
    if not existing:
        print("[SKIP] mypy: no target paths found")
        return 0
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        "pyproject.toml",
        "--no-error-summary",
        "--show-column-numbers",
        "--pretty",
        *existing,
    ]
    rc, _, _ = _run(cmd, label="mypy — type checking")
    return rc


def run_bandit(targets: list[str], severity: str = "HIGH") -> int:
    """Run bandit security linter; fail only on HIGH severity by default."""
    existing = [t for t in targets if Path(t).exists()]
    if not existing:
        print("[SKIP] bandit: no target paths found")
        return 0
    # Check bandit is available
    check = subprocess.run(
        [sys.executable, "-m", "bandit", "--version"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print(
            "[WARN] bandit is not installed; skipping security gate. "
            "Add bandit>=1.7.0 to requirements.txt to enable this check.",
            file=sys.stderr,
        )
        return 0  # non-blocking if bandit is absent — warn but don't fail

    cmd = [
        sys.executable,
        "-m",
        "bandit",
        "-r",
        *existing,
        "--severity-level",
        severity.lower(),
        "--confidence-level",
        "medium",
        "-f",
        "screen",
        "--quiet",
    ]
    rc, _, _ = _run(cmd, label=f"bandit — security linting (≥{severity} severity)")
    return rc


def run_complexity(targets: list[str], max_complexity: int = 10) -> int:
    """Run radon cc; fail if any function exceeds *max_complexity*."""
    existing = [t for t in targets if Path(t).exists()]
    if not existing:
        print("[SKIP] radon: no target paths found")
        return 0
    # Check radon is available
    check = subprocess.run(
        [sys.executable, "-m", "radon", "--version"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        print(
            "[WARN] radon is not installed; skipping complexity gate. "
            "Add radon>=6.0.0 to requirements.txt to enable this check.",
            file=sys.stderr,
        )
        return 0  # non-blocking if radon is absent

    # radon cc -n C exits 0 even with violations; we parse output manually
    cmd = [
        sys.executable,
        "-m",
        "radon",
        "cc",
        *existing,
        "--min",
        "C",  # C = complexity 5–10, D/E/F = higher
        "--average",
        "--show-complexity",
        "--no-assert",
    ]
    rc, stdout, _ = _run(cmd, label=f"radon — cyclomatic complexity (max={max_complexity})")

    # Parse violations: lines like "    F 42:0 some_function - D (15)"
    violations = []
    for line in stdout.splitlines():
        # Extract complexity score from the trailing "(N)"
        import re

        m = re.search(r"\((\d+)\)\s*$", line.strip())
        if m and int(m.group(1)) > max_complexity:
            violations.append(line.strip())

    if violations:
        print(
            f"\n[FAIL] {len(violations)} function(s) exceed complexity threshold "
            f"({max_complexity}):"
        )
        for v in violations:
            print(f"  {v}")
        print(
            "\nHint: refactor long functions into smaller helpers. "
            "Use 'radon cc <file> --show-complexity' for per-file details."
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repository-wide static analysis gate (mypy + bandit + radon).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=ANALYSIS_TARGETS,
        metavar="PATH",
        help="Paths to analyse (default: detection ingestion streaming ci_metrics benchmarks utils config.py)",
    )
    parser.add_argument(
        "--skip-mypy",
        action="store_true",
        help="Skip the mypy type-checking gate.",
    )
    parser.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Skip the bandit security gate.",
    )
    parser.add_argument(
        "--skip-complexity",
        action="store_true",
        help="Skip the radon complexity gate.",
    )
    parser.add_argument(
        "--bandit-severity",
        default="HIGH",
        choices=["LOW", "MEDIUM", "HIGH"],
        help="Minimum bandit severity that fails the gate (default: HIGH).",
    )
    parser.add_argument(
        "--complexity-max",
        type=int,
        default=10,
        metavar="N",
        help="Maximum allowed cyclomatic complexity per function (default: 10).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    targets = args.targets

    exit_codes: list[int] = []

    if not args.skip_mypy:
        exit_codes.append(run_mypy(targets))

    if not args.skip_bandit:
        exit_codes.append(run_bandit(targets, severity=args.bandit_severity))

    if not args.skip_complexity:
        exit_codes.append(run_complexity(targets, max_complexity=args.complexity_max))

    passed = all(c == 0 for c in exit_codes)
    print("\n" + "=" * 60)
    print(f"Static analysis gate: {'ALL PASSED ✓' if passed else 'FAILED ✗'}")
    print("=" * 60)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
