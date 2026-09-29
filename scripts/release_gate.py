"""Release blocking gates for critical test failures (Issue #559).

Overview
--------
Not all test failures are equal.  A broken test for an experimental feature
should produce a warning; a broken test for the core Benford scoring path
or the risk-score persistence layer should *block the release*.

This module implements a three-layer gate system:

1. **Criticality taxonomy** (``data/test_criticality.json``) — a JSON file
   that maps pytest *node IDs* (or glob patterns) to one of three tiers:

   * ``CRITICAL`` — blocks release; must be all-green.
   * ``HIGH`` — blocks release by default but can be overridden with
     ``--allow-high-failures`` (useful for non-production branches).
   * ``LOW`` — informational only; never blocks.

2. ``ReleaseGateEvaluator`` — parses a JUnit XML report (produced by
   ``pytest --junitxml=reports/junit.xml``) and applies the taxonomy.
   Returns a structured ``GateResult`` with human-readable diagnostics.

3. **CLI entry point** — ``python -m scripts.release_gate`` exits with code
   1 when the gate fails (intended for CI ``run:`` steps).

CI integration
--------------
See ``ci.yml`` — a new ``release-gate`` job runs after the main ``test``
job, parses the JUnit XML, and fails the pipeline if any CRITICAL test is
red.

Taxonomy format (``data/test_criticality.json``)
-------------------------------------------------
::

    {
      "version": 1,
      "entries": [
        {
          "pattern": "tests/test_benford.py",
          "criticality": "CRITICAL",
          "reason": "Core Benford scoring path — release blocker"
        },
        {
          "pattern": "tests/test_persistence.py::*",
          "criticality": "CRITICAL",
          "reason": "DB persistence layer"
        },
        {
          "pattern": "tests/test_adversarial.py",
          "criticality": "HIGH",
          "reason": "Model robustness"
        }
      ]
    }

Pattern matching
----------------
Patterns are matched against the pytest node ID
(``tests/test_benford.py::test_leading_digits_basic``) using fnmatch glob
rules.  A ``tests/test_benford.py`` pattern (no ``::`` suffix) matches all
tests in that file.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default taxonomy path — relative to the repo root
# ---------------------------------------------------------------------------

DEFAULT_TAXONOMY_PATH = Path(__file__).parent.parent / "data" / "test_criticality.json"
DEFAULT_JUNIT_PATH = Path("reports") / "junit.xml"

VALID_CRITICALITY_LEVELS = {"CRITICAL", "HIGH", "LOW"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TaxonomyEntry:
    pattern: str
    criticality: str  # "CRITICAL" | "HIGH" | "LOW"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.criticality not in VALID_CRITICALITY_LEVELS:
            raise ValueError(
                f"Invalid criticality {self.criticality!r} for pattern {self.pattern!r}. "
                f"Must be one of {VALID_CRITICALITY_LEVELS}."
            )


@dataclass
class TestResult:
    node_id: str
    classname: str
    name: str
    outcome: str  # "passed" | "failed" | "error" | "skipped"
    message: str = ""


@dataclass
class CriticalFailure:
    node_id: str
    criticality: str
    reason: str
    message: str


@dataclass
class GateResult:
    passed: bool
    critical_failures: list[CriticalFailure] = field(default_factory=list)
    high_failures: list[CriticalFailure] = field(default_factory=list)
    evaluated_tests: int = 0
    total_failures: int = 0
    blocked_by: str = ""

    def summary(self) -> str:
        lines = [
            f"Release gate: {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  Evaluated tests : {self.evaluated_tests}",
            f"  Total failures  : {self.total_failures}",
            f"  Critical failures: {len(self.critical_failures)}",
            f"  High failures   : {len(self.high_failures)}",
        ]
        if self.critical_failures:
            lines.append("\n  CRITICAL test failures (release BLOCKED):")
            for f in self.critical_failures:
                lines.append(f"    [{f.criticality}] {f.node_id}")
                if f.reason:
                    lines.append(f"      reason: {f.reason}")
                if f.message:
                    lines.append(f"      failure: {f.message[:200]}")
        if self.high_failures:
            lines.append("\n  HIGH test failures (release blocked unless --allow-high-failures):")
            for f in self.high_failures:
                lines.append(f"    [{f.criticality}] {f.node_id}")
        if self.blocked_by:
            lines.append(f"\n  Blocked by: {self.blocked_by}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Taxonomy loader
# ---------------------------------------------------------------------------


class TestCriticalityTaxonomy:
    """Loads and queries the criticality taxonomy.

    Parameters
    ----------
    path:
        Path to the ``test_criticality.json`` file.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        effective_path = Path(path) if path else DEFAULT_TAXONOMY_PATH
        self._entries: list[TaxonomyEntry] = []
        self._load(effective_path)

    def _load(self, path: Path) -> None:
        if not path.exists():
            logger.warning("Criticality taxonomy not found at %s — all tests default to LOW", path)
            return
        with open(path) as fh:
            data = json.load(fh)
        for raw in data.get("entries", []):
            try:
                self._entries.append(
                    TaxonomyEntry(
                        pattern=raw["pattern"],
                        criticality=raw.get("criticality", "LOW"),
                        reason=raw.get("reason", ""),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping invalid taxonomy entry %r: %s", raw, exc)

    def criticality_for(self, node_id: str) -> str:
        """Return the highest criticality level that matches *node_id*.

        Matching priority: CRITICAL > HIGH > LOW.  If no pattern matches,
        returns ``"LOW"``.
        """
        matched: str = "LOW"
        priority = {"LOW": 0, "HIGH": 1, "CRITICAL": 2}

        for entry in self._entries:
            pattern = entry.pattern
            # If pattern has no "::" and node_id does, match the file part only.
            if "::" not in pattern:
                file_part = node_id.split("::")[0]
                matches = fnmatch.fnmatch(file_part, pattern)
            else:
                matches = fnmatch.fnmatch(node_id, pattern)
            if matches and priority[entry.criticality] > priority[matched]:
                matched = entry.criticality
        return matched

    def reason_for(self, node_id: str) -> str:
        """Return the reason string for the highest-priority matching entry."""
        priority = {"LOW": 0, "HIGH": 1, "CRITICAL": 2}
        best: TaxonomyEntry | None = None
        for entry in self._entries:
            pattern = entry.pattern
            if "::" not in pattern:
                file_part = node_id.split("::")[0]
                matches = fnmatch.fnmatch(file_part, pattern)
            else:
                matches = fnmatch.fnmatch(node_id, pattern)
            if matches:
                if best is None or priority[entry.criticality] > priority[best.criticality]:
                    best = entry
        return best.reason if best else ""

    @property
    def entries(self) -> list[TaxonomyEntry]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# JUnit XML parser
# ---------------------------------------------------------------------------


def parse_junit_xml(path: str | Path) -> list[TestResult]:
    """Parse a JUnit XML report and return a flat list of ``TestResult``s.

    Handles both ``<testsuite>`` and ``<testsuites>`` root elements.
    """
    results: list[TestResult] = []
    tree = ElementTree.parse(str(path))
    root = tree.getroot()

    suites = []
    if root.tag == "testsuites":
        suites = list(root)
    elif root.tag == "testsuite":
        suites = [root]
    else:
        # Some pytest versions nest differently
        suites = root.findall("testsuite") or [root]

    for suite in suites:
        for case in suite.findall("testcase"):
            classname = case.get("classname", "")
            name = case.get("name", "")
            # Build the node_id the same way pytest does.
            # classname is typically "tests.test_benford" → convert dots to /
            module_path = classname.replace(".", "/") + ".py" if classname else ""
            node_id = f"{module_path}::{name}" if module_path else name

            failure = case.find("failure")
            error = case.find("error")
            skipped = case.find("skipped")

            if failure is not None:
                outcome = "failed"
                message = failure.get("message", "") or (failure.text or "")
            elif error is not None:
                outcome = "error"
                message = error.get("message", "") or (error.text or "")
            elif skipped is not None:
                outcome = "skipped"
                message = skipped.get("message", "") or ""
            else:
                outcome = "passed"
                message = ""

            results.append(
                TestResult(
                    node_id=node_id,
                    classname=classname,
                    name=name,
                    outcome=outcome,
                    message=str(message).strip(),
                )
            )
    return results


# ---------------------------------------------------------------------------
# ReleaseGateEvaluator
# ---------------------------------------------------------------------------


class ReleaseGateEvaluator:
    """Applies the criticality taxonomy to a parsed test run.

    Parameters
    ----------
    taxonomy:
        ``TestCriticalityTaxonomy`` instance.
    allow_high_failures:
        When True, HIGH-tier failures do not block the release.  CRITICAL
        failures always block regardless of this flag.
    """

    def __init__(
        self,
        taxonomy: TestCriticalityTaxonomy,
        allow_high_failures: bool = False,
    ) -> None:
        self._taxonomy = taxonomy
        self._allow_high = allow_high_failures

    def evaluate(self, results: list[TestResult]) -> GateResult:
        """Evaluate *results* and return a ``GateResult``.

        Parameters
        ----------
        results:
            Test results as produced by ``parse_junit_xml``.
        """
        failures = [r for r in results if r.outcome in ("failed", "error")]
        critical_failures: list[CriticalFailure] = []
        high_failures: list[CriticalFailure] = []

        for tr in failures:
            level = self._taxonomy.criticality_for(tr.node_id)
            reason = self._taxonomy.reason_for(tr.node_id)
            cf = CriticalFailure(
                node_id=tr.node_id,
                criticality=level,
                reason=reason,
                message=tr.message,
            )
            if level == "CRITICAL":
                critical_failures.append(cf)
            elif level == "HIGH":
                high_failures.append(cf)
            # LOW failures are counted but don't block.

        blocked = bool(critical_failures)
        if not self._allow_high:
            blocked = blocked or bool(high_failures)

        blocked_by = ""
        if critical_failures:
            blocked_by = f"{len(critical_failures)} CRITICAL test failure(s)"
        elif high_failures and not self._allow_high:
            blocked_by = (
                f"{len(high_failures)} HIGH test failure(s) "
                "(pass --allow-high-failures to override)"
            )

        return GateResult(
            passed=not blocked,
            critical_failures=critical_failures,
            high_failures=high_failures,
            evaluated_tests=len(results),
            total_failures=len(failures),
            blocked_by=blocked_by,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate release gate: block release if critical tests failed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  — Gate passed; release may proceed.
  1  — Gate failed; critical (or unexcepted high) test failures detected.
  2  — Input error (missing JUnit XML or taxonomy file).

Examples:
  # Run tests with JUnit output then evaluate the gate:
  pytest --junitxml=reports/junit.xml
  python -m scripts.release_gate --junit reports/junit.xml

  # Allow HIGH failures (e.g. on a feature branch):
  python -m scripts.release_gate --junit reports/junit.xml --allow-high-failures
        """,
    )
    parser.add_argument(
        "--junit",
        default=str(DEFAULT_JUNIT_PATH),
        help=f"Path to JUnit XML report (default: {DEFAULT_JUNIT_PATH})",
    )
    parser.add_argument(
        "--taxonomy",
        default=str(DEFAULT_TAXONOMY_PATH),
        help=f"Path to test_criticality.json taxonomy (default: {DEFAULT_TAXONOMY_PATH})",
    )
    parser.add_argument(
        "--allow-high-failures",
        action="store_true",
        help="Allow HIGH-tier test failures (CRITICAL still blocks)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the GateResult as JSON to this path (optional)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    junit_path = Path(args.junit)
    taxonomy_path = Path(args.taxonomy)

    if not junit_path.exists():
        logger.error("JUnit XML not found: %s", junit_path)
        print(f"ERROR: JUnit XML not found: {junit_path}", file=sys.stderr)
        print("Run pytest with --junitxml=reports/junit.xml first.", file=sys.stderr)
        return 2

    taxonomy = TestCriticalityTaxonomy(taxonomy_path)
    evaluator = ReleaseGateEvaluator(taxonomy, allow_high_failures=args.allow_high_failures)

    try:
        results = parse_junit_xml(junit_path)
    except Exception as exc:
        logger.error("Failed to parse JUnit XML %s: %s", junit_path, exc)
        print(f"ERROR: Failed to parse {junit_path}: {exc}", file=sys.stderr)
        return 2

    gate_result = evaluator.evaluate(results)
    print(gate_result.summary())

    if args.output_json:
        output = {
            "passed": gate_result.passed,
            "blocked_by": gate_result.blocked_by,
            "evaluated_tests": gate_result.evaluated_tests,
            "total_failures": gate_result.total_failures,
            "critical_failures": [
                {
                    "node_id": f.node_id,
                    "criticality": f.criticality,
                    "reason": f.reason,
                    "message": f.message,
                }
                for f in gate_result.critical_failures
            ],
            "high_failures": [
                {
                    "node_id": f.node_id,
                    "criticality": f.criticality,
                    "reason": f.reason,
                    "message": f.message,
                }
                for f in gate_result.high_failures
            ],
        }
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as fh:
            json.dump(output, fh, indent=2)
        logger.info("Gate result written to %s", args.output_json)

    return 0 if gate_result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
