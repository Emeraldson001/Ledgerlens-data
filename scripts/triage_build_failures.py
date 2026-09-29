"""Build Failure Triage Report Generator — Issue #539.

Parses CI test output (JUnit XML, pytest JSON, or raw stdout logs) and produces
a structured triage report that groups failures by:
  - Failure category (import error, assertion, timeout, fixture, unknown)
  - Test module
  - Recurrence (failures that appear in more than one run are flagged as
    "recurring")

The report is written to ``reports/triage/triage_<timestamp>.json`` and an
optional Markdown summary to ``reports/triage/triage_<timestamp>.md``.

Usage::

    # Parse a pytest JSON report (generated with: pytest --json-report)
    python -m scripts.triage_build_failures \\
        --input .report.json \\
        --format pytest-json

    # Parse a JUnit XML file (generated with: pytest --junitxml=results.xml)
    python -m scripts.triage_build_failures \\
        --input results.xml \\
        --format junit-xml

    # Read raw CI log from stdin and detect failures
    python -m scripts.triage_build_failures \\
        --input - \\
        --format raw-log

    # Compare two reports to find newly introduced vs resolved failures
    python -m scripts.triage_build_failures \\
        --input results.xml --format junit-xml \\
        --baseline reports/triage/triage_baseline.json \\
        --compare

Exit codes:
    0 — no failures found
    1 — failures found (triage report written)
    2 — fatal error (bad input, unrecognised format, etc.)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

REPORTS_DIR = Path("reports/triage")

# ---------------------------------------------------------------------------
# Failure category patterns (order matters — first match wins)
# ---------------------------------------------------------------------------
_CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("import_error", re.compile(r"(ImportError|ModuleNotFoundError|cannot import)", re.I)),
    ("timeout", re.compile(r"(TimeoutError|timed out|timeout)", re.I)),
    (
        "fixture_error",
        re.compile(r"(fixture '.*' not found|ScopeMismatch|FixtureLookupError)", re.I),
    ),
    ("attribute_error", re.compile(r"AttributeError", re.I)),
    ("type_error", re.compile(r"TypeError", re.I)),
    ("value_error", re.compile(r"ValueError", re.I)),
    ("assertion_error", re.compile(r"AssertionError|assert ", re.I)),
    ("connection_error", re.compile(r"(ConnectionRefusedError|ConnectionError|socket)", re.I)),
    ("schema_mismatch", re.compile(r"(schema|feature_columns|hash mismatch|RuntimeError)", re.I)),
]

_RECURRING_MARKER = "RECURRING"


# ---------------------------------------------------------------------------
# Data classes (plain dicts for JSON serialisation)
# ---------------------------------------------------------------------------


def _categorise(message: str) -> str:
    for category, pattern in _CATEGORY_PATTERNS:
        if pattern.search(message):
            return category
    return "unknown"


def _extract_module(nodeid: str) -> str:
    """Return the test module portion of a pytest node id.

    E.g. ``tests/test_benford.py::TestBenford::test_chi_square`` → ``test_benford``
    """
    parts = nodeid.split("::")
    path = parts[0] if parts else nodeid
    return Path(path).stem


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def _parse_pytest_json(raw: str) -> list[dict[str, Any]]:
    """Parse a pytest --json-report JSON blob."""
    data = json.loads(raw)
    failures: list[dict[str, Any]] = []
    for test in data.get("tests", []):
        if test.get("outcome") not in ("failed", "error"):
            continue
        nodeid = test.get("nodeid", "unknown")
        call = test.get("call") or test.get("setup") or {}
        longrepr = call.get("longrepr", "") or ""
        if isinstance(longrepr, dict):
            longrepr = longrepr.get("reprcrash", {}).get("message", "")
        failures.append(
            {
                "nodeid": nodeid,
                "module": _extract_module(nodeid),
                "outcome": test.get("outcome", "failed"),
                "category": _categorise(longrepr),
                "message": longrepr[:1000],
                "duration_s": round(call.get("duration", 0.0), 3),
            }
        )
    return failures


def _parse_junit_xml(raw: str) -> list[dict[str, Any]]:
    """Parse a JUnit XML report (pytest --junitxml)."""
    root = ET.fromstring(raw)
    failures: list[dict[str, Any]] = []
    suites = list(root.iter("testsuite"))
    if not suites:
        suites = [root]

    for suite in suites:
        for case in suite.iter("testcase"):
            classname = case.get("classname", "")
            testname = case.get("name", "unknown")
            nodeid = f"{classname}::{testname}" if classname else testname

            failure_el = case.find("failure")
            error_el = case.find("error")
            el = failure_el if failure_el is not None else error_el
            if el is None:
                continue

            message = el.get("message", "") or (el.text or "")
            failures.append(
                {
                    "nodeid": nodeid,
                    "module": _extract_module(classname or testname),
                    "outcome": "error" if error_el is not None else "failed",
                    "category": _categorise(message),
                    "message": message[:1000],
                    "duration_s": round(float(case.get("time", 0) or 0), 3),
                }
            )
    return failures


def _parse_raw_log(raw: str) -> list[dict[str, Any]]:
    """Heuristic parser for raw pytest stdout output."""
    failures: list[dict[str, Any]] = []
    # Look for FAILED / ERROR lines that contain a nodeid
    failed_re = re.compile(r"^(FAILED|ERROR)\s+([\w/.:]+::\S*)", re.M)
    # Collect short error context after each FAILED line
    lines = raw.splitlines()
    line_index: dict[int, re.Match[str]] = {}
    for m in failed_re.finditer(raw):
        lineno = raw[: m.start()].count("\n")
        line_index[lineno] = m

    for lineno, m in line_index.items():
        outcome = m.group(1).lower()
        nodeid = m.group(2)
        # Grab up to 5 surrounding lines as context
        context_lines = lines[max(0, lineno - 2) : lineno + 5]
        context = "\n".join(context_lines)
        failures.append(
            {
                "nodeid": nodeid,
                "module": _extract_module(nodeid),
                "outcome": outcome,
                "category": _categorise(context),
                "message": context[:1000],
                "duration_s": 0.0,
            }
        )
    return failures


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_triage_report(
    failures: list[dict[str, Any]],
    baseline: dict[str, Any] | None = None,
    source_file: str = "",
) -> dict[str, Any]:
    """Group failures and annotate recurring ones.

    If *baseline* is provided (a previously generated triage report), failures
    that also appeared in the baseline are flagged as ``RECURRING``.
    """
    baseline_nodeids: set[str] = set()
    if baseline:
        for f in baseline.get("failures", []):
            baseline_nodeids.add(f.get("nodeid", ""))

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)

    annotated: list[dict[str, Any]] = []
    for f in failures:
        entry = dict(f)
        entry["recurring"] = f["nodeid"] in baseline_nodeids
        if entry["recurring"]:
            entry["recurrence_label"] = _RECURRING_MARKER
        annotated.append(entry)
        by_category[f["category"]].append(entry)
        by_module[f["module"]].append(entry)

    recurring_count = sum(1 for f in annotated if f["recurring"])
    new_failures = [f for f in annotated if not f["recurring"]] if baseline else annotated

    category_summary = {
        cat: {"count": len(items), "recurring": sum(1 for i in items if i["recurring"])}
        for cat, items in sorted(by_category.items())
    }
    module_summary = {
        mod: {"count": len(items), "recurring": sum(1 for i in items if i["recurring"])}
        for mod, items in sorted(by_module.items(), key=lambda kv: -len(kv[1]))
    }

    return {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source_file": source_file,
        "total_failures": len(annotated),
        "recurring_failures": recurring_count,
        "new_failures": len(new_failures),
        "has_baseline": baseline is not None,
        "category_summary": category_summary,
        "module_summary": module_summary,
        "failures": annotated,
    }


def render_markdown(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown summary of the triage report."""
    lines = [
        "# LedgerLens Build Failure Triage Report",
        "",
        f"**Generated:** {report['generated_at']}",
        f"**Source:** `{report['source_file']}`",
        f"**Total failures:** {report['total_failures']}",
        f"**Recurring failures:** {report['recurring_failures']}",
        f"**New failures (vs baseline):** {report['new_failures']}",
        "",
        "## Failures by Category",
        "",
        "| Category | Total | Recurring |",
        "|---|---|---|",
    ]
    for cat, info in report["category_summary"].items():
        lines.append(f"| `{cat}` | {info['count']} | {info['recurring']} |")

    lines += [
        "",
        "## Failures by Module (top 10)",
        "",
        "| Module | Total | Recurring |",
        "|---|---|---|",
    ]
    for mod, info in list(report["module_summary"].items())[:10]:
        lines.append(f"| `{mod}` | {info['count']} | {info['recurring']} |")

    lines += ["", "## Failure Details", ""]
    for f in report["failures"]:
        recurring_tag = " 🔁 **RECURRING**" if f.get("recurring") else ""
        lines.append(
            f"### `{f['nodeid']}`{recurring_tag}\n"
            f"- **Category:** `{f['category']}`\n"
            f"- **Module:** `{f['module']}`\n"
            f"- **Duration:** {f['duration_s']}s\n"
            f"- **Message:**\n```\n{f['message'][:500]}\n```\n"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a build failure triage report from CI test output.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the test report file, or '-' to read from stdin.",
    )
    p.add_argument(
        "--format",
        "-f",
        choices=["pytest-json", "junit-xml", "raw-log"],
        default="junit-xml",
        help="Format of the input report (default: junit-xml).",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        default=str(REPORTS_DIR),
        help=f"Directory to write triage reports (default: {REPORTS_DIR}).",
    )
    p.add_argument(
        "--baseline",
        "-b",
        default=None,
        help="Path to a previous triage JSON report to compare against.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Print a comparison summary when --baseline is provided.",
    )
    p.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip generating the Markdown summary file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        raw = _read_input(args.input)
    except (FileNotFoundError, OSError) as exc:
        logger.error("Cannot read input: %s", exc)
        return 2

    # Parse
    try:
        if args.format == "pytest-json":
            failures = _parse_pytest_json(raw)
        elif args.format == "junit-xml":
            failures = _parse_junit_xml(raw)
        else:
            failures = _parse_raw_log(raw)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse %s input: %s", args.format, exc)
        return 2

    # Load baseline if provided
    baseline: dict[str, Any] | None = None
    if args.baseline:
        try:
            baseline = json.loads(Path(args.baseline).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Could not load baseline %s: %s", args.baseline, exc)

    # Build report
    report = build_triage_report(
        failures,
        baseline=baseline,
        source_file=args.input,
    )

    # Write outputs
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"triage_{ts}.json"
    json_path.write_text(json.dumps(report, indent=2))
    logger.info("Triage report written → %s", json_path)

    if not args.no_markdown:
        md_path = out_dir / f"triage_{ts}.md"
        md_path.write_text(render_markdown(report))
        logger.info("Markdown summary written → %s", md_path)

    # Print brief summary to stdout for CI visibility
    print(
        f"[triage] {report['total_failures']} failure(s) found "
        f"({report['recurring_failures']} recurring, "
        f"{report['new_failures']} new) — report: {json_path}"
    )

    if args.compare and baseline is not None:
        resolved = set(f["nodeid"] for f in baseline.get("failures", [])) - set(
            f["nodeid"] for f in report["failures"]
        )
        newly_broken = [f["nodeid"] for f in report["failures"] if not f["recurring"]]
        print(f"\n[triage] Resolved vs baseline: {len(resolved)}")
        for nid in sorted(resolved):
            print(f"  ✓ {nid}")
        print(f"[triage] Newly introduced failures: {len(newly_broken)}")
        for nid in sorted(newly_broken):
            print(f"  ✗ {nid}")

    return 1 if report["total_failures"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
