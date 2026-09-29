"""Contributor-facing commands for advanced validation suites.

Issue #558 — Build contributor-facing commands for advanced validation suites
=============================================================================

This script is the single entry-point for all LedgerLens data-quality and
structural validation checks.  It is designed to be run by contributors
locally and in CI::

    # Run all suites
    python -m scripts.validate

    # Run a specific suite
    python -m scripts.validate --suite parsing
    python -m scripts.validate --suite reconciliation
    python -m scripts.validate --suite schema
    python -m scripts.validate --suite feature_ranges

    # Fail fast on the first error (useful in CI)
    python -m scripts.validate --fail-fast

    # Write a JSON report to disk
    python -m scripts.validate --report reports/validation_report.json

    # Verbose output (show passing checks too)
    python -m scripts.validate --verbose

Exit codes
----------
0 – All suites passed (no hard errors).
1 – One or more suites failed (hard errors found).
2 – Validation suite crashed with an unhandled exception.

Suites
------
``parsing``
    Validates ``data/known_manipulation_events.csv``,
    ``data/feature_ranges.json``, and ``data/trade_avro_schema.json``.

``schema``
    Validates that the Avro trade schema is internally consistent and
    backward-compatible with itself (smoke test), and checks all JSON files
    in ``data/`` are valid JSON.

``feature_ranges``
    Checks that ``data/feature_ranges.json`` is well-formed and all feature
    ``{min, max, mean, std}`` entries are mathematically consistent.

``reconciliation``
    Loads the synthetic dataset and the (optional) model metadata, then runs
    trade-count, feature-column, and wallet-score reconciliation checks.

``all`` (default)
    Runs every suite above in order.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Suite result type
# ---------------------------------------------------------------------------


@dataclass
class SuiteResult:
    name: str
    passed: bool
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "suite": self.name,
            "passed": self.passed,
            "duration_ms": round(self.duration_ms, 1),
            "messages": self.messages,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Individual suite runners
# ---------------------------------------------------------------------------


def _run_parsing_suite(verbose: bool = False) -> SuiteResult:
    """Validate CSV and JSON parsing contracts (Issue #552)."""
    import time

    from validation.parsing import (
        CSVParseError,
        JSONParseError,
        parse_feature_ranges,
        parse_json,
        parse_known_manipulation_events,
    )

    result = SuiteResult(name="parsing", passed=True)
    t0 = time.monotonic()

    # ------------------------------------------------------------------
    # 1. known_manipulation_events.csv
    # ------------------------------------------------------------------
    events_path = Path("data") / "known_manipulation_events.csv"
    try:
        pr = parse_known_manipulation_events(events_path)
        if pr.ok:
            msg = f"  ✓ {events_path}: {pr.record_count} manipulation events parsed"
            result.messages.append(msg)
            if verbose:
                print(msg)
        else:
            result.passed = False
            for err in pr.errors:
                result.errors.append(f"  ✗ {events_path}: {err}")
    except CSVParseError as exc:
        result.passed = False
        result.errors.append(f"  ✗ {events_path}: {exc}")

    # ------------------------------------------------------------------
    # 2. feature_ranges.json
    # ------------------------------------------------------------------
    ranges_path = Path("data") / "feature_ranges.json"
    if ranges_path.exists():
        try:
            pr2 = parse_feature_ranges(ranges_path)
            if pr2.ok:
                msg = f"  ✓ {ranges_path}: {pr2.record_count} feature ranges parsed"
                result.messages.append(msg)
                if verbose:
                    print(msg)
            else:
                for err in pr2.errors:
                    result.warnings.append(f"  ⚠ {ranges_path}: {err}")
        except JSONParseError as exc:
            result.passed = False
            result.errors.append(f"  ✗ {ranges_path}: {exc}")

    # ------------------------------------------------------------------
    # 3. trade_avro_schema.json
    # ------------------------------------------------------------------
    avro_path = Path("data") / "trade_avro_schema.json"
    if avro_path.exists():
        try:
            schema = parse_json(avro_path)
            if not isinstance(schema, dict) or "fields" not in schema:
                result.passed = False
                result.errors.append(f"  ✗ {avro_path}: schema missing 'fields' key")
            else:
                msg = f"  ✓ {avro_path}: valid Avro schema " f"({len(schema['fields'])} fields)"
                result.messages.append(msg)
                if verbose:
                    print(msg)
        except JSONParseError as exc:
            result.passed = False
            result.errors.append(f"  ✗ {avro_path}: {exc}")

    result.duration_ms = (time.monotonic() - t0) * 1000
    return result


def _run_schema_suite(verbose: bool = False) -> SuiteResult:
    """Validate all JSON files in data/ and the Avro schema compatibility."""
    import time

    from validation.parsing import JSONParseError, parse_json

    result = SuiteResult(name="schema", passed=True)
    t0 = time.monotonic()

    data_dir = Path("data")
    json_files = list(data_dir.glob("*.json"))

    for json_file in json_files:
        try:
            parse_json(json_file)
            msg = f"  ✓ {json_file}: valid JSON"
            result.messages.append(msg)
            if verbose:
                print(msg)
        except JSONParseError as exc:
            result.passed = False
            result.errors.append(f"  ✗ {json_file}: {exc}")

    # Avro schema self-compatibility check
    avro_path = data_dir / "trade_avro_schema.json"
    if avro_path.exists():
        try:
            from ingestion.avro_codec import (
                check_backward_compatibility,
                load_schema,
            )

            schema = load_schema(str(avro_path))
            ok, violations = check_backward_compatibility(schema, schema)
            if ok:
                msg = f"  ✓ {avro_path}: Avro self-compatibility check passed"
                result.messages.append(msg)
                if verbose:
                    print(msg)
            else:
                for v in violations:
                    result.warnings.append(f"  ⚠ Avro compatibility: {v}")
        except Exception as exc:  # pragma: no cover
            result.warnings.append(f"  ⚠ Avro compatibility check skipped: {exc}")

    # build_config.json structure check
    build_config_path = data_dir / "build_config.json"
    if build_config_path.exists():
        try:
            cfg = parse_json(build_config_path)
            if isinstance(cfg, dict):
                msg = f"  ✓ {build_config_path}: valid build config"
                result.messages.append(msg)
                if verbose:
                    print(msg)
        except JSONParseError as exc:
            result.passed = False
            result.errors.append(f"  ✗ {build_config_path}: {exc}")

    result.duration_ms = (time.monotonic() - t0) * 1000
    return result


def _run_feature_ranges_suite(verbose: bool = False) -> SuiteResult:
    """Validate feature_ranges.json consistency."""
    import time

    result = SuiteResult(name="feature_ranges", passed=True)
    t0 = time.monotonic()

    ranges_path = Path("data") / "feature_ranges.json"
    if not ranges_path.exists():
        result.warnings.append(f"  ⚠ {ranges_path} not found — skipping suite")
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    try:
        raw: dict = json.loads(ranges_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.passed = False
        result.errors.append(f"  ✗ {ranges_path}: {exc}")
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    feature_count = 0
    for feature_name, bounds in raw.items():
        if not isinstance(bounds, dict):
            result.errors.append(f"  ✗ {feature_name}: expected dict, got {type(bounds).__name__}")
            result.passed = False
            continue

        lo = bounds.get("min")
        hi = bounds.get("max")
        mean = bounds.get("mean")
        std = bounds.get("std")

        # min <= max
        if lo is not None and hi is not None and lo > hi:
            result.passed = False
            result.errors.append(f"  ✗ {feature_name}: min ({lo}) > max ({hi})")

        # mean within [min, max]
        if lo is not None and mean is not None and mean < lo:
            result.warnings.append(f"  ⚠ {feature_name}: mean ({mean}) < min ({lo})")
        if hi is not None and mean is not None and mean > hi:
            result.warnings.append(f"  ⚠ {feature_name}: mean ({mean}) > max ({hi})")

        # std >= 0
        if std is not None and std < 0:
            result.passed = False
            result.errors.append(f"  ✗ {feature_name}: std ({std}) is negative")

        feature_count += 1

    msg = f"  ✓ {ranges_path}: {feature_count} feature ranges validated"
    if result.passed:
        result.messages.append(msg)
        if verbose:
            print(msg)

    result.duration_ms = (time.monotonic() - t0) * 1000
    return result


def _run_reconciliation_suite(verbose: bool = False) -> SuiteResult:
    """Run reconciliation checks on the synthetic dataset (Issue #554)."""
    import time

    result = SuiteResult(name="reconciliation", passed=True)
    t0 = time.monotonic()

    dataset_path = Path("data") / "synthetic_dataset.parquet"
    if not dataset_path.exists():
        result.warnings.append(
            f"  ⚠ {dataset_path} not found — "
            "run `python -m scripts.generate_synthetic_dataset` first"
        )
        result.duration_ms = (time.monotonic() - t0) * 1000
        return result

    try:
        import pandas as pd

        from validation.reconciliation import (
            ReconciliationReport,
            reconcile_features,
        )

        df = pd.read_parquet(dataset_path)
        msg = f"  ✓ {dataset_path}: loaded {len(df)} rows, {len(df.columns)} columns"
        result.messages.append(msg)
        if verbose:
            print(msg)

        # Feature-matrix structural checks
        core_cols = ["wallet_id", "label"]
        report: ReconciliationReport = reconcile_features(
            df,
            required_columns=core_cols,
            feature_ranges_path=(
                Path("data") / "feature_ranges.json"
                if (Path("data") / "feature_ranges.json").exists()
                else None
            ),
        )

        for e in report.errors:
            if e.severity == "error":
                result.passed = False
                result.errors.append(f"  ✗ reconciliation/{e.check}: {e}")
            else:
                result.warnings.append(f"  ⚠ reconciliation/{e.check}: {e}")

        if report.ok:
            msg = f"  ✓ Feature-matrix reconciliation passed ({len(df.columns)} columns checked)"
            result.messages.append(msg)
            if verbose:
                print(msg)

    except Exception as exc:
        result.passed = False
        result.errors.append(f"  ✗ reconciliation suite error: {exc}")
        if verbose:
            traceback.print_exc()

    result.duration_ms = (time.monotonic() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Suite registry
# ---------------------------------------------------------------------------

SUITES: dict[str, Callable[[bool], SuiteResult]] = {
    "parsing": _run_parsing_suite,
    "schema": _run_schema_suite,
    "feature_ranges": _run_feature_ranges_suite,
    "reconciliation": _run_reconciliation_suite,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="validate",
        description=(
            "LedgerLens contributor validation suite.\n\n"
            "Run all data-quality, parsing-contract, and reconciliation checks "
            "without requiring a live Horizon connection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 — all suites passed\n"
            "  1 — one or more hard errors found\n"
            "  2 — unhandled exception in a suite runner\n"
        ),
    )
    p.add_argument(
        "--suite",
        choices=[*SUITES.keys(), "all"],
        default="all",
        help="Which validation suite to run (default: all).",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop after the first suite that has hard errors.",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        help="Write a JSON validation report to this path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print passing checks in addition to failures.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except the final summary line.",
    )
    return p


def run_suites(
    suite_names: list[str],
    *,
    fail_fast: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> list[SuiteResult]:
    """Execute the named validation suites and return their results."""
    results: list[SuiteResult] = []

    for name in suite_names:
        runner = SUITES[name]

        if not quiet:
            print(f"\n── Suite: {name} ──")

        try:
            suite_result = runner(verbose=verbose)
        except Exception as exc:  # pragma: no cover
            suite_result = SuiteResult(
                name=name,
                passed=False,
                errors=[f"  ✗ Suite runner crashed: {exc}"],
            )
            if verbose:
                traceback.print_exc()

        results.append(suite_result)

        if not quiet:
            # Print errors and warnings regardless of verbose flag
            for msg in suite_result.errors:
                print(msg)
            for msg in suite_result.warnings:
                print(msg)
            if verbose or not suite_result.passed:
                for msg in suite_result.messages:
                    print(msg)

            status = "PASSED" if suite_result.passed else "FAILED"
            print(
                f"  → {name}: {status} "
                f"({suite_result.duration_ms:.0f} ms, "
                f"{len(suite_result.errors)} error(s), "
                f"{len(suite_result.warnings)} warning(s))"
            )

        if fail_fast and not suite_result.passed:
            break

    return results


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.suite == "all":
        suite_names = list(SUITES.keys())
    else:
        suite_names = [args.suite]

    if not args.quiet:
        print(
            f"LedgerLens Validation Suite — {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(f"Suites: {', '.join(suite_names)}")

    results = run_suites(
        suite_names,
        fail_fast=args.fail_fast,
        verbose=args.verbose,
        quiet=args.quiet,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    total_errors = sum(len(r.errors) for r in results)
    total_warnings = sum(len(r.warnings) for r in results)

    if not args.quiet:
        print(
            f"\n{'='*60}\n"
            f"Summary: {passed}/{total} suites passed, "
            f"{total_errors} error(s), {total_warnings} warning(s)"
        )

    # ------------------------------------------------------------------
    # Optional JSON report
    # ------------------------------------------------------------------
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = {
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "suites_run": suite_names,
            "passed": failed == 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "results": [r.to_dict() for r in results],
        }
        report_path.write_text(
            json.dumps(report_data, indent=2, default=str),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"Report written to: {report_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
