"""Engineering quality scorecard for LedgerLens (Issue #604).

Overview
--------
Tracks repository maturity through a composite set of engineering quality
metrics, writes a machine-readable ``reports/quality_metrics.json`` report,
optionally emits Shields.io-compatible CI badge data, and exits with a
non-zero code when the composite score drops below the configured threshold.

Metrics collected
-----------------

+--------------------------+--------------------------------------------------+
| Metric                   | Source                                           |
+==========================+==================================================+
| ``test_coverage``        | Cobertura XML (``--cov-report=xml``)            |
| ``mutation_score``       | mutmut SQLite cache (``.mutmut-cache``)          |
| ``critical_pass_rate``   | JUnit XML + criticality taxonomy JSON            |
| ``drift_rate``           | PSI drift report (``reports/drift_report.json``) |
| ``cycle_time_hours``     | Git log (time between last two merges to main)   |
+--------------------------+--------------------------------------------------+

Composite score
---------------
Each metric is normalised to [0, 100] and weighted::

    composite = Σ(metric_score × weight) / Σ(weights)

Default weights (configurable via ``--weights`` JSON argument):

+------------------+--------+
| Metric           | Weight |
+==================+========+
| test_coverage    | 0.25   |
| mutation_score   | 0.30   |
| critical_pass    | 0.25   |
| drift_rate       | 0.10   |
| cycle_time       | 0.10   |
+------------------+--------+

Threshold gate
--------------
``--threshold N`` (default: 70) causes the script to exit 1 if the composite
score is below *N*.  CI uses this to block merges when quality regresses.

CI badge reporter
-----------------
Pass ``--badge`` to write ``reports/quality_badge.json`` in the Shields.io
endpoint format so a dynamic badge can be rendered in the README.

Usage
-----
::

    # Full scorecard from all sources:
    pytest --junitxml=reports/junit.xml --cov --cov-report=xml:reports/coverage.xml
    python -m scripts.quality_metrics \\
        --junit    reports/junit.xml \\
        --coverage reports/coverage.xml \\
        --output   reports/quality_metrics.json \\
        --badge \\
        --threshold 70

    # Skip metrics that are not available (they get a neutral score of 100):
    python -m scripts.quality_metrics --junit reports/junit.xml --skip-missing
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_JUNIT = Path("reports/junit.xml")
DEFAULT_COVERAGE = Path("reports/coverage.xml")
DEFAULT_MUTMUT_CACHE = Path(".mutmut-cache")
DEFAULT_DRIFT_REPORT = Path("reports/drift_report.json")
DEFAULT_TAXONOMY = Path("data/test_criticality.json")
DEFAULT_OUTPUT = Path("reports/quality_metrics.json")
DEFAULT_BADGE = Path("reports/quality_badge.json")
DEFAULT_THRESHOLD = 70.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "test_coverage": 0.25,
    "mutation_score": 0.30,
    "critical_pass_rate": 0.25,
    "drift_rate": 0.10,
    "cycle_time": 0.10,
}

# Cycle time target (hours).  A cycle time ≤ this gets a score of 100.
TARGET_CYCLE_TIME_HOURS = 72.0
# Worst-case cycle time that maps to a score of 0.
WORST_CYCLE_TIME_HOURS = 336.0  # 2 weeks


# ---------------------------------------------------------------------------
# Metric collectors
# ---------------------------------------------------------------------------


@dataclass
class MetricReading:
    name: str
    raw_value: float | None  # None means not available
    score: float  # normalised 0-100
    source: str  # where the value came from
    notes: str = ""


def collect_test_coverage(coverage_xml: Path, skip_missing: bool = False) -> MetricReading:
    """Parse Cobertura XML and extract overall line-coverage percentage."""
    if not coverage_xml.exists():
        msg = f"Coverage XML not found: {coverage_xml}"
        if skip_missing:
            logger.info("%s — skipping (neutral score 100)", msg)
            return MetricReading("test_coverage", None, 100.0, str(coverage_xml), "skipped")
        logger.warning(msg)
        return MetricReading("test_coverage", None, 0.0, str(coverage_xml), msg)

    try:
        tree = ElementTree.parse(str(coverage_xml))
        root = tree.getroot()
        # Cobertura root attribute is "line-rate" (0.0 – 1.0)
        rate = root.get("line-rate")
        if rate is None:
            raise ValueError("No 'line-rate' attribute on Cobertura root element")
        raw = float(rate) * 100.0
        return MetricReading("test_coverage", raw, raw, str(coverage_xml))
    except Exception as exc:
        notes = f"Parse error: {exc}"
        logger.warning("collect_test_coverage: %s", notes)
        return MetricReading("test_coverage", None, 0.0, str(coverage_xml), notes)


def collect_mutation_score(cache_path: Path, skip_missing: bool = False) -> MetricReading:
    """Read the mutmut SQLite cache and compute killed/(killed+survived)."""
    if not cache_path.exists():
        msg = f"mutmut cache not found: {cache_path}"
        if skip_missing:
            logger.info("%s — skipping (neutral score 100)", msg)
            return MetricReading("mutation_score", None, 100.0, str(cache_path), "skipped")
        logger.warning(msg)
        return MetricReading("mutation_score", None, 0.0, str(cache_path), msg)

    try:
        conn = sqlite3.connect(str(cache_path))
        cursor = conn.execute("SELECT status FROM mutant")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError as exc:
        notes = f"SQLite error: {exc}"
        logger.warning("collect_mutation_score: %s", notes)
        return MetricReading("mutation_score", None, 0.0, str(cache_path), notes)

    killed = sum(1 for (s,) in rows if s in ("ok", "suspicious", "timeout"))
    survived = sum(1 for (s,) in rows if s == "survived")
    total = killed + survived
    if total == 0:
        notes = "No mutations found in cache"
        logger.warning("collect_mutation_score: %s", notes)
        return MetricReading("mutation_score", None, 0.0, str(cache_path), notes)

    raw = (killed / total) * 100.0
    return MetricReading("mutation_score", raw, raw, str(cache_path))


def collect_critical_pass_rate(
    junit_xml: Path,
    taxonomy_json: Path,
    skip_missing: bool = False,
) -> MetricReading:
    """Fraction (0–100) of CRITICAL tests that passed."""
    if not junit_xml.exists():
        msg = f"JUnit XML not found: {junit_xml}"
        if skip_missing:
            return MetricReading("critical_pass_rate", None, 100.0, str(junit_xml), "skipped")
        return MetricReading("critical_pass_rate", None, 0.0, str(junit_xml), msg)

    # Import taxonomy helpers from release_gate — avoids duplicating logic.
    try:
        from scripts.release_gate import TestCriticalityTaxonomy, parse_junit_xml

        taxonomy = TestCriticalityTaxonomy(taxonomy_json)
        results = parse_junit_xml(junit_xml)
    except Exception as exc:
        notes = f"Parse/taxonomy error: {exc}"
        logger.warning("collect_critical_pass_rate: %s", notes)
        return MetricReading("critical_pass_rate", None, 0.0, str(junit_xml), notes)

    critical = [r for r in results if taxonomy.criticality_for(r.node_id) == "CRITICAL"]
    if not critical:
        return MetricReading(
            "critical_pass_rate",
            None,
            100.0,
            str(junit_xml),
            "No CRITICAL tests found in taxonomy; defaulting to 100",
        )

    passed = sum(1 for r in critical if r.outcome == "passed")
    raw = (passed / len(critical)) * 100.0
    notes = f"{passed}/{len(critical)} CRITICAL tests passed"
    return MetricReading("critical_pass_rate", raw, raw, str(junit_xml), notes)


def collect_drift_rate(
    drift_report: Path,
    skip_missing: bool = False,
) -> MetricReading:
    """Read the PSI drift report and compute a quality score.

    A drift rate of 0% (no features drifting) maps to score 100.
    A drift rate of 100% (all features drifting) maps to score 0.
    """
    if not drift_report.exists():
        if skip_missing:
            return MetricReading("drift_rate", None, 100.0, str(drift_report), "skipped")
        return MetricReading(
            "drift_rate",
            None,
            100.0,
            str(drift_report),
            "No drift report found; assuming no drift (score 100)",
        )

    try:
        with open(drift_report) as fh:
            data = json.load(fh)
        # Expected keys: "features_checked", "features_drifted" or "drift_fraction"
        if "drift_fraction" in data:
            raw_fraction = float(data["drift_fraction"])
        elif "features_drifted" in data and "features_checked" in data:
            checked = int(data["features_checked"])
            drifted = int(data["features_drifted"])
            raw_fraction = drifted / checked if checked > 0 else 0.0
        else:
            return MetricReading(
                "drift_rate",
                None,
                100.0,
                str(drift_report),
                "drift_fraction key not found; assuming no drift",
            )
        raw_pct = raw_fraction * 100.0
        # Invert: high drift → low quality score
        score = max(0.0, 100.0 - raw_pct)
        return MetricReading("drift_rate", raw_pct, score, str(drift_report))
    except Exception as exc:
        notes = f"Parse error: {exc}"
        logger.warning("collect_drift_rate: %s", notes)
        return MetricReading("drift_rate", None, 100.0, str(drift_report), notes)


def collect_cycle_time(
    repo_root: Path = Path("."),
    skip_missing: bool = False,
) -> MetricReading:
    """Estimate engineering cycle time from git log.

    Computes the average time between the last 10 merge commits on the
    default branch (main/master).  Returns hours.

    A cycle time ≤ TARGET_CYCLE_TIME_HOURS scores 100;
    ≥ WORST_CYCLE_TIME_HOURS scores 0; linear interpolation in between.
    """
    try:
        result = subprocess.run(
            ["git", "log", "--merges", "--format=%ct", "-20"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=15,
        )
        timestamps = [int(t.strip()) for t in result.stdout.strip().splitlines() if t.strip()]
    except Exception as exc:
        notes = f"git log failed: {exc}"
        logger.info("collect_cycle_time: %s", notes)
        if skip_missing:
            return MetricReading("cycle_time", None, 100.0, "git log", "skipped")
        return MetricReading("cycle_time", None, 100.0, "git log", notes)

    if len(timestamps) < 2:
        return MetricReading(
            "cycle_time",
            None,
            100.0,
            "git log",
            "Fewer than 2 merge commits found; defaulting to score 100",
        )

    diffs_hours = [(timestamps[i] - timestamps[i + 1]) / 3600.0 for i in range(len(timestamps) - 1)]
    avg_hours = sum(diffs_hours) / len(diffs_hours)
    # Clamp
    avg_hours = max(0.0, avg_hours)

    if avg_hours <= TARGET_CYCLE_TIME_HOURS:
        score = 100.0
    elif avg_hours >= WORST_CYCLE_TIME_HOURS:
        score = 0.0
    else:
        span = WORST_CYCLE_TIME_HOURS - TARGET_CYCLE_TIME_HOURS
        score = 100.0 * (WORST_CYCLE_TIME_HOURS - avg_hours) / span

    return MetricReading("cycle_time", avg_hours, score, "git log")


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_composite_score(
    readings: list[MetricReading],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute the weighted composite quality score (0–100)."""
    effective_weights = weights or DEFAULT_WEIGHTS
    total_weight = 0.0
    weighted_sum = 0.0
    for reading in readings:
        w = effective_weights.get(reading.name, 0.0)
        weighted_sum += reading.score * w
        total_weight += w
    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    composite_score: float
    passed_threshold: bool
    threshold: float
    metrics: list[MetricReading]
    weights: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_score": round(self.composite_score, 2),
            "passed_threshold": self.passed_threshold,
            "threshold": self.threshold,
            "metrics": [
                {
                    "name": m.name,
                    "raw_value": round(m.raw_value, 2) if m.raw_value is not None else None,
                    "score": round(m.score, 2),
                    "source": m.source,
                    "notes": m.notes or None,
                }
                for m in self.metrics
            ],
            "weights": self.weights,
        }

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  LedgerLens Engineering Quality Scorecard",
            "=" * 60,
        ]
        for m in self.metrics:
            raw_str = f"{m.raw_value:.1f}" if m.raw_value is not None else "N/A"
            lines.append(
                f"  {m.name:<25} raw={raw_str:<8} score={m.score:>5.1f}/100"
                + (f"  [{m.notes}]" if m.notes else "")
            )
        lines += [
            "-" * 60,
            f"  Composite score: {self.composite_score:.1f}/100",
            f"  Threshold:       {self.threshold:.1f}",
            f"  Result:          {'PASS ✓' if self.passed_threshold else 'FAIL ✗'}",
            "=" * 60,
        ]
        return "\n".join(lines)


def build_badge_json(report: QualityReport) -> dict[str, str]:
    """Return a Shields.io endpoint JSON for the quality badge."""
    score = report.composite_score
    if score >= 90:
        colour = "brightgreen"
        label_val = f"{score:.0f}% excellent"
    elif score >= 75:
        colour = "green"
        label_val = f"{score:.0f}% good"
    elif score >= 60:
        colour = "yellow"
        label_val = f"{score:.0f}% fair"
    else:
        colour = "red"
        label_val = f"{score:.0f}% poor"

    return {
        "schemaVersion": 1,
        "label": "quality score",
        "message": label_val,
        "color": colour,
    }


def report_to_csv(report: QualityReport) -> str:
    """Convert a QualityReport to CSV format.

    Returns a CSV string with one row per metric, including columns for
    metric name, raw value, score, source, and notes.
    """
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["metric_name", "raw_value", "score", "source", "notes"],
    )
    writer.writeheader()

    for metric in report.metrics:
        writer.writerow({
            "metric_name": metric.name,
            "raw_value": f"{metric.raw_value:.2f}" if metric.raw_value is not None else "",
            "score": f"{metric.score:.2f}",
            "source": metric.source,
            "notes": metric.notes or "",
        })

    return output.getvalue()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect and report LedgerLens engineering quality metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit codes:
  0  — Composite score meets or exceeds threshold.
  1  — Composite score is below the threshold.
  2  — Input error.

Example:
  pytest --junitxml=reports/junit.xml --cov --cov-report=xml:reports/coverage.xml
  python -m scripts.quality_metrics --badge --threshold 70
  python -m scripts.quality_metrics --output-format csv --output metrics.csv
        """,
    )
    parser.add_argument("--junit", default=str(DEFAULT_JUNIT))
    parser.add_argument("--coverage", default=str(DEFAULT_COVERAGE))
    parser.add_argument("--mutmut-cache", default=str(DEFAULT_MUTMUT_CACHE))
    parser.add_argument("--drift-report", default=str(DEFAULT_DRIFT_REPORT))
    parser.add_argument("--taxonomy", default=str(DEFAULT_TAXONOMY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--output-format",
        choices=["json", "csv"],
        default="json",
        help="Output format (json or csv)",
    )
    parser.add_argument("--badge", action="store_true", help="Write quality_badge.json")
    parser.add_argument("--badge-output", default=str(DEFAULT_BADGE))
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum composite score to pass (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="JSON string overriding metric weights, e.g. '{\"mutation_score\": 0.4}'",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Give neutral scores (100) to metrics whose source files are absent",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            overrides = json.loads(args.weights)
            weights.update(overrides)
        except json.JSONDecodeError as exc:
            print(f"ERROR: --weights is not valid JSON: {exc}", file=sys.stderr)
            return 2

    skip = args.skip_missing

    readings = [
        collect_test_coverage(Path(args.coverage), skip),
        collect_mutation_score(Path(args.mutmut_cache), skip),
        collect_critical_pass_rate(Path(args.junit), Path(args.taxonomy), skip),
        collect_drift_rate(Path(args.drift_report), skip),
        collect_cycle_time(skip_missing=skip),
    ]

    composite = compute_composite_score(readings, weights)
    passed = composite >= args.threshold

    report = QualityReport(
        composite_score=composite,
        passed_threshold=passed,
        threshold=args.threshold,
        metrics=readings,
        weights=weights,
    )

    print(report.summary())

    # Write report in requested format
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "csv":
        with open(output_path, "w", newline="") as fh:
            fh.write(report_to_csv(report))
    else:
        # Default JSON format
        with open(output_path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)

    logger.info("Quality metrics written to %s", output_path)

    # Write badge JSON
    if args.badge:
        badge_path = Path(args.badge_output)
        badge_path.parent.mkdir(parents=True, exist_ok=True)
        with open(badge_path, "w") as fh:
            json.dump(build_badge_json(report), fh, indent=2)
        logger.info("CI badge JSON written to %s", badge_path)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
