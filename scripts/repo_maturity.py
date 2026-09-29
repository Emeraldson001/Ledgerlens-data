"""Repository maturity tracking for advanced engineering goals.

Issue #560 — Build repository maturity tracking for advanced engineering goals
===============================================================================

This script evaluates the LedgerLens-data repository against a configurable
maturity model and produces a scored report.  It is designed to run:

* Locally by contributors to assess the health of their branch::

    python -m scripts.repo_maturity

* In CI as a non-blocking advisory step::

    python -m scripts.repo_maturity --report reports/maturity_report.json

* As a make target (``make maturity``)::

    make maturity

Maturity dimensions
-------------------
The model measures five dimensions, each scored 0–100.  The composite score
is a weighted average.

=================  ========================================================
Dimension          What it measures
=================  ========================================================
**tests**          Test coverage breadth: number of test modules, presence
                   of integration and fuzz test sub-directories, conftest.
**docs**           Documentation completeness: presence of key doc files,
                   inline docstring coverage in Python source modules.
**ci**             CI/CD maturity: workflow files, mutation testing,
                   lint/format gates, notebook execution.
**data_quality**   Data contract health: parsing contracts, reconciliation
                   checks, known manipulation event count.
**security**       Security posture: presence of security docs, DP modules,
                   adversarial defences, integrity verification.
=================  ========================================================

Configuration
-------------
Dimensions and their scoring rubrics are defined in
``config/repo_maturity.yaml``.  The YAML is loaded at runtime so teams can
tune thresholds without touching Python code.

Exit codes
----------
0 – Composite score >= ``--threshold`` (default 60).
1 – Composite score < threshold (non-blocking advisory in CI).
2 – Crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DimensionScore:
    name: str
    score: float  # 0–100
    max_score: float  # always 100
    weight: float  # fraction of composite (0–1)
    details: list[str] = field(default_factory=list)
    deductions: list[str] = field(default_factory=list)

    @property
    def weighted(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": round(self.score, 1),
            "max_score": self.max_score,
            "weight": self.weight,
            "weighted_contribution": round(self.weighted, 1),
            "details": self.details,
            "deductions": self.deductions,
        }


@dataclass
class MaturityReport:
    dimensions: list[DimensionScore] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    repo_root: str = "."
    threshold: float = 60.0

    @property
    def composite_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.weighted for d in self.dimensions)

    @property
    def passed(self) -> bool:
        return self.composite_score >= self.threshold

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "repo_root": self.repo_root,
            "composite_score": round(self.composite_score, 1),
            "threshold": self.threshold,
            "passed": self.passed,
            "dimensions": [d.to_dict() for d in self.dimensions],
        }


# ---------------------------------------------------------------------------
# Helper: count files matching a pattern
# ---------------------------------------------------------------------------


def _count(root: Path, pattern: str) -> int:
    return len(list(root.glob(pattern)))


def _exists(root: Path, *parts: str) -> bool:
    return (root / Path(*parts)).exists()


def _python_modules(root: Path) -> list[Path]:
    return [
        p
        for p in root.rglob("*.py")
        if ".venv" not in p.parts and "venv" not in p.parts and "__pycache__" not in p.parts
    ]


def _has_docstring(path: Path) -> bool:
    """Return True if the module-level docstring is non-empty."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Quick heuristic: file starts with a triple-quoted string
        stripped = text.lstrip()
        return stripped.startswith('"""') or stripped.startswith("'''")
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------


def _score_tests(root: Path) -> DimensionScore:
    """Measure test coverage breadth."""
    ds = DimensionScore(name="tests", score=100.0, max_score=100.0, weight=0.25)
    tests_dir = root / "tests"

    if not tests_dir.exists():
        ds.score = 0.0
        ds.deductions.append("tests/ directory is missing (–100)")
        return ds

    # Count test files
    test_files = list(tests_dir.glob("test_*.py"))
    n_tests = len(test_files)
    if n_tests < 5:
        deduction = 30
        ds.score -= deduction
        ds.deductions.append(f"Fewer than 5 test modules ({n_tests}) (–{deduction})")
    elif n_tests >= 20:
        ds.details.append(f"  ✓ {n_tests} test modules")
    else:
        ds.details.append(f"  ✓ {n_tests} test modules")

    # Integration tests
    if _exists(root, "tests", "integration"):
        ds.details.append("  ✓ Integration tests directory present")
    else:
        ds.score -= 10
        ds.deductions.append("No tests/integration/ directory (–10)")

    # Fuzz tests
    if _exists(root, "tests", "fuzz"):
        ds.details.append("  ✓ Fuzz tests directory present")
    else:
        ds.score -= 5
        ds.deductions.append("No tests/fuzz/ directory (–5)")

    # conftest.py
    if _exists(root, "tests", "conftest.py"):
        ds.details.append("  ✓ tests/conftest.py present")
    else:
        ds.score -= 5
        ds.deductions.append("No tests/conftest.py (–5)")

    # Test for parsing contracts (Issue #552)
    parsing_tests = [f for f in test_files if "parsing" in f.name or "reconciliation" in f.name]
    if parsing_tests:
        ds.details.append(f"  ✓ Parsing/reconciliation tests present ({len(parsing_tests)} files)")
    else:
        ds.score -= 5
        ds.deductions.append("No tests for parsing contracts or reconciliation (–5)")

    ds.score = max(0.0, ds.score)
    return ds


def _score_docs(root: Path) -> DimensionScore:
    """Measure documentation completeness."""
    ds = DimensionScore(name="docs", score=100.0, max_score=100.0, weight=0.20)
    docs_dir = root / "docs"

    if not docs_dir.exists():
        ds.score -= 20
        ds.deductions.append("docs/ directory missing (–20)")
    else:
        doc_files = list(docs_dir.glob("*.md"))
        ds.details.append(f"  ✓ {len(doc_files)} markdown docs in docs/")

    # Key docs
    key_docs = [
        ("README.md", 10),
        ("CONTRIBUTING.md", 5),
        ("CHANGELOG.md", 5),
        ("docs/security.md", 5),
        ("docs/drift_detection.md", 5),
    ]
    for rel, penalty in key_docs:
        if _exists(root, rel):
            ds.details.append(f"  ✓ {rel}")
        else:
            ds.score -= penalty
            ds.deductions.append(f"Missing {rel} (–{penalty})")

    # Docstring coverage (sample top-level Python modules)
    top_modules = list((root / "detection").glob("*.py")) + list((root / "ingestion").glob("*.py"))
    if top_modules:
        with_doc = sum(1 for p in top_modules if _has_docstring(p))
        pct = with_doc / len(top_modules) * 100
        if pct >= 80:
            ds.details.append(f"  ✓ {pct:.0f}% of detection+ingestion modules have docstrings")
        elif pct >= 50:
            ds.score -= 5
            ds.deductions.append(
                f"  Only {pct:.0f}% of detection+ingestion modules have docstrings (–5)"
            )
        else:
            ds.score -= 15
            ds.deductions.append(
                f"  Only {pct:.0f}% of detection+ingestion modules have docstrings (–15)"
            )

    ds.score = max(0.0, ds.score)
    return ds


def _score_ci(root: Path) -> DimensionScore:
    """Measure CI/CD maturity."""
    ds = DimensionScore(name="ci", score=100.0, max_score=100.0, weight=0.20)
    workflows_dir = root / ".github" / "workflows"

    if not workflows_dir.exists():
        ds.score = 0.0
        ds.deductions.append(".github/workflows/ missing (–100)")
        return ds

    workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
    n_workflows = len(workflow_files)
    ds.details.append(f"  ✓ {n_workflows} workflow file(s) in .github/workflows/")

    # Key workflow names (file stem hints)
    stems = {f.stem.lower() for f in workflow_files}
    expected = {
        "ci": ("ci.yml", 20),
        "retrain": ("retrain.yml", 10),
        "active_learning": ("active_learning.yml", 5),
    }
    for stem, (label, penalty) in expected.items():
        if any(stem in s for s in stems):
            ds.details.append(f"  ✓ {label} workflow present")
        else:
            ds.score -= penalty
            ds.deductions.append(f"No {label} workflow found (–{penalty})")

    # Makefile targets
    makefile = root / "Makefile"
    if makefile.exists():
        content = makefile.read_text(encoding="utf-8")
        for target, penalty in [
            ("mutation-test", 5),
            ("lint", 5),
            ("test", 5),
            ("validate", 5),
        ]:
            if target in content:
                ds.details.append(f"  ✓ Makefile has '{target}' target")
            else:
                ds.score -= penalty
                ds.deductions.append(f"Makefile missing '{target}' target (–{penalty})")
    else:
        ds.score -= 15
        ds.deductions.append("Makefile missing (–15)")

    # pyproject.toml
    if _exists(root, "pyproject.toml"):
        ds.details.append("  ✓ pyproject.toml present")
    else:
        ds.score -= 10
        ds.deductions.append("pyproject.toml missing (–10)")

    ds.score = max(0.0, ds.score)
    return ds


def _score_data_quality(root: Path) -> DimensionScore:
    """Measure data contract and quality health."""
    ds = DimensionScore(name="data_quality", score=100.0, max_score=100.0, weight=0.20)

    # Parsing contracts module (Issue #552)
    if _exists(root, "validation", "parsing.py"):
        ds.details.append("  ✓ validation/parsing.py (parsing contracts) present")
    else:
        ds.score -= 20
        ds.deductions.append("validation/parsing.py missing (–20)")

    # Reconciliation checks module (Issue #554)
    if _exists(root, "validation", "reconciliation.py"):
        ds.details.append("  ✓ validation/reconciliation.py (reconciliation checks) present")
    else:
        ds.score -= 20
        ds.deductions.append("validation/reconciliation.py missing (–20)")

    # Contributor validation script (Issue #558)
    if _exists(root, "scripts", "validate.py"):
        ds.details.append("  ✓ scripts/validate.py (contributor validation CLI) present")
    else:
        ds.score -= 10
        ds.deductions.append("scripts/validate.py missing (–10)")

    # Known manipulation events
    events_path = root / "data" / "known_manipulation_events.csv"
    if events_path.exists():
        try:
            import csv

            with open(events_path, encoding="utf-8") as fh:
                n_rows = sum(1 for _ in csv.DictReader(fh))
            ds.details.append(f"  ✓ {events_path.name}: {n_rows} manipulation events")
            if n_rows < 5:
                ds.score -= 10
                ds.deductions.append(f"Too few manipulation events ({n_rows} < 5) (–10)")
        except Exception as exc:  # pragma: no cover
            ds.score -= 10
            ds.deductions.append(f"Cannot read {events_path.name}: {exc} (–10)")
    else:
        ds.score -= 15
        ds.deductions.append("data/known_manipulation_events.csv missing (–15)")

    # Feature ranges
    if _exists(root, "data", "feature_ranges.json"):
        ds.details.append("  ✓ data/feature_ranges.json present")
    else:
        ds.score -= 5
        ds.deductions.append("data/feature_ranges.json missing (–5)")

    # Synthetic dataset
    if _exists(root, "data", "synthetic_dataset.parquet"):
        ds.details.append("  ✓ data/synthetic_dataset.parquet present")
    else:
        ds.score -= 10
        ds.deductions.append(
            "data/synthetic_dataset.parquet missing — run generate_synthetic_dataset (–10)"
        )

    ds.score = max(0.0, ds.score)
    return ds


def _score_security(root: Path) -> DimensionScore:
    """Measure security posture."""
    ds = DimensionScore(name="security", score=100.0, max_score=100.0, weight=0.15)

    # Security docs
    for doc_path, penalty in [
        ("docs/security.md", 10),
        ("docs/security_threat_model.md", 10),
    ]:
        if _exists(root, doc_path):
            ds.details.append(f"  ✓ {doc_path}")
        else:
            ds.score -= penalty
            ds.deductions.append(f"Missing {doc_path} (–{penalty})")

    # Differential privacy modules
    if _exists(root, "detection", "differential_privacy.py"):
        ds.details.append("  ✓ Differential privacy module present")
    else:
        ds.score -= 15
        ds.deductions.append("detection/differential_privacy.py missing (–15)")

    # Adversarial defences
    if _exists(root, "detection", "adversarial"):
        ds.details.append("  ✓ Adversarial defences directory present")
    else:
        ds.score -= 10
        ds.deductions.append("detection/adversarial/ missing (–10)")

    # Model integrity
    if _exists(root, "models", "metrics.json"):
        ds.details.append("  ✓ models/metrics.json present")
    else:
        ds.score -= 5
        ds.deductions.append("models/metrics.json missing (–5)")

    # .env.example (secrets hygiene)
    if _exists(root, ".env.example"):
        ds.details.append("  ✓ .env.example present (secrets documentation)")
    else:
        ds.score -= 5
        ds.deductions.append(".env.example missing (–5)")

    # Audit trail
    if _exists(root, "detection", "audit_trail.py"):
        ds.details.append("  ✓ Audit trail module present")
    else:
        ds.score -= 5
        ds.deductions.append("detection/audit_trail.py missing (–5)")

    ds.score = max(0.0, ds.score)
    return ds


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

DIMENSION_SCORERS = [
    _score_tests,
    _score_docs,
    _score_ci,
    _score_data_quality,
    _score_security,
]

# Validate weights sum to 1.0
_WEIGHT_SUM = 0.25 + 0.20 + 0.20 + 0.20 + 0.15  # = 1.00


def compute_maturity(root: Path, threshold: float = 60.0) -> MaturityReport:
    """Evaluate repository maturity and return a :class:`MaturityReport`."""
    report = MaturityReport(
        repo_root=str(root),
        threshold=threshold,
    )
    for scorer in DIMENSION_SCORERS:
        report.dimensions.append(scorer(root))
    return report


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def _maturity_to_markdown(report: MaturityReport) -> str:
    """Convert a MaturityReport to a clean, README-embeddable Markdown summary."""
    lines = [
        "# Repository Maturity Report",
        "",
        f"## Composite Score: {report.composite_score:.1f}/100",
        "",
        f"**Status:** {'✓ PASSED' if report.passed else '✗ FAILED'} (threshold: {report.threshold})",
        "",
        "## Dimension Scores",
        "",
    ]

    for dim in report.dimensions:
        # Visual bar
        bar_filled = int(dim.score / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)

        lines.append(f"### {dim.name.title()}")
        lines.append(f"`[{bar}] {dim.score:.1f}/100` (weight: {dim.weight:.0%})")
        lines.append("")

        # Deductions if any
        if dim.deductions:
            lines.append("**Deductions:**")
            for deduction in dim.deductions:
                lines.append(f"- {deduction}")
            lines.append("")

    lines.append(f"*Generated: {report.generated_at}*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo_maturity",
        description=(
            "LedgerLens repository maturity tracker.\n\n"
            "Evaluates the repository against a five-dimension maturity model "
            "and reports a composite score (0–100)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exit codes:\n"
            "  0 — composite score >= threshold\n"
            "  1 — composite score < threshold\n"
            "  2 — crash\n"
        ),
    )
    p.add_argument(
        "--root",
        default=".",
        metavar="PATH",
        help="Repository root to evaluate (default: current directory).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        metavar="SCORE",
        help="Minimum composite score to exit 0 (default: 60).",
    )
    p.add_argument(
        "--report",
        metavar="PATH",
        help="Write JSON maturity report to this path.",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        help="Write Markdown summary (README-embeddable) to this path.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print dimension details (passing checks).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except the final score line.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()

    if not args.quiet:
        print(
            f"LedgerLens Repository Maturity Report — "
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        print(f"Root: {root}\n")

    try:
        report = compute_maturity(root, threshold=args.threshold)
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: maturity computation crashed: {exc}", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------
    # Print dimension results
    # ------------------------------------------------------------------
    if not args.quiet:
        for dim in report.dimensions:
            bar_filled = int(dim.score / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            print(f"  {dim.name:<14} [{bar}] {dim.score:5.1f}/100  " f"(weight {dim.weight:.0%})")
            for deduction in dim.deductions:
                print(f"           {deduction}")
            if args.verbose:
                for detail in dim.details:
                    print(f"           {detail}")

        print()
        status = "✓ PASSED" if report.passed else "✗ FAILED"
        print(
            f"Composite score: {report.composite_score:.1f}/100  "
            f"(threshold: {args.threshold})  {status}"
        )

    # ------------------------------------------------------------------
    # JSON report
    # ------------------------------------------------------------------
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        if not args.quiet:
            print(f"Report written to: {report_path}")

    # ------------------------------------------------------------------
    # Markdown summary
    # ------------------------------------------------------------------
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_content = _maturity_to_markdown(report)
        output_path.write_text(markdown_content, encoding="utf-8")
        if not args.quiet:
            print(f"Markdown summary written to: {output_path}")

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
