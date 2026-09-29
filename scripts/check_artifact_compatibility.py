"""Validate that a candidate model artifact is backward compatible with a
previously stored baseline (Issue #510).

Compares ``model_metadata.json``/``metrics.json`` in two model directories
(a baseline, e.g. an entry under ``models/archive/``, and a candidate, e.g.
the current ``models/`` directory produced by the latest training run) using
:func:`detection.artifact_compatibility.check_backward_compatibility`.

Usage:
    python -m scripts.check_artifact_compatibility \\
        --baseline-dir models/archive/2026-06-01 \\
        --candidate-dir models

If *baseline_dir* has no ``model_metadata.json`` (e.g. no version has been
archived yet), the check is skipped and the script exits 0 — there is
nothing to be backward compatible with yet.

Exit codes:
    0  Compatible (or no baseline to compare against).
    1  Breaking incompatibility found.
    2  Candidate directory is missing required artifact files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from detection.artifact_compatibility import (
    DEFAULT_MAX_METRIC_DROP,
    DEFAULT_METRIC_KEY,
    check_backward_compatibility,
)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate backward compatibility of a candidate model artifact "
        "against a previously stored baseline."
    )
    parser.add_argument(
        "--baseline-dir",
        required=True,
        help="Directory containing the previously stored artifact "
        "(e.g. models/archive/<version>).",
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Directory containing the candidate artifact (e.g. models/).",
    )
    parser.add_argument(
        "--max-metric-drop",
        type=float,
        default=DEFAULT_MAX_METRIC_DROP,
        help=f"Maximum allowed metric regression (default: {DEFAULT_MAX_METRIC_DROP}).",
    )
    parser.add_argument(
        "--metric-key",
        default=DEFAULT_METRIC_KEY,
        help=f"Metric name to compare per-model (default: {DEFAULT_METRIC_KEY}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)

    old_metadata = _load_json(baseline_dir / "model_metadata.json")
    if old_metadata is None:
        print(
            f"No baseline artifact found at {baseline_dir} — nothing to compare "
            "against, skipping backward compatibility check."
        )
        return 0

    new_metadata = _load_json(candidate_dir / "model_metadata.json")
    if new_metadata is None:
        print(
            f"ERROR: candidate artifact at {candidate_dir} has no model_metadata.json.",
            file=sys.stderr,
        )
        return 2

    old_metrics = _load_json(baseline_dir / "metrics.json")
    new_metrics = _load_json(candidate_dir / "metrics.json")

    report = check_backward_compatibility(
        old_metadata,
        new_metadata,
        old_metrics,
        new_metrics,
        max_metric_drop=args.max_metric_drop,
        metric_key=args.metric_key,
    )

    print(f"Baseline:  {baseline_dir}")
    print(f"Candidate: {candidate_dir}")
    print(report.format())

    return 0 if report.compatible else 1


if __name__ == "__main__":
    sys.exit(main())
