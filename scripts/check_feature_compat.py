"""Feature Compatibility Check CLI — Issue #532.

Checks whether the current feature pipeline is compatible with one or more
trained model versions, and produces a machine-readable + human-readable
schema diff report.

Can also compare two archived model versions to understand what changed
between training runs (useful during PR review and post-retrain validation).

Usage::

    # Check current pipeline against the active model
    python -m scripts.check_feature_compat

    # Check against an explicit model directory
    python -m scripts.check_feature_compat --model-dir models/

    # Check against a specific archived version
    python -m scripts.check_feature_compat \\
        --model-dir models/archive/20240601T120000Z

    # Compare two archived versions
    python -m scripts.check_feature_compat \\
        --model-dir models/archive/20240601T120000Z \\
        --compare-with models/archive/20240801T120000Z

    # Diff ALL archived versions in sequence
    python -m scripts.check_feature_compat --diff-all

    # CI mode: exit 0 only if schema matches exactly, exit 1 on warnings, exit 2 on errors
    python -m scripts.check_feature_compat --strict --ci

Exit codes:
    0 — fully compatible (no errors or warnings)
    1 — compatible with warnings (extra/unused features present)
    2 — incompatible (missing required features) or fatal error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from detection.feature_compat import (
    FeatureCompatibilityChecker,
    check_current_pipeline,
    diff_model_versions,
    load_metadata,
)
from utils.logging import get_logger

logger = get_logger(__name__)

MODELS_DIR = Path("models")
ARCHIVE_DIR = MODELS_DIR / "archive"
REPORTS_DIR = Path("reports/feature_compat")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    """Pretty-print a compatibility report to stdout."""
    compatible = report.get("is_compatible", False)
    status = "✓ COMPATIBLE" if compatible else "✗ INCOMPATIBLE"
    print(f"\n{'=' * 60}")
    print(f"  {status}")
    print(f"  {report['source_label']} → {report['target_label']}")
    print(f"{'=' * 60}")
    print(
        f"  Features: {report['n_source_features']} source, "
        f"{report['n_target_features']} target, "
        f"{report['n_common_features']} common"
    )
    print(f"  Issues: {report['error_count']} error(s), {report['warning_count']} warning(s)")

    added = report.get("features_added_in_target", [])
    removed = report.get("features_removed_from_target", [])
    if added:
        print(f"\n  + Features added in target ({len(added)}):")
        for f in added:
            print(f"      + {f}")
    if removed:
        print(f"\n  - Features removed from target ({len(removed)}):")
        for f in removed:
            print(f"      - {f}")

    issues = report.get("issues", [])
    errors = [i for i in issues if i["severity"] == "error"]
    warnings = [i for i in issues if i["severity"] == "warning"]

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for i in errors:
            feat = f" [{i['feature']}]" if i["feature"] else ""
            print(f"    ✗ [{i['code']}]{feat}: {i['message']}")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for i in warnings[:10]:
            feat = f" [{i['feature']}]" if i["feature"] else ""
            print(f"    ⚠ [{i['code']}]{feat}: {i['message']}")
        if len(warnings) > 10:
            print(f"    … and {len(warnings) - 10} more warning(s)")

    print()


def _print_diff(diff: dict[str, Any]) -> None:
    """Print a multi-version diff summary."""
    print(f"\n{'=' * 60}")
    print("  MULTI-VERSION SCHEMA DIFF")
    print(f"  Versions: {' → '.join(diff['versions'])}")
    print(f"{'=' * 60}")

    for pw in diff.get("pairwise_diffs", []):
        if "error" in pw:
            print(f"\n  {pw['source_label']} → {pw['target_label']}: ERROR — {pw['error']}")
            continue
        compat = "✓" if pw.get("is_compatible") else "✗"
        print(
            f"\n  {compat} {pw['source_label']} → {pw['target_label']}: "
            f"{pw['error_count']} error(s), {pw['warning_count']} warning(s)"
        )
        added = pw.get("features_added_in_target", [])
        removed = pw.get("features_removed_from_target", [])
        if added:
            print(f"      Added:   {', '.join(added[:5])}" + ("…" if len(added) > 5 else ""))
        if removed:
            print(f"      Removed: {', '.join(removed[:5])}" + ("…" if len(removed) > 5 else ""))

    # Timeline for features that changed
    timeline = diff.get("feature_timeline", {})
    changed_features = {
        feat: states for feat, states in timeline.items() if len(set(states.values())) > 1
    }
    if changed_features:
        print(f"\n  Features that changed across versions ({len(changed_features)}):")
        for feat, states in sorted(changed_features.items()):
            state_str = " | ".join(f"{v}:{s}" for v, s in states.items())
            print(f"    {feat}: {state_str}")


# ---------------------------------------------------------------------------
# Archive helpers
# ---------------------------------------------------------------------------


def _load_all_archive_versions() -> list[tuple[str, dict[str, Any]]]:
    """Load all model_metadata.json files from models/archive/, sorted chronologically."""
    if not ARCHIVE_DIR.exists():
        return []
    results: list[tuple[str, dict[str, Any]]] = []
    for entry in sorted(ARCHIVE_DIR.iterdir()):
        if not entry.is_dir():
            continue
        meta_path = entry / "model_metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            results.append((entry.name, meta))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load %s: %s", meta_path, exc)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check feature compatibility across LedgerLens model versions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-dir",
        default=str(MODELS_DIR),
        help=f"Model directory to check against (default: {MODELS_DIR}).",
    )
    p.add_argument(
        "--compare-with",
        default=None,
        help="Second model directory to compare with --model-dir (version diff mode).",
    )
    p.add_argument(
        "--diff-all",
        action="store_true",
        help="Diff all archived model versions in sequence.",
    )
    p.add_argument(
        "--features-file",
        default=None,
        help="JSON file listing the source feature columns. "
        "If omitted, the current pipeline features are inferred automatically.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat extra features in source as errors (not just warnings).",
    )
    p.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit 2 on any error, exit 1 on warnings, exit 0 on full match.",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPORTS_DIR),
        help=f"Directory for report files (default: {REPORTS_DIR}).",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Skip writing report files (print to stdout only).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)

    # ------------------------------------------------------------------
    # Load source features if provided
    # ------------------------------------------------------------------
    pipeline_features: list[str] | None = None
    if args.features_file:
        try:
            pipeline_features = json.loads(Path(args.features_file).read_text())
            if not isinstance(pipeline_features, list):
                logger.error("--features-file must contain a JSON array of strings")
                return 2
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.error("Cannot read features file: %s", exc)
            return 2

    # ------------------------------------------------------------------
    # Mode: diff all archived versions
    # ------------------------------------------------------------------
    if args.diff_all:
        versions = _load_all_archive_versions()
        # Also include the active model
        try:
            active_meta = load_metadata(args.model_dir)
            versions.append(("active", active_meta))
        except FileNotFoundError:
            logger.warning("Active model metadata not found at %s", args.model_dir)

        if len(versions) < 2:
            print("[compat] Fewer than 2 versions found — nothing to diff.")
            return 0

        diff = diff_model_versions(versions)
        _print_diff(diff.to_dict())

        if not args.no_report:
            output_dir.mkdir(parents=True, exist_ok=True)
            from datetime import UTC, datetime

            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            out_path = output_dir / f"feature_compat_diff_all_{ts}.json"
            out_path.write_text(json.dumps(diff.to_dict(), indent=2))
            print(f"\n[compat] Report written → {out_path}")

        any_error = any(
            pw.get("error_count", 0) > 0 for pw in diff.to_dict().get("pairwise_diffs", [])
        )
        return 2 if any_error else 0

    # ------------------------------------------------------------------
    # Mode: compare two specific versions
    # ------------------------------------------------------------------
    if args.compare_with:
        try:
            a_meta = load_metadata(args.model_dir)
        except FileNotFoundError as exc:
            logger.error("Cannot load model dir %s: %s", args.model_dir, exc)
            return 2
        try:
            b_meta = load_metadata(args.compare_with)
        except FileNotFoundError as exc:
            logger.error("Cannot load --compare-with %s: %s", args.compare_with, exc)
            return 2

        a_label = Path(args.model_dir).name or args.model_dir
        b_label = Path(args.compare_with).name or args.compare_with

        diff = diff_model_versions([(a_label, a_meta), (b_label, b_meta)])
        _print_diff(diff.to_dict())

        if not args.no_report:
            output_dir.mkdir(parents=True, exist_ok=True)
            from datetime import UTC, datetime

            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            out_path = output_dir / f"feature_compat_{a_label}_vs_{b_label}_{ts}.json"
            out_path.write_text(json.dumps(diff.to_dict(), indent=2))
            print(f"\n[compat] Report written → {out_path}")

        any_error = any(
            pw.get("error_count", 0) > 0 for pw in diff.to_dict().get("pairwise_diffs", [])
        )
        return 2 if any_error else 0

    # ------------------------------------------------------------------
    # Mode: check current pipeline against a single model version (default)
    # ------------------------------------------------------------------
    try:
        report = check_current_pipeline(
            model_dir=args.model_dir,
            pipeline_features=pipeline_features,
        )
        if args.strict:
            # Re-run with strict=True
            target_meta = load_metadata(args.model_dir)
            src_feats = pipeline_features or report.source_features
            checker = FeatureCompatibilityChecker(
                target_metadata=target_meta,
                source_features=src_feats,
                source_label="current_pipeline",
                target_label=Path(args.model_dir).name or args.model_dir,
                strict=True,
            )
            report = checker.check()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    report_dict = report.to_dict()
    _print_report(report_dict)

    if not args.no_report:
        output_dir.mkdir(parents=True, exist_ok=True)
        from datetime import UTC, datetime

        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_path = output_dir / f"feature_compat_{ts}.json"
        out_path.write_text(json.dumps(report_dict, indent=2))
        print(f"[compat] Report written → {out_path}")

    if args.ci:
        if report.error_count > 0:
            return 2
        if report.warning_count > 0:
            return 1
        return 0

    return 2 if not report.is_compatible else 0


if __name__ == "__main__":
    sys.exit(main())
