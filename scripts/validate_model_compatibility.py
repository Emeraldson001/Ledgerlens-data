"""Validate feature contracts across model metadata versions.

Examples:
    python -m scripts.validate_model_compatibility \
        --reference models \
        --candidate models/archive/20260720_120000

    python -m scripts.validate_model_compatibility \
        --reference models/model_metadata.json \
        --candidate /tmp/candidate/model_metadata.json \
        --json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from detection.model_compatibility import (
    FeatureContractError,
    validate_feature_compatibility,
)

METADATA_FILENAME = "model_metadata.json"


def _metadata_path(path: str) -> str:
    return os.path.join(path, METADATA_FILENAME) if os.path.isdir(path) else path


def load_metadata(path: str) -> tuple[dict[str, Any], str]:
    """Load a metadata file, accepting either a directory or JSON path."""
    resolved = _metadata_path(path)
    try:
        with open(resolved) as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureContractError(f"{resolved}: cannot load metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise FeatureContractError(f"{resolved}: metadata root must be an object")
    return metadata, resolved


def validate_paths(
    reference_path: str,
    candidate_paths: list[str],
    *,
    allow_additive: bool = False,
) -> list[dict[str, Any]]:
    """Compare one reference metadata contract with each candidate path."""
    reference, resolved_reference = load_metadata(reference_path)
    results: list[dict[str, Any]] = []

    for candidate_path in candidate_paths:
        try:
            candidate, resolved_candidate = load_metadata(candidate_path)
            report = validate_feature_compatibility(
                reference,
                candidate,
                allow_additive=allow_additive,
                reference_source=resolved_reference,
                candidate_source=resolved_candidate,
            )
            result = report.to_dict()
        except FeatureContractError as exc:
            resolved_candidate = _metadata_path(candidate_path)
            result = {
                "compatible": False,
                "status": "invalid_metadata",
                "validation_scope": "none",
                "added_features": [],
                "removed_features": [],
                "reordered_features": [],
                "dtype_changes": {},
                "unchecked_dtypes": [],
                "errors": [str(exc)],
                "diagnostics": [str(exc)],
            }
        result["reference"] = resolved_reference
        result["candidate"] = resolved_candidate
        results.append(result)

    return results


def _print_human(results: list[dict[str, Any]]) -> None:
    for result in results:
        outcome = "PASS" if result["compatible"] else "FAIL"
        print(
            f"{outcome} {result['candidate']} against {result['reference']}: "
            f"{result['status']} ({result['validation_scope']})"
        )
        for diagnostic in result["diagnostics"]:
            print(f"  - {diagnostic}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate model feature compatibility from metadata sidecars"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference model directory or model_metadata.json path",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        action="append",
        help="Candidate model directory or metadata path (repeatable)",
    )
    parser.add_argument(
        "--allow-additive",
        action="store_true",
        help="Permit candidate-only features while still rejecting other changes",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = validate_paths(
            args.reference,
            args.candidate,
            allow_additive=args.allow_additive,
        )
    except FeatureContractError as exc:
        results = [
            {
                "compatible": False,
                "status": "invalid_metadata",
                "validation_scope": "none",
                "reference": _metadata_path(args.reference),
                "candidate": None,
                "added_features": [],
                "removed_features": [],
                "reordered_features": [],
                "dtype_changes": {},
                "unchecked_dtypes": [],
                "errors": [str(exc)],
                "diagnostics": [str(exc)],
            }
        ]

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
    else:
        _print_human(results)
    return 0 if all(result["compatible"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
