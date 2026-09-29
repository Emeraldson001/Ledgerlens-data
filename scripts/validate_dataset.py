"""Run the data quality validation framework over a JSON-lines dataset.

Usage:
    python -m scripts.validate_dataset --input data/some_export.jsonl \\
        --required wallet --required score \\
        --range score:0:100

Prints a summary and exits non-zero if any record fails validation, so it
can be used as a CI/pre-ingest gate.
"""

import argparse
import json
import sys

from utils.data_quality import DataQualityValidator, RangeRule, RequiredFieldRule


def _parse_range(spec: str) -> RangeRule:
    field_name, minimum, maximum = spec.split(":")
    return RangeRule(
        field_name,
        minimum=float(minimum) if minimum else None,
        maximum=float(maximum) if maximum else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to a JSON-lines file")
    parser.add_argument(
        "--required", action="append", default=[], help="Field that must be present (repeatable)"
    )
    parser.add_argument(
        "--range", action="append", default=[], dest="ranges", help="field:min:max (repeatable)"
    )
    parser.add_argument("--feature-ranges", default="data/feature_ranges.json")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    rules: list = [RequiredFieldRule(f) for f in args.required]
    rules += [_parse_range(spec) for spec in args.ranges]
    rules += RangeRule.from_feature_ranges(args.feature_ranges)

    validator = DataQualityValidator(rules)

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    report = validator.validate_batch(records, fail_fast=args.fail_fast)
    print(json.dumps(report.as_dict(), indent=2))

    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
