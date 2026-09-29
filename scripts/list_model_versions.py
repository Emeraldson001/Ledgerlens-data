"""CLI to list archived model versions with training dates and metrics.

Usage:
    python -m scripts.list_model_versions
    python -m scripts.list_model_versions --max-rows 20
"""

import argparse
import json
import os
from datetime import UTC, datetime

ARCHIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "archive"
)

# Versions whose metadata is missing or carries an unparseable ``trained_at``
# sort to the very end under the newest-first ordering rather than crashing.
_UNKNOWN_TRAINED_AT = datetime.min.replace(tzinfo=UTC)


def load_version_metadata(version_dir: str) -> dict:
    """Load metrics.json and model_metadata.json from an archived version."""
    metrics_path = os.path.join(version_dir, "metrics.json")
    metadata_path = os.path.join(version_dir, "model_metadata.json")

    data = {"version": os.path.basename(version_dir)}

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        data["trained_at"] = meta.get("trained_at", "unknown")
        data["n_training_rows"] = meta.get("n_training_rows", "?")
        data["n_test_rows"] = meta.get("n_test_rows", "?")

    return data


def parse_trained_at(value: object) -> datetime:
    """Parse a ``trained_at`` metadata value into a timezone-aware datetime.

    Accepts ISO-8601 strings, including a trailing ``Z`` (as documented in the
    README's ``model_metadata.json`` schema). Naive timestamps are assumed to
    be UTC. Missing (``None``), ``"unknown"``, or unparseable values return
    :data:`_UNKNOWN_TRAINED_AT` so they sort last instead of raising.
    """
    if not isinstance(value, str):
        return _UNKNOWN_TRAINED_AT

    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return _UNKNOWN_TRAINED_AT

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def list_versions(archive_dir: str = ARCHIVE_DIR) -> list[dict]:
    """Return archived model versions ordered by training date, newest first.

    Ordering is by the ``trained_at`` timestamp parsed from each version's
    ``model_metadata.json`` — not by directory name. A lexicographic directory
    sort only coincidentally matches chronological order for zero-padded
    timestamp names and would silently break if the archive naming scheme ever
    changed. The directory name is used only as a stable tie-breaker (and for
    versions with unknown training dates).
    """
    if not os.path.isdir(archive_dir):
        return []

    versions = [
        load_version_metadata(os.path.join(archive_dir, entry))
        for entry in os.listdir(archive_dir)
        if os.path.isdir(os.path.join(archive_dir, entry))
    ]

    versions.sort(key=lambda v: v["version"])
    versions.sort(key=lambda v: parse_trained_at(v.get("trained_at")), reverse=True)
    return versions


def print_table(versions: list[dict], max_rows: int | None = None) -> None:
    if not versions:
        print("No archived model versions found.")
        return

    if max_rows is not None:
        versions = versions[:max_rows]

    header = (
        f"{'Version':<20} {'Trained At':<28} {'Rows (train/test)':<20} {'AUC-ROC':<20} {'F1':<20}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for v in versions:
        version = v["version"]
        trained_at = v.get("trained_at", "unknown")
        n_train = v.get("n_training_rows", "?")
        n_test = v.get("n_test_rows", "?")
        rows_str = f"{n_train}/{n_test}"

        metrics = v.get("metrics", {})
        auc_strs = []
        f1_strs = []
        for model_name in sorted(metrics.keys()):
            m = metrics[model_name]
            auc_strs.append(f"{m.get('auc_roc', '?'):.4f}")
            f1_strs.append(f"{m.get('f1', '?'):.4f}")

        auc_display = ", ".join(auc_strs) if auc_strs else "?"
        f1_display = ", ".join(f1_strs) if f1_strs else "?"

        print(f"{version:<20} {trained_at:<28} {rows_str:<20} {auc_display:<20} {f1_display:<20}")


def print_json(versions: list[dict], max_rows: int | None = None) -> None:
    """Output versions as JSON."""
    if max_rows is not None:
        versions = versions[:max_rows]
    print(json.dumps(versions, indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List archived model versions")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit number of versions shown")
    parser.add_argument(
        "--archive-dir",
        default=ARCHIVE_DIR,
        help="Archive directory (default: models/archive)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format: table (default) or json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    versions = list_versions(args.archive_dir)
    if args.format == "json":
        print_json(versions, max_rows=args.max_rows)
    else:
        print_table(versions, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
