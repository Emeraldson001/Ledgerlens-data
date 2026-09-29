"""CLI to list registered data connectors and their config health.

Discovers every built-in connector plus any installed out-of-tree plugins
(entry point group ``ledgerlens.connectors``) and reports, per connector:
id, source, record type, and whether ``validate_config()`` currently
passes — so a contributor can tell *why* a connector isn't usable without
reading source or triggering a network call.

Usage:
    python -m scripts.list_connectors
    python -m scripts.list_connectors --json
"""

import argparse
import json
import sys

from ingestion.connectors import registry


def collect_rows() -> list[dict]:
    rows = []
    for metadata in registry.list_metadata():
        connector_cls = registry.get(metadata.connector_id)
        instance = connector_cls()
        health = instance.health_check()
        rows.append(
            {
                "connector_id": metadata.connector_id,
                "source": metadata.source,
                "record_type": metadata.record_type.__name__,
                "description": metadata.description,
                "required_env": list(metadata.required_env),
                "ok": health.ok,
                "detail": health.detail,
            }
        )
    return sorted(rows, key=lambda r: r["connector_id"])


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No connectors registered.")
        return

    header = f"{'Connector ID':<28} {'Source':<14} {'Record Type':<16} {'Status':<8} Detail"
    print(header)
    print("-" * len(header))
    for row in rows:
        status = "OK" if row["ok"] else "FAIL"
        print(
            f"{row['connector_id']:<28} {row['source']:<14} {row['record_type']:<16} "
            f"{status:<8} {row['detail']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON instead of a table"
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit with status 1 if any registered connector fails its health check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_rows()

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows)

    if args.fail_on_error and any(not row["ok"] for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
