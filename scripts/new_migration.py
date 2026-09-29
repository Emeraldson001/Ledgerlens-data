"""Scaffold a new migration file under ``migrations/versions/``.

Determines the next sequential ID, creates a stub migration file from a
template, and prints the registry entry you need to add to
``migrations/registry.py``.

Usage::

    python -m scripts.new_migration "Add foo column to risk_scores"
    python -m scripts.new_migration --id 0007 "Backfill missing timestamps"
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_VERSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "migrations", "versions")

_TEMPLATE = '''\
"""Migration {id}: {description}.

TODO: fill in rationale / link to the issue that requires this migration.
"""

from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from migrations.base import Migration


class {class_name}(Migration):
    id = "{id}"
    description = "{description}"

    def up(self, conn: Connection) -> None:
        # TODO: implement the migration.
        # Use inspect(conn) to check the current schema before executing DDL.
        # Example:
        #   inspector = inspect(conn)
        #   if "my_table" not in inspector.get_table_names():
        #       return
        #   existing = {{col["name"] for col in inspector.get_columns("my_table")}}
        #   if "new_column" not in existing:
        #       conn.execute(text("ALTER TABLE my_table ADD COLUMN new_column VARCHAR"))
        raise NotImplementedError("Migration {id} has not been implemented yet")


migration = {class_name}()
'''


def _next_id(versions_dir: str) -> str:
    """Return the next unused 4-digit migration ID based on existing files."""
    existing: list[int] = []
    if os.path.isdir(versions_dir):
        for fname in os.listdir(versions_dir):
            m = re.match(r"^(\d{4})_", fname)
            if m:
                existing.append(int(m.group(1)))
    return str(max(existing, default=0) + 1).zfill(4)


def _slug(description: str) -> str:
    """Convert a description string to a snake_case filename slug."""
    slug = description.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    slug = slug.strip("_")
    # Truncate to avoid overly long filenames
    return slug[:60]


def _class_name(description: str) -> str:
    """Convert description to a PascalCase class name."""
    parts = re.sub(r"[^a-zA-Z0-9]+", " ", description).split()
    return "".join(p.capitalize() for p in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scaffold a new migration file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "description", help="Short description of the migration (used in filename and class)"
    )
    parser.add_argument(
        "--id",
        metavar="NNNN",
        default=None,
        help="Override the auto-incremented 4-digit ID",
    )
    args = parser.parse_args(argv)

    versions_dir = os.path.abspath(_VERSIONS_DIR)
    os.makedirs(versions_dir, exist_ok=True)

    migration_id = args.id if args.id else _next_id(versions_dir)
    if not re.fullmatch(r"\d{4}", migration_id):
        print(f"ERROR: ID must be exactly 4 digits, got {migration_id!r}", file=sys.stderr)
        return 1

    slug = _slug(args.description)
    filename = f"{migration_id}_{slug}.py"
    filepath = os.path.join(versions_dir, filename)

    if os.path.exists(filepath):
        print(f"ERROR: File already exists: {filepath}", file=sys.stderr)
        return 1

    class_name = _class_name(args.description)
    content = _TEMPLATE.format(
        id=migration_id,
        description=args.description,
        class_name=class_name,
    )

    with open(filepath, "w") as fh:
        fh.write(content)

    print(f"Created: {filepath}")
    print()
    print("Next steps:")
    print(f"  1. Implement the `up` method in {filename}")
    print(f"  2. Add '{migration_id}' to REGISTRY in migrations/registry.py")
    print("  3. Run: python -m scripts.migrate --dry-run  to verify")
    return 0


if __name__ == "__main__":
    sys.exit(main())
