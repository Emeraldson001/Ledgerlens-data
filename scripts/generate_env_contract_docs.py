#!/usr/bin/env python3
"""Generate/verify docs/environment_contract.md from config.py (Issue #544).

Statically parses the `Config` class in `config.py` (no import, no side
effects) and renders a typed environment-variable contract to markdown.

Usage:
    python scripts/generate_env_contract_docs.py            # regenerate the doc
    python scripts/generate_env_contract_docs.py --check    # verify, don't write

Exit codes:
    0  Doc written (default mode) or already up to date (--check mode).
    1  --check mode found drift between config.py and the committed doc.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.env_contract import (  # noqa: E402
    DEFAULT_CONFIG_SOURCE,
    DEFAULT_DOCS_PATH,
    build_env_contract,
    render_markdown,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_CONFIG_SOURCE)
    parser.add_argument("--output", default=DEFAULT_DOCS_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Don't write; exit 1 if the generated doc differs from --output.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    entries = build_env_contract(args.source)
    rendered = render_markdown(entries)
    output_path = Path(args.output)

    if args.check:
        existing = output_path.read_text() if output_path.exists() else None
        if existing != rendered:
            print(
                f"{args.output} is out of date with {args.source}. "
                "Run `make env-docs` to regenerate it."
            )
            return 1
        print(f"{args.output} is up to date ({len(entries)} entries).")
        return 0

    output_path.write_text(rendered)
    print(f"Wrote {len(entries)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
