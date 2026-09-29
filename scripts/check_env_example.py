"""Validate that .env.example lists every environment variable config.py reads.

Usage:
    python -m scripts.check_env_example

Exit codes:
    0  All config.py env vars are present in .env.example
    1  One or more variables are missing
    2  Usage error
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.py"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"


def extract_env_vars_from_config(config_path: Path) -> set[str]:
    """Return the set of env var names read via os.getenv in config.py."""
    text = config_path.read_text()
    # Match os.getenv("VAR") and os.getenv('VAR')
    vars_found = set(re.findall(r'os\.getenv\("([^"]+)"', text))
    vars_found |= set(re.findall(r"os\.getenv\('([^']+)'", text))
    return vars_found


def extract_env_vars_from_example(env_example_path: Path) -> set[str]:
    """Return the set of env var names present (even commented out) in .env.example."""
    text = env_example_path.read_text()
    # Match lines like VAR=value or # VAR=value
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.MULTILINE))


def main() -> int:
    if not CONFIG_PATH.exists():
        print("❌ config.py not found")
        return 2
    if not ENV_EXAMPLE_PATH.exists():
        print("❌ .env.example not found")
        return 2

    config_vars = extract_env_vars_from_config(CONFIG_PATH)
    example_vars = extract_env_vars_from_example(ENV_EXAMPLE_PATH)

    missing = sorted(config_vars - example_vars)

    print(f"config.py references {len(config_vars)} env vars")
    print(f".env.example lists {len(example_vars)} vars")
    print()

    if not missing:
        print("✅ .env.example covers every variable config.py reads.")
        return 0

    print(f"❌ {len(missing)} variable(s) missing from .env.example:")
    for var in missing:
        print(f"   - {var}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
