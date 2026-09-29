#!/usr/bin/env python
"""Validate that the operational scripts in ``scripts/`` still match their
declared CLI contracts in ``scripts/cli_contracts.py``.

This parses each contracted script with :mod:`ast` (no execution, so it
carries none of the script's own runtime dependencies -- important since
several operational scripts import Kafka clients, ML frameworks, etc. that
aren't guaranteed to be installed in every environment that just wants to
lint the CLI surface) and extracts every ``parser.add_argument(...)`` /
``sub_parser.add_argument(...)`` call: its name(s) and whether
``required=True`` was passed.

It then diffs that against the contract for the same script and reports,
per script:

* **missing**: a contract argument that no longer appears in the script's
  source at all (renamed or removed without updating the contract).
* **undeclared**: a flag the script defines that isn't in the contract
  (added without documenting it as part of the operational surface).
* **required mismatch**: the contract and the script disagree on whether
  the argument is required.

Usage
-----
    python scripts/check_cli_contracts.py
    python scripts/check_cli_contracts.py --script score_wallet.py
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
from cli_contracts import CONTRACTS, CliContract  # noqa: E402


@dataclass(frozen=True)
class ActualArgument:
    aliases: tuple[str, ...]
    required: bool
    lineno: int

    def matches(self, name: str) -> bool:
        return name in self.aliases


def _extract_actual_arguments(path: Path) -> list[ActualArgument]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    actual: list[ActualArgument] = []

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue

        aliases = tuple(
            a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
        )
        if not aliases:
            continue

        required = False
        for kw in node.keywords:
            if (
                kw.arg == "required"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
            ):
                required = True

        actual.append(ActualArgument(aliases=aliases, required=required, lineno=node.lineno))

    return actual


def check_contract(contract: CliContract, actual: list[ActualArgument]) -> list[str]:
    diagnostics: list[str] = []

    for declared in contract.arguments:
        match = next((a for a in actual if a.matches(declared.name)), None)
        if match is None:
            diagnostics.append(
                f"[{contract.script}] contract declares '{declared.name}' but no matching "
                f"add_argument() call was found -- update scripts/cli_contracts.py or restore the flag."
            )
            continue
        if declared.required != match.required:
            diagnostics.append(
                f"[{contract.script}:{match.lineno}] '{declared.name}' required mismatch: "
                f"contract says required={declared.required}, source says required={match.required}."
            )

    declared_names = contract.argument_names()
    for found in actual:
        if not any(alias in declared_names for alias in found.aliases):
            diagnostics.append(
                f"[{contract.script}:{found.lineno}] undeclared argument {found.aliases!r} -- "
                f"add it to its CliContract in scripts/cli_contracts.py (or remove it from the script)."
            )

    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default=None, help="Only check a single contracted script")
    args = parser.parse_args()

    scripts = [args.script] if args.script else sorted(CONTRACTS)
    all_diagnostics: list[str] = []

    for script_name in scripts:
        contract = CONTRACTS.get(script_name)
        if contract is None:
            print(f"No contract declared for '{script_name}' in scripts/cli_contracts.py")
            return 1
        script_path = SCRIPTS_DIR / contract.script
        if not script_path.is_file():
            all_diagnostics.append(
                f"[{contract.script}] contract references a script that no longer exists at "
                f"{script_path.relative_to(REPO_ROOT)}."
            )
            continue
        actual = _extract_actual_arguments(script_path)
        all_diagnostics.extend(check_contract(contract, actual))

    if all_diagnostics:
        print(f"CLI contract check FAILED: {len(all_diagnostics)} issue(s)\n")
        for d in all_diagnostics:
            print(f"  - {d}")
        return 1

    print(f"CLI contract check passed for {len(scripts)} script(s): {', '.join(scripts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
