"""Validate the current environment against a runtime mode's config contract.

Runs the same checks each entry point (run_pipeline.py, api/app.py,
scripts/stream.py, ...) enforces at startup — see config/contracts.py — but
without starting the service. Useful in CI, in a pre-deploy check, or when
debugging "why won't this start" locally.

Usage::

    python -m scripts.check_env --mode api
    python -m scripts.check_env --mode pipeline_onchain
    python -m scripts.check_env --all
    python -m scripts.check_env --all --json
    python -m scripts.check_env --explain RISK_SCORE_DB_URL
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys

from config import Config
from config.contracts import RUNTIME_MODES, _CONTRACTS, validate_mode


def _extract_var_name_from_check(check_func) -> str | None:
    """Extract Config attribute name from a check function if possible."""
    source = inspect.getsource(check_func)
    # Look for patterns like getattr(cls, "ATTR_NAME") or cls.ATTR_NAME
    import re

    # Try cls.ATTR_NAME pattern first
    match = re.search(r"cls\.([A-Z_][A-Z0-9_]*)", source)
    if match:
        return match.group(1)

    # Try getattr(cls, "ATTR_NAME") pattern
    match = re.search(r'getattr\(cls,\s*["\']([A-Z_][A-Z0-9_]*)["\']', source)
    if match:
        return match.group(1)

    return None


def _build_variable_map() -> dict[str, dict[str, any]]:
    """Build a map of Config variables to their usage info across all modes."""
    var_map: dict[str, dict[str, any]] = {}

    # Iterate through all modes and their checks
    for mode_name, contract in _CONTRACTS.items():
        for check in contract.checks:
            var_name = _extract_var_name_from_check(check)
            if var_name:
                if var_name not in var_map:
                    # Get the default value from Config
                    try:
                        default = getattr(Config, var_name)
                    except AttributeError:
                        default = "<not defined in Config>"

                    var_map[var_name] = {
                        "modes": [],
                        "default": default,
                        "reason": None,
                    }

                var_map[var_name]["modes"].append(mode_name)

    return var_map


def cmd_explain(var_name: str) -> int:
    """Explain a single contract variable and its requirements."""
    var_map = _build_variable_map()

    if var_name not in var_map:
        print(f"not a recognised contract variable: {var_name}", file=sys.stderr)
        return 1

    info = var_map[var_name]

    # Print the explanation
    print(f"Variable: {var_name}")
    print(f"Default: {info['default']}")
    print(f"Required in modes:")
    for mode in sorted(info["modes"]):
        contract = _CONTRACTS[mode]
        print(f"  • {mode} — {contract.description}")

    return 0


def _format_error_group(error: str) -> list[str]:
    """Break a ``validate_mode()`` OSError into its individual check bullets.

    ``validate_mode()`` raises one OSError whose message is::

        LedgerLens configuration errors for mode='api' (uvicorn api.app:app):
        - RISK_SCORE_DB_URL is not set ...
        - API_KEYS is not set ...

    This splits it into just the discrete bullets (dropping the redundant
    leading banner and the ``- `` list markers) so that each mode's section in
    a multi-mode ``--all`` run lists its failures as individual items.
    """
    bullets: list[str] = []
    for line in error.splitlines()[1:]:  # skip the "configuration errors for mode=..." banner
        stripped = line.lstrip()
        if stripped.startswith("- "):
            stripped = stripped[2:]
        if stripped:
            bullets.append(stripped)
    return bullets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate LedgerLens runtime-mode config contracts"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mode", choices=sorted(RUNTIME_MODES), help="Validate a single runtime mode"
    )
    group.add_argument("--all", action="store_true", help="Validate every known runtime mode")
    group.add_argument("--explain", metavar="VAR_NAME", help="Explain a single contract variable")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args(argv)

    if args.explain:
        return cmd_explain(args.explain)

    modes = sorted(RUNTIME_MODES) if args.all else [args.mode]
    results = []

    for mode in modes:
        try:
            validate_mode(mode)
        except OSError as exc:
            results.append({"mode": mode, "ok": False, "error": str(exc)})
        else:
            results.append({"mode": mode, "ok": True, "error": None})

    ok = all(r["ok"] for r in results)

    if args.json:
        # JSON mirrors the per-mode structure with a "mode" key on each issue.
        print(json.dumps({"ok": ok, "results": results}, indent=2))
    else:
        # Human-readable output groups each mode's failures under a clear
        # per-mode heading, so a multi-mode `--all` run never reads as one
        # flat list of unrelated errors.
        for r in results:
            print(f"=== {r['mode']} ===")
            if r["ok"]:
                print("    OK — all checks passed")
            else:
                for bullet in _format_error_group(r["error"]):
                    print(f"    FAIL: {bullet}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
