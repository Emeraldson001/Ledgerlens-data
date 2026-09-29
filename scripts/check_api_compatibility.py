#!/usr/bin/env python
"""Detect breaking changes to the public API surface of consumer-facing
packages.

Several packages in this repo declare an explicit ``__all__`` in their
``__init__.py`` (``detection``, ``ingestion``, ``privacy``, ``reporting``,
``streaming``, ``utils``, ``analysis``) -- that list is the intended public
contract for anything importing ``from detection import ...`` etc., whether
that's another package in this repo, a notebook, or an external consumer of
this codebase as a library. Nothing today stops a signature edit on one of
those exported symbols from silently breaking every caller.

This script statically extracts the public API surface of those packages
with :mod:`ast` (functions: their signature; classes: their ``__init__``
signature plus public method signatures), and compares it against a
checked-in baseline snapshot (``tests/fixtures/api_baseline.json``).

Usage
-----
    python scripts/check_api_compatibility.py
        Compare the current public API against the baseline. Exits 1 and
        prints one diagnostic per break (removed symbol, changed signature)
        if the baseline no longer matches.

    python scripts/check_api_compatibility.py --update-baseline
        Regenerate the baseline from the current source tree. Run this
        deliberately after a reviewed, intentional public API change.

    python scripts/check_api_compatibility.py --package detection
        Scope extraction/comparison to a single package.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "tests" / "fixtures" / "api_baseline.json"

# Packages whose __init__.py declares __all__ and is treated as a public,
# compatibility-checked contract. Add a package here once its __init__.py
# exports a stable __all__ list.
PUBLIC_PACKAGES = [
    "analysis",
    "detection",
    "ingestion",
    "privacy",
    "reporting",
    "streaming",
    "utils",
]


def _module_file(dotted: str) -> Path | None:
    parts = dotted.split(".")
    as_module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
    if as_module.is_file():
        return as_module
    as_package = REPO_ROOT.joinpath(*parts, "__init__.py")
    if as_package.is_file():
        return as_package
    return None


def _format_args(args: ast.arguments) -> str:
    parts: list[str] = []

    def fmt(arg: ast.arg, default: ast.expr | None = None) -> str:
        piece = arg.arg
        if arg.annotation is not None:
            piece += f": {ast.unparse(arg.annotation)}"
        if default is not None:
            piece += f" = {ast.unparse(default)}"
        return piece

    n_pos = len(args.posonlyargs) + len(args.args)
    defaults = [None] * (n_pos - len(args.defaults)) + list(args.defaults)
    pos_args = args.posonlyargs + args.args
    for arg, default in zip(pos_args, defaults, strict=False):
        parts.append(fmt(arg, default))
        if args.posonlyargs and arg is args.posonlyargs[-1]:
            parts.append("/")

    if args.vararg:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=False):
        parts.append(fmt(arg, default))

    if args.kwarg:
        parts.append("**" + args.kwarg.arg)

    return "(" + ", ".join(parts) + ")"


def _find_top_level_def(tree: ast.Module, name: str) -> ast.AST | None:
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.name == name
        ):
            return node
    return None


def _describe_symbol(node: ast.AST) -> dict:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {"kind": "function", "signature": _format_args(node.args)}
    if isinstance(node, ast.ClassDef):
        init_sig = "(self)"
        methods: dict[str, str] = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name == "__init__":
                    init_sig = _format_args(item.args)
                elif not item.name.startswith("_"):
                    methods[item.name] = _format_args(item.args)
        return {"kind": "class", "init_signature": init_sig, "public_methods": methods}
    return {"kind": "unknown", "signature": "<unresolved>"}


def _import_map(tree: ast.Module, package: str) -> dict[str, tuple[str, str]]:
    """name -> (dotted_module, original_name) for every `from X import Y` in __init__.py."""
    mapping: dict[str, tuple[str, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        dotted = node.module if node.level == 0 else f"{package}.{node.module}"
        for alias in node.names:
            local = alias.asname or alias.name
            mapping[local] = (dotted, alias.name)
    return mapping


def extract_public_api(package: str) -> dict[str, dict]:
    """Extract {symbol_name: description} for every name in `package.__all__`."""
    init_path = REPO_ROOT / package / "__init__.py"
    if not init_path.is_file():
        return {}

    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))

    all_names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            if isinstance(node.value, (ast.List, ast.Tuple)):
                all_names = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]

    if not all_names:
        return {}

    imports = _import_map(tree, package)
    result: dict[str, dict] = {}

    for name in all_names:
        local_def = _find_top_level_def(tree, name)
        if local_def is not None:
            result[name] = _describe_symbol(local_def)
            continue

        if name in imports:
            dotted, original = imports[name]
            target_file = _module_file(dotted)
            if target_file is not None:
                target_tree = ast.parse(
                    target_file.read_text(encoding="utf-8"), filename=str(target_file)
                )
                target_def = _find_top_level_def(target_tree, original)
                if target_def is not None:
                    result[name] = _describe_symbol(target_def)
                    continue

        result[name] = {
            "kind": "unresolved",
            "signature": "<could not statically resolve definition>",
        }

    return result


def extract_all(packages: list[str]) -> dict[str, dict[str, dict]]:
    return {pkg: extract_public_api(pkg) for pkg in packages}


def compare(baseline: dict, current: dict) -> list[str]:
    diagnostics: list[str] = []
    for package, symbols in baseline.items():
        if package not in current:
            diagnostics.append(f"[{package}] entire package missing from current API surface")
            continue
        for name, old_desc in symbols.items():
            if name not in current[package]:
                diagnostics.append(
                    f"[{package}.{name}] public symbol removed (was in __all__, no longer resolvable)"
                )
                continue
            new_desc = current[package][name]
            if old_desc != new_desc:
                diagnostics.append(
                    f"[{package}.{name}] signature changed:\n"
                    f"      before: {json.dumps(old_desc)}\n"
                    f"      after:  {json.dumps(new_desc)}"
                )
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--package", default=None, help="Only check/update a single package")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Regenerate the baseline file from the current source tree",
    )
    args = parser.parse_args()

    packages = [args.package] if args.package else PUBLIC_PACKAGES
    current = extract_all(packages)

    if args.update_baseline:
        existing = {}
        if args.baseline.is_file():
            existing = json.loads(args.baseline.read_text())
        existing.update(current)
        args.baseline.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"Updated {args.baseline} for package(s): {', '.join(packages)}")
        return 0

    if not args.baseline.is_file():
        print(f"No baseline found at {args.baseline}. Run with --update-baseline to create one.")
        return 1

    baseline = json.loads(args.baseline.read_text())
    if args.package:
        baseline = {args.package: baseline.get(args.package, {})}

    diagnostics = compare(baseline, current)

    if diagnostics:
        print(f"API compatibility check FAILED: {len(diagnostics)} issue(s)\n")
        for d in diagnostics:
            print(f"  - {d}")
        print(
            "\nIf this change is intentional, review it carefully (this is a public API "
            "break for anything importing these packages) then run:\n"
            "  python scripts/check_api_compatibility.py --update-baseline"
        )
        return 1

    print(f"API compatibility check passed for: {', '.join(packages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
