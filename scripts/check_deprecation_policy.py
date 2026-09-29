"""Enforce structured deprecation policy across public modules (Issue #511).

Statically scans every public top-level package (any repo-root directory with
an `__init__.py`, excluding `tests/` and `scripts/`) for two policy
violations, without importing the packages (so it works even when optional
heavy dependencies like `torch`/`xgboost` aren't installed):

1. **Past-due `@utils.deprecation.deprecated` symbols** — a deprecation whose
   declared `removal_version` is less than or equal to the current
   `[project] version` in `pyproject.toml` should have been removed already.
2. **Unstructured deprecations** — a function that raises
   `warnings.warn(..., DeprecationWarning)` directly (the pattern used before
   Issue #511, e.g. `detection/wallet_graph.py`) must document the removal
   via a `.. deprecated::` docstring note, so at least a human reviewer has
   the same information the structured decorator would provide automatically.

Usage:
    python -m scripts.check_deprecation_policy
    python -m scripts.check_deprecation_policy --root /path/to/repo

Exit codes:
    0  No policy violations found.
    1  One or more policy violations found (see printed report).
"""

from __future__ import annotations

import argparse
import ast
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_DIR_NAMES = {"tests", "scripts", ".venv", "venv", "__pycache__", "node_modules", ".git"}

#: The decorator's own implementation legitimately calls warnings.warn() with
#: DeprecationWarning inside its wrapper closure — that is the sanctioned
#: mechanism, not a violation of the "unstructured deprecation" rule.
DECORATOR_IMPLEMENTATION_FILE = Path("utils/deprecation.py")


@dataclass(frozen=True)
class PolicyViolation:
    file: Path
    line: int
    symbol: str
    message: str


def _current_version(pyproject_path: Path) -> str:
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _parse_version(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for segment in version.strip().split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def discover_public_packages(root: Path) -> list[Path]:
    """Return every top-level package directory under *root* (has `__init__.py`)."""
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir() and p.name not in EXCLUDED_DIR_NAMES and (p / "__init__.py").exists()
    )


def _decorator_name(decorator: ast.expr) -> str | None:
    """Return the bare name of a decorator expression, resolving calls like
    `@deprecated(...)` to `"deprecated"`."""
    node = decorator
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _keyword_str(call: ast.Call, name: str) -> str | None:
    for kw in call.keywords:
        is_str_constant = isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str)
        if kw.arg == name and is_str_constant:
            return kw.value.value
    return None


def _check_deprecated_decorator(
    file: Path, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef, current_version: str
) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    for decorator in node.decorator_list:
        if _decorator_name(decorator) != "deprecated" or not isinstance(decorator, ast.Call):
            continue

        removal_version = _keyword_str(decorator, "removal_version")
        reason = _keyword_str(decorator, "reason")

        if not removal_version:
            violations.append(
                PolicyViolation(
                    file,
                    node.lineno,
                    node.name,
                    "@deprecated is missing a literal 'removal_version' argument.",
                )
            )
            continue
        if not reason:
            violations.append(
                PolicyViolation(
                    file,
                    node.lineno,
                    node.name,
                    "@deprecated is missing a literal 'reason' argument.",
                )
            )

        if _parse_version(removal_version) <= _parse_version(current_version):
            violations.append(
                PolicyViolation(
                    file,
                    node.lineno,
                    node.name,
                    f"removal_version={removal_version!r} has already passed "
                    f"(current version is {current_version!r}) — this symbol should "
                    "be removed, not just deprecated.",
                )
            )
    return violations


def _calls_deprecation_warning(node: ast.AST) -> ast.Call | None:
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "warn"
        ):
            args = list(child.args) + [kw.value for kw in child.keywords]
            for arg in args:
                if isinstance(arg, ast.Name) and arg.id == "DeprecationWarning":
                    return child
    return None


def _check_unstructured_warning(
    file: Path, node: ast.FunctionDef | ast.AsyncFunctionDef
) -> list[PolicyViolation]:
    if file == DECORATOR_IMPLEMENTATION_FILE:
        return []
    if _calls_deprecation_warning(node) is None:
        return []
    docstring = ast.get_docstring(node) or ""
    if ".. deprecated::" in docstring:
        return []
    return [
        PolicyViolation(
            file,
            node.lineno,
            node.name,
            "raises DeprecationWarning but has no '.. deprecated::' docstring note "
            "documenting the reason/replacement (or migrate to @utils.deprecation.deprecated).",
        )
    ]


def check_file(file: Path, current_version: str) -> list[PolicyViolation]:
    try:
        tree = ast.parse(file.read_text(), filename=str(file))
    except SyntaxError:
        return []

    violations: list[PolicyViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            violations.extend(_check_deprecated_decorator(file, node, current_version))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_unstructured_warning(file, node))
    return violations


def check_repository(root: Path) -> list[PolicyViolation]:
    current_version = _current_version(root / "pyproject.toml")
    violations: list[PolicyViolation] = []
    for package in discover_public_packages(root):
        for py_file in sorted(package.rglob("*.py")):
            relative = py_file.relative_to(root)
            violations.extend(check_file(relative, current_version))
    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enforce structured deprecation policy across public modules."
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root to scan (default: repo root of this script).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()
    violations = check_repository(root)

    if not violations:
        print("Deprecation policy check passed — no violations found.")
        return 0

    print(f"Deprecation policy violations ({len(violations)}):")
    for v in sorted(violations, key=lambda x: (str(x.file), x.line)):
        print(f"  {v.file}:{v.line}: {v.symbol}: {v.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
