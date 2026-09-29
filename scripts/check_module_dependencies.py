#!/usr/bin/env python
"""Enforce the module layering/dependency rules declared in
``config/module_boundaries.yml``.

Ledgerlens-data has grown into ~15 top-level packages (utils, config,
detection, ingestion, api, scripts, ...) with no machine-checked rule
preventing, say, ``utils`` from quietly depending on ``detection``, or a
domain package from reaching into the HTTP API layer. That kind of drift is
invisible until someone tries to import ``utils`` in a lightweight context
(a Lambda, a notebook, a different service) and pulls in half the repo.

This script parses every ``.py`` file under each package listed in
``config/module_boundaries.yml`` with :mod:`ast` (no execution, so it has no
runtime dependency requirements), resolves each ``import`` /
``from ... import`` statement to a local top-level package, and checks it
against:

1. **Layering** -- a package may only import from its own layer or a lower
   one (see the YAML file's ``layers`` list, ordered lowest-first).
2. **Explicit forbidden pairs** -- one-off rules in ``forbidden_imports``
   for packages that share a layer but should still stay decoupled.

Usage
-----
    python scripts/check_module_dependencies.py
    python scripts/check_module_dependencies.py --config config/module_boundaries.yml
    python scripts/check_module_dependencies.py --package detection  # scope to one package

Exit codes: 0 = no violations, 1 = one or more violations (printed as
``path:lineno: message`` so they open directly in an editor/CI annotation).
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "module_boundaries.yml"


@dataclass(frozen=True)
class Violation:
    file: Path
    lineno: int
    message: str

    def __str__(self) -> str:
        rel = self.file.relative_to(REPO_ROOT) if self.file.is_absolute() else self.file
        return f"{rel}:{self.lineno}: {self.message}"


class BoundaryConfig:
    """Typed view over config/module_boundaries.yml."""

    def __init__(self, raw: dict):
        self.layer_of: dict[str, int] = {}
        self.layer_names: list[str] = []
        for index, layer in enumerate(raw.get("layers", [])):
            name = layer["name"]
            self.layer_names.append(name)
            for package in layer.get("packages", []):
                self.layer_of[package] = index

        self.forbidden_pairs: set[tuple[str, str]] = {
            (rule["importer"], rule["forbidden"]) for rule in raw.get("forbidden_imports", [])
        }
        self.excluded_packages: set[str] = set(raw.get("excluded_packages", []))

    @classmethod
    def load(cls, path: Path) -> BoundaryConfig:
        with open(path) as fh:
            return cls(yaml.safe_load(fh) or {})

    def known_packages(self) -> set[str]:
        return set(self.layer_of)


def _resolve_local_package(module_name: str, known_packages: set[str]) -> str | None:
    top = module_name.split(".", 1)[0]
    return top if top in known_packages else None


def _iter_python_files(package_dir: Path):
    for path in sorted(package_dir.rglob("*.py")):
        # Skip test doubles/fixtures that may live under a package's own dir.
        if any(part in {"__pycache__"} for part in path.parts):
            continue
        yield path


def _imports_in_file(path: Path) -> list[tuple[str, int]]:
    """Return (module_name, lineno) pairs for every import in a file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as exc:
        return [(f"<unparseable: {exc}>", 1)]

    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import; always within the same package
            if node.module:
                found.append((node.module, node.lineno))
    return found


def check(config: BoundaryConfig, only_package: str | None = None) -> list[Violation]:
    violations: list[Violation] = []
    packages = [only_package] if only_package else sorted(config.layer_of)

    for package in packages:
        if package not in config.layer_of:
            continue
        package_dir = REPO_ROOT / package
        if not package_dir.is_dir():
            continue
        importer_layer = config.layer_of[package]

        for file_path in _iter_python_files(package_dir):
            for module_name, lineno in _imports_in_file(file_path):
                target = _resolve_local_package(module_name, config.known_packages())
                if target is None or target == package:
                    continue

                target_layer = config.layer_of[target]
                if target_layer > importer_layer:
                    violations.append(
                        Violation(
                            file_path,
                            lineno,
                            f"layering violation: '{package}' (layer "
                            f"'{config.layer_names[importer_layer]}') imports '{module_name}' "
                            f"which lives in higher layer '{config.layer_names[target_layer]}'. "
                            f"See config/module_boundaries.yml.",
                        )
                    )
                elif (package, target) in config.forbidden_pairs:
                    violations.append(
                        Violation(
                            file_path,
                            lineno,
                            f"forbidden import: '{package}' must not import '{target}' "
                            f"(see forbidden_imports in config/module_boundaries.yml).",
                        )
                    )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--package", default=None, help="Only check imports originating in this package"
    )
    args = parser.parse_args()

    config = BoundaryConfig.load(args.config)
    violations = check(config, only_package=args.package)

    if violations:
        print(f"Module dependency check FAILED: {len(violations)} violation(s)\n")
        for v in violations:
            print(f"  {v}")
        return 1

    scope = args.package or f"{len(config.known_packages())} packages"
    print(f"Module dependency check passed ({scope} within declared boundaries).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
