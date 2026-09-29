#!/usr/bin/env python3
"""Validate advanced work-item mappings against statically collected pytest tests."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MarkedTest:
    """A pytest node ID and the work-item IDs declared on it."""

    node_id: str
    issue_ids: frozenset[str]


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _issue_ids(decorators: list[ast.expr]) -> frozenset[str]:
    issue_ids: set[str] = set()
    for decorator in decorators:
        if not isinstance(decorator, ast.Call):
            continue
        if _qualified_name(decorator.func) not in {"pytest.mark.issue", "mark.issue"}:
            continue
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            value = decorator.args[0].value
            if isinstance(value, (str, int)):
                issue_ids.add(str(value))
    return frozenset(issue_ids)


def collect_marked_tests(tests_root: Path, repository_root: Path) -> dict[str, MarkedTest]:
    """Collect pytest node IDs and issue markers without importing test modules."""
    collected: dict[str, MarkedTest] = {}
    tests_root = tests_root.resolve()
    repository_root = repository_root.resolve()
    for path in sorted(tests_root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative_path = path.resolve().relative_to(repository_root).as_posix()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                node_id = f"{relative_path}::{node.name}"
                collected[node_id] = MarkedTest(node_id, _issue_ids(node.decorator_list))
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                class_issues = _issue_ids(node.decorator_list)
                for child in node.body:
                    if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if not child.name.startswith("test_"):
                        continue
                    node_id = f"{relative_path}::{node.name}::{child.name}"
                    collected[node_id] = MarkedTest(
                        node_id,
                        class_issues | _issue_ids(child.decorator_list),
                    )
    return collected


def load_manifest(path: Path) -> dict[str, Any]:
    """Load the versioned JSON traceability manifest."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load traceability manifest {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Traceability manifest root must be an object")
    return data


def validate_traceability(
    manifest: dict[str, Any],
    collected: dict[str, MarkedTest],
) -> list[str]:
    """Return all manifest/marker consistency errors."""
    errors: list[str] = []
    if manifest.get("version") != 1:
        errors.append("manifest.version must be 1")

    work_items = manifest.get("work_items")
    if not isinstance(work_items, list):
        return [*errors, "manifest.work_items must be a list"]

    seen_ids: set[str] = set()
    mapped_nodes: set[str] = set()
    for index, item in enumerate(work_items):
        prefix = f"work_items[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        issue_id = str(item.get("id", "")).strip()
        if not issue_id:
            errors.append(f"{prefix}.id is required")
            continue
        if issue_id in seen_ids:
            errors.append(f"duplicate work-item id: {issue_id}")
        seen_ids.add(issue_id)
        if not str(item.get("title", "")).strip():
            errors.append(f"{prefix}.title is required")
        if item.get("tier") != "advanced":
            errors.append(f"{prefix}.tier must be 'advanced'")

        selectors = item.get("tests")
        if not isinstance(selectors, list) or not selectors:
            errors.append(f"advanced work item {issue_id} must map to at least one test")
            continue
        for selector in selectors:
            if not isinstance(selector, str):
                errors.append(f"{prefix}.tests entries must be strings")
                continue
            mapped_nodes.add(selector)
            test = collected.get(selector)
            if test is None:
                errors.append(f"{issue_id}: test node does not exist: {selector}")
            elif issue_id not in test.issue_ids:
                errors.append(f"{issue_id}: {selector} is missing @pytest.mark.issue({issue_id!r})")

    for node_id, test in collected.items():
        for issue_id in test.issue_ids:
            if issue_id in seen_ids and node_id not in mapped_nodes:
                errors.append(f"{issue_id}: marked test is absent from manifest: {node_id}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tests/issue_traceability.json"),
    )
    parser.add_argument("--tests-root", type=Path, default=Path("tests"))
    args = parser.parse_args(argv)

    repository_root = Path.cwd().resolve()
    try:
        manifest = load_manifest(args.manifest)
        collected = collect_marked_tests(args.tests_root, repository_root)
        errors = validate_traceability(manifest, collected)
    except (OSError, SyntaxError, ValueError) as exc:
        print(f"Traceability check failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print("Issue-to-test traceability errors:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Traceability valid: {len(manifest['work_items'])} advanced work items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
