"""
scripts/check_import_cycles.py — Import Cycle Detection (Issue #546)
=====================================================================
Detects circular imports across the LedgerLens Python package architecture
by statically analysing ``import`` and ``from … import`` statements without
executing any code.

Design
------
* Builds a directed dependency graph where each node is a dotted module path
  (e.g. ``detection.benford_engine``) and each directed edge A → B means
  "module A imports module B".
* Uses DFS-based cycle detection (Tarjan's strongly-connected components).
  Any SCC with more than one member is a cycle; an SCC of size one whose
  sole edge is a self-loop is also reported.
* Only intra-repo imports are tracked — stdlib and third-party packages are
  ignored so the report stays actionable.
* Emits a machine-readable JSON report to ``import_cycle_report_<ts>.json``
  alongside human-readable console output.

Usage
-----
    # Check all packages (default)
    python scripts/check_import_cycles.py

    # Check specific packages only
    python scripts/check_import_cycles.py --packages detection ingestion

    # Emit JSON report to a custom path
    python scripts/check_import_cycles.py --report-path reports/cycles.json

    # Fail only on cycles that cross package boundaries
    python scripts/check_import_cycles.py --cross-package-only

Exit codes
----------
0  No import cycles found.
1  Fatal error (file unreadable, bad argument, etc.).
2  One or more import cycles detected.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import pathlib
import sys
from collections import defaultdict
from collections.abc import Generator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Top-level Python packages that live inside this repo.
REPO_PACKAGES: list[str] = [
    "detection",
    "ingestion",
    "streaming",
    "utils",
    "scripts",
    "integrations",
    "monitoring",
    "reporting",
    "training",
    "evaluation",
    "privacy",
    "features",
    "alerts",
    "config",
    "api",
    "analysis",
    "data",
]

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()


# ---------------------------------------------------------------------------
# Module discovery
# ---------------------------------------------------------------------------


def _find_python_files(root: pathlib.Path, packages: list[str]) -> list[pathlib.Path]:
    """Return all .py files that belong to the requested packages."""
    files: list[pathlib.Path] = []
    for pkg in packages:
        pkg_dir = root / pkg
        if pkg_dir.is_dir():
            files.extend(sorted(pkg_dir.rglob("*.py")))
        # Also check single-file modules at repo root
        single = root / f"{pkg}.py"
        if single.is_file():
            files.append(single)
    return files


def _path_to_module(path: pathlib.Path, root: pathlib.Path) -> str:
    """Convert a filesystem path to a dotted module string."""
    rel = path.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # strip .py
    return ".".join(parts)


# ---------------------------------------------------------------------------
# Static import extraction
# ---------------------------------------------------------------------------


def _extract_imports(source: str, module_name: str) -> Generator[str, None, None]:
    """
    Yield the dotted names of every intra-repo module imported by *source*.

    Handles:
    * ``import foo.bar``
    * ``from foo.bar import baz``
    * ``from . import baz`` (relative — resolved against *module_name*)
    * ``from .. import baz`` (relative — resolved two levels up)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    ".".join(module_name.split(".")[:-1])  # parent package

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in REPO_PACKAGES:
                    yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                # Absolute import
                if node.module and node.module.split(".")[0] in REPO_PACKAGES:
                    yield node.module
            else:
                # Relative import — resolve manually
                parts = module_name.split(".")
                # Go up `level` levels (level=1 → same package)
                base_parts = parts[: max(0, len(parts) - node.level)]
                if node.module:
                    resolved = ".".join(base_parts + node.module.split("."))
                else:
                    resolved = ".".join(base_parts)
                if resolved.split(".")[0] in REPO_PACKAGES:
                    yield resolved


# ---------------------------------------------------------------------------
# Dependency graph builder
# ---------------------------------------------------------------------------


def build_dependency_graph(files: list[pathlib.Path], root: pathlib.Path) -> dict[str, set[str]]:
    """
    Return a ``{module: {imported_modules}}`` adjacency dict for *files*.

    Modules that are imported but not present as source files still appear
    as nodes with an empty adjacency set (they cannot introduce new cycles).
    """
    graph: dict[str, set[str]] = defaultdict(set)

    for path in files:
        module = _path_to_module(path, root)
        graph[module]  # ensure node exists even with no imports
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[WARN] Cannot read {path}: {exc}", file=sys.stderr)
            continue
        for dep in _extract_imports(source, module):
            if dep != module:  # ignore self-imports
                graph[module].add(dep)
                graph[dep]  # ensure target node exists

    return graph


# ---------------------------------------------------------------------------
# Cycle detection (Tarjan's SCC algorithm)
# ---------------------------------------------------------------------------


class _TarjanSCC:
    """Iterative Tarjan strongly-connected components."""

    def __init__(self, graph: dict[str, set[str]]) -> None:
        self.graph = graph
        self.index_counter = [0]
        self.index: dict[str, int] = {}
        self.lowlink: dict[str, int] = {}
        self.on_stack: dict[str, bool] = {}
        self.stack: list[str] = []
        self.sccs: list[list[str]] = []

    def run(self) -> list[list[str]]:
        for node in self.graph:
            if node not in self.index:
                self._strongconnect(node)
        return self.sccs

    def _strongconnect(self, start: str) -> None:
        # Iterative DFS to avoid Python recursion limit on large graphs
        call_stack: list[tuple[str, str | None]] = [(start, None)]
        iter_map: dict[str, iter] = {}  # type: ignore[type-arg]

        while call_stack:
            node, parent = call_stack[-1]

            if node not in self.index:
                idx = self.index_counter[0]
                self.index[node] = idx
                self.lowlink[node] = idx
                self.index_counter[0] += 1
                self.on_stack[node] = True
                self.stack.append(node)
                iter_map[node] = iter(sorted(self.graph.get(node, set())))

            advanced = False
            for neighbour in iter_map[node]:
                if neighbour not in self.index:
                    call_stack.append((neighbour, node))
                    advanced = True
                    break
                elif self.on_stack.get(neighbour, False):
                    self.lowlink[node] = min(self.lowlink[node], self.index[neighbour])

            if not advanced:
                # Pop — update parent's lowlink
                call_stack.pop()
                if call_stack:
                    parent_node = call_stack[-1][0]
                    self.lowlink[parent_node] = min(self.lowlink[parent_node], self.lowlink[node])
                # Emit SCC if this is a root
                if self.lowlink[node] == self.index[node]:
                    scc: list[str] = []
                    while True:
                        w = self.stack.pop()
                        self.on_stack[w] = False
                        scc.append(w)
                        if w == node:
                            break
                    self.sccs.append(scc)


def find_cycles(
    graph: dict[str, set[str]],
    cross_package_only: bool = False,
) -> list[list[str]]:
    """
    Return a list of cycles.  Each cycle is a list of module names forming
    a strongly-connected component of size > 1 (or a self-loop).

    If *cross_package_only* is True, single-package internal cycles are
    omitted — only cycles that span two or more top-level packages are kept.
    """
    sccs = _TarjanSCC(graph).run()
    cycles: list[list[str]] = []

    for scc in sccs:
        if len(scc) > 1:
            cycles.append(sorted(scc))
        elif len(scc) == 1:
            node = scc[0]
            if node in graph.get(node, set()):
                cycles.append(scc)  # self-loop

    if cross_package_only:
        filtered = []
        for cycle in cycles:
            packages = {m.split(".")[0] for m in cycle}
            if len(packages) > 1:
                filtered.append(cycle)
        cycles = filtered

    return cycles


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _format_cycle(cycle: list[str]) -> str:
    """Return a human-readable one-line description of a cycle."""
    return " → ".join(cycle) + " → " + cycle[0]


def _write_json_report(
    cycles: list[list[str]],
    packages_checked: list[str],
    elapsed_ms: float,
    report_path: pathlib.Path,
) -> None:
    report = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "packages_checked": packages_checked,
        "elapsed_ms": round(elapsed_ms, 1),
        "cycle_count": len(cycles),
        "cycles": [
            {
                "modules": cycle,
                "description": _format_cycle(cycle),
                "package_count": len({m.split(".")[0] for m in cycle}),
            }
            for cycle in cycles
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect circular imports in the LedgerLens Python codebase.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--packages",
        nargs="+",
        default=REPO_PACKAGES,
        metavar="PKG",
        help="Top-level packages to scan (default: all repo packages).",
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=REPO_ROOT,
        help="Repository root directory (default: auto-detected).",
    )
    parser.add_argument(
        "--report-path",
        type=pathlib.Path,
        default=None,
        metavar="PATH",
        help=(
            "Write a JSON report to this path.  "
            "Default: import_cycle_report_<timestamp>.json in repo root."
        ),
    )
    parser.add_argument(
        "--cross-package-only",
        action="store_true",
        help="Only report cycles that span two or more top-level packages.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cycle console output; only print the summary line.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:  # noqa: C901
    import time

    args = _parse_args(argv)
    root: pathlib.Path = args.root.resolve()

    # Validate packages
    unknown = [p for p in args.packages if p not in REPO_PACKAGES]
    if unknown:
        print(
            f"[ERROR] Unknown package(s): {', '.join(unknown)}. "
            f"Valid options: {', '.join(REPO_PACKAGES)}",
            file=sys.stderr,
        )
        return 1

    print(f"[check_import_cycles] Scanning packages: {', '.join(args.packages)}")
    print(f"[check_import_cycles] Root: {root}")

    t0 = time.perf_counter()
    files = _find_python_files(root, args.packages)
    print(f"[check_import_cycles] Found {len(files)} Python source files.")

    graph = build_dependency_graph(files, root)
    print(f"[check_import_cycles] Dependency graph: {len(graph)} nodes.")

    cycles = find_cycles(graph, cross_package_only=args.cross_package_only)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── JSON report ──────────────────────────────────────────────────────────
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_path: pathlib.Path = args.report_path or (root / f"import_cycle_report_{ts}.json")
    _write_json_report(cycles, args.packages, elapsed_ms, report_path)

    # ── Console output ───────────────────────────────────────────────────────
    if cycles:
        if not args.quiet:
            print(f"\n[check_import_cycles] ✗ {len(cycles)} import cycle(s) detected:\n")
            for i, cycle in enumerate(cycles, 1):
                packages_in_cycle = sorted({m.split(".")[0] for m in cycle})
                print(
                    f"  Cycle {i} ({len(cycle)} modules, "
                    f"packages: {', '.join(packages_in_cycle)}):"
                )
                for mod in cycle:
                    deps_in_cycle = sorted(graph.get(mod, set()) & set(cycle))
                    print(f"    {mod}  →  {', '.join(deps_in_cycle) or '(self)'}")
                print(f"    ↻ {_format_cycle(cycle)}\n")
        else:
            print(f"[check_import_cycles] ✗ {len(cycles)} import cycle(s) detected.")

        print(f"[check_import_cycles] Report written to: {report_path}")
        print(
            "[check_import_cycles] How to fix: break the cycle by extracting "
            "shared types into a dedicated module (e.g. detection/types.py) "
            "that neither side imports from the other, or use lazy imports."
        )
        return 2

    print(
        f"\n[check_import_cycles] ✓ No import cycles found "
        f"({len(graph)} modules checked in {elapsed_ms:.0f} ms)."
    )
    print(f"[check_import_cycles] Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
