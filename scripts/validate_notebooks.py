"""
scripts/validate_notebooks.py — Notebook & Research Artifact Validation (Issue #549)
======================================================================================
Validates Jupyter notebooks and other research artefacts in the ``notebooks/``
directory to ensure they meet the quality bar required for a production-grade,
contributor-friendly repository.

Checks performed
----------------
**Structural integrity**
* JSON parses as valid notebook (``.ipynb``) format.
* ``nbformat`` version field is present and supported (≥ 4).
* ``kernelspec`` metadata is present.

**Cell hygiene**
* No empty code cells (a cell with only whitespace contributes nothing and
  breaks executed-notebook comparisons).
* No cells with ``TODO`` / ``FIXME`` / ``HACK`` markers left in source
  (configurable; warns by default, errors in strict mode).
* Source length sanity — a single code cell should not exceed
  ``MAX_CELL_LINES`` (default 200) without a module-level comment explaining
  why (raises a warning, not an error).

**Output hygiene** (``--check-outputs`` mode)
* Notebooks committed to the repo must have cleared outputs — large base64
  blobs inflate diffs and can accidentally leak data.
* An exception is made for notebooks that carry a ``"keep_outputs": true``
  key in their top-level metadata.

**Execution freshness** (``--check-execution-count`` mode)
* If outputs are present, execution counts must be sequential (1, 2, 3, …)
  with no gaps — a non-sequential count indicates a notebook was partially
  re-run, which makes reproducibility audits unreliable.

**Dependency declaration** (best-effort)
* Detects ``import`` statements in code cells and cross-references them
  against ``requirements.txt``; reports any import that has no corresponding
  entry (useful catch for undocumented notebook-only deps).

Usage
-----
    # Validate all notebooks in notebooks/ (default)
    python scripts/validate_notebooks.py

    # Validate specific file(s)
    python scripts/validate_notebooks.py --notebooks notebooks/benford_explainer.ipynb

    # Require cleared outputs (CI mode)
    python scripts/validate_notebooks.py --check-outputs

    # Require sequential execution counts
    python scripts/validate_notebooks.py --check-execution-count

    # Strict mode — TODO/FIXME markers become errors
    python scripts/validate_notebooks.py --strict

    # Emit JSON report
    python scripts/validate_notebooks.py --json

    # All CI checks at once
    python scripts/validate_notebooks.py --check-outputs --check-execution-count --strict

Exit codes
----------
0  All notebooks pass.
1  Fatal error (file unreadable, bad argument).
2  One or more notebooks failed validation.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"

MAX_CELL_LINES = 200
MARKER_PATTERN = re.compile(r"\b(TODO|FIXME|HACK|XXX)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------


@dataclass
class NotebookFinding:
    """A single validation finding for a notebook."""

    level: str  # "error" | "warning"
    notebook: str  # relative path
    cell_index: int | None  # 1-based, None for notebook-level findings
    check: str  # short check name
    message: str

    def __str__(self) -> str:
        indicator = "✗" if self.level == "error" else "⚠"
        cell_ref = f" cell {self.cell_index}" if self.cell_index is not None else ""
        return f"  [{indicator}] {self.notebook}{cell_ref}: {self.message}"


# ---------------------------------------------------------------------------
# Notebook validator
# ---------------------------------------------------------------------------


class NotebookValidator:
    """
    Validates a single ``.ipynb`` file.

    Parameters
    ----------
    path:                   Path to the notebook file.
    root:                   Repository root (for relative path display).
    check_outputs:          Fail if cells have non-empty outputs.
    check_execution_count:  Fail if execution counts are non-sequential.
    strict:                 Treat TODO/FIXME markers as errors.
    requirements_pkgs:      Set of top-level import names from requirements.txt.
    """

    def __init__(
        self,
        path: pathlib.Path,
        root: pathlib.Path = REPO_ROOT,
        check_outputs: bool = False,
        check_execution_count: bool = False,
        strict: bool = False,
        requirements_pkgs: set[str] | None = None,
    ) -> None:
        self.path = path
        self.root = root
        self.rel_path = str(path.relative_to(root))
        self.check_outputs = check_outputs
        self.check_execution_count = check_execution_count
        self.strict = strict
        self.requirements_pkgs = requirements_pkgs or set()

    def validate(self) -> list[NotebookFinding]:
        findings: list[NotebookFinding] = []

        # ── Load JSON ─────────────────────────────────────────────────────
        try:
            raw = self.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="io",
                    message=f"Cannot read file: {exc}",
                )
            )
            return findings

        try:
            nb: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="json",
                    message=f"Invalid JSON: {exc}",
                )
            )
            return findings

        # ── Structural checks ─────────────────────────────────────────────
        findings.extend(self._check_structure(nb))
        if any(f.level == "error" for f in findings):
            # If structure is broken, cell-level checks are unreliable
            return findings

        cells: list[dict[str, Any]] = nb.get("cells", [])
        nb_metadata: dict[str, Any] = nb.get("metadata", {})
        keep_outputs: bool = bool(nb_metadata.get("keep_outputs", False))

        execution_counts: list[int] = []

        for i, cell in enumerate(cells, 1):
            cell_type: str = cell.get("cell_type", "")
            source_lines: list[str] = cell.get("source", [])
            source: str = "".join(source_lines) if isinstance(source_lines, list) else source_lines

            if cell_type == "code":
                findings.extend(self._check_code_cell(i, cell, source, keep_outputs))
                ec = cell.get("execution_count")
                if ec is not None:
                    execution_counts.append(ec)
            elif cell_type == "markdown":
                findings.extend(self._check_markdown_cell(i, source))

        # ── Execution count check ─────────────────────────────────────────
        if self.check_execution_count and execution_counts:
            findings.extend(self._check_execution_counts(execution_counts))

        return findings

    # ── Structural ────────────────────────────────────────────────────────

    def _check_structure(self, nb: dict[str, Any]) -> list[NotebookFinding]:
        findings = []

        # nbformat
        nbformat_version = nb.get("nbformat")
        if nbformat_version is None:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="nbformat",
                    message="Missing 'nbformat' field — not a valid notebook.",
                )
            )
        elif nbformat_version < 4:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="nbformat",
                    message=(
                        f"nbformat version {nbformat_version} is not supported. "
                        "Upgrade with: jupyter nbconvert --to notebook "
                        "--nbformat=4 <notebook>"
                    ),
                )
            )

        # kernelspec
        metadata = nb.get("metadata", {})
        if "kernelspec" not in metadata:
            findings.append(
                NotebookFinding(
                    level="warning",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="kernelspec",
                    message=(
                        "Missing 'kernelspec' in metadata — notebook may not "
                        "execute correctly.  Add a kernel via: "
                        "Kernel → Change kernel in Jupyter UI."
                    ),
                )
            )

        # cells list
        if "cells" not in nb:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="cells",
                    message="Missing 'cells' field.",
                )
            )

        return findings

    # ── Code cell ─────────────────────────────────────────────────────────

    def _check_code_cell(
        self,
        idx: int,
        cell: dict[str, Any],
        source: str,
        keep_outputs: bool,
    ) -> list[NotebookFinding]:
        findings = []

        # Empty cell
        if not source.strip():
            findings.append(
                NotebookFinding(
                    level="warning",
                    notebook=self.rel_path,
                    cell_index=idx,
                    check="empty_cell",
                    message=(
                        "Empty code cell.  Remove it or add placeholder "
                        "content — empty cells break executed-notebook diffs."
                    ),
                )
            )
            return findings  # further checks don't apply to empty cells

        # TODO/FIXME markers
        markers = MARKER_PATTERN.findall(source)
        if markers:
            level = "error" if self.strict else "warning"
            unique = sorted(set(m.upper() for m in markers))
            findings.append(
                NotebookFinding(
                    level=level,
                    notebook=self.rel_path,
                    cell_index=idx,
                    check="todo_marker",
                    message=(
                        f"Cell contains unresolved marker(s): {', '.join(unique)}. "
                        "Resolve or remove before committing."
                    ),
                )
            )

        # Cell length
        n_lines = len(source.splitlines())
        if n_lines > MAX_CELL_LINES:
            findings.append(
                NotebookFinding(
                    level="warning",
                    notebook=self.rel_path,
                    cell_index=idx,
                    check="cell_length",
                    message=(
                        f"Cell has {n_lines} lines (max recommended: "
                        f"{MAX_CELL_LINES}).  Consider extracting logic into "
                        "a library module."
                    ),
                )
            )

        # Output hygiene
        if self.check_outputs and not keep_outputs:
            outputs = cell.get("outputs", [])
            if outputs:
                findings.append(
                    NotebookFinding(
                        level="error",
                        notebook=self.rel_path,
                        cell_index=idx,
                        check="outputs_not_cleared",
                        message=(
                            "Cell has outputs that were not cleared before commit. "
                            "Clear with: jupyter nbconvert --ClearOutputPreprocessor"
                            ".enabled=True --to notebook --inplace <notebook>. "
                            "Or set metadata.keep_outputs=true if outputs must "
                            "be preserved."
                        ),
                    )
                )

        # Undeclared imports
        if self.requirements_pkgs:
            findings.extend(self._check_undeclared_imports(idx, source))

        return findings

    def _check_undeclared_imports(self, idx: int, source: str) -> list[NotebookFinding]:
        findings = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return findings

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if not self._is_stdlib_or_known(top):
                        findings.append(
                            NotebookFinding(
                                level="warning",
                                notebook=self.rel_path,
                                cell_index=idx,
                                check="undeclared_import",
                                message=(
                                    f"Import '{top}' not found in requirements.txt. "
                                    "Add it if it's a real dependency."
                                ),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    top = node.module.split(".")[0]
                    if not self._is_stdlib_or_known(top):
                        findings.append(
                            NotebookFinding(
                                level="warning",
                                notebook=self.rel_path,
                                cell_index=idx,
                                check="undeclared_import",
                                message=(
                                    f"Import '{top}' not found in requirements.txt. "
                                    "Add it if it's a real dependency."
                                ),
                            )
                        )
        return findings

    def _is_stdlib_or_known(self, name: str) -> bool:
        """Return True for stdlib modules and known repo/requirements packages."""
        _STDLIB = {
            "sys",
            "os",
            "re",
            "json",
            "math",
            "time",
            "datetime",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "typing",
            "dataclasses",
            "abc",
            "io",
            "copy",
            "random",
            "string",
            "struct",
            "hashlib",
            "hmac",
            "logging",
            "warnings",
            "traceback",
            "inspect",
            "importlib",
            "contextlib",
            "threading",
            "multiprocessing",
            "subprocess",
            "argparse",
            "csv",
            "textwrap",
            "enum",
            "uuid",
            "base64",
            "urllib",
            "http",
            "html",
            "xml",
            "ast",
            "dis",
            "gc",
            "operator",
            "pprint",
            "queue",
            "shutil",
            "tempfile",
            "glob",
            "fnmatch",
            "pickle",
            "shelve",
            "heapq",
            "bisect",
            "decimal",
            "fractions",
            "statistics",
            "cmath",
            "array",
            "weakref",
            "signal",
            "platform",
            "socket",
            "ssl",
            "ipaddress",
            "email",
            "mimetypes",
        }
        if name in _STDLIB:
            return True
        if name in self.requirements_pkgs:
            return True
        # Repo-internal packages
        _REPO_PKGS = {
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
        }
        return name in _REPO_PKGS

    # ── Markdown cell ─────────────────────────────────────────────────────

    def _check_markdown_cell(self, idx: int, source: str) -> list[NotebookFinding]:
        findings = []
        if not source.strip():
            findings.append(
                NotebookFinding(
                    level="warning",
                    notebook=self.rel_path,
                    cell_index=idx,
                    check="empty_cell",
                    message="Empty markdown cell — remove it.",
                )
            )
        return findings

    # ── Execution counts ──────────────────────────────────────────────────

    def _check_execution_counts(self, counts: list[int]) -> list[NotebookFinding]:
        findings = []
        expected = list(range(1, len(counts) + 1))
        if counts != expected:
            findings.append(
                NotebookFinding(
                    level="error",
                    notebook=self.rel_path,
                    cell_index=None,
                    check="execution_count",
                    message=(
                        f"Non-sequential execution counts: {counts[:10]}"
                        f"{'…' if len(counts) > 10 else ''}. "
                        "Re-run the notebook from top to bottom and clear "
                        "outputs before committing: "
                        "Kernel → Restart & Run All, then Cell → All Output → Clear."
                    ),
                )
            )
        return findings


# ---------------------------------------------------------------------------
# Requirements parser
# ---------------------------------------------------------------------------


def parse_requirements_pkgs(root: pathlib.Path) -> set[str]:
    """
    Return the set of top-level import names inferred from requirements.txt.

    Maps pip package names to their likely import names (e.g.
    ``scikit-learn`` → ``sklearn``, ``python-louvain`` → ``community``).
    """
    req_file = root / "requirements.txt"
    if not req_file.is_file():
        return set()

    _PIP_TO_IMPORT: dict[str, str] = {
        "scikit-learn": "sklearn",
        "python-louvain": "community",
        "stellar-sdk": "stellar_sdk",
        "torch-geometric": "torch_geometric",
        "confluent-kafka": "confluent_kafka",
        "stable-baselines3": "stable_baselines3",
        "causal-learn": "causallearn",
        "dice-ml": "dice",
        "prometheus-client": "prometheus_client",
        "imbalanced-learn": "imblearn",
        "python-dotenv": "dotenv",
        "python-jose": "jose",
        "python-json-logger": "pythonjsonlogger",
        "factory-boy": "factory",
    }

    pkgs: set[str] = set()
    for line in req_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip version specifiers (>=, ==, ~=, <=, etc.)
        pkg = re.split(r"[>=<!~\[]", line)[0].strip().lower()
        import_name = _PIP_TO_IMPORT.get(pkg, pkg.replace("-", "_"))
        pkgs.add(import_name)
    return pkgs


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


@dataclass
class AggregatedReport:
    """Aggregated findings across all validated notebooks."""

    notebook_results: dict[str, list[NotebookFinding]] = field(default_factory=dict)

    @property
    def all_findings(self) -> list[NotebookFinding]:
        findings = []
        for ff in self.notebook_results.values():
            findings.extend(ff)
        return findings

    @property
    def errors(self) -> list[NotebookFinding]:
        return [f for f in self.all_findings if f.level == "error"]

    @property
    def warnings(self) -> list[NotebookFinding]:
        return [f for f in self.all_findings if f.level == "warning"]

    def summary(self) -> str:
        lines = ["Notebook Validation Report", "=" * 40]
        for nb_path, findings in self.notebook_results.items():
            if findings:
                lines.append(f"\n  {nb_path}:")
                for f in findings:
                    lines.append(str(f))
            else:
                lines.append(f"  ✓  {nb_path}")
        lines.append("=" * 40)
        lines.append(
            f"  {len(self.notebook_results)} notebook(s) checked, "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "notebook_count": len(self.notebook_results),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "notebooks": {
                nb: [
                    {
                        "level": f.level,
                        "cell_index": f.cell_index,
                        "check": f.check,
                        "message": f.message,
                    }
                    for f in findings
                ]
                for nb, findings in self.notebook_results.items()
            },
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Jupyter notebooks and research artefacts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Notebook files or directories to validate "
            "(default: notebooks/ directory). "
            "Directories are searched recursively for *.ipynb files."
        ),
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=REPO_ROOT,
        help="Repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--check-outputs",
        action="store_true",
        help="Fail if any notebook cell has non-empty outputs.",
    )
    parser.add_argument(
        "--check-execution-count",
        action="store_true",
        help="Fail if execution counts are non-sequential.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat TODO/FIXME/HACK markers as errors.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the report as JSON.",
    )
    return parser.parse_args(argv)


def _collect_notebooks(paths: list[str] | None, root: pathlib.Path) -> list[pathlib.Path]:
    if paths is None:
        nb_dir = root / "notebooks"
        if nb_dir.is_dir():
            return sorted(nb_dir.rglob("*.ipynb"))
        return []
    result = []
    for p in paths:
        target = pathlib.Path(p)
        if not target.is_absolute():
            target = root / target
        if target.is_dir():
            result.extend(sorted(target.rglob("*.ipynb")))
        elif target.is_file():
            result.append(target)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root: pathlib.Path = args.root.resolve()

    notebooks = _collect_notebooks(args.notebooks, root)
    if not notebooks:
        print(
            "[validate_notebooks] No .ipynb files found.",
            file=sys.stderr,
        )
        return 1

    print(f"[validate_notebooks] Validating {len(notebooks)} notebook(s).")
    requirements_pkgs = parse_requirements_pkgs(root)

    aggregated = AggregatedReport()
    for nb_path in notebooks:
        validator = NotebookValidator(
            path=nb_path,
            root=root,
            check_outputs=args.check_outputs,
            check_execution_count=args.check_execution_count,
            strict=args.strict,
            requirements_pkgs=requirements_pkgs,
        )
        findings = validator.validate()
        rel = str(nb_path.relative_to(root))
        aggregated.notebook_results[rel] = findings

    if args.as_json:
        print(json.dumps(aggregated.to_dict(), indent=2))
    else:
        print(aggregated.summary())

    if aggregated.errors:
        print(
            f"\n[validate_notebooks] ✗ {len(aggregated.errors)} error(s). "
            "Fix the issues above before merging.",
            file=sys.stderr,
        )
        return 2

    if aggregated.warnings:
        print(
            f"[validate_notebooks] ⚠  {len(aggregated.warnings)} warning(s). "
            "Review the report above."
        )

    print(f"[validate_notebooks] ✓ All {len(notebooks)} notebook(s) passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
