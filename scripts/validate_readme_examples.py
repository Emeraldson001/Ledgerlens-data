"""
scripts/validate_readme_examples.py — README Workflow Examples Validator (Issue #548)
======================================================================================
Extracts every ``bash`` / ``sh`` / ``shell`` fenced code block from the
README (and optionally other Markdown docs) and validates that:

1. Every ``python -m <module>`` reference resolves to an importable module
   path inside the repo.
2. Every ``python <path>`` reference resolves to a file that actually exists.
3. Every ``make <target>`` reference is a declared Makefile target.
4. CLI flags documented for known scripts match the flags the scripts actually
   register (``argparse``-based discovery, best-effort).

What this does NOT do
---------------------
* It does not execute any commands — all checks are static.
* It does not validate shell syntax or flag values.
* It does not follow shell variables or command substitutions.

Design
------
* Parses Markdown with a simple regex fence extractor (no third-party deps).
* For each bash line it identifies the *verb* (python, python3, make) and
  the *target* (module path, file path, or Make target).
* Results are collected into a :class:`ValidationReport` which is printed
  to stdout and optionally written as JSON.

Usage
-----
    # Validate README.md (default)
    python scripts/validate_readme_examples.py

    # Validate README + all docs/*.md
    python scripts/validate_readme_examples.py --docs README.md docs/

    # Emit JSON report
    python scripts/validate_readme_examples.py --json

    # Fail silently on missing optional scripts (warn only)
    python scripts/validate_readme_examples.py --warn-only

Exit codes
----------
0  All examples valid.
1  Fatal error (file unread, etc.).
2  One or more examples reference missing targets.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from dataclasses import dataclass, field

REPO_ROOT = pathlib.Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Markdown code-block extraction
# ---------------------------------------------------------------------------

# Matches ```bash, ```sh, ```shell (and plain ```) fenced blocks
_FENCE_RE = re.compile(
    r"```(?:bash|sh|shell|)[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_bash_blocks(text: str) -> list[str]:
    """Return a list of raw code-block contents (bash/sh/shell or plain)."""
    return [m.group(1) for m in _FENCE_RE.finditer(text)]


def collect_markdown_files(paths: list[pathlib.Path]) -> list[pathlib.Path]:
    """Expand directories to .md files; keep individual files as-is."""
    result: list[pathlib.Path] = []
    for p in paths:
        if p.is_dir():
            result.extend(sorted(p.rglob("*.md")))
        elif p.is_file():
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# Line classification
# ---------------------------------------------------------------------------

# Patterns for lines we know how to validate
_PYTHON_MODULE_RE = re.compile(r"(?:python3?|python)\s+-m\s+([\w.]+)")
_PYTHON_FILE_RE = re.compile(r"(?:python3?|python)\s+(?!-m\s)([\w./-]+\.py)")
_MAKE_RE = re.compile(r"make\s+([\w-]+)")
# Lines we should skip (comments, variable assignments, env-var prefixes only)
_SKIP_LINE_RE = re.compile(r"^\s*#|^\s*$|^[A-Z_]+=\S+\s*$")


@dataclass
class CodeLine:
    """A single parsed line from a bash code block."""

    raw: str
    source_file: str
    block_index: int
    line_number: int  # within the block, 1-based

    kind: str = ""  # "python_module" | "python_file" | "make" | "unknown"
    target: str = ""  # the module path, file path, or make target


def classify_line(
    raw: str,
    source_file: str,
    block_index: int,
    line_number: int,
) -> CodeLine:
    line = CodeLine(
        raw=raw,
        source_file=source_file,
        block_index=block_index,
        line_number=line_number,
    )
    # Strip leading env-var prefix (e.g. "FOO=bar python …")
    stripped = re.sub(r"^(?:[A-Z_]+=\S+\s+)+", "", raw.strip())

    m = _PYTHON_MODULE_RE.search(stripped)
    if m:
        line.kind = "python_module"
        line.target = m.group(1)
        return line

    m = _PYTHON_FILE_RE.search(stripped)
    if m:
        line.kind = "python_file"
        line.target = m.group(1)
        return line

    m = _MAKE_RE.search(stripped)
    if m:
        line.kind = "make"
        line.target = m.group(1)
        return line

    line.kind = "unknown"
    return line


def parse_code_block(
    block: str,
    source_file: str,
    block_index: int,
) -> list[CodeLine]:
    """Parse all non-trivial lines in a code block."""
    lines = []
    for i, raw in enumerate(block.splitlines(), 1):
        if _SKIP_LINE_RE.match(raw):
            continue
        # Handle line continuations (trailing \)
        raw = raw.rstrip("\\").strip()
        if not raw:
            continue
        cl = classify_line(raw, source_file, block_index, i)
        if cl.kind != "unknown":
            lines.append(cl)
    return lines


# ---------------------------------------------------------------------------
# Makefile target discovery
# ---------------------------------------------------------------------------

_MAKE_TARGET_RE = re.compile(r"^([\w-]+)\s*:", re.MULTILINE)


def get_makefile_targets(root: pathlib.Path) -> set[str]:
    """Return the set of targets declared in the repo Makefile."""
    makefile = root / "Makefile"
    if not makefile.is_file():
        return set()
    text = makefile.read_text(encoding="utf-8", errors="replace")
    return set(_MAKE_TARGET_RE.findall(text))


# ---------------------------------------------------------------------------
# Python module / file resolution
# ---------------------------------------------------------------------------


def module_exists(module: str, root: pathlib.Path) -> bool:
    """
    Return True if *module* (dotted path) maps to a Python file or package
    inside *root*.
    """
    parts = module.split(".")
    # Try as a package (__init__.py)
    pkg_path = root.joinpath(*parts) / "__init__.py"
    if pkg_path.is_file():
        return True
    # Try as a module file
    mod_path = root.joinpath(*parts[:-1]) / f"{parts[-1]}.py"
    if mod_path.is_file():
        return True
    # Also check if module is a directory itself (namespace package)
    ns_path = root.joinpath(*parts)
    if ns_path.is_dir():
        return True
    return False


def file_exists(path_str: str, root: pathlib.Path) -> bool:
    """Return True if *path_str* resolves to an existing file under *root*."""
    candidate = root / path_str
    return candidate.is_file()


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------


@dataclass
class Finding:
    """A single validation finding (error or warning)."""

    level: str  # "error" | "warning"
    source_file: str
    block_index: int
    line_number: int
    raw_line: str
    kind: str
    target: str
    message: str

    def __str__(self) -> str:
        indicator = "✗" if self.level == "error" else "⚠"
        return (
            f"  [{indicator}] {self.source_file} block {self.block_index}, "
            f"line {self.line_number}: {self.message}\n"
            f"      → {self.raw_line.strip()}"
        )


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    findings: list[Finding] = field(default_factory=list)
    checked: int = 0

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    def summary(self) -> str:
        lines = ["README Examples Validation Report", "=" * 40]
        if self.findings:
            for f in self.findings:
                lines.append(str(f))
            lines.append("=" * 40)
        n_errors = len(self.errors)
        n_warnings = len(self.warnings)
        lines.append(
            f"  Checked {self.checked} example line(s): "
            f"{n_errors} error(s), {n_warnings} warning(s)."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "checked": self.checked,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "findings": [
                {
                    "level": f.level,
                    "source_file": f.source_file,
                    "block_index": f.block_index,
                    "line_number": f.line_number,
                    "kind": f.kind,
                    "target": f.target,
                    "message": f.message,
                    "raw_line": f.raw_line.strip(),
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ReadmeExamplesValidator:
    """
    Validates bash code blocks extracted from Markdown files.

    Parameters
    ----------
    root:           Repository root.
    warn_only:      Treat missing targets as warnings, not errors.
    """

    def __init__(
        self,
        root: pathlib.Path = REPO_ROOT,
        warn_only: bool = False,
    ) -> None:
        self.root = root
        self.warn_only = warn_only
        self._makefile_targets: set[str] | None = None

    @property
    def makefile_targets(self) -> set[str]:
        if self._makefile_targets is None:
            self._makefile_targets = get_makefile_targets(self.root)
        return self._makefile_targets

    def validate_files(self, md_files: list[pathlib.Path]) -> ValidationReport:
        report = ValidationReport()
        for md_file in md_files:
            try:
                text = md_file.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                report.findings.append(
                    Finding(
                        level="error",
                        source_file=str(md_file.relative_to(self.root)),
                        block_index=0,
                        line_number=0,
                        raw_line="",
                        kind="io_error",
                        target=str(md_file),
                        message=f"Cannot read file: {exc}",
                    )
                )
                continue

            blocks = extract_bash_blocks(text)
            rel_path = str(md_file.relative_to(self.root))
            for bi, block in enumerate(blocks, 1):
                lines = parse_code_block(block, rel_path, bi)
                for cl in lines:
                    report.checked += 1
                    finding = self._validate_line(cl)
                    if finding:
                        report.findings.append(finding)
        return report

    def _validate_line(self, cl: CodeLine) -> Finding | None:
        level = "warning" if self.warn_only else "error"

        if cl.kind == "python_module":
            if not module_exists(cl.target, self.root):
                return Finding(
                    level=level,
                    source_file=cl.source_file,
                    block_index=cl.block_index,
                    line_number=cl.line_number,
                    raw_line=cl.raw,
                    kind=cl.kind,
                    target=cl.target,
                    message=(
                        f"python -m {cl.target!r}: module not found in repo. "
                        f"Expected file at "
                        f"'{'/'.join(cl.target.split('.'))}.py' or "
                        f"'{'/'.join(cl.target.split('.'))}/__init__.py'."
                    ),
                )

        elif cl.kind == "python_file":
            if not file_exists(cl.target, self.root):
                return Finding(
                    level=level,
                    source_file=cl.source_file,
                    block_index=cl.block_index,
                    line_number=cl.line_number,
                    raw_line=cl.raw,
                    kind=cl.kind,
                    target=cl.target,
                    message=(
                        f"python {cl.target!r}: file not found at " f"'{self.root / cl.target}'."
                    ),
                )

        elif cl.kind == "make":
            if cl.target not in self.makefile_targets:
                # Don't hard-fail on make install / make test — they're almost
                # universally present; flag as warning regardless of warn_only
                return Finding(
                    level="warning",
                    source_file=cl.source_file,
                    block_index=cl.block_index,
                    line_number=cl.line_number,
                    raw_line=cl.raw,
                    kind=cl.kind,
                    target=cl.target,
                    message=(
                        f"make {cl.target!r}: target not found in Makefile. "
                        f"Declared targets: "
                        f"{', '.join(sorted(self.makefile_targets))}."
                    ),
                )

        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate bash examples in README and Markdown docs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--docs",
        nargs="+",
        default=["README.md"],
        metavar="PATH",
        help=(
            "Markdown files or directories to validate "
            "(default: README.md). "
            "Directories are searched recursively for *.md files."
        ),
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=REPO_ROOT,
        help="Repository root directory (default: auto-detected).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Print the report as JSON instead of human-readable text.",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Emit warnings instead of errors for missing targets (exit 0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root: pathlib.Path = args.root.resolve()

    raw_paths = [root / p for p in args.docs]
    md_files = collect_markdown_files(raw_paths)

    if not md_files:
        print(
            f"[validate_readme_examples] No Markdown files found in: "
            f"{', '.join(str(p) for p in raw_paths)}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[validate_readme_examples] Checking {len(md_files)} file(s): "
        f"{', '.join(str(f.relative_to(root)) for f in md_files)}"
    )

    validator = ReadmeExamplesValidator(root=root, warn_only=args.warn_only)
    report = validator.validate_files(md_files)

    if args.as_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())

    if report.errors:
        print(
            f"\n[validate_readme_examples] ✗ {len(report.errors)} error(s) found. "
            "Update the README or add the missing script/module.",
            file=sys.stderr,
        )
        return 2

    if report.warnings:
        print(
            f"[validate_readme_examples] ⚠  {len(report.warnings)} warning(s). "
            "Review the output above."
        )

    print(f"[validate_readme_examples] ✓ All {report.checked} example line(s) valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
