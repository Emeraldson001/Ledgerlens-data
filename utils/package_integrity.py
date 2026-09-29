"""Source package integrity checks, run before test execution.

LedgerLens has already been bitten once by a class of bug this module exists
to catch early: a merge on ``config.py`` silently dropped several attributes
that call sites across the codebase still referenced (see the "Restored
config attributes" note in config.py). That failure mode — a bad merge or a
half-applied edit leaving a source tree that *looks* fine but is structurally
broken — doesn't show up as a clean, single test failure. It shows up as a
scatter of unrelated `AttributeError`/`ImportError` failures across the
suite, which is expensive to root-cause.

This module performs a fast, dependency-free, no-import structural sweep of
the source tree and reports every issue with a clear diagnosis, so a broken
tree fails once, loudly, with a pointer to the exact file — instead of as
noise across unrelated test failures. It never imports project code (so it
has no side effects and needs no dependencies installed), it only reads
files and parses them with ``ast``.

Checks performed, per file/package:

* **Missing ``__init__.py``** — a configured source package directory that
  contains ``.py`` files but has no ``__init__.py`` will silently fail to
  be treated as a regular package by some import machinery/tools.
* **Unresolved merge conflict markers** — ``<<<<<<<``, ``=======``,
  ``>>>>>>>`` left in a committed file after a bad merge/rebase.
* **Syntax errors** — every ``.py`` file must parse with ``ast.parse``.
* **Empty non-``__init__`` modules** — a zero-byte ``.py`` file (other than
  ``__init__.py``) usually indicates a truncated merge or a botched save.

Use :func:`check_source_package_integrity` directly, or via
``scripts/check_package_integrity.py`` / the ``pytest_sessionstart`` hook in
``tests/conftest.py`` (which runs this before any test collects, aborting
the whole run with a readable report on failure).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path

#: Top-level source packages checked by default. Kept as an explicit list
#: (rather than "every directory in the repo") so generated/vendored/data
#: directories are never swept by accident.
DEFAULT_SOURCE_PACKAGES: tuple[str, ...] = (
    "alerts",
    "analysis",
    "api",
    "config",
    "detection",
    "evaluation",
    "features",
    "ingestion",
    "integrations",
    "monitoring",
    "privacy",
    "reporting",
    "streaming",
    "training",
    "utils",
)

_CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


@dataclass
class IntegrityIssue:
    """A single, actionable integrity finding."""

    file: Path
    check: str
    message: str

    def __str__(self) -> str:
        return f"[{self.check}] {self.file}: {self.message}"


@dataclass
class IntegrityReport:
    """Result of a full integrity sweep."""

    issues: list[IntegrityIssue] = field(default_factory=list)
    files_checked: int = 0
    packages_checked: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        if self.ok:
            return (
                f"Source package integrity OK — {self.files_checked} file(s) "
                f"across {len(self.packages_checked)} package(s) checked "
                f"({', '.join(self.packages_checked)})."
            )
        lines = [
            f"Source package integrity FAILED — {len(self.issues)} issue(s) "
            f"found across {self.files_checked} file(s) checked:",
        ]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


def _check_file(path: Path) -> list[IntegrityIssue]:
    issues: list[IntegrityIssue] = []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        return [IntegrityIssue(path, "encoding", f"file is not valid UTF-8: {exc}")]

    if path.name != "__init__.py" and text.strip() == "":
        issues.append(
            IntegrityIssue(
                path, "empty-module", "non-__init__ module is empty (0 bytes of content)"
            )
        )

    for marker in _CONFLICT_MARKERS:
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.rstrip()
            # Exact match only: a real conflict marker is either the bare
            # "=======" separator or "<<<<<<<"/">>>>>>>" followed by a ref
            # name. Prefix-only matching false-positives on text dividers
            # like a long "====...====" docstring underline.
            is_marker = stripped == marker or (
                marker != "=======" and stripped.startswith(marker + " ")
            )
            if is_marker:
                issues.append(
                    IntegrityIssue(
                        path,
                        "merge-conflict-marker",
                        f"unresolved merge conflict marker {marker!r} at line {lineno}",
                    )
                )
                break  # one finding per marker type per file is enough signal

    try:
        ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        issues.append(
            IntegrityIssue(
                path,
                "syntax-error",
                f"failed to parse: {exc.msg} (line {exc.lineno}, col {exc.offset})",
            )
        )

    return issues


def check_source_package_integrity(
    root: Path | str = ".",
    packages: tuple[str, ...] = DEFAULT_SOURCE_PACKAGES,
) -> IntegrityReport:
    """Sweep ``packages`` under ``root`` for structural integrity issues.

    Never imports the checked code — purely filesystem + ``ast`` based, so
    it's safe to run with no project dependencies installed and has no
    side effects.
    """
    root_path = Path(root)
    report = IntegrityReport(packages_checked=packages)

    for package in packages:
        package_dir = root_path / package
        if not package_dir.is_dir():
            report.issues.append(
                IntegrityIssue(
                    package_dir,
                    "missing-package",
                    "configured source package directory does not exist",
                )
            )
            continue

        py_files = sorted(package_dir.rglob("*.py"))
        if py_files and not (package_dir / "__init__.py").exists():
            report.issues.append(
                IntegrityIssue(
                    package_dir,
                    "missing-init",
                    "directory contains .py files but has no __init__.py",
                )
            )

        # Also check every nested directory that itself contains .py files
        # but skip __pycache__ and similar generated dirs.
        nested_dirs = {p.parent for p in py_files if p.parent != package_dir}
        for nested in sorted(nested_dirs):
            if nested.name == "__pycache__":
                continue
            if not (nested / "__init__.py").exists():
                report.issues.append(
                    IntegrityIssue(
                        nested,
                        "missing-init",
                        "nested directory contains .py files but has no __init__.py",
                    )
                )

        for py_file in py_files:
            if "__pycache__" in py_file.parts:
                continue
            report.files_checked += 1
            report.issues.extend(_check_file(py_file))

    return report
