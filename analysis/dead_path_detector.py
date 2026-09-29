"""Dead-path detection for retired modules (Issue #547).

LedgerLens has accumulated 150+ modules across its source packages over many
merged feature branches. When a feature is superseded (e.g. a scoring
approach replaced by a newer one) nothing currently tells a contributor that
the old module is now unreachable — it just sits there, still shipped,
still counted in coverage, still a maintenance burden.

This module performs a static, read-only sweep of the source tree and
reports every module that:

  1. Has zero inbound Python `import`/`from ... import` references anywhere
     in the source packages, scripts, or tests, AND
  2. Is not an entry point (no `if __name__ == "__main__":` guard), AND
  3. Is not referenced by file path or dotted name from a non-Python
     surface (Makefile, CI workflows, docs), AND
  4. Is not explicitly allowlisted in `analysis/dead_path_ignorelist.yaml`
     (for legitimate dynamic-import/plugin-style modules).

It never deletes or modifies anything — it only reports. Every module in
the sweep (not just candidates) carries its reference counts and every
signal checked, so a "why was/wasn't this flagged" question always has a
concrete answer in the report.

Limitations (documented, not hidden): this is a static, import-statement
based analysis. It cannot see `importlib.import_module(dynamic_string)`
calls, string-based entry-point registration, or reflection. Use the
ignore list for known false positives rather than trusting the tool blindly
for irreversible deletions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from utils.package_integrity import DEFAULT_SOURCE_PACKAGES

#: Packages whose modules are candidates for dead-path reporting.
CANDIDATE_PACKAGES: tuple[str, ...] = DEFAULT_SOURCE_PACKAGES

#: Extra directories/files scanned only to count *references* (imports),
#: never themselves reported as dead-path candidates.
REFERRER_ONLY_DIRS: tuple[str, ...] = ("scripts", "tests")
REFERRER_ONLY_FILES: tuple[str, ...] = ("run_pipeline.py", "config.py")

#: Non-Python surfaces checked for a textual reference (module invoked via
#: `python -m pkg.module`, mentioned in a Makefile target, CI step, or doc)
#: before a zero-Python-reference module is treated as a real candidate.
EXTERNAL_REFERENCE_GLOBS: tuple[str, ...] = (
    "Makefile",
    ".github/workflows/*.yml",
    ".github/workflows/*.yaml",
    "docs/*.md",
    "pyproject.toml",
    "README.md",
)

DEFAULT_IGNORELIST_PATH = "analysis/dead_path_ignorelist.yaml"


@dataclass
class ModuleReference:
    """Everything checked for a single candidate module."""

    module: str
    file: Path
    python_refs: int = 0
    is_entrypoint: bool = False
    external_ref_hits: list[str] = field(default_factory=list)
    ignored_reason: str | None = None

    @property
    def is_dead_path_candidate(self) -> bool:
        return (
            self.python_refs == 0
            and not self.is_entrypoint
            and not self.external_ref_hits
            and self.ignored_reason is None
        )


@dataclass
class DeadPathReport:
    modules: list[ModuleReference] = field(default_factory=list)

    @property
    def candidates(self) -> list[ModuleReference]:
        return [m for m in self.modules if m.is_dead_path_candidate]

    def render(self) -> str:
        candidates = self.candidates
        lines = [
            f"Dead-path scan: {len(self.modules)} module(s) checked, "
            f"{len(candidates)} candidate(s) for retirement.",
        ]
        if candidates:
            lines.append("")
            lines.append("Candidates (no inbound Python import, not an entry point,")
            lines.append("not referenced from Makefile/CI/docs, not ignorelisted):")
            for m in sorted(candidates, key=lambda m: str(m.file)):
                lines.append(f"  - {m.module}  ({m.file})")
        return "\n".join(lines)

    def render_markdown(self) -> str:
        candidates = self.candidates
        lines = [
            "# Dead-Path Detection Report",
            "",
            f"Modules checked: **{len(self.modules)}**  ",
            f"Candidates for retirement: **{len(candidates)}**",
            "",
            "Static analysis only — see module docstring for limitations. "
            "Verify with `grep`/git history before deleting anything.",
            "",
        ]
        if candidates:
            lines.append("| Module | File |")
            lines.append("|---|---|")
            for m in sorted(candidates, key=lambda m: str(m.file)):
                lines.append(f"| `{m.module}` | `{m.file}` |")
        else:
            lines.append("_No dead-path candidates found._")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict:
        return {
            "modules_checked": len(self.modules),
            "candidates": [{"module": m.module, "file": str(m.file)} for m in self.candidates],
            "all_modules": [
                {
                    "module": m.module,
                    "file": str(m.file),
                    "python_refs": m.python_refs,
                    "is_entrypoint": m.is_entrypoint,
                    "external_ref_hits": m.external_ref_hits,
                    "ignored_reason": m.ignored_reason,
                }
                for m in self.modules
            ],
        }


def _module_name_for(root: Path, file: Path) -> str:
    rel = file.relative_to(root)
    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][: -len(".py")]
    return ".".join(parts)


def _iter_py_files(root: Path, dirs: tuple[str, ...]) -> list[Path]:
    files = []
    for d in dirs:
        base = root / d
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            files.append(f)
    return files


def _resolve_relative_module(file: Path, root: Path, level: int, module: str | None) -> str:
    rel = file.relative_to(root)
    package_parts = list(rel.parts[:-1])  # containing directory, dotted
    # level=1 means "from . import x" -> current package.
    # level=2 means "from .. import x" -> parent package. Etc.
    up = level - 1
    if up > 0:
        package_parts = package_parts[: len(package_parts) - up] if up <= len(package_parts) else []
    base = ".".join(package_parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def _collect_python_references(root: Path, files: list[Path]) -> Counter:
    refs: Counter = Counter()
    for file in files:
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    refs[alias.name] += 1
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    resolved = _resolve_relative_module(file, root, node.level, node.module)
                else:
                    resolved = node.module or ""
                if resolved:
                    refs[resolved] += 1
                for alias in node.names:
                    if resolved:
                        refs[f"{resolved}.{alias.name}"] += 1
    return refs


def _matches_reference(candidate: str, refs: Counter) -> int:
    """Count references to ``candidate`` — exact hits or submodule imports."""
    count = 0
    for referenced, n in refs.items():
        if referenced == candidate or referenced.startswith(candidate + "."):
            count += n
    return count


def _has_main_guard(file: Path) -> bool:
    try:
        text = file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    return '__name__ == "__main__"' in text or "__name__ == '__main__'" in text


def _load_ignorelist(root: Path, path: str) -> dict[str, str]:
    ignorelist_path = root / path
    if not ignorelist_path.exists():
        return {}
    import yaml

    data = yaml.safe_load(ignorelist_path.read_text()) or {}
    return {str(k): str(v) for k, v in (data.get("ignored") or {}).items()}


def _external_reference_hits(root: Path, module: str, file: Path) -> list[str]:
    needles = {module, str(file.relative_to(root)).replace("\\", "/")}
    hits = []
    for pattern in EXTERNAL_REFERENCE_GLOBS:
        for surface in root.glob(pattern):
            if not surface.is_file():
                continue
            try:
                text = surface.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for needle in needles:
                if needle in text:
                    hits.append(str(surface.relative_to(root)))
                    break
    return hits


def detect_dead_paths(
    root: Path | str = ".",
    candidate_packages: tuple[str, ...] = CANDIDATE_PACKAGES,
    ignorelist_path: str = DEFAULT_IGNORELIST_PATH,
) -> DeadPathReport:
    """Run the full dead-path sweep and return a :class:`DeadPathReport`."""
    root_path = Path(root).resolve()

    referrer_dirs = tuple(dict.fromkeys((*candidate_packages, *REFERRER_ONLY_DIRS)))
    referrer_files = [root_path / f for f in REFERRER_ONLY_FILES if (root_path / f).is_file()]

    all_files = _iter_py_files(root_path, referrer_dirs) + referrer_files
    refs = _collect_python_references(root_path, all_files)
    ignorelist = _load_ignorelist(root_path, ignorelist_path)

    report = DeadPathReport()
    for file in _iter_py_files(root_path, candidate_packages):
        if file.name == "__init__.py":
            continue  # a package itself is never a "dead path"
        module = _module_name_for(root_path, file)

        mref = ModuleReference(module=module, file=file.relative_to(root_path))
        mref.python_refs = _matches_reference(module, refs)
        mref.is_entrypoint = _has_main_guard(file)
        if mref.python_refs == 0 and not mref.is_entrypoint:
            mref.external_ref_hits = _external_reference_hits(root_path, module, file)
        mref.ignored_reason = ignorelist.get(module)
        report.modules.append(mref)

    return report
