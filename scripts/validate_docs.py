"""Documentation validation for advanced contributor paths (Issue #512).

CONTRIBUTING.md directs contributors toward specific docs before making
higher-risk changes — e.g. the Security Threat Model before touching API
endpoints/model loading/persistence, or the Feature Contributor Guide before
adding a new ML feature. Those "advanced contributor path" docs are
discovered dynamically from CONTRIBUTING.md's own relative markdown links
(no hardcoded list to go stale), then validated for:

1. **Existence** — the doc itself, and every local (non-http) markdown link
   inside it, must resolve to a real repo file. Catches the class of dead
   link the existing `markdown-link-check` CI step only covers for 3
   hardcoded files.
2. **Structure** — the doc must be non-empty and contain at least one
   Markdown heading (a stub file is not a usable contributor path).
3. **Code examples** — every fenced ```python code block must be valid
   Python (`ast.parse`), so a worked example like
   `docs/contributor_feature_guide.md`'s `counterparty_variance` walkthrough
   can't silently rot into broken sample code.

Usage:
    python -m scripts.validate_docs
    python -m scripts.validate_docs --entry-point CONTRIBUTING.md

Exit codes:
    0  All discovered docs pass validation.
    1  One or more validation failures found (see printed report).
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENTRY_POINT = "CONTRIBUTING.md"

_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_PYTHON_BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)

#: Angle-bracket placeholders (e.g. `<Type>`, `<optional_data>`) mark an
#: illustrative signature template, not runnable code — skip strict
#: validation for those blocks instead of flagging intentional pseudocode.
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"<[A-Za-z_][\w ]*>")


def _python_block_error(code: str) -> str | None:
    """Return a syntax error message for *code*, or None if it's acceptable.

    Tolerates two common documentation conventions in addition to full
    modules: signature templates containing `<placeholder>` markers, and
    bare `"key": value,` dict-entry fragments meant to be copied into an
    existing dict literal (which don't parse standalone as a module).
    """
    if _TEMPLATE_PLACEHOLDER_RE.search(code):
        return None
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        try:
            ast.parse("{" + code + "}", mode="eval")
            return None
        except SyntaxError:
            return exc.msg


@dataclass(frozen=True)
class DocViolation:
    file: Path
    line: int
    message: str


def _is_local_link(target: str) -> bool:
    if target.startswith(("http://", "https://", "mailto:", "#")):
        return False
    return True


def extract_local_links(doc: Path) -> list[tuple[int, str]]:
    """Return [(line_number, link_target)] for every local markdown link in *doc*."""
    text = doc.read_text()
    links = []
    for match in _LINK_RE.finditer(text):
        target = match.group(1)
        if not _is_local_link(target):
            continue
        line = text.count("\n", 0, match.start()) + 1
        links.append((line, target))
    return links


def discover_advanced_contributor_docs(
    root: Path, entry_point: str = DEFAULT_ENTRY_POINT
) -> list[Path]:
    """Return the entry point plus every local .md file it links to.

    These are the docs CONTRIBUTING.md sends contributors to before making an
    advanced/high-risk change (security, new features, integration tests,
    artifact compatibility, deprecations) — the "advanced contributor path".
    """
    entry = root / entry_point
    docs = [entry]
    if not entry.exists():
        return docs

    for _line, target in extract_local_links(entry):
        clean_target = target.split("#", 1)[0]
        if not clean_target.endswith(".md"):
            continue
        resolved = (entry.parent / clean_target).resolve()
        if resolved not in docs:
            docs.append(resolved)
    return docs


def validate_doc(doc: Path, root: Path) -> list[DocViolation]:
    if not doc.exists():
        return [DocViolation(doc, 0, "referenced doc does not exist")]

    text = doc.read_text()
    violations: list[DocViolation] = []

    if not text.strip():
        violations.append(DocViolation(doc, 0, "file is empty"))
        return violations

    if not _HEADING_RE.search(text):
        violations.append(DocViolation(doc, 1, "no Markdown heading found"))

    for line, target in extract_local_links(doc):
        clean_target = target.split("#", 1)[0]
        if not clean_target:
            continue  # pure in-page anchor
        resolved = (doc.parent / clean_target).resolve()
        if not resolved.exists():
            violations.append(DocViolation(doc, line, f"broken local link: {target}"))

    for match in _PYTHON_BLOCK_RE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        code = match.group(1)
        error = _python_block_error(code)
        if error is not None:
            violations.append(
                DocViolation(doc, line, f"invalid Python in fenced code block: {error}")
            )

    return violations


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate documentation for advanced contributor paths."
    )
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repository root to scan (default: repo root of this script).",
    )
    parser.add_argument(
        "--entry-point",
        default=DEFAULT_ENTRY_POINT,
        help=f"Doc whose local links define the contributor path (default: {DEFAULT_ENTRY_POINT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).resolve()

    docs = discover_advanced_contributor_docs(root, args.entry_point)
    violations: list[DocViolation] = []
    for doc in docs:
        violations.extend(validate_doc(doc, root))

    print(f"Validated {len(docs)} advanced contributor path doc(s):")
    for doc in docs:
        try:
            print(f"  - {doc.relative_to(root)}")
        except ValueError:
            print(f"  - {doc}")

    if not violations:
        print("\nDocumentation validation passed — no issues found.")
        return 0

    print(f"\nDocumentation validation failures ({len(violations)}):")
    for v in violations:
        try:
            display = v.file.relative_to(root)
        except ValueError:
            display = v.file
        print(f"  {display}:{v.line}: {v.message}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
