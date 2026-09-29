"""Environment contract docs generated from config schemas (Issue #544).

``config.py`` is the single source of truth for every environment variable
LedgerLens reads — but that truth is scattered across 500+ lines as
``os.getenv(...)`` calls with inline comments, and nothing keeps a
human-readable contract of it in sync. New contributors (and operators
writing a ``.env`` for a new deployment) either read the whole file or
guess.

This module statically parses the ``Config`` class in ``config.py`` with
``ast`` (no import, no side effects, no dependency on the app actually
running) and produces a typed contract: for every attribute, its Python
type annotation, the env var name it reads (if any), whether it's required
(no default) or optional, its default value expression, and the
description pulled from the comment block directly above it in the source.

``scripts/generate_env_contract_docs.py`` renders this contract to
``docs/environment_contract.md`` and supports ``--check`` (drift
detection, wired into CI via ``make env-docs-check``) so the doc can never
silently go stale.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG_SOURCE = "config.py"
DEFAULT_DOCS_PATH = "docs/environment_contract.md"
DEFAULT_CONFIG_CLASS = "Config"

_DASH_RULE_RE = re.compile(r"^#\s*-{5,}\s*$")


@dataclass(frozen=True)
class EnvVarContractEntry:
    """One row of the environment contract — one ``Config`` attribute."""

    name: str
    type_annotation: str
    env_var: str | None
    required: bool
    default: str | None
    description: str
    section: str


def _comment_block_above(lines: list[str], lineno: int) -> list[str]:
    """Contiguous ``#``-only lines directly above 1-indexed ``lineno``."""
    collected: list[str] = []
    i = lineno - 2
    while i >= 0:
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            collected.append(stripped)
            i -= 1
        else:
            break
    collected.reverse()
    return collected


def _split_header(comment_lines: list[str]) -> tuple[str | None, list[str]]:
    """Split a ``# ---\\n# Title\\n# ---`` header off the front, if present."""
    if (
        len(comment_lines) >= 3
        and _DASH_RULE_RE.match(comment_lines[0])
        and _DASH_RULE_RE.match(comment_lines[2])
    ):
        title = comment_lines[1].lstrip("#").strip()
        return title, comment_lines[3:]
    return None, comment_lines


def _describe(comment_lines: list[str]) -> str:
    return " ".join(
        line.lstrip("#").strip() for line in comment_lines if not _DASH_RULE_RE.match(line)
    ).strip()


def _find_getenv_call(node: ast.AST) -> ast.Call | None:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            is_getenv = (isinstance(func, ast.Attribute) and func.attr == "getenv") or (
                isinstance(func, ast.Name) and func.id == "getenv"
            )
            if is_getenv:
                return sub
    return None


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def build_env_contract(
    source_path: str | Path = DEFAULT_CONFIG_SOURCE,
    class_name: str = DEFAULT_CONFIG_CLASS,
) -> list[EnvVarContractEntry]:
    """Parse ``class_name`` in ``source_path`` into a typed env var contract.

    Entries are returned in declaration order. When an attribute name is
    assigned more than once in the class body (config.py does this — the
    last assignment wins at runtime, matching normal Python class semantics)
    only the *last* occurrence is kept, positioned where it last appears.

    Raises:
        ValueError: if ``class_name`` isn't found in ``source_path``.
    """
    source = Path(source_path).read_text()
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(source_path))

    class_node = next(
        (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name),
        None,
    )
    if class_node is None:
        raise ValueError(f"No {class_name!r} class found in {source_path}")

    entries: list[EnvVarContractEntry] = []
    current_section = "General"

    for node in class_node.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue

        name = node.target.id
        comment_lines = _comment_block_above(lines, node.lineno)
        header_title, remaining = _split_header(comment_lines)
        if header_title:
            current_section = header_title
        description = _describe(remaining)

        getenv_call = _find_getenv_call(node.value) if node.value is not None else None
        env_var: str | None = None
        default_src: str | None = None
        required = False

        if getenv_call is not None and getenv_call.args:
            first_arg = getenv_call.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                env_var = first_arg.value
            if len(getenv_call.args) >= 2:
                default_src = _unparse(getenv_call.args[1])
            else:
                required = True
        else:
            default_src = _unparse(node.value)

        entry = EnvVarContractEntry(
            name=name,
            type_annotation=_unparse(node.annotation) or "",
            env_var=env_var,
            required=required,
            default=default_src,
            description=description,
            section=current_section,
        )
        entries = [e for e in entries if e.name != name]
        entries.append(entry)

    return entries


def render_markdown(entries: list[EnvVarContractEntry]) -> str:
    """Render the contract as grouped markdown tables, one per section."""
    lines = [
        "# Environment Variable Contract",
        "",
        "Auto-generated from `config.py` by `scripts/generate_env_contract_docs.py` "
        "(Issue #544). **Do not hand-edit.** Regenerate with `make env-docs` after "
        "changing `config.py`; `make env-docs-check` (wired into CI) fails the build "
        "if this file has drifted from the source.",
        "",
    ]

    sections: dict[str, list[EnvVarContractEntry]] = {}
    order: list[str] = []
    for entry in entries:
        if entry.section not in sections:
            sections[entry.section] = []
            order.append(entry.section)
        sections[entry.section].append(entry)

    for section in order:
        lines.append(f"## {section}")
        lines.append("")
        lines.append("| Variable | Env Var | Type | Required | Default | Description |")
        lines.append("|---|---|---|---|---|---|")
        for e in sections[section]:
            env_var = f"`{e.env_var}`" if e.env_var else "—"
            required = "Yes" if e.required else "No"
            default = f"`{e.default}`" if e.default not in (None, "") else "—"
            description = (e.description or "—").replace("|", "\\|")
            lines.append(
                f"| `{e.name}` | {env_var} | `{e.type_annotation}` | {required} "
                f"| {default} | {description} |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
