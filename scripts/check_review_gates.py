#!/usr/bin/env python3
"""Require a written acknowledgement when a PR touches high-risk paths.

Reads the gate definitions in `.github/review-gates.yml`, matches them against
the files a pull request changes, and checks that the PR body contains a
substantive note under each triggered gate's heading.

Why prose rather than a checkbox: a tick is a one-bit signal that costs nothing
to fake. A sentence naming the migration plan or the evaluation run leaves an
artifact a reviewer can evaluate, and it persists in the PR record.

A PR may bypass every gate with a `Review-Gate-Override: <reason>` line. The
override stays visible in the PR body and is echoed in the bot comment, so it
is auditable rather than a silently disabled check.

Usage:
    python scripts/check_review_gates.py --changed-paths-from changed.txt \\
        --pr-body-file body.md
    python scripts/check_review_gates.py --changed-paths a.py b.py --pr-body "..."
    python scripts/check_review_gates.py --changed-paths-from f --pr-body-file b --dry-run

Exit codes:
    0  No gate fired, every fired gate is acknowledged, or an override applies.
    1  A gate fired without an acknowledgement — CI failure.
    2  The gate configuration is missing or invalid.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = ".github/review-gates.yml"

EXIT_OK = 0
EXIT_UNACKNOWLEDGED = 1
EXIT_CONFIG_ERROR = 2

#: Text that looks like an acknowledgement but says nothing.
_PLACEHOLDERS = {
    "",
    "-",
    "_",
    "...",
    "n/a",
    "na",
    "none",
    "tbd",
    "todo",
    "todo.",
    "<reason>",
    "<describe>",
    "xxx",
}

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*#*\s*$")
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]")


class ConfigError(Exception):
    """The gate configuration is missing or malformed."""


@dataclass(frozen=True)
class Gate:
    id: str
    heading: str
    paths: tuple[str, ...]
    prompt: str = ""


@dataclass
class GateConfig:
    gates: tuple[Gate, ...]
    min_prose_chars: int = 20
    override_prefix: str = "Review-Gate-Override:"


@dataclass
class Result:
    """Outcome of evaluating one PR."""

    triggered: list[Gate] = field(default_factory=list)
    acknowledged: list[Gate] = field(default_factory=list)
    missing: list[Gate] = field(default_factory=list)
    override_reason: str | None = None

    @property
    def overridden(self) -> bool:
        return self.override_reason is not None

    @property
    def exit_code(self) -> int:
        if self.missing and not self.overridden:
            return EXIT_UNACKNOWLEDGED
        return EXIT_OK


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> GateConfig:
    """Parse the gate definitions.

    Raises:
        ConfigError: The file is missing, unparseable, or structurally invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"gate configuration not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path} is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a mapping at the top level")

    raw_gates = raw.get("gates")
    if not isinstance(raw_gates, list) or not raw_gates:
        raise ConfigError(f"{config_path} must define a non-empty 'gates' list")

    gates: list[Gate] = []
    seen: set[str] = set()
    for entry in raw_gates:
        if not isinstance(entry, dict):
            raise ConfigError(f"{config_path}: each gate must be a mapping")
        for required in ("id", "heading", "paths"):
            if not entry.get(required):
                raise ConfigError(f"{config_path}: gate is missing '{required}'")
        gate_id = str(entry["id"])
        if gate_id in seen:
            raise ConfigError(f"{config_path}: duplicate gate id {gate_id!r}")
        seen.add(gate_id)
        paths = entry["paths"]
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ConfigError(f"{config_path}: gate {gate_id!r} 'paths' must be a list of strings")
        gates.append(
            Gate(
                id=gate_id,
                heading=str(entry["heading"]),
                paths=tuple(paths),
                prompt=str(entry.get("prompt", "")).strip(),
            )
        )

    return GateConfig(
        gates=tuple(gates),
        min_prose_chars=int(raw.get("min_prose_chars", 20)),
        override_prefix=str(raw.get("override_prefix", "Review-Gate-Override:")),
    )


def gate_matches(gate: Gate, changed_paths: list[str]) -> list[str]:
    """Return the changed paths that cause *gate* to fire."""
    hits = []
    for path in changed_paths:
        normalised = path.strip().lstrip("./")
        if not normalised:
            continue
        if any(fnmatch.fnmatchcase(normalised, pattern) for pattern in gate.paths):
            hits.append(normalised)
    return hits


def _is_substantive(text: str, min_chars: int) -> bool:
    """True when *text* reads as a real note rather than a placeholder."""
    cleaned = " ".join(
        line.strip() for line in text.splitlines() if line.strip() and not _CHECKBOX_RE.match(line)
    ).strip()
    if cleaned.lower() in _PLACEHOLDERS:
        return False
    # Strip markdown emphasis/backticks before measuring length.
    measured = re.sub(r"[`*_>#\-\[\]()]", "", cleaned).strip()
    return len(measured) >= min_chars


def section_body(pr_body: str, heading: str) -> str | None:
    """Return the text under the markdown heading matching *heading*.

    Matching is case-insensitive and ignores heading level. Returns ``None``
    when the heading is absent.
    """
    lines = pr_body.splitlines()
    target = heading.strip().lower()
    collected: list[str] | None = None

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            title = match.group("title").strip().lower()
            if collected is not None:
                break
            if title == target:
                collected = []
            continue
        if collected is not None:
            collected.append(line)

    return "\n".join(collected) if collected is not None else None


def find_override(pr_body: str, config: GateConfig) -> str | None:
    """Return the override reason if a valid override line is present."""
    prefix = config.override_prefix.lower()
    for line in pr_body.splitlines():
        stripped = line.strip().lstrip("-*> ").strip()
        if stripped.lower().startswith(prefix):
            reason = stripped[len(config.override_prefix) :].strip()
            if _is_substantive(reason, config.min_prose_chars):
                return reason
    return None


def evaluate(changed_paths: list[str], pr_body: str, config: GateConfig) -> Result:
    """Evaluate one PR against the configured gates."""
    result = Result()

    for gate in config.gates:
        if not gate_matches(gate, changed_paths):
            continue
        result.triggered.append(gate)
        body = section_body(pr_body, gate.heading)
        if body is not None and _is_substantive(body, config.min_prose_chars):
            result.acknowledged.append(gate)
        else:
            result.missing.append(gate)

    if result.missing:
        result.override_reason = find_override(pr_body, config)

    return result


def render_report(result: Result, config: GateConfig) -> str:
    """Render a human-readable report, also used as the PR comment body."""
    if not result.triggered:
        return "No high-risk paths touched — review gates do not apply."

    lines = ["## Review gates", ""]

    if result.overridden:
        lines += [
            "Gates were **overridden** for this PR.",
            "",
            f"> {result.override_reason}",
            "",
        ]

    for gate in result.acknowledged:
        lines.append(f"* Acknowledged — **{gate.heading}**")
    for gate in result.missing:
        status = "Overridden" if result.overridden else "Missing"
        lines.append(f"* {status} — **{gate.heading}**")

    if result.missing and not result.overridden:
        lines += [
            "",
            "Add the following heading(s) to the pull-request description, each "
            f"followed by at least {config.min_prose_chars} characters of prose "
            "(a checkbox is not enough):",
            "",
        ]
        for gate in result.missing:
            lines.append(f"### {gate.heading}")
            if gate.prompt:
                lines.append(f"<!-- {gate.prompt} -->")
            lines.append("")
        lines += [
            "See `.github/review-checklists.md` for what each gate is asking.",
            "",
            f"To bypass deliberately, add a line: `{config.override_prefix} <reason>`",
        ]

    return "\n".join(lines).rstrip() + "\n"


def _read_changed_paths(args: argparse.Namespace) -> list[str]:
    if args.changed_paths_from:
        text = Path(args.changed_paths_from).read_text(encoding="utf-8")
        return [line for line in text.splitlines() if line.strip()]
    return list(args.changed_paths or [])


def _read_pr_body(args: argparse.Namespace) -> str:
    if args.pr_body_file:
        return Path(args.pr_body_file).read_text(encoding="utf-8")
    return args.pr_body or ""


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Require acknowledgement when a PR touches high-risk paths.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Gate configuration file.")
    parser.add_argument("--changed-paths", nargs="*", help="Changed paths, space separated.")
    parser.add_argument("--changed-paths-from", help="File with one changed path per line.")
    parser.add_argument("--pr-body", help="Pull-request body text.")
    parser.add_argument("--pr-body-file", help="File containing the pull-request body.")
    parser.add_argument(
        "--report-file",
        help="Write the rendered report here (used as the PR comment body).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report but always exit 0.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}")
        return EXIT_OK if args.dry_run else EXIT_CONFIG_ERROR

    changed_paths = _read_changed_paths(args)
    pr_body = _read_pr_body(args)

    result = evaluate(changed_paths, pr_body, config)
    report = render_report(result, config)
    print(report)

    if args.report_file:
        Path(args.report_file).write_text(report, encoding="utf-8")

    if args.dry_run:
        print(f"(dry run — would exit {result.exit_code})")
        return EXIT_OK
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
