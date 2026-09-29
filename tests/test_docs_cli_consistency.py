"""CI docs-vs-CLI consistency check (Grand 2 / issue #671 acceptance
criterion): every `--flag` mentioned in `docs/model_rollback_runbook.md` and
`docs/model_artifact_lifecycle.md` in the context of a specific script must
actually exist on that script's real argparse parser.

The bug this guards against already happened once: the runbook documented
`--check-shadow`/`--no-shadow` for `scripts/retrain_if_drifted.py` while
those flags did not exist in `parse_args()` at all, so every invocation of
the script crashed with `AttributeError` before reaching any drift-detection
logic. This test would have caught that immediately.
"""

from __future__ import annotations

import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")

# Flags that are legitimately generic/example placeholders in prose, not
# real CLI flags of the script being discussed in that code block.
_IGNORE = {"--help"}


def _flags_in_code_blocks_mentioning(doc_text: str, script_ref: str) -> set[str]:
    """Return every `--flag` token found inside fenced code blocks that also
    mention *script_ref* somewhere in the same block."""
    flags: set[str] = set()
    for block in re.findall(r"```(?:bash)?\n(.*?)```", doc_text, flags=re.DOTALL):
        if script_ref not in block:
            continue
        for match in _FLAG_RE.findall(block):
            if match not in _IGNORE:
                flags.add(match)
    return flags


def test_retrain_if_drifted_runbook_flags_exist_on_the_real_parser():
    from scripts.retrain_if_drifted import build_parser

    parser = build_parser()
    real_flags = {opt for action in parser._actions for opt in action.option_strings}

    for doc_name in ["model_rollback_runbook.md"]:
        doc_path = os.path.join(REPO_ROOT, "docs", doc_name)
        with open(doc_path) as f:
            doc_text = f.read()
        documented = _flags_in_code_blocks_mentioning(doc_text, "scripts.retrain_if_drifted")
        missing = documented - real_flags
        assert not missing, (
            f"{doc_name} documents flag(s) {sorted(missing)} for "
            "scripts.retrain_if_drifted that do not exist on its real argparse "
            "parser — this is exactly the --check-shadow/--no-shadow bug Grand 2 "
            f"(issue #671) fixed. Real flags: {sorted(real_flags)}"
        )
        # And the reverse: the two flags this issue was specifically about
        # must be documented (a script can gain undocumented internal flags
        # without that being a doc bug, but these two are load-bearing for
        # the runbook's entire "how do I promote/rollback" narrative).
        for required in ("--check-shadow", "--no-shadow"):
            assert required in documented, (
                f"{doc_name} no longer documents {required} for "
                "scripts.retrain_if_drifted — the runbook must describe every "
                "shipped promotion/rollback code path."
            )


def test_manage_artifact_lifecycle_runbook_flags_exist_on_the_real_parser():
    from scripts.manage_artifact_lifecycle import build_parser

    parser = build_parser()
    real_flags: set[str] = {opt for action in parser._actions for opt in action.option_strings}
    for sub_action in parser._subparsers._group_actions:
        for sub_parser in sub_action.choices.values():
            for action in sub_parser._actions:
                real_flags.update(action.option_strings)

    for doc_name in ["model_artifact_lifecycle.md", "model_rollback_runbook.md"]:
        doc_path = os.path.join(REPO_ROOT, "docs", doc_name)
        with open(doc_path) as f:
            doc_text = f.read()
        documented = _flags_in_code_blocks_mentioning(doc_text, "scripts.manage_artifact_lifecycle")
        missing = documented - real_flags
        assert not missing, (
            f"{doc_name} documents flag(s) {sorted(missing)} for "
            f"scripts.manage_artifact_lifecycle that do not exist on its real "
            f"argparse parser(s). Real flags: {sorted(real_flags)}"
        )


def test_manage_artifact_lifecycle_promote_and_rollback_require_actor_and_credential():
    """Acceptance criterion: promotion/rollback are authenticated actions —
    a docs-vs-CLI check specifically for the authorization surface, since an
    operator following the runbook must be able to trust that --actor/
    --credential are real, required-in-spirit flags on both commands."""
    from scripts.manage_artifact_lifecycle import build_parser

    parser = build_parser()
    for command in ("promote", "rollback"):
        sub_parser = parser._subparsers._group_actions[0].choices[command]
        flags = {opt for action in sub_parser._actions for opt in action.option_strings}
        assert "--actor" in flags, f"{command} must accept --actor"
        assert "--credential" in flags, f"{command} must accept --credential"
