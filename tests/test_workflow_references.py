"""Assert that CI workflows reference things that actually exist (issue #607).

Workflow YAML cannot be unit tested directly, but its realistic failure mode
can be: a script or Makefile target is renamed, the `run:` line still points at
the old name, and the check silently stops running. Nothing else in the suite
would notice, because a workflow step that never executes reports no failure.

This scans every workflow for references to repo files, Python modules, and
Makefile targets, and fails when one does not resolve.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
MAKEFILE = REPO_ROOT / "Makefile"

#: `python scripts/foo.py` / `python3 scripts/foo.py`
_SCRIPT_PATH_RE = re.compile(r"\bpython3?\s+(?P<path>[\w./-]+\.py)\b")
#: `python -m scripts.foo`
_MODULE_RE = re.compile(r"\bpython3?\s+-m\s+(?P<module>[\w.]+)\b")
#: `make target` (excluding `make -C`, variable assignments)
_MAKE_TARGET_RE = re.compile(r"\bmake\s+(?P<target>[a-z][\w-]*)\b")


def _workflow_files() -> list[Path]:
    return sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))


def _run_blocks(workflow: Path) -> list[str]:
    """Every `run:` script body in *workflow*."""
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    blocks: list[str] = []

    for job in (data.get("jobs") or {}).values():
        for step in job.get("steps") or []:
            run = step.get("run")
            if isinstance(run, str):
                blocks.append(run)
    return blocks


def _makefile_targets() -> set[str]:
    targets: set[str] = set()
    for line in MAKEFILE.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^(?P<name>[a-zA-Z][\w-]*)\s*:(?!=)", line)
        if match:
            targets.add(match.group("name"))
    return targets


def _module_exists(module: str) -> bool:
    parts = module.split(".")
    return (REPO_ROOT / Path(*parts).with_suffix(".py")).is_file() or (
        REPO_ROOT / Path(*parts) / "__init__.py"
    ).is_file()


def test_workflow_directory_is_present():
    assert _workflow_files(), "no workflow files found"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_workflow_is_valid_yaml(workflow):
    assert yaml.safe_load(workflow.read_text(encoding="utf-8"))


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_referenced_script_paths_exist(workflow):
    missing = []
    for block in _run_blocks(workflow):
        for match in _SCRIPT_PATH_RE.finditer(block):
            path = match.group("path")
            # Skip heredoc-generated temp scripts and absolute paths.
            if path.startswith(("/", "$")):
                continue
            if not (REPO_ROOT / path).is_file():
                missing.append(path)
    assert not missing, f"{workflow.name} references missing script(s): {missing}"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_referenced_python_modules_exist(workflow):
    missing = []
    for block in _run_blocks(workflow):
        for match in _MODULE_RE.finditer(block):
            module = match.group("module")
            # Third-party tools invoked as modules are not repo paths.
            if not module.startswith(("scripts.", "detection.", "ingestion.", "streaming.")):
                continue
            if not _module_exists(module):
                missing.append(module)
    assert not missing, f"{workflow.name} references missing module(s): {missing}"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_referenced_make_targets_exist(workflow):
    targets = _makefile_targets()
    missing = []
    for block in _run_blocks(workflow):
        for match in _MAKE_TARGET_RE.finditer(block):
            target = match.group("target")
            if target not in targets:
                missing.append(target)
    assert not missing, f"{workflow.name} references missing make target(s): {missing}"


# ---------------------------------------------------------------------------
# The review gates specifically — these are the checks this PR adds, so assert
# they are actually wired rather than merely present on disk.
# ---------------------------------------------------------------------------


def test_schema_compatibility_is_wired_into_ci():
    blocks = " ".join(_run_blocks(WORKFLOW_DIR / "ci.yml"))
    assert "check-schema-compatibility" in blocks


def test_review_gates_workflow_runs_the_gate_script():
    blocks = " ".join(_run_blocks(WORKFLOW_DIR / "review-gates.yml"))
    assert "scripts/check_review_gates.py" in blocks


def test_review_gates_workflow_reacts_to_description_edits():
    """Without `edited` the gate would not re-run when the body is fixed."""
    data = yaml.safe_load((WORKFLOW_DIR / "review-gates.yml").read_text(encoding="utf-8"))
    # `on` is parsed as the boolean True by YAML 1.1.
    triggers = data.get("on") or data.get(True)
    assert "edited" in triggers["pull_request"]["types"]


def test_review_gates_workflow_can_comment_on_pull_requests():
    data = yaml.safe_load((WORKFLOW_DIR / "review-gates.yml").read_text(encoding="utf-8"))
    assert data["permissions"]["pull-requests"] == "write"


def test_gate_config_and_checklist_doc_are_present():
    assert (REPO_ROOT / ".github" / "review-gates.yml").is_file()
    assert (REPO_ROOT / ".github" / "review-checklists.md").is_file()


def test_every_gate_has_a_section_in_the_checklist_doc():
    """A gate CI can fire but the doc never explains is a dead end for the author."""
    config = yaml.safe_load(
        (REPO_ROOT / ".github" / "review-gates.yml").read_text(encoding="utf-8")
    )
    doc = (REPO_ROOT / ".github" / "review-checklists.md").read_text(encoding="utf-8")

    missing = [g["heading"] for g in config["gates"] if f"## {g['heading']}" not in doc]
    assert not missing, f"checklist doc has no section for: {missing}"
