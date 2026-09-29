"""Acceptance test for advanced work-item issue-to-test traceability."""

from pathlib import Path

import pytest


@pytest.mark.issue("ADV-004")
def test_advanced_work_item_traceability_is_complete():
    from scripts.check_issue_test_traceability import (
        collect_marked_tests,
        load_manifest,
        validate_traceability,
    )

    repository_root = Path(__file__).parent.parent.resolve()
    manifest = load_manifest(repository_root / "tests" / "issue_traceability.json")
    collected = collect_marked_tests(repository_root / "tests", repository_root)

    assert validate_traceability(manifest, collected) == []
