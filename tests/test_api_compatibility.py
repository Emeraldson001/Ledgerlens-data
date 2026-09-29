"""Tests for scripts/check_api_compatibility.py.

Extraction logic is exercised against synthetic package trees so these
tests don't depend on (or get broken by) unrelated changes to the real
public API surface; a smoke test at the bottom validates the checked-in
tests/fixtures/api_baseline.json is still internally consistent.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import check_api_compatibility as api_check  # noqa: E402


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(api_check, "REPO_ROOT", tmp_path)
    return tmp_path


def _write(root: Path, relpath: str, source: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source))


class TestExtraction:
    def test_extracts_function_signature_defined_locally_in_init(self, fake_repo, tmp_path):
        _write(
            tmp_path,
            "widgets/__init__.py",
            """
            def build(name: str, count: int = 1) -> list:
                return [name] * count

            __all__ = ["build"]
            """,
        )
        api = api_check.extract_public_api("widgets")
        assert api == {"build": {"kind": "function", "signature": "(name: str, count: int = 1)"}}

    def test_extracts_function_imported_from_submodule(self, fake_repo, tmp_path):
        _write(tmp_path, "widgets/__init__.py", "from .core import score\n\n__all__ = ['score']\n")
        _write(
            tmp_path,
            "widgets/core.py",
            "def score(x, *, threshold=0.5):\n    return x > threshold\n",
        )
        api = api_check.extract_public_api("widgets")
        assert api == {"score": {"kind": "function", "signature": "(x, *, threshold = 0.5)"}}

    def test_extracts_class_init_and_public_methods_only(self, fake_repo, tmp_path):
        _write(
            tmp_path,
            "widgets/__init__.py",
            """
            from widgets.core import Widget

            __all__ = ["Widget"]
            """,
        )
        _write(
            tmp_path,
            "widgets/core.py",
            """
            class Widget:
                def __init__(self, name):
                    self.name = name

                def spin(self):
                    ...

                def _private(self):
                    ...
            """,
        )
        api = api_check.extract_public_api("widgets")
        assert api == {
            "Widget": {
                "kind": "class",
                "init_signature": "(self, name)",
                "public_methods": {"spin": "(self)"},
            }
        }

    def test_unresolvable_export_is_reported_not_crashed(self, fake_repo, tmp_path):
        _write(
            tmp_path, "widgets/__init__.py", "from .core import UNKNOWN\n\n__all__ = ['UNKNOWN']\n"
        )
        _write(tmp_path, "widgets/core.py", "x = 1\n")
        api = api_check.extract_public_api("widgets")
        assert api["UNKNOWN"]["kind"] == "unresolved"

    def test_package_without_all_yields_empty_surface(self, fake_repo, tmp_path):
        _write(tmp_path, "widgets/__init__.py", "def helper():\n    ...\n")
        assert api_check.extract_public_api("widgets") == {}


class TestCompare:
    def test_no_diagnostics_when_identical(self):
        snapshot = {"pkg": {"f": {"kind": "function", "signature": "(x)"}}}
        assert api_check.compare(snapshot, snapshot) == []

    def test_removed_symbol_is_flagged(self):
        baseline = {"pkg": {"f": {"kind": "function", "signature": "(x)"}}}
        current = {"pkg": {}}
        diagnostics = api_check.compare(baseline, current)
        assert len(diagnostics) == 1
        assert "removed" in diagnostics[0]

    def test_changed_signature_is_flagged_with_before_after(self):
        baseline = {"pkg": {"f": {"kind": "function", "signature": "(x)"}}}
        current = {"pkg": {"f": {"kind": "function", "signature": "(x, y)"}}}
        diagnostics = api_check.compare(baseline, current)
        assert len(diagnostics) == 1
        assert "before" in diagnostics[0] and "after" in diagnostics[0]

    def test_new_symbol_addition_is_not_flagged(self):
        baseline = {"pkg": {"f": {"kind": "function", "signature": "(x)"}}}
        current = {
            "pkg": {
                "f": {"kind": "function", "signature": "(x)"},
                "g": {"kind": "function", "signature": "()"},
            }
        }
        assert api_check.compare(baseline, current) == []


class TestUpdateBaseline:
    def test_update_baseline_writes_current_extraction(self, fake_repo, tmp_path):
        _write(tmp_path, "widgets/__init__.py", "def build():\n    ...\n\n__all__ = ['build']\n")
        baseline_path = tmp_path / "baseline.json"
        # Exercise the underlying helpers directly rather than argv parsing.
        current = api_check.extract_all(["widgets"])
        baseline_path.write_text(json.dumps(current, indent=2, sort_keys=True))
        reloaded = json.loads(baseline_path.read_text())
        assert reloaded == {"widgets": {"build": {"kind": "function", "signature": "()"}}}


class TestRealBaselineFixture:
    def test_checked_in_baseline_is_internally_consistent(self):
        baseline = json.loads(api_check.DEFAULT_BASELINE.read_text())
        assert set(baseline) <= set(api_check.PUBLIC_PACKAGES)
        for _package, symbols in baseline.items():
            assert isinstance(symbols, dict)
            for _name, desc in symbols.items():
                assert desc["kind"] in {"function", "class", "unresolved", "unknown"}

    def test_current_source_tree_matches_baseline(self):
        """Guards against unreviewed public API drift: fails with an
        actionable diff if a symbol in api_baseline.json was removed or had
        its signature changed without updating the baseline."""
        baseline = json.loads(api_check.DEFAULT_BASELINE.read_text())
        current = api_check.extract_all(list(baseline.keys()))
        diagnostics = api_check.compare(baseline, current)
        assert diagnostics == [], "\n".join(diagnostics)
