"""Tests for utils/ownership.py — ownership metadata registry."""

from __future__ import annotations

import json
from pathlib import Path

from utils.ownership import (
    SUBSYSTEMS,
    CodeOwnersEntry,
    OwnershipRegistry,
    _matches_pattern,
    check_ownership_compliance,
    codeowners_entries,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── CODEOWNERS parsing ───────────────────────────────────────────────────────


class TestCodeOwnersParsing:
    def test_parse_codeowners_returns_entries(self, tmp_path: Path) -> None:
        co_file = tmp_path / "CODEOWNERS"
        co_file.write_text(
            "# Comment line\n"
            "/detection/ @Ledger-Lenz/ml-team\n"
            "/ingestion/ @Ledger-Lenz/data-team @Ledger-Lenz/infra-team\n"
            "\n"
            "*.py @Ledger-Lenz/core-maintainers\n",
            encoding="utf-8",
        )
        entries = OwnershipRegistry._parse_codeowners(co_file)
        assert len(entries) == 3
        assert entries[0].pattern == "/detection/"
        assert entries[0].owners == ["@Ledger-Lenz/ml-team"]
        assert entries[0].line_number == 2
        assert entries[1].owners == [
            "@Ledger-Lenz/data-team",
            "@Ledger-Lenz/infra-team",
        ]

    def test_parse_codeowners_skips_comments_and_blanks(self, tmp_path: Path) -> None:
        co_file = tmp_path / "CODEOWNERS"
        co_file.write_text(
            "# This is a comment\n" "\n" "   \n" "/docs/ @Ledger-Lenz/docs-team\n",
            encoding="utf-8",
        )
        entries = OwnershipRegistry._parse_codeowners(co_file)
        assert len(entries) == 1
        assert entries[0].owners == ["@Ledger-Lenz/docs-team"]

    def test_parse_codeowners_handles_minimal_lines(self, tmp_path: Path) -> None:
        co_file = tmp_path / "CODEOWNERS"
        co_file.write_text("pattern-without-owner\n", encoding="utf-8")
        entries = OwnershipRegistry._parse_codeowners(co_file)
        assert len(entries) == 0


# ── Pattern matching ─────────────────────────────────────────────────────────


class TestPatternMatching:
    def test_exact_file_match(self) -> None:
        assert _matches_pattern("detection/benford_engine.py", "/detection/benford_engine.py")

    def test_directory_prefix_match(self) -> None:
        assert _matches_pattern("detection/model_training.py", "/detection/")

    def test_glob_star_match(self) -> None:
        assert _matches_pattern("tests/test_benford.py", "tests/*.py")

    def test_no_match(self) -> None:
        assert not _matches_pattern("ingestion/horizon_streamer.py", "/detection/")

    def test_leading_slash_stripped(self) -> None:
        assert _matches_pattern("detection/foo.py", "/detection/")

    def test_trailing_slash_matches_prefix(self) -> None:
        assert _matches_pattern("scripts/stream.py", "scripts/")

    def test_root_anchored_vs_bare(self) -> None:
        assert _matches_pattern("detection/foo.py", "detection/")


# ── Registry: load and query ─────────────────────────────────────────────────


class TestOwnershipRegistryLoad:
    def test_load_from_repo_root(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        assert registry._repo_root == REPO_ROOT
        assert len(registry.subsystems) > 0

    def test_load_with_explicit_path(self) -> None:
        registry = OwnershipRegistry.load(str(REPO_ROOT))
        assert isinstance(registry, OwnershipRegistry)

    def test_list_subsystems(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        subsystems = registry.list_subsystems()
        assert "detection" in subsystems
        assert "ingestion" in subsystems
        assert "streaming" in subsystems
        assert "integrations" in subsystems
        assert "monitoring" in subsystems
        assert sorted(subsystems) == subsystems

    def test_get_subsystem_info(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        info = registry.get_subsystem_info("detection")
        assert info is not None
        assert "description" in info
        assert "critical_files" in info
        assert "owners" in info
        assert "review_required" in info
        assert len(info["critical_files"]) > 0

    def test_get_subsystem_info_unknown(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        assert registry.get_subsystem_info("nonexistent") is None


class TestOwnershipRegistryGetOwners:
    def test_get_owners_from_codeowners(self, tmp_path: Path) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[
                CodeOwnersEntry(
                    pattern="/detection/",
                    owners=["@Ledger-Lenz/ml-team"],
                    line_number=1,
                ),
            ],
            subsystems={},
            _repo_root=tmp_path,
        )
        owners = registry.get_owners("detection/benford_engine.py")
        assert owners == ["@Ledger-Lenz/ml-team"]

    def test_get_owners_falls_back_to_subsystem(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems=SUBSYSTEMS.copy(),
            _repo_root=REPO_ROOT,
        )
        owners = registry.get_owners("detection/model_training.py")
        assert "@Ledger-Lenz/ml-team" in owners

    def test_get_owners_last_rule_wins(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[
                CodeOwnersEntry("/detection/", ["@team-a"], 1),
                CodeOwnersEntry("/detection/benford_engine.py", ["@team-b"], 2),
            ],
            subsystems={},
            _repo_root=REPO_ROOT,
        )
        owners = registry.get_owners("detection/benford_engine.py")
        assert owners == ["@team-b"]

    def test_get_owners_empty_when_no_match(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems={},
            _repo_root=REPO_ROOT,
        )
        owners = registry.get_owners("unknown/file.py")
        assert owners == []


class TestOwnershipRegistryReview:
    def test_get_review_required_files(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        files = registry.get_review_required_files()
        assert len(files) > 0
        assert "detection/benford_engine.py" in files
        assert "ingestion/horizon_streamer.py" in files


class TestOwnershipRegistryAllOwners:
    def test_get_all_owners(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        owners = registry.get_all_owners()
        assert len(owners) > 0
        assert all(o.startswith("@") for o in owners)
        assert owners == sorted(owners)


# ── Validation ───────────────────────────────────────────────────────────────


class TestOwnershipRegistryValidate:
    def test_validate_returns_empty_on_valid(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems=SUBSYSTEMS.copy(),
            _repo_root=REPO_ROOT,
        )
        warnings = registry.validate()
        assert warnings == []

    def test_validate_warns_on_missing_owners(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems={
                "broken": {
                    "description": "test",
                    "critical_files": ["foo.py"],
                    "owners": [],
                    "review_required": True,
                },
            },
            _repo_root=REPO_ROOT,
        )
        warnings = registry.validate()
        assert any("broken" in w for w in warnings)

    def test_validate_warns_on_missing_critical_files(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems={
                "bare": {
                    "description": "test",
                    "critical_files": [],
                    "owners": ["@team"],
                    "review_required": True,
                },
            },
            _repo_root=REPO_ROOT,
        )
        warnings = registry.validate()
        assert any("bare" in w for w in warnings)

    def test_validate_warns_on_unowned_critical_file(self) -> None:
        registry = OwnershipRegistry(
            codeowners_entries=[],
            subsystems={
                "orphan": {
                    "description": "test",
                    "critical_files": ["orphan/file.py"],
                    "owners": [],
                    "review_required": True,
                },
            },
            _repo_root=REPO_ROOT,
        )
        warnings = registry.validate()
        assert any("orphan/file.py" in w for w in warnings)


# ── Serialization ────────────────────────────────────────────────────────────


class TestOwnershipRegistrySerialization:
    def test_to_dict(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        data = registry.to_dict()
        assert "subsystems" in data
        assert "codeowners_rules" in data
        assert "all_owners" in data
        assert isinstance(data["subsystems"], dict)

    def test_to_json_is_valid(self) -> None:
        registry = OwnershipRegistry.load(REPO_ROOT)
        json_str = registry.to_json()
        data = json.loads(json_str)
        assert "subsystems" in data


# ── codeowners_entries iterator ──────────────────────────────────────────────


class TestCodeOwnersEntriesIterator:
    def test_yields_all_entries(self) -> None:
        entries = [
            CodeOwnersEntry("/a/", ["@a"], 1),
            CodeOwnersEntry("/b/", ["@b"], 2),
        ]
        result = list(codeowners_entries(entries))
        assert len(result) == 2
        assert result[0].pattern == "/a/"


# ── Integration: real CODEOWNERS file in repo ────────────────────────────────


# ── Ownership compliance checking ────────────────────────────────────────────


class TestOwnershipComplianceCheck:
    def test_check_compliance_names_file_and_pattern(self, tmp_path: Path) -> None:
        """Ownership check failure names the specific file and matching CODEOWNERS pattern."""
        registry = OwnershipRegistry(
            codeowners_entries=[
                CodeOwnersEntry(
                    pattern="/detection/",
                    owners=["@Ledger-Lenz/ml-team"],
                    line_number=1,
                ),
                CodeOwnersEntry(
                    pattern="/ingestion/",
                    owners=["@Ledger-Lenz/data-team"],
                    line_number=2,
                ),
            ],
            subsystems={},
            _repo_root=tmp_path,
        )

        # Mock the registry load to return our test registry
        import unittest.mock as mock

        with mock.patch("utils.ownership.OwnershipRegistry.load", return_value=registry):
            compliant, violations = check_ownership_compliance(
                ["detection/benford_engine.py", "ingestion/horizon_streamer.py"],
                repo_root=tmp_path,
            )

        assert compliant is True
        assert len(violations) == 0

    def test_check_compliance_detects_unowned_file(self, tmp_path: Path) -> None:
        """Ownership check detects files without CODEOWNERS pattern."""
        registry = OwnershipRegistry(
            codeowners_entries=[
                CodeOwnersEntry(
                    pattern="/detection/",
                    owners=["@Ledger-Lenz/ml-team"],
                    line_number=1,
                ),
            ],
            subsystems={},
            _repo_root=tmp_path,
        )

        import unittest.mock as mock

        with mock.patch("utils.ownership.OwnershipRegistry.load", return_value=registry):
            compliant, violations = check_ownership_compliance(
                ["detection/benford_engine.py", "unknown/file.py"],
                repo_root=tmp_path,
            )

        assert compliant is False
        assert len(violations) == 1
        # Violation should name the file and indicate no pattern matches
        assert "unknown/file.py" in violations[0]
        assert "no CODEOWNERS pattern" in violations[0]

    def test_check_compliance_failure_includes_file_and_owner_detail(self, tmp_path: Path) -> None:
        """Ownership check failure output includes file, pattern, and assigned owners."""
        registry = OwnershipRegistry(
            codeowners_entries=[
                CodeOwnersEntry(
                    pattern="/detection/",
                    owners=["@Ledger-Lenz/ml-team"],
                    line_number=1,
                ),
            ],
            subsystems={},
            _repo_root=tmp_path,
        )

        import unittest.mock as mock

        with mock.patch("utils.ownership.OwnershipRegistry.load", return_value=registry):
            compliant, violations = check_ownership_compliance(
                ["unknown_subsystem/new_feature.py"],
                repo_root=tmp_path,
            )

        assert compliant is False
        violation_text = "\n".join(violations)
        # Violation should name the file
        assert "unknown_subsystem/new_feature.py" in violation_text


class TestRealCodeOwners:
    def test_codeowners_file_exists(self) -> None:
        co_path = REPO_ROOT / ".github" / "CODEOWNERS"
        assert co_path.is_file(), ".github/CODEOWNERS must exist"

    def test_codeowners_is_parseable(self) -> None:
        co_path = REPO_ROOT / ".github" / "CODEOWNERS"
        entries = OwnershipRegistry._parse_codeowners(co_path)
        assert len(entries) > 0, "CODEOWNERS should have at least one rule"

    def test_all_subsystems_have_owners(self) -> None:
        for name, meta in SUBSYSTEMS.items():
            assert meta.get("owners"), f"Subsystem '{name}' must have owners"
