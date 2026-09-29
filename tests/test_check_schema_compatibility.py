"""Tests for the Avro schema compatibility gate (issue #607).

The compatibility rules themselves live in `ingestion/avro_codec.py`; these
tests cover the gate around them — baseline resolution, the new-schema and
unchanged short circuits, and the exit codes CI depends on.

Baseline retrieval is injected, so no test shells out to git.
"""

import json
from pathlib import Path

import pytest

from scripts.check_schema_compatibility import (
    EXIT_BASELINE_UNRESOLVED,
    EXIT_INCOMPATIBLE,
    EXIT_OK,
    BaselineRefError,
    BaselineUnavailable,
    default_baseline_ref,
    evaluate,
    main,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_SCHEMA = REPO_ROOT / "data" / "trade_avro_schema.json"


def base_schema() -> dict:
    return {
        "type": "record",
        "name": "Trade",
        "fields": [
            {"name": "trade_id", "type": "string"},
            {"name": "price", "type": "double"},
        ],
    }


def _with_fields(fields: list[dict]) -> dict:
    schema = base_schema()
    schema["fields"] = fields
    return schema


def _reader(schema: dict):
    """Build an injectable baseline reader returning *schema*."""

    def read(ref: str, path: str) -> dict:
        return schema

    return read


def _raiser(exc: Exception):
    def read(ref: str, path: str) -> dict:
        raise exc

    return read


# ---------------------------------------------------------------------------
# evaluate() — compatibility rules as wired by the gate
# ---------------------------------------------------------------------------


def test_identical_schemas_are_compatible():
    ok, violations = evaluate(base_schema(), base_schema())
    assert ok
    assert violations == []


def test_adding_an_optional_field_with_a_default_is_compatible():
    new = base_schema()
    new["fields"].append({"name": "extra", "type": ["null", "string"], "default": None})

    ok, violations = evaluate(base_schema(), new)
    assert ok, violations


def test_adding_a_required_field_is_incompatible():
    new = base_schema()
    new["fields"].append({"name": "extra", "type": "string"})

    ok, violations = evaluate(base_schema(), new)
    assert not ok
    assert any("extra" in v for v in violations)


def test_removing_a_required_field_is_incompatible():
    new = _with_fields([{"name": "trade_id", "type": "string"}])

    ok, violations = evaluate(base_schema(), new)
    assert not ok
    assert any("price" in v for v in violations)


def test_renaming_a_field_is_incompatible_both_ways():
    new = _with_fields([{"name": "trade_id", "type": "string"}, {"name": "cost", "type": "double"}])

    ok, violations = evaluate(base_schema(), new)
    assert not ok
    assert any("[backward]" in v for v in violations)
    assert any("[forward]" in v for v in violations)


def test_mode_restricts_which_direction_is_reported():
    new = base_schema()
    new["fields"].append({"name": "extra", "type": "string"})

    _, backward_only = evaluate(base_schema(), new, mode="backward")
    _, forward_only = evaluate(base_schema(), new, mode="forward")

    assert all(v.startswith("[backward]") for v in backward_only)
    assert all(v.startswith("[forward]") for v in forward_only)


def test_removing_an_optional_field_is_compatible():
    old = _with_fields(
        [
            {"name": "trade_id", "type": "string"},
            {"name": "extra", "type": ["null", "string"], "default": None},
        ]
    )
    new = _with_fields([{"name": "trade_id", "type": "string"}])

    ok, violations = evaluate(old, new)
    assert ok, violations


# ---------------------------------------------------------------------------
# Baseline resolution
# ---------------------------------------------------------------------------


def test_default_baseline_ref_uses_github_base_ref(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "release-1.x")
    assert default_baseline_ref() == "origin/release-1.x"


def test_default_baseline_ref_falls_back_to_origin_main(monkeypatch):
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)
    assert default_baseline_ref() == "origin/main"


def test_blank_github_base_ref_falls_back(monkeypatch):
    monkeypatch.setenv("GITHUB_BASE_REF", "   ")
    assert default_baseline_ref() == "origin/main"


# ---------------------------------------------------------------------------
# main() — exit codes CI depends on
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, schema: dict) -> Path:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


def test_unchanged_schema_exits_ok(tmp_path):
    path = _write(tmp_path, base_schema())
    code = main(["--schema-path", str(path)], baseline_reader=_reader(base_schema()))
    assert code == EXIT_OK


def test_compatible_change_exits_ok(tmp_path):
    new = base_schema()
    new["fields"].append({"name": "extra", "type": ["null", "string"], "default": None})
    path = _write(tmp_path, new)

    code = main(["--schema-path", str(path)], baseline_reader=_reader(base_schema()))
    assert code == EXIT_OK


def test_incompatible_change_exits_one(tmp_path):
    new = base_schema()
    new["fields"].append({"name": "extra", "type": "string"})
    path = _write(tmp_path, new)

    code = main(["--schema-path", str(path)], baseline_reader=_reader(base_schema()))
    assert code == EXIT_INCOMPATIBLE


def test_new_schema_without_a_baseline_exits_ok(tmp_path):
    """A file that does not exist on the base branch is compatible by definition."""
    path = _write(tmp_path, base_schema())
    code = main(
        ["--schema-path", str(path)],
        baseline_reader=_raiser(BaselineUnavailable("absent")),
    )
    assert code == EXIT_OK


def test_unresolvable_ref_exits_two_not_one(tmp_path):
    """Infrastructure failure must be distinguishable from a schema break."""
    path = _write(tmp_path, base_schema())
    code = main(
        ["--schema-path", str(path)],
        baseline_reader=_raiser(BaselineRefError("no such ref")),
    )
    assert code == EXIT_BASELINE_UNRESOLVED


def test_missing_schema_file_exits_two(tmp_path):
    code = main(
        ["--schema-path", str(tmp_path / "absent.json")],
        baseline_reader=_reader(base_schema()),
    )
    assert code == EXIT_BASELINE_UNRESOLVED


@pytest.mark.parametrize(
    "reader",
    [
        _raiser(BaselineRefError("no such ref")),
        _reader(base_schema()),
    ],
)
def test_dry_run_never_fails(tmp_path, reader):
    new = base_schema()
    new["fields"].append({"name": "extra", "type": "string"})
    path = _write(tmp_path, new)

    assert main(["--schema-path", str(path), "--dry-run"], baseline_reader=reader) == EXIT_OK


def test_incompatible_change_prints_violations(tmp_path, capsys):
    new = base_schema()
    new["fields"].append({"name": "extra", "type": "string"})
    path = _write(tmp_path, new)

    main(["--schema-path", str(path)], baseline_reader=_reader(base_schema()))
    out = capsys.readouterr().out

    assert "extra" in out
    assert "data/schema_evolution.md" in out


# ---------------------------------------------------------------------------
# The real schema
# ---------------------------------------------------------------------------


def test_repo_schema_is_valid_json_and_self_compatible():
    schema = json.loads(REAL_SCHEMA.read_text(encoding="utf-8"))
    ok, violations = evaluate(schema, schema)
    assert ok, violations


def test_repo_schema_checked_against_itself_exits_ok():
    schema = json.loads(REAL_SCHEMA.read_text(encoding="utf-8"))
    code = main(["--schema-path", str(REAL_SCHEMA)], baseline_reader=_reader(schema))
    assert code == EXIT_OK
