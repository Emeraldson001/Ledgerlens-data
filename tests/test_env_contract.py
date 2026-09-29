"""Tests for environment contract docs generation (Issue #544)."""

from pathlib import Path

from config.env_contract import build_env_contract, render_markdown

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_build_env_contract_parses_real_config():
    entries = build_env_contract(PROJECT_ROOT / "config.py")
    assert len(entries) > 100
    names = [e.name for e in entries]
    assert "HORIZON_URL" in names
    assert "STELLAR_NETWORK" in names


def test_build_env_contract_dedupes_repeated_attribute_last_wins(tmp_path):
    source = tmp_path / "cfg.py"
    source.write_text(
        "import os\n\n"
        "class Config:\n"
        "    FOO: int = int(os.getenv('FOO', '1'))\n"
        "    FOO: int = int(os.getenv('FOO', '2'))\n"
    )

    entries = build_env_contract(source)

    matches = [e for e in entries if e.name == "FOO"]
    assert len(matches) == 1
    assert matches[0].default == "'2'"


def test_build_env_contract_marks_no_default_as_required(tmp_path):
    source = tmp_path / "cfg.py"
    source.write_text(
        "import os\n\n" "class Config:\n" "    SECRET: str | None = os.getenv('SECRET')\n"
    )

    entries = build_env_contract(source)

    assert entries[0].required is True
    assert entries[0].env_var == "SECRET"
    assert entries[0].default is None


def test_build_env_contract_extracts_description_and_section(tmp_path):
    source = tmp_path / "cfg.py"
    source.write_text(
        "import os\n\n"
        "class Config:\n"
        "    # ---------------------------------------------------------------\n"
        "    # Widget subsystem\n"
        "    # ---------------------------------------------------------------\n"
        "    # Number of widgets to spin up at boot.\n"
        "    WIDGET_COUNT: int = int(os.getenv('WIDGET_COUNT', '3'))\n"
    )

    entries = build_env_contract(source)

    assert entries[0].section == "Widget subsystem"
    assert entries[0].description == "Number of widgets to spin up at boot."


def test_build_env_contract_missing_class_raises(tmp_path):
    source = tmp_path / "cfg.py"
    source.write_text("class NotConfig:\n    pass\n")

    import pytest

    with pytest.raises(ValueError):
        build_env_contract(source)


def test_render_markdown_contains_table_header_and_entries():
    entries = build_env_contract(PROJECT_ROOT / "config.py")
    rendered = render_markdown(entries)

    assert rendered.startswith("# Environment Variable Contract")
    assert "| Variable | Env Var | Type | Required | Default | Description |" in rendered
    assert "`HORIZON_URL`" in rendered


def test_committed_environment_contract_doc_matches_generator():
    """Doc-drift guard: docs/environment_contract.md must equal what the
    generator produces from the current config.py — CI's `make
    env-docs-check` runs the same comparison; this keeps it enforced
    under plain `pytest` too."""
    entries = build_env_contract(PROJECT_ROOT / "config.py")
    rendered = render_markdown(entries)
    committed = (PROJECT_ROOT / "docs" / "environment_contract.md").read_text()
    assert rendered == committed, (
        "docs/environment_contract.md is out of date with config.py — "
        "run `make env-docs` to regenerate it."
    )
