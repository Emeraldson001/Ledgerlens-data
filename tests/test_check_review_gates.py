"""Tests for the high-risk path review gate (issue #607).

The gate's value depends on it firing for the right paths and *not* firing for
paths that only look risky, so both directions are asserted explicitly. The
deliberate exclusions recorded in `.github/review-gates.yml` are encoded here as
executable assertions rather than prose.
"""

from pathlib import Path

import pytest

from scripts.check_review_gates import (
    EXIT_CONFIG_ERROR,
    EXIT_OK,
    EXIT_UNACKNOWLEDGED,
    ConfigError,
    Gate,
    GateConfig,
    evaluate,
    find_override,
    gate_matches,
    load_config,
    main,
    render_report,
    section_body,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / ".github" / "review-gates.yml"
FIXTURES = Path(__file__).parent / "fixtures" / "review_gates"


@pytest.fixture
def config() -> GateConfig:
    return load_config(CONFIG_PATH)


def _body_with(heading: str, prose: str) -> str:
    return f"## Summary\n\nSomething.\n\n### {heading}\n\n{prose}\n"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_repo_config_loads(config):
    assert config.gates
    assert config.min_prose_chars > 0
    assert config.override_prefix


def test_repo_config_gate_ids_are_unique(config):
    ids = [g.id for g in config.gates]
    assert len(ids) == len(set(ids))


def test_missing_config_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("gates: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_config_without_gates_raises(tmp_path):
    bad = tmp_path / "empty.yml"
    bad.write_text("version: 1\ngates: []\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="non-empty"):
        load_config(bad)


def test_config_with_duplicate_ids_raises(tmp_path):
    bad = tmp_path / "dup.yml"
    bad.write_text(
        "gates:\n"
        "  - id: a\n    heading: A\n    paths: ['x.py']\n"
        "  - id: a\n    heading: B\n    paths: ['y.py']\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate"):
        load_config(bad)


def test_config_gate_missing_field_raises(tmp_path):
    bad = tmp_path / "partial.yml"
    bad.write_text("gates:\n  - id: a\n    heading: A\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="paths"):
        load_config(bad)


# ---------------------------------------------------------------------------
# Path matching — what fires
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected_gate"),
    [
        ("data/trade_avro_schema.json", "kafka-wire-schema"),
        ("ingestion/data_models.py", "shared-contract"),
        ("detection/model_inference.py", "shared-contract"),
        ("detection/feature_engineering.py", "model-behaviour"),
        ("detection/model_training.py", "model-behaviour"),
        ("detection/persistence.py", "database-schema"),
        ("scripts/migrate_add_ring_id.py", "database-schema"),
        ("config/tenants.yaml", "tenant-config"),
        ("models/metrics.json", "model-metadata"),
    ],
)
def test_high_risk_paths_trigger_expected_gate(config, path, expected_gate):
    result = evaluate([path], "", config)
    assert [g.id for g in result.triggered] == [expected_gate]


# ---------------------------------------------------------------------------
# Path matching — deliberate exclusions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        # Generated statistics, read by nothing at runtime; FEATURE_RANGES is
        # hardcoded in detection/feature_engineering.py.
        "data/feature_ranges.json",
        # Gitignored — can never appear in a diff.
        "models/xgboost.joblib",
        "models/random_forest.pkl",
        # Already enforced at runtime by reporting/model_card_generator.py.
        "reporting/schemas/model_metadata.json",
        # Ordinary changes must not be gated.
        "README.md",
        "tests/test_benford.py",
        "detection/benford_engine.py",
        "ingestion/horizon_streamer.py",
    ],
)
def test_excluded_paths_do_not_trigger_any_gate(config, path):
    result = evaluate([path], "", config)
    assert result.triggered == []
    assert result.exit_code == EXIT_OK


def test_no_changed_paths_passes_with_empty_body(config):
    assert evaluate([], "", config).exit_code == EXIT_OK


def test_gate_matches_normalises_leading_dot_slash():
    gate = Gate(id="g", heading="H", paths=("config/tenants.yaml",))
    assert gate_matches(gate, ["./config/tenants.yaml"]) == ["config/tenants.yaml"]


def test_gate_matches_ignores_blank_lines():
    gate = Gate(id="g", heading="H", paths=("config/tenants.yaml",))
    assert gate_matches(gate, ["", "  ", "config/tenants.yaml"]) == ["config/tenants.yaml"]


# ---------------------------------------------------------------------------
# Acknowledgement
# ---------------------------------------------------------------------------


def test_substantive_prose_satisfies_the_gate(config):
    body = _body_with(
        "Kafka wire schema",
        "Dual-write for one retention window; consumers updated in core#412.",
    )
    result = evaluate(["data/trade_avro_schema.json"], body, config)

    assert [g.id for g in result.acknowledged] == ["kafka-wire-schema"]
    assert result.missing == []
    assert result.exit_code == EXIT_OK


def test_missing_heading_fails_the_gate(config):
    result = evaluate(["data/trade_avro_schema.json"], "## Summary\n\nA change.\n", config)

    assert [g.id for g in result.missing] == ["kafka-wire-schema"]
    assert result.exit_code == EXIT_UNACKNOWLEDGED


def test_heading_matching_is_case_and_level_insensitive(config):
    body = "###### kafka WIRE schema\n\nDual-write across one full retention window.\n"
    result = evaluate(["data/trade_avro_schema.json"], body, config)
    assert result.missing == []


@pytest.mark.parametrize("prose", ["", "   ", "TODO", "n/a", "...", "-"])
def test_placeholder_text_does_not_satisfy_the_gate(config, prose):
    body = _body_with("Kafka wire schema", prose)
    result = evaluate(["data/trade_avro_schema.json"], body, config)
    assert [g.id for g in result.missing] == ["kafka-wire-schema"]


def test_a_bare_checkbox_does_not_satisfy_the_gate(config):
    """A tick is a one-bit signal; the gate requires prose."""
    body = _body_with("Kafka wire schema", "- [x] done")
    result = evaluate(["data/trade_avro_schema.json"], body, config)
    assert [g.id for g in result.missing] == ["kafka-wire-schema"]


def test_prose_shorter_than_the_minimum_does_not_satisfy(config):
    body = _body_with("Kafka wire schema", "ok")
    result = evaluate(["data/trade_avro_schema.json"], body, config)
    assert [g.id for g in result.missing] == ["kafka-wire-schema"]


def test_each_triggered_gate_needs_its_own_acknowledgement(config):
    body = _body_with("Model behaviour", "Retrained all three models; AUC within 0.4% of baseline.")
    result = evaluate(
        ["detection/feature_engineering.py", "detection/persistence.py"], body, config
    )

    assert [g.id for g in result.acknowledged] == ["model-behaviour"]
    assert [g.id for g in result.missing] == ["database-schema"]
    assert result.exit_code == EXIT_UNACKNOWLEDGED


def test_section_body_stops_at_the_next_heading():
    body = "### One\n\nfirst section\n\n### Two\n\nsecond section\n"
    assert "first section" in section_body(body, "One")
    assert "second section" not in section_body(body, "One")


def test_section_body_returns_none_when_absent():
    assert section_body("## Summary\n\ntext\n", "Nope") is None


# ---------------------------------------------------------------------------
# Override
# ---------------------------------------------------------------------------


def test_override_with_a_reason_passes(config):
    body = (
        "## Summary\n\nHotfix.\n\n"
        "Review-Gate-Override: Incident INC-2291, migration note follows same day.\n"
    )
    result = evaluate(["data/trade_avro_schema.json"], body, config)

    assert result.overridden
    assert result.missing
    assert result.exit_code == EXIT_OK


def test_override_without_a_substantive_reason_does_not_pass(config):
    body = "## Summary\n\nHotfix.\n\nReview-Gate-Override: TODO\n"
    result = evaluate(["data/trade_avro_schema.json"], body, config)

    assert not result.overridden
    assert result.exit_code == EXIT_UNACKNOWLEDGED


def test_override_is_ignored_when_nothing_is_missing(config):
    body = (
        _body_with("Kafka wire schema", "Dual-write for one retention window; core#412 updated.")
        + "\nReview-Gate-Override: not needed but present anyway, ignore me.\n"
    )
    result = evaluate(["data/trade_avro_schema.json"], body, config)

    assert result.missing == []
    assert not result.overridden


def test_find_override_tolerates_list_and_quote_markers(config):
    assert find_override("- Review-Gate-Override: incident hotfix, note follows same day\n", config)
    assert find_override("> Review-Gate-Override: incident hotfix, note follows same day\n", config)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_report_is_quiet_when_no_gate_fires(config):
    assert "do not apply" in render_report(evaluate(["README.md"], "", config), config)


def test_report_names_the_missing_heading_and_the_override_hint(config):
    result = evaluate(["config/tenants.yaml"], "", config)
    report = render_report(result, config)

    assert "Tenant configuration" in report
    assert config.override_prefix in report
    assert ".github/review-checklists.md" in report


def test_report_shows_the_override_reason(config):
    body = "Review-Gate-Override: Incident INC-2291, migration note follows same day.\n"
    report = render_report(evaluate(["config/tenants.yaml"], body, config), config)
    assert "overridden" in report.lower()
    assert "INC-2291" in report


# ---------------------------------------------------------------------------
# CLI — the exact surface CI invokes
# ---------------------------------------------------------------------------


def test_cli_fails_on_unacknowledged_fixture():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_schema.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_missing.md"),
        ]
    )
    assert code == EXIT_UNACKNOWLEDGED


def test_cli_passes_on_acknowledged_fixture():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_schema.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_acknowledged.md"),
        ]
    )
    assert code == EXIT_OK


def test_cli_rejects_placeholder_fixture():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_schema.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_placeholder.md"),
        ]
    )
    assert code == EXIT_UNACKNOWLEDGED


def test_cli_accepts_override_fixture():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_schema.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_override.md"),
        ]
    )
    assert code == EXIT_OK


def test_cli_passes_on_safe_paths_fixture():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_safe.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_missing.md"),
        ]
    )
    assert code == EXIT_OK


def test_cli_dry_run_never_fails():
    code = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths-from",
            str(FIXTURES / "changed_paths_schema.txt"),
            "--pr-body-file",
            str(FIXTURES / "pr_body_missing.md"),
            "--dry-run",
        ]
    )
    assert code == EXIT_OK


def test_cli_reports_config_error(tmp_path):
    assert main(["--config", str(tmp_path / "absent.yml")]) == EXIT_CONFIG_ERROR


def test_cli_writes_the_report_file(tmp_path):
    report = tmp_path / "report.md"
    main(
        [
            "--config",
            str(CONFIG_PATH),
            "--changed-paths",
            "config/tenants.yaml",
            "--pr-body",
            "",
            "--report-file",
            str(report),
        ]
    )
    assert "Tenant configuration" in report.read_text(encoding="utf-8")
