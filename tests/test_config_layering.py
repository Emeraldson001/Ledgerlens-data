"""Tests for config/layering.py — advanced configuration layering."""

import pytest

from config.layering import ConfigSource, LayeredConfig, detect_environment
from utils.errors import ConfigurationError


@pytest.fixture
def config_dir(tmp_path):
    d = tmp_path / "environments"
    d.mkdir()
    return d


def _write(path, content: str):
    path.write_text(content)


def test_defaults_used_when_no_files_or_env(config_dir, monkeypatch):
    monkeypatch.delenv("LEDGERLENS_LOG_LEVEL", raising=False)
    cfg = LayeredConfig({"log_level": "INFO"}, environment="local", config_dir=config_dir)
    assert cfg.get("log_level") == "INFO"
    assert cfg.source("log_level") == ConfigSource.DEFAULT


def test_base_file_overrides_defaults(config_dir):
    _write(config_dir / "base.yaml", "log_level: WARNING\n")
    cfg = LayeredConfig({"log_level": "INFO"}, environment="local", config_dir=config_dir)
    assert cfg.get("log_level") == "WARNING"
    assert cfg.source("log_level") == ConfigSource.BASE_FILE


def test_env_specific_file_overrides_base_file(config_dir):
    _write(config_dir / "base.yaml", "log_level: WARNING\ndb_pool_size: 5\n")
    _write(config_dir / "ci.yaml", "log_level: ERROR\n")
    cfg = LayeredConfig({"log_level": "INFO"}, environment="ci", config_dir=config_dir)
    assert cfg.get("log_level") == "ERROR"
    assert cfg.get("db_pool_size") == 5
    assert cfg.source("log_level") == ConfigSource.ENV_FILE
    assert cfg.source("db_pool_size") == ConfigSource.BASE_FILE


def test_env_var_overrides_files_and_coerces_int(config_dir, monkeypatch):
    _write(config_dir / "base.yaml", "db_pool_size: 5\n")
    monkeypatch.setenv("LEDGERLENS_DB_POOL_SIZE", "20")
    cfg = LayeredConfig({"db_pool_size": 5}, environment="local", config_dir=config_dir)
    assert cfg.get("db_pool_size") == 20
    assert isinstance(cfg.get("db_pool_size"), int)
    assert cfg.source("db_pool_size") == ConfigSource.ENV_VAR


def test_env_var_coerces_bool_and_list(config_dir, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_FEATURE_FLAG", "true")
    monkeypatch.setenv("LEDGERLENS_TAGS", "a, b, c")
    cfg = LayeredConfig(
        {"feature_flag": False, "tags": []}, environment="local", config_dir=config_dir
    )
    assert cfg.get("feature_flag") is True
    assert cfg.get("tags") == ["a", "b", "c"]


def test_unknown_env_var_key_is_captured_as_string(config_dir, monkeypatch):
    monkeypatch.setenv("LEDGERLENS_NEW_SETTING", "hello")
    cfg = LayeredConfig({}, environment="local", config_dir=config_dir)
    assert cfg.get("new_setting") == "hello"
    assert cfg.source("new_setting") == ConfigSource.ENV_VAR


def test_explicit_override_has_highest_precedence(config_dir, monkeypatch):
    _write(config_dir / "base.yaml", "log_level: WARNING\n")
    monkeypatch.setenv("LEDGERLENS_LOG_LEVEL", "ERROR")
    cfg = LayeredConfig(
        {"log_level": "INFO"},
        environment="local",
        config_dir=config_dir,
        overrides={"log_level": "CRITICAL"},
    )
    assert cfg.get("log_level") == "CRITICAL"
    assert cfg.source("log_level") == ConfigSource.OVERRIDE


def test_require_raises_configuration_error_with_all_missing_keys(config_dir):
    cfg = LayeredConfig({"log_level": "INFO"}, environment="local", config_dir=config_dir)
    with pytest.raises(ConfigurationError) as exc_info:
        cfg.require("risk_score_db_url", "jwt_public_key_path")

    exc = exc_info.value
    assert exc.code == "CFG-002"
    assert exc.context["missing_keys"] == ["risk_score_db_url", "jwt_public_key_path"]


def test_require_passes_when_all_keys_present(config_dir):
    cfg = LayeredConfig(
        {"risk_score_db_url": "sqlite:///x.db"}, environment="local", config_dir=config_dir
    )
    cfg.require("risk_score_db_url")  # should not raise


def test_invalid_yaml_top_level_raises_configuration_error(config_dir):
    _write(config_dir / "base.yaml", "- item1\n- item2\n")
    with pytest.raises(ConfigurationError) as exc_info:
        LayeredConfig({}, environment="local", config_dir=config_dir)
    assert exc_info.value.code == "CFG-001"


def test_explain_lists_every_key_with_its_source(config_dir):
    _write(config_dir / "base.yaml", "log_level: WARNING\n")
    cfg = LayeredConfig(
        {"log_level": "INFO", "db_pool_size": 5}, environment="local", config_dir=config_dir
    )
    output = cfg.explain()
    assert "log_level = 'WARNING'  [base_file]" in output
    assert "db_pool_size = 5  [default]" in output


def test_detect_environment_prefers_explicit_var(monkeypatch):
    monkeypatch.setenv("LEDGERLENS_ENV", "staging")
    monkeypatch.setenv("CI", "true")
    assert detect_environment() == "staging"


def test_detect_environment_falls_back_to_ci_indicator(monkeypatch):
    monkeypatch.delenv("LEDGERLENS_ENV", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert detect_environment() == "ci"


def test_detect_environment_defaults_to_local(monkeypatch):
    monkeypatch.delenv("LEDGERLENS_ENV", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert detect_environment() == "local"


def test_missing_environment_file_is_tolerated(config_dir):
    cfg = LayeredConfig({"log_level": "INFO"}, environment="nonexistent_env", config_dir=config_dir)
    assert cfg.get("log_level") == "INFO"
