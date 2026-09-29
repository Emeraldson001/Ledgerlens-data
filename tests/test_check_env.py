"""Tests for scripts/check_env.py, the `make check-env` dev command."""

import json

import pytest

from config import Config
from scripts import check_env


def test_single_passing_mode_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", ["$2b$somehash"])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 60)

    code = check_env.main(["--mode", "api"])

    out = capsys.readouterr().out
    assert code == 0
    assert "=== api ===" in out
    assert "all checks passed" in out


def test_single_failing_mode_exits_nonzero_and_prints_reason(monkeypatch, capsys):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", [])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 60)

    code = check_env.main(["--mode", "api"])

    out = capsys.readouterr().out
    assert code == 1
    assert "=== api ===" in out
    assert "FAIL: API_KEYS" in out


def test_all_flag_checks_every_registered_mode(monkeypatch, capsys):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [("USDC", "native")])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")

    code = check_env.main(["--all"])

    out = capsys.readouterr().out
    for mode in ("pipeline", "api", "streaming_sse", "streaming_kafka", "ws_server", "training"):
        assert mode in out
    assert code == 1  # onchain/ws_server modes are unconfigured in this test


def test_json_output_is_valid_and_matches_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", ["$2b$somehash"])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 60)

    code = check_env.main(["--mode", "api", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["ok"] is True
    assert payload["results"] == [{"mode": "api", "ok": True, "error": None}]


def test_mode_and_all_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        check_env.main(["--mode", "api", "--all"])


def test_requires_mode_or_all():
    with pytest.raises(SystemExit):
        check_env.main([])


def test_explain_known_variable_prints_info(capsys):
    """Test that --explain prints info for a known contract variable."""
    code = check_env.main(["--explain", "RISK_SCORE_DB_URL"])

    assert code == 0
    out = capsys.readouterr().out

    # Should show the variable name and default
    assert "RISK_SCORE_DB_URL" in out
    assert "Default:" in out
    assert "api" in out  # Should mention the api mode


def test_explain_unknown_variable_exits_with_error(capsys):
    """Test that --explain on unknown variable gives clear error message."""
    code = check_env.main(["--explain", "NONEXISTENT_VAR"])

    assert code == 1
    err = capsys.readouterr().err

    # Should have clear error message, not a traceback
    assert "not a recognised contract variable" in err
    assert "NONEXISTENT_VAR" in err
    assert "Traceback" not in err


def test_explain_lists_all_applicable_modes(capsys):
    """Test that --explain shows all modes that use the variable."""
    code = check_env.main(["--explain", "MODEL_DIR"])

    assert code == 0
    out = capsys.readouterr().out

    # MODEL_DIR is used in multiple modes
    assert "api" in out
    assert "streaming_sse" in out or "training" in out  # At least one more mode
