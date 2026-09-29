"""Tests for config/contracts.py — per-runtime-mode environment config contracts.

Each `validate_mode(mode)` call should raise a single `OSError` listing every
violation of that mode's contract (mirroring `Config.validate()`'s existing
"collect all errors, then raise once" behavior), and should pass silently once
every required var for that mode is set. See config/contracts.py for the
rationale: before this module existed, only run_pipeline.py validated
anything at startup, and it did so *before* parsing --submit-onchain.
"""

import pytest

from config import Config
from config.contracts import RUNTIME_MODES, validate_mode

VALID_PAIRS = [("USDC", "GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN")]


def _set_pipeline_baseline(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "DP_AGGREGATOR_EPSILON", 1.0)
    monkeypatch.setattr(Config, "DP_AGGREGATOR_DELTA", 1e-5)


# ---------------------------------------------------------------------------
# validate_mode() dispatch
# ---------------------------------------------------------------------------


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError, match="Unknown runtime mode"):
        validate_mode("not-a-real-mode")


def test_all_runtime_modes_are_registered():
    assert set(RUNTIME_MODES) == {
        "pipeline",
        "pipeline_onchain",
        "api",
        "streaming_sse",
        "streaming_kafka",
        "ws_server",
        "training",
    }


# ---------------------------------------------------------------------------
# pipeline / pipeline_onchain
# ---------------------------------------------------------------------------


def test_pipeline_passes_with_baseline_config(monkeypatch):
    _set_pipeline_baseline(monkeypatch)
    validate_mode("pipeline")


def test_pipeline_reports_each_missing_var_on_its_own_line(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "DP_AGGREGATOR_EPSILON", 1.0)
    monkeypatch.setattr(Config, "DP_AGGREGATOR_DELTA", 1e-5)

    with pytest.raises(OSError) as exc:
        validate_mode("pipeline")

    lines = [line for line in str(exc.value).splitlines() if line.startswith("- ")]
    assert "- WATCHED_ASSET_PAIRS is not set." in lines
    assert "- RISK_SCORE_DB_URL is not set." in lines


def test_pipeline_onchain_requires_contract_vars_even_though_pipeline_does_not(monkeypatch):
    _set_pipeline_baseline(monkeypatch)
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "")

    validate_mode("pipeline")  # not required for the plain pipeline mode

    with pytest.raises(OSError, match="LEDGERLENS_CONTRACT_ID"):
        validate_mode("pipeline_onchain")


def test_pipeline_onchain_passes_once_fully_configured(monkeypatch):
    _set_pipeline_baseline(monkeypatch)
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "contract-id")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "secret")
    monkeypatch.setattr(Config, "SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    monkeypatch.setattr(Config, "STELLAR_NETWORK", "TESTNET")

    validate_mode("pipeline_onchain")


def test_pipeline_onchain_rejects_invalid_stellar_network(monkeypatch):
    _set_pipeline_baseline(monkeypatch)
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "contract-id")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "secret")
    monkeypatch.setattr(Config, "SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    monkeypatch.setattr(Config, "STELLAR_NETWORK", "MAINNET")  # not a real Stellar network name

    with pytest.raises(OSError, match="STELLAR_NETWORK"):
        validate_mode("pipeline_onchain")


def test_pipeline_onchain_fails_fast_without_submitter_secret(monkeypatch):
    """Verify that --submit-onchain without LEDGERLENS_SUBMITTER_SECRET fails at startup.

    This test confirms that the error message specifically names the missing variable,
    and occurs before any pipeline work (ingestion/scoring) is invoked.
    """
    _set_pipeline_baseline(monkeypatch)
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "contract-id")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "")  # Empty secret
    monkeypatch.setattr(Config, "SOROBAN_RPC_URL", "https://soroban-testnet.stellar.org")
    monkeypatch.setattr(Config, "STELLAR_NETWORK", "TESTNET")

    with pytest.raises(OSError) as exc:
        validate_mode("pipeline_onchain")

    error_msg = str(exc.value)
    assert "LEDGERLENS_SUBMITTER_SECRET" in error_msg
    assert "not set" in error_msg


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------


def test_api_passes_when_configured(monkeypatch):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", ["$2b$somehash"])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 60)

    validate_mode("api")


def test_api_rejects_empty_api_keys(monkeypatch):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", [])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 60)

    with pytest.raises(OSError, match="API_KEYS"):
        validate_mode("api")


def test_api_rejects_non_positive_rate_limit(monkeypatch):
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "sqlite:///test.db")
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "API_KEYS", ["$2b$somehash"])
    monkeypatch.setattr(Config, "API_RATE_LIMIT_RPM", 0)

    with pytest.raises(OSError, match="API_RATE_LIMIT_RPM"):
        validate_mode("api")


# ---------------------------------------------------------------------------
# streaming_sse
# ---------------------------------------------------------------------------


def test_streaming_sse_requires_watched_pairs(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "ALERT_CHANNEL", "stdout")

    with pytest.raises(OSError, match="WATCHED_ASSET_PAIRS"):
        validate_mode("streaming_sse")


def test_streaming_sse_passes_with_defaults(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "ALERT_CHANNEL", "stdout")

    validate_mode("streaming_sse")


def test_streaming_sse_webhook_channel_requires_webhook_url(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "ALERT_WEBHOOK_URL", None)

    with pytest.raises(OSError, match="ALERT_WEBHOOK_URL"):
        validate_mode("streaming_sse", alert_channel="webhook")


def test_streaming_sse_webhook_channel_passes_with_url_set(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "ALERT_WEBHOOK_URL", "https://example.com/hook")

    validate_mode("streaming_sse", alert_channel="webhook")


def test_streaming_sse_stdout_channel_ignores_missing_webhook_url(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "ALERT_WEBHOOK_URL", None)

    validate_mode("streaming_sse", alert_channel="stdout")


# ---------------------------------------------------------------------------
# streaming_kafka
# ---------------------------------------------------------------------------


def test_streaming_kafka_worker_role_does_not_need_watched_pairs(monkeypatch):
    # scripts/stream.py: a Kafka worker discovers topics dynamically.
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")
    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", None)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", None)

    validate_mode("streaming_kafka", role="worker", backend="kafka")


def test_streaming_kafka_producer_role_does_not_need_model_dir(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")
    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", None)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", None)

    validate_mode("streaming_kafka", role="producer", backend="kafka")


def test_streaming_kafka_all_role_needs_both_pairs_and_model_dir(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "MODEL_DIR", "")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")
    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", None)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", None)

    with pytest.raises(OSError) as exc:
        validate_mode("streaming_kafka", role="all", backend="kafka")

    msg = str(exc.value)
    assert "WATCHED_ASSET_PAIRS" in msg
    assert "MODEL_DIR" in msg


def test_streaming_kafka_requires_avro_schema_file_to_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", str(tmp_path / "does_not_exist.json"))
    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", None)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", None)

    with pytest.raises(OSError, match="TRADE_AVRO_SCHEMA_PATH"):
        validate_mode("streaming_kafka", role="all", backend="kafka")

    schema_file = tmp_path / "schema.json"
    schema_file.write_text("{}")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", str(schema_file))
    validate_mode("streaming_kafka", role="all", backend="kafka")


@pytest.mark.parametrize(
    ("username", "password"),
    [("alice", None), (None, "hunter2")],
)
def test_streaming_kafka_rejects_partial_sasl_credentials(monkeypatch, username, password):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")
    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", username)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", password)

    with pytest.raises(OSError, match="KAFKA_SASL"):
        validate_mode("streaming_kafka", role="all", backend="kafka")


def test_streaming_kafka_allows_both_or_neither_sasl_credentials(monkeypatch):
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", VALID_PAIRS)
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", "data/trade_avro_schema.json")

    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", None)
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", None)
    validate_mode("streaming_kafka", role="all", backend="kafka")

    monkeypatch.setattr(Config, "KAFKA_SASL_USERNAME", "alice")
    monkeypatch.setattr(Config, "KAFKA_SASL_PASSWORD", "hunter2")
    validate_mode("streaming_kafka", role="all", backend="kafka")


# ---------------------------------------------------------------------------
# ws_server
# ---------------------------------------------------------------------------


def test_ws_server_requires_jwt_public_key_file_to_exist(monkeypatch, tmp_path):
    monkeypatch.setattr(Config, "JWT_PUBLIC_KEY_PATH", str(tmp_path / "missing.pem"))
    monkeypatch.setattr(Config, "WS_MAX_CLIENTS", 200)

    with pytest.raises(OSError, match="JWT_PUBLIC_KEY_PATH"):
        validate_mode("ws_server")


def test_ws_server_passes_once_key_file_exists(monkeypatch, tmp_path):
    key_file = tmp_path / "jwt_public_key.pem"
    key_file.write_text("-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----\n")
    monkeypatch.setattr(Config, "JWT_PUBLIC_KEY_PATH", str(key_file))
    monkeypatch.setattr(Config, "WS_MAX_CLIENTS", 200)

    validate_mode("ws_server")


def test_ws_server_rejects_non_positive_max_clients(monkeypatch, tmp_path):
    key_file = tmp_path / "jwt_public_key.pem"
    key_file.write_text("...")
    monkeypatch.setattr(Config, "JWT_PUBLIC_KEY_PATH", str(key_file))
    monkeypatch.setattr(Config, "WS_MAX_CLIENTS", 0)

    with pytest.raises(OSError, match="WS_MAX_CLIENTS"):
        validate_mode("ws_server")


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def test_training_passes_with_defaults(monkeypatch):
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "CALIBRATION_SPLIT", 0.20)
    monkeypatch.setattr(Config, "DP_TARGET_EPSILON", 8.0)
    monkeypatch.setattr(Config, "DP_TARGET_DELTA", 1e-5)

    validate_mode("training")


def test_training_model_dir_context_override_beats_empty_config(monkeypatch):
    # detection/model_training.py: `--model-dir` overrides config.MODEL_DIR.
    monkeypatch.setattr(Config, "MODEL_DIR", "")
    monkeypatch.setattr(Config, "CALIBRATION_SPLIT", 0.20)
    monkeypatch.setattr(Config, "DP_TARGET_EPSILON", 8.0)
    monkeypatch.setattr(Config, "DP_TARGET_DELTA", 1e-5)

    with pytest.raises(OSError, match="MODEL_DIR"):
        validate_mode("training")

    validate_mode("training", model_dir="/tmp/some-run-specific-dir")


def test_training_rejects_calibration_split_out_of_range(monkeypatch):
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "CALIBRATION_SPLIT", 1.5)
    monkeypatch.setattr(Config, "DP_TARGET_EPSILON", 8.0)
    monkeypatch.setattr(Config, "DP_TARGET_DELTA", 1e-5)

    with pytest.raises(OSError, match="CALIBRATION_SPLIT"):
        validate_mode("training")


def test_training_rejects_non_positive_dp_epsilon(monkeypatch):
    monkeypatch.setattr(Config, "MODEL_DIR", "./models")
    monkeypatch.setattr(Config, "CALIBRATION_SPLIT", 0.20)
    monkeypatch.setattr(Config, "DP_TARGET_EPSILON", 0.0)
    monkeypatch.setattr(Config, "DP_TARGET_DELTA", 1e-5)

    with pytest.raises(OSError, match="DP_TARGET_EPSILON"):
        validate_mode("training")


# ---------------------------------------------------------------------------
# Multi-variable missing error aggregation & runtime mode assertions
# ---------------------------------------------------------------------------


def test_api_multi_variable_missing_lists_all_vars_and_runtime_mode(monkeypatch):
    """When multiple required env vars are missing in api mode, a single OSError

    is raised listing every missing variable and naming the runtime mode.
    """
    monkeypatch.setattr(Config, "RISK_SCORE_DB_URL", "")
    monkeypatch.setattr(Config, "MODEL_DIR", "")
    monkeypatch.setattr(Config, "API_KEYS", [])

    with pytest.raises(OSError) as exc_info:
        validate_mode("api")

    err_msg = str(exc_info.value)
    assert "mode='api'" in err_msg
    assert "RISK_SCORE_DB_URL is not set" in err_msg
    assert "MODEL_DIR is not set" in err_msg
    assert "API_KEYS is not set" in err_msg

    # Confirm all errors are in a single exception message
    bullet_lines = [line for line in err_msg.splitlines() if line.startswith("- ")]
    assert len(bullet_lines) >= 3


def test_streaming_kafka_multi_variable_missing_lists_all_vars_and_runtime_mode(
    monkeypatch, tmp_path
):
    """When multiple required env vars are missing in streaming_kafka mode, a

    single OSError is raised listing every violation and naming the mode.
    """
    monkeypatch.setattr(Config, "WATCHED_ASSET_PAIRS", [])
    monkeypatch.setattr(Config, "KAFKA_BOOTSTRAP_SERVERS", "")
    monkeypatch.setattr(Config, "TRADE_AVRO_SCHEMA_PATH", str(tmp_path / "missing_schema.json"))

    with pytest.raises(OSError) as exc_info:
        validate_mode("streaming_kafka", role="all", backend="kafka")

    err_msg = str(exc_info.value)
    assert "mode='streaming_kafka'" in err_msg
    assert "WATCHED_ASSET_PAIRS is not set" in err_msg
    assert "KAFKA_BOOTSTRAP_SERVERS is not set" in err_msg
    assert "TRADE_AVRO_SCHEMA_PATH" in err_msg


def test_pipeline_onchain_multi_variable_missing_lists_all_vars_and_runtime_mode(monkeypatch):
    """When multiple required env vars are missing in pipeline_onchain mode, all

    are reported in a single message alongside mode='pipeline_onchain'.
    """
    _set_pipeline_baseline(monkeypatch)
    monkeypatch.setattr(Config, "LEDGERLENS_CONTRACT_ID", "")
    monkeypatch.setattr(Config, "LEDGERLENS_SUBMITTER_SECRET", "")
    monkeypatch.setattr(Config, "STELLAR_NETWORK", "INVALID_NET")

    with pytest.raises(OSError) as exc_info:
        validate_mode("pipeline_onchain")

    err_msg = str(exc_info.value)
    assert "mode='pipeline_onchain'" in err_msg
    assert "LEDGERLENS_CONTRACT_ID is not set" in err_msg
    assert "LEDGERLENS_SUBMITTER_SECRET is not set" in err_msg
    assert "STELLAR_NETWORK" in err_msg

