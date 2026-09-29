"""Tests for typed deployment-mode configuration fixtures (Issue #543)."""

import pytest

from config import Config
from config.deployment_modes import (
    DEPLOYMENT_MODE_FIXTURES,
    DeploymentMode,
    DeploymentModeFixture,
    DeploymentModeValidationError,
    UnknownDeploymentModeError,
    apply_deployment_mode,
    get_deployment_mode_fixture,
)


def test_every_deployment_mode_has_a_registered_fixture():
    for mode in DeploymentMode:
        fixture = DEPLOYMENT_MODE_FIXTURES[mode]
        assert isinstance(fixture, DeploymentModeFixture)
        assert fixture.mode is mode
        assert fixture.description


@pytest.mark.parametrize("mode", list(DeploymentMode))
def test_get_deployment_mode_fixture_by_enum(mode):
    fixture = get_deployment_mode_fixture(mode)
    assert fixture.mode is mode


@pytest.mark.parametrize("mode", ["local", "testnet", "production"])
def test_get_deployment_mode_fixture_by_string(mode):
    fixture = get_deployment_mode_fixture(mode)
    assert fixture.mode.value == mode


def test_get_deployment_mode_fixture_unknown_mode_raises_typed_error():
    with pytest.raises(UnknownDeploymentModeError) as exc:
        get_deployment_mode_fixture("staging-eu-west")

    assert "staging-eu-west" in str(exc.value)
    assert "local" in str(exc.value)
    assert "testnet" in str(exc.value)
    assert "production" in str(exc.value)


@pytest.mark.parametrize("mode", list(DeploymentMode))
def test_apply_deployment_mode_validates_successfully(mode):
    with apply_deployment_mode(mode) as fixture:
        assert fixture.mode is mode
        # Overrides must actually be applied onto Config while inside the block.
        for name, value in fixture.overrides.items():
            assert getattr(Config, name) == value


def test_apply_deployment_mode_restores_previous_values_on_exit():
    original = Config.STELLAR_NETWORK
    with apply_deployment_mode(DeploymentMode.TESTNET):
        assert Config.STELLAR_NETWORK == "TESTNET"
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_restores_on_exception():
    original = Config.STELLAR_NETWORK
    with pytest.raises(RuntimeError):
        with apply_deployment_mode(DeploymentMode.LOCAL):
            raise RuntimeError("boom")
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_unknown_mode_raises_before_mutating_config():
    original = Config.STELLAR_NETWORK
    with pytest.raises(UnknownDeploymentModeError):
        with apply_deployment_mode("does-not-exist"):
            pass  # pragma: no cover - should never be reached
    assert Config.STELLAR_NETWORK == original


def test_apply_deployment_mode_surfaces_validation_failures(monkeypatch):
    broken = DeploymentModeFixture(
        mode=DeploymentMode.LOCAL,
        description="Deliberately invalid fixture for the negative test path.",
        overrides={"WATCHED_ASSET_PAIRS": [], "RISK_SCORE_DB_URL": "", "MODEL_DIR": ""},
        require_onchain=False,
    )
    monkeypatch.setitem(DEPLOYMENT_MODE_FIXTURES, DeploymentMode.LOCAL, broken)
    original = Config.RISK_SCORE_DB_URL

    with pytest.raises(DeploymentModeValidationError) as exc:
        with apply_deployment_mode(DeploymentMode.LOCAL):
            pass  # pragma: no cover - validation fails before the body runs

    assert exc.value.mode is DeploymentMode.LOCAL
    # Overrides applied during the failed validation must still be rolled back.
    assert Config.RISK_SCORE_DB_URL == original


def test_apply_deployment_mode_validate_false_skips_validation(monkeypatch):
    broken = DeploymentModeFixture(
        mode=DeploymentMode.LOCAL,
        description="Deliberately invalid fixture for the negative test path.",
        overrides={"WATCHED_ASSET_PAIRS": [], "RISK_SCORE_DB_URL": "", "MODEL_DIR": ""},
        require_onchain=False,
    )
    monkeypatch.setitem(DEPLOYMENT_MODE_FIXTURES, DeploymentMode.LOCAL, broken)

    with apply_deployment_mode(DeploymentMode.LOCAL, validate=False) as fixture:
        assert fixture is broken
        assert Config.RISK_SCORE_DB_URL == ""


def test_local_deployment_config_fixture(local_deployment_config):
    assert local_deployment_config.mode is DeploymentMode.LOCAL
    assert Config.STREAMING_BACKEND == "sse"
    assert Config.HORIZON_DEV_MODE is True


def test_testnet_deployment_config_fixture(testnet_deployment_config):
    assert testnet_deployment_config.mode is DeploymentMode.TESTNET
    assert Config.STELLAR_NETWORK == "TESTNET"
    assert Config.LEDGERLENS_CONTRACT_ID


def test_production_deployment_config_fixture(production_deployment_config):
    assert production_deployment_config.mode is DeploymentMode.PRODUCTION
    assert Config.STELLAR_NETWORK == "PUBLIC"
    assert Config.HORIZON_DEV_MODE is False
