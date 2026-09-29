"""Tests for integrations.offline_stubs — offline dev stand-ins for LedgerLensContractClient."""

import pytest

from integrations.offline_stubs import StubContractClient, get_contract_client


def make_score(**overrides) -> dict:
    base = {"score": 80, "benford_flag": True, "ml_flag": True, "confidence": 76, "timestamp": 123}
    base.update(overrides)
    return base


def test_submit_and_get_score_round_trips():
    client = StubContractClient()
    wallet, pair = "GWALLET", "USDC:issuer/XLM:native"
    client.submit_score(wallet, pair, make_score())
    assert client.get_score(wallet, pair)["score"] == 80


def test_get_score_missing_raises_lookup_error():
    client = StubContractClient()
    with pytest.raises(LookupError):
        client.get_score("GUNKNOWN", "USDC:issuer/XLM:native")


def test_submit_score_requires_submitter_secret():
    client = StubContractClient(submitter_secret="")
    with pytest.raises(ValueError):
        client.submit_score("GWALLET", "USDC:issuer/XLM:native", make_score())


def test_submit_score_with_commitment_stores_metadata():
    client = StubContractClient()
    wallet, pair = "GWALLET", "USDC:issuer/XLM:native"
    client.submit_score_with_commitment(wallet, pair, make_score(), "c1", "h1", "m1")
    stored = client.get_score(wallet, pair)
    assert stored["commitment"] == "c1"
    assert stored["trade_data_hash"] == "h1"


def test_submit_score_commitment_requires_hashes():
    client = StubContractClient()
    with pytest.raises(ValueError):
        client.submit_score("GWALLET", "USDC:issuer/XLM:native", make_score(), commitment="c1")


def test_threshold_change_propose_and_approve():
    client = StubContractClient()
    proposal_id = client.propose_threshold_change("GGOV", 90, "SPROPOSER")
    assert client.approve_threshold_change("GGOV", proposal_id, "SAPPROVER") is True


def test_approve_unknown_proposal_raises():
    client = StubContractClient()
    with pytest.raises(LookupError):
        client.approve_threshold_change("GGOV", 999, "SAPPROVER")


def test_calls_are_recorded_for_diagnostics():
    client = StubContractClient()
    client.submit_score("GWALLET", "USDC:issuer/XLM:native", make_score())
    assert client.calls[0][0] == "submit_score"


def test_get_contract_client_returns_stub_when_offline_true():
    client = get_contract_client(offline=True)
    assert isinstance(client, StubContractClient)


def test_get_contract_client_respects_env_var(monkeypatch):
    monkeypatch.setenv("LEDGERLENS_OFFLINE", "1")
    client = get_contract_client()
    assert isinstance(client, StubContractClient)


def test_get_contract_client_defaults_to_real_client(monkeypatch):
    monkeypatch.delenv("LEDGERLENS_OFFLINE", raising=False)
    client = get_contract_client(
        offline=False,
        contract_id="CCONTRACT",
        rpc_url="https://soroban-testnet.stellar.org",
        network_passphrase="Test SDF Network ; September 2015",
        submitter_secret="SAUQSDM4BPSOWVJJM7RAHPSGXDX5YLRYNZCZ5QP33EVB6WDAAVJJRJHG",
    )
    from integrations.contract_client import LedgerLensContractClient

    assert isinstance(client, LedgerLensContractClient)
