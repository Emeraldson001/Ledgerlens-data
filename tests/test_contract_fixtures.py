"""Unit and integration tests for contract-driven test fixtures (Issue #465)."""

import json

import pandas as pd
import pytest

from detection.feature_engineering import compute_benford_features, compute_trade_pattern_features
from tests.contract_fixtures import (
    AccountSpec,
    LedgerScenarioBuilder,
    OperationType,
    ScenarioContract,
    ScenarioPatternType,
    TransactionSpec,
    make_mev_sandwich_contract,
    make_wash_trade_ring_contract,
)


@pytest.fixture
def wash_ring_builder() -> LedgerScenarioBuilder:
    contract = make_wash_trade_ring_contract()
    return LedgerScenarioBuilder(contract)


def test_scenario_contract_validation():
    # Valid contract
    contract = make_wash_trade_ring_contract()
    assert len(contract.validate_schema()) == 0

    # Invalid contract: unknown source account
    bad_contract = ScenarioContract(
        scenario_id="bad_1",
        name="Bad Contract",
        pattern_type=ScenarioPatternType.WASH_TRADE_RING,
        description="Bad",
        accounts=[AccountSpec("G1")],
        timeline=[
            TransactionSpec("tx1", 100, 0.0, "G_UNKNOWN", "G1", OperationType.PAYMENT, 100.0)
        ],
        expectations=[],
    )
    errors = bad_contract.validate_schema()
    assert len(errors) > 0
    assert any("source_account 'G_UNKNOWN' not in defined accounts" in err for err in errors)

    with pytest.raises(ValueError, match="Invalid ScenarioContract"):
        LedgerScenarioBuilder(bad_contract)


def test_build_trades_dataframe(wash_ring_builder):
    df = wash_ring_builder.build_trades_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40
    assert "trade_id" in df.columns
    assert "account" in df.columns
    assert "amount" in df.columns
    assert "ledger_close_time" in df.columns
    assert (df["amount"] == 5000.0).all()


def test_wash_trade_ring_scenario_execution(wash_ring_builder):
    df = wash_ring_builder.build_trades_dataframe()

    # Adapt columns for compute_graph_features
    df["base_account"] = df["account"]
    df["counter_account"] = df["counterparty"]

    # Run feature extraction on generated scenario data
    metrics_by_account = {}
    for acct, group in df.groupby("account"):
        # Compute Benford features
        benford_feats = compute_benford_features(group, decompose=False)
        # Compute trade pattern features
        pattern_feats = compute_trade_pattern_features(acct, group)
        metrics_by_account[str(acct)] = {
            **benford_feats,
            **pattern_feats,
        }

    res = wash_ring_builder.verify_expectations(metrics_by_account)

    assert res.all_passed is True
    assert "high_counterparty_concentration" in res.passed_expectations
    assert "perfect_counterparty_concentration" in res.passed_expectations
    assert len(res.failed_expectations) == 0


def test_mev_sandwich_contract():
    contract = make_mev_sandwich_contract()
    builder = LedgerScenarioBuilder(contract)
    df = builder.build_trades_dataframe()

    assert len(df) == 3
    assert set(df["account"]) == {"GMEV_BOT", "GVICTIM_TRADER", "GDEX_POOL"}


def test_contract_json_serialization_roundtrip(tmp_path):
    contract = make_wash_trade_ring_contract()
    json_path = tmp_path / "scenario_wash_ring.json"

    # Export to JSON
    json_str = json.dumps(contract.to_dict(), indent=2)
    json_path.write_text(json_str, encoding="utf-8")

    # Reload builder from JSON file
    reloaded_builder = LedgerScenarioBuilder.from_json(json_path)
    assert reloaded_builder.contract.scenario_id == "scenario_wash_ring_v1"
    assert len(reloaded_builder.contract.accounts) == 4
    assert len(reloaded_builder.contract.timeline) == 40
