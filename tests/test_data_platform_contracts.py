from __future__ import annotations

from datetime import UTC, datetime

import pytest

from features.store_interface import (
    FeatureKey,
    FeatureRecord,
    InMemoryFeatureStore,
    require_feature_schema,
)
from ingestion.data_models import Asset, Trade
from ingestion.source_connectors import (
    SourceBatch,
    SourceCursor,
    StaticLedgerSourceConnector,
    validate_connector_batch,
)
from mlops.experiment_tracking import ExperimentRun, JsonlExperimentTracker


def _trade(trade_id: str) -> Trade:
    asset = Asset(code="XLM")
    return Trade(
        trade_id=trade_id,
        ledger_close_time=datetime(2026, 1, 1, tzinfo=UTC),
        base_account="GA",
        counter_account="GB",
        base_asset=asset,
        counter_asset=Asset(code="USDC", issuer="issuer"),
        base_amount=10.0,
        counter_amount=2.0,
        price=0.2,
    )


def test_static_source_connector_paginates_with_provider_cursor():
    connector = StaticLedgerSourceConnector("horizon", [_trade("1"), _trade("2")])
    first = connector.fetch_since(limit=1)
    assert [record.trade_id for record in first] == ["1"]
    assert first.next_cursor == SourceCursor(provider="horizon", position="1")

    second = connector.fetch_since(first.next_cursor, limit=1)
    assert [record.trade_id for record in second] == ["2"]
    assert second.next_cursor is None


def test_connector_validation_rejects_wrong_provider_cursor():
    connector = StaticLedgerSourceConnector("horizon", [])
    batch = SourceBatch(records=(), next_cursor=SourceCursor(provider="archive", position="10"))
    with pytest.raises(ValueError, match="expected 'horizon'"):
        validate_connector_batch(connector, batch)


def test_feature_store_contract_preserves_schema_version():
    key = FeatureKey(wallet_id="GA", pair_id="XLM:native/USDC:issuer", window_hours=24)
    record = FeatureRecord(key=key, features={"risk_score": 0.9}, schema_version="2")
    store = InMemoryFeatureStore()

    store.put(record)

    cached = store.get(key)
    assert cached == record
    assert require_feature_schema(cached, "2") == record
    with pytest.raises(ValueError, match="schema version mismatch"):
        require_feature_schema(cached, "1")


def test_experiment_tracker_writes_reproducible_jsonl(tmp_path):
    run = ExperimentRun(
        name="baseline-rf",
        params={"seed": 7, "model": "rf"},
        feature_schema_hash="schema123",
        dataset_sha256="data456",
        git_sha="abc123",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    tracker = JsonlExperimentTracker(tmp_path / "experiments.jsonl")

    record = tracker.log_run(run, metrics={"auc_roc": 0.91}, artifacts={"model": "rf.joblib"})

    assert record["run_id"] == run.run_id
    assert tracker.list_runs() == [record]


def test_experiment_tracker_rejects_non_numeric_metrics(tmp_path):
    tracker = JsonlExperimentTracker(tmp_path / "experiments.jsonl")
    run = ExperimentRun(
        name="bad-metric",
        params={},
        feature_schema_hash="schema123",
        dataset_sha256="data456",
    )

    with pytest.raises(TypeError, match="metrics must be numeric"):
        tracker.log_run(run, metrics={"auc_roc": "high"})  # type: ignore[arg-type]
