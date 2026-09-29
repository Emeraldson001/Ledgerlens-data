"""Tests for package-boundary contracts."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections.abc import Iterator
from typing import Any

from contracts import (
    AlertChannel,
    AlertEvent,
    FeatureStore,
    RiskScore,
    Scorer,
    TradeSource,
)


class TestRiskScore:
    def test_minimal_risk_score(self):
        rs: RiskScore = {
            "wallet": "GABCD123",
            "asset_pair": "USDC_native",
            "score": 85,
            "benford_flag": True,
            "ml_flag": True,
            "confidence": 90,
            "timestamp": 1700000000,
        }
        assert rs["wallet"] == "GABCD123"
        assert rs["score"] == 85
        assert rs.get("propagated_risk") is None

    def test_extended_risk_score(self):
        rs: RiskScore = {
            "wallet": "GXYZ",
            "asset_pair": "BTC_GBP",
            "score": 42,
            "benford_flag": False,
            "ml_flag": True,
            "confidence": 75,
            "timestamp": 1700000001,
            "propagated_risk": 0.3,
            "ring_id": 7,
            "score_lower": 0.1,
            "score_upper": 0.9,
            "coverage_guarantee": 0.95,
            "replay_model_version": "v2.1",
            "model_name": "ensemble_v3",
            "feature_contributions": {"volume": 0.5, "velocity": 0.3},
        }
        assert rs["ring_id"] == 7
        assert rs["feature_contributions"]["volume"] == 0.5
        assert rs["replay_model_version"] == "v2.1"

    def test_risk_score_type(self):
        rs: RiskScore = {
            "wallet": "x",
            "asset_pair": "y",
            "score": 0,
            "benford_flag": False,
            "ml_flag": False,
            "confidence": 0,
            "timestamp": 0,
        }
        assert isinstance(rs, dict)
        assert rs["wallet"] == "x"


class TestScorerProtocol:
    def test_scorer_protocol_structural(self):
        class MyScorer:
            def score(self, feature_row: Any, **kwargs: Any) -> RiskScore:
                return {
                    "wallet": "test",
                    "asset_pair": "USDC_native",
                    "score": 50,
                    "benford_flag": False,
                    "ml_flag": False,
                    "confidence": 100,
                    "timestamp": 1700000000,
                }

        scorer: Scorer = MyScorer()
        rs = scorer.score({})
        assert rs["score"] == 50
        assert isinstance(scorer, Scorer)

    def test_scorer_protocol_not_satisfied(self):
        class NotAScorer:
            pass

        assert not isinstance(NotAScorer(), Scorer)


class TestTradeSourceProtocol:
    def test_trade_source_protocol_structural(self):
        class MockSource:
            def stream_trades(
                self, asset_pair: str, since: float | None = None
            ) -> Iterator[dict[str, Any]]:
                yield {"asset_pair": asset_pair, "amount": 100.0}

        source: TradeSource = MockSource()
        trades = list(source.stream_trades("USDC_native"))
        assert len(trades) == 1
        assert trades[0]["amount"] == 100.0
        assert isinstance(source, TradeSource)

    def test_trade_source_not_satisfied(self):
        class NotASource:
            pass

        assert not isinstance(NotASource(), TradeSource)


class TestFeatureStoreProtocol:
    def test_feature_store_protocol_structural(self):
        class MockStore:
            def get_or_compute(self, wallet: str, pair: str, compute_fn):
                result = compute_fn()
                return {**result, "wallet": wallet, "pair": pair}

        store: FeatureStore = MockStore()
        result = store.get_or_compute("wallet1", "USDC_native", lambda: {"vol": 42.0})
        assert result["vol"] == 42.0
        assert result["wallet"] == "wallet1"
        assert isinstance(store, FeatureStore)

    def test_feature_store_not_satisfied(self):
        class NotAStore:
            pass

        assert not isinstance(NotAStore(), FeatureStore)


class TestAlertEvent:
    def test_minimal_alert_event(self):
        event: AlertEvent = {
            "wallet": "GABCD",
            "asset_pair": "USDC_native",
            "score": 95,
            "detectors": ["benford", "ml"],
            "timestamp": 1700000000,
        }
        assert event.get("severity") is None  # total=False, optional

    def test_full_alert_event(self):
        event: AlertEvent = {
            "wallet": "GABCD",
            "asset_pair": "USDC_native",
            "score": 95,
            "detectors": ["benford"],
            "timestamp": 1700000000,
            "severity": "high",
            "message": "Suspicious activity detected",
        }
        assert event["severity"] == "high"


class TestAlertChannelProtocol:
    def test_alert_channel_protocol_structural(self):
        dispatched: list[AlertEvent] = []

        class MockChannel:
            def dispatch(self, event: AlertEvent) -> None:
                dispatched.append(event)

        channel: AlertChannel = MockChannel()
        event: AlertEvent = {
            "wallet": "G123",
            "asset_pair": "USDC_native",
            "score": 99,
            "detectors": ["ml"],
            "timestamp": 1700000000,
        }
        channel.dispatch(event)
        assert len(dispatched) == 1
        assert isinstance(channel, AlertChannel)

    def test_alert_channel_not_satisfied(self):
        class NotAChannel:
            pass

        assert not isinstance(NotAChannel(), AlertChannel)
