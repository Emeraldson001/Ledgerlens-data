"""Tests for streaming.alert_ledger.AlertDeliveryLedger and its wiring into
AlertDispatcher (Issue #670, required scope E)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from streaming.alert_dispatcher import AlertDispatcher
from streaming.alert_ledger import AlertDeliveryLedger

THRESHOLD = 70


@pytest.fixture
def ledger(tmp_path):
    return AlertDeliveryLedger(db_url=f"sqlite:///{tmp_path / 'alert_ledger.db'}")


def test_record_and_query_delivered(ledger):
    ledger.record("GABC", "USDC:.../XLM:native", {"score": 80}, "delivered", channel="stdout")
    records = ledger.for_wallet_pair("GABC", "USDC:.../XLM:native")
    assert len(records) == 1
    assert records[0].outcome == "delivered"


def test_multiple_attempts_for_same_wallet_all_recorded(ledger):
    for score in (75, 80, 90):
        ledger.record(
            "GABC", "USDC:.../XLM:native", {"score": score}, "delivered", channel="stdout"
        )
    records = ledger.for_wallet_pair("GABC", "USDC:.../XLM:native")
    assert len(records) == 3
    assert {r.score for r in records} == {75, 80, 90}


def test_all_records_spans_wallets(ledger):
    ledger.record("GA", "pair", {"score": 80}, "delivered", channel="stdout")
    ledger.record(
        "GB", "pair", {"score": 90}, "dead_lettered", channel="webhook", reason="HTTP 500"
    )
    records = ledger.all_records()
    assert len(records) == 2
    outcomes = {r.wallet: r.outcome for r in records}
    assert outcomes == {"GA": "delivered", "GB": "dead_lettered"}


class TestAlertDispatcherLedgerIntegration:
    def test_stdout_delivery_recorded(self, ledger):
        dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD, delivery_ledger=ledger)
        dispatcher.dispatch("GABC", {"score": 80, "benford_flag": True, "ml_flag": True}, "pair-1")

        records = ledger.for_wallet_pair("GABC", "pair-1")
        assert len(records) == 1
        assert records[0].outcome == "delivered"
        assert records[0].channel == "stdout"

    def test_cooldown_suppression_recorded(self, ledger):
        dispatcher = AlertDispatcher(
            channel="stdout",
            threshold=THRESHOLD,
            alert_cooldown_seconds=3600,
            delivery_ledger=ledger,
        )
        risk_score = {"score": 80, "benford_flag": True, "ml_flag": True}
        dispatcher.dispatch("GABC", risk_score, "pair-1")
        dispatcher.dispatch("GABC", risk_score, "pair-1")  # within cooldown

        records = ledger.for_wallet_pair("GABC", "pair-1")
        outcomes = [r.outcome for r in records]
        assert outcomes.count("delivered") == 1
        assert outcomes.count("suppressed_cooldown") == 1

    def test_below_threshold_not_recorded(self, ledger):
        dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD, delivery_ledger=ledger)
        dispatcher.dispatch(
            "GABC", {"score": 10, "benford_flag": False, "ml_flag": False}, "pair-1"
        )

        assert ledger.for_wallet_pair("GABC", "pair-1") == []

    def test_webhook_dead_letter_recorded(self, ledger):
        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://example.com/hook",
            threshold=THRESHOLD,
            max_retries=0,
            base_delay=0.0,
            delivery_ledger=ledger,
        )
        resp = MagicMock()
        resp.raise_for_status.side_effect = __import__("requests").HTTPError(
            response=MagicMock(status_code=500)
        )
        with patch("streaming.alert_dispatcher.requests.post", return_value=resp):
            dispatcher.dispatch(
                "GABC", {"score": 90, "benford_flag": True, "ml_flag": True}, "pair-1"
            )

        records = ledger.for_wallet_pair("GABC", "pair-1")
        assert len(records) == 1
        assert records[0].outcome == "dead_lettered"
        assert records[0].channel == "webhook"

    def test_webhook_delivered_recorded(self, ledger):
        dispatcher = AlertDispatcher(
            channel="webhook",
            webhook_url="https://example.com/hook",
            threshold=THRESHOLD,
            delivery_ledger=ledger,
        )
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        with patch("streaming.alert_dispatcher.requests.post", return_value=resp):
            dispatcher.dispatch(
                "GABC", {"score": 90, "benford_flag": True, "ml_flag": True}, "pair-1"
            )

        records = ledger.for_wallet_pair("GABC", "pair-1")
        assert len(records) == 1
        assert records[0].outcome == "delivered"

    def test_no_ledger_is_backward_compatible(self):
        """Omitting delivery_ledger must not change existing dispatch behavior."""
        dispatcher = AlertDispatcher(channel="stdout", threshold=THRESHOLD)
        dispatcher.dispatch("GABC", {"score": 80, "benford_flag": True, "ml_flag": True}, "pair-1")
        # No assertion needed beyond "did not raise" — this is the regression guard.
