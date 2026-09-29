"""Unit and integration tests for data quality scoring and ledger import readiness (Issue #464)."""

import datetime

import pandas as pd
import pytest

from ingestion.data_quality import (
    LedgerQualityScorer,
    ReadinessStatus,
)


@pytest.fixture
def valid_trade_df() -> pd.DataFrame:
    """Fixture providing clean, valid Stellar ledger trade records."""
    now = datetime.datetime.now(tz=datetime.UTC)
    return pd.DataFrame(
        {
            "trade_id": [f"t_{i}" for i in range(20)],
            "account": ["GBRPYHIL2CI3FNQ4BXLFMNDLFJUNPU2HY3ZMFXYSFZW2BV3FL224GKO7"] * 20,
            "seller": ["GAHK7EEG2WWHVKTZB2LHVDT6EB2S6HREL5Y2FQZFYAZTXWCD4MGJQBOX"] * 20,
            "amount": [100.0 + i * 5 for i in range(20)],
            "ledger_close_time": [
                (now - datetime.timedelta(minutes=i)).isoformat() for i in range(20)
            ],
        }
    )


def test_quality_scorer_clean_dataset(valid_trade_df):
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(valid_trade_df)

    assert report.status == ReadinessStatus.READY
    assert report.overall_score >= 95.0
    assert report.total_records == 20
    assert all(r.passed for r in report.rule_results)
    assert len(report.diagnostics) == 0


def test_completeness_failure():
    df = pd.DataFrame(
        {
            "trade_id": ["t1", "t2", "t3"],
            "amount": [10.0, None, None],  # 66% nulls
        }
    )
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(df)

    completeness_res = [r for r in report.rule_results if r.rule_name == "completeness_check"][0]
    assert completeness_res.passed is False
    assert completeness_res.failed_records_count == 2
    assert report.overall_score < scorer.pass_threshold


def test_stellar_address_validity_failure():
    df = pd.DataFrame(
        {
            "trade_id": ["t1", "t2"],
            "account": [
                "INVALID_ACCOUNT_ADDRESS_123",
                "GBRPYHIL2CI3FNQ4BXLFMNDLFJUNPU2HY3ZMFXYSFZW2BV3FL224GKO7",
            ],
            "amount": [100.0, 50.0],
            "ledger_close_time": [datetime.datetime.now(tz=datetime.UTC).isoformat()] * 2,
        }
    )
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(df)

    address_res = [r for r in report.rule_results if r.rule_name == "stellar_address_validity"][0]
    assert address_res.passed is False
    assert address_res.failed_records_count == 1
    assert any("invalid Stellar account keys" in d for d in report.diagnostics)


def test_amount_validity_failure():
    df = pd.DataFrame(
        {
            "amount": [10.0, -50.0, float("nan")],
        }
    )
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(df)

    amt_res = [r for r in report.rule_results if r.rule_name == "amount_validity"][0]
    assert amt_res.passed is False
    assert amt_res.failed_records_count == 2


def test_timeliness_future_and_stale_timestamps():
    now = datetime.datetime.now(tz=datetime.UTC)
    future_dt = (now + datetime.timedelta(days=2)).isoformat()
    stale_dt = (now - datetime.timedelta(days=500)).isoformat()

    df = pd.DataFrame(
        {
            "amount": [10.0, 20.0],
            "ledger_close_time": [future_dt, stale_dt],
        }
    )
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(df)

    timeliness_res = [r for r in report.rule_results if r.rule_name == "timeliness_check"][0]
    assert timeliness_res.passed is False
    assert timeliness_res.details["future_count"] == 1
    assert timeliness_res.details["stale_count"] == 1


def test_uniqueness_duplicate_detection():
    df = pd.DataFrame(
        {
            "trade_id": ["t1", "t1", "t1", "t2"],  # 50% duplicate ratio
            "amount": [10.0, 10.0, 10.0, 20.0],
        }
    )
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(df)

    uniq_res = [r for r in report.rule_results if r.rule_name == "uniqueness_check"][0]
    assert uniq_res.passed is False
    assert uniq_res.details["duplicate_count"] == 2


test_orderbook_spread_crossed_markets_data = pd.DataFrame(
    {
        "best_bid": [100.0, 105.0],
        "best_ask": [102.0, 101.0],  # 105 > 101 crossed market
        "amount": [10.0, 20.0],
    }
)


def test_orderbook_spread_consistency():
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(test_orderbook_spread_crossed_markets_data)

    spread_res = [r for r in report.rule_results if r.rule_name == "orderbook_spread_consistency"][
        0
    ]
    assert spread_res.passed is False
    assert spread_res.failed_records_count == 1


def test_empty_dataset_quarantine_rejection():
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(pd.DataFrame())

    assert report.status == ReadinessStatus.QUARANTINE_REJECTED
    assert report.overall_score == 0.0
    assert any("CRITICAL: Input dataset is empty" in d for d in report.diagnostics)


def test_report_serialization(valid_trade_df):
    scorer = LedgerQualityScorer()
    report = scorer.evaluate_import_readiness(valid_trade_df)
    report_dict = report.to_dict()

    assert isinstance(report_dict["overall_score"], float)
    assert report_dict["status"] == "READY"
    assert "COMPLETENESS" in report_dict["dimension_scores"]
    assert isinstance(report_dict["rule_results"], list)
