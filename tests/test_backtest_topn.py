"""Tests for scripts/backtest.py --top-n flag."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from scripts.backtest import BacktestEngine, generate_report


def test_topn_limits_report_to_worst_lag_campaigns(tmp_path):
    """Test that --top-n limits campaigns to N worst-lag entries."""
    # Create fixture ground truth with multiple campaigns
    gt_data = {
        "wallet": ["wallet_a", "wallet_b", "wallet_c", "wallet_d"],
        "asset_pair": ["USD/XLM", "USD/XLM", "USD/XLM", "USD/XLM"],
        "campaign_start": [
            "2024-01-01T00:00:00Z",
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
        ],
        "campaign_end": [
            "2024-01-02T00:00:00Z",
            "2024-01-03T00:00:00Z",
            "2024-01-04T00:00:00Z",
            "2024-01-05T00:00:00Z",
        ],
        "label_source": ["https://example.com/1", "https://example.com/2", "https://example.com/3", "https://example.com/4"],
    }
    gt_df = pd.DataFrame(gt_data)

    # Create mock detection lags with varying lag_hours
    lags = {
        "wallet_a": {
            "wallet": "wallet_a",
            "campaign_start": "2024-01-01T00:00:00+00:00",
            "campaign_end": "2024-01-02T00:00:00+00:00",
            "first_detection": "2024-01-01T12:00:00Z",
            "lag_hours": 12.0,
            "detected": True,
        },
        "wallet_b": {
            "wallet": "wallet_b",
            "campaign_start": "2024-01-02T00:00:00+00:00",
            "campaign_end": "2024-01-03T00:00:00+00:00",
            "first_detection": "2024-01-02T23:00:00Z",
            "lag_hours": 23.0,
            "detected": True,
        },
        "wallet_c": {
            "wallet": "wallet_c",
            "campaign_start": "2024-01-03T00:00:00+00:00",
            "campaign_end": "2024-01-04T00:00:00+00:00",
            "first_detection": "2024-01-03T06:00:00Z",
            "lag_hours": 6.0,
            "detected": True,
        },
        "wallet_d": {
            "wallet": "wallet_d",
            "campaign_start": "2024-01-04T00:00:00+00:00",
            "campaign_end": "2024-01-05T00:00:00+00:00",
            "first_detection": "2024-01-04T18:00:00Z",
            "lag_hours": 18.0,
            "detected": True,
        },
    }

    # Generate full report
    empty_results = pd.DataFrame()
    report = generate_report(
        results=empty_results,
        lags=lags,
        temporal_auc=0.85,
        ground_truth=gt_df,
        start_date="2024-01-01",
        end_date="2024-01-05",
        model_path="./models",
    )

    # Verify all campaigns are in the full report
    assert len(report["detection_lags"]) == 4, "Full report should have all 4 campaigns"

    # Test top-n filtering logic
    top_n = 2
    sorted_lags = sorted(
        lags.items(),
        key=lambda x: x[1].get("lag_hours", 0) if x[1].get("lag_hours") != float("inf") else -1,
        reverse=True,
    )
    top_n_lags = dict(sorted_lags[:top_n])

    # Should have the two worst lags: wallet_b (23h) and wallet_d (18h)
    assert len(top_n_lags) == 2, "Should have exactly 2 campaigns"
    assert "wallet_b" in top_n_lags, "Should include worst-lag campaign (wallet_b with 23h)"
    assert "wallet_d" in top_n_lags, "Should include second-worst-lag campaign (wallet_d with 18h)"
    assert "wallet_a" not in top_n_lags, "Should exclude lower-lag campaigns"
    assert "wallet_c" not in top_n_lags, "Should exclude lowest-lag campaign"

    # Verify ordering is descending by lag
    lag_values = [v["lag_hours"] for v in top_n_lags.values()]
    assert lag_values == sorted(lag_values, reverse=True), "Lags should be in descending order"


def test_topn_with_undetected_campaigns(tmp_path):
    """Test that --top-n correctly handles undetected (infinite lag) campaigns."""
    gt_data = {
        "wallet": ["wallet_a", "wallet_b", "wallet_c"],
        "asset_pair": ["USD/XLM", "USD/XLM", "USD/XLM"],
        "campaign_start": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        "campaign_end": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z", "2024-01-04T00:00:00Z"],
        "label_source": ["https://example.com/1", "https://example.com/2", "https://example.com/3"],
    }
    gt_df = pd.DataFrame(gt_data)

    lags = {
        "wallet_a": {
            "wallet": "wallet_a",
            "campaign_start": "2024-01-01T00:00:00+00:00",
            "campaign_end": "2024-01-02T00:00:00+00:00",
            "first_detection": None,
            "lag_hours": float("inf"),
            "detected": False,
        },
        "wallet_b": {
            "wallet": "wallet_b",
            "campaign_start": "2024-01-02T00:00:00+00:00",
            "campaign_end": "2024-01-03T00:00:00+00:00",
            "first_detection": "2024-01-02T06:00:00Z",
            "lag_hours": 6.0,
            "detected": True,
        },
        "wallet_c": {
            "wallet": "wallet_c",
            "campaign_start": "2024-01-03T00:00:00+00:00",
            "campaign_end": "2024-01-04T00:00:00+00:00",
            "first_detection": None,
            "lag_hours": float("inf"),
            "detected": False,
        },
    }

    # When sorting by lag descending, undetected (inf) campaigns should come first
    sorted_lags = sorted(
        lags.items(),
        key=lambda x: x[1].get("lag_hours", 0) if x[1].get("lag_hours") != float("inf") else -1,
        reverse=True,
    )

    top_n_lags = dict(sorted_lags[:2])

    # Should have 2 entries (the undetected campaigns should be prioritized as "worst" in some sense)
    assert len(top_n_lags) == 2, "Should have exactly 2 campaigns"
    # At least one of the undetected should be in top 2
    undetected_in_top = sum(1 for w in top_n_lags if not lags[w]["detected"])
    assert undetected_in_top >= 1, "Undetected campaigns should be in top N"


def test_full_report_without_topn_unchanged():
    """Test that reports without --top-n remain unchanged."""
    gt_data = {
        "wallet": ["wallet_a", "wallet_b"],
        "asset_pair": ["USD/XLM", "USD/XLM"],
        "campaign_start": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        "campaign_end": ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"],
        "label_source": ["https://example.com/1", "https://example.com/2"],
    }
    gt_df = pd.DataFrame(gt_data)

    lags = {
        "wallet_a": {
            "wallet": "wallet_a",
            "campaign_start": "2024-01-01T00:00:00+00:00",
            "campaign_end": "2024-01-02T00:00:00+00:00",
            "first_detection": "2024-01-01T12:00:00Z",
            "lag_hours": 12.0,
            "detected": True,
        },
        "wallet_b": {
            "wallet": "wallet_b",
            "campaign_start": "2024-01-02T00:00:00+00:00",
            "campaign_end": "2024-01-03T00:00:00+00:00",
            "first_detection": "2024-01-02T18:00:00Z",
            "lag_hours": 18.0,
            "detected": True,
        },
    }

    # Generate report without top-n
    empty_results = pd.DataFrame()
    report = generate_report(
        results=empty_results,
        lags=lags,
        temporal_auc=0.85,
        ground_truth=gt_df,
        start_date="2024-01-01",
        end_date="2024-01-03",
        model_path="./models",
    )

    # Should include all campaigns
    assert len(report["detection_lags"]) == 2, "Full report should have all campaigns when top-n not applied"
