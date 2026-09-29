import math

import pandas as pd
import pytest

from features.ohlcv_features import compute_ohlcv_features


def test_compute_ohlcv_features_expected_values_1m_known_5_trades():
    # Five trades all fall into the same 1m candle (minute 00).
    ts0 = pd.Timestamp("2024-01-01T00:00:05Z")
    trades = pd.DataFrame(
        {
            "ledger_close_time": [
                ts0,
                ts0 + pd.Timedelta(seconds=10),
                ts0 + pd.Timedelta(seconds=20),
                ts0 + pd.Timedelta(seconds=30),
                ts0 + pd.Timedelta(seconds=45),
            ],
            "price": [10.0, 12.0, 11.0, 13.0, 14.0],
            "amount": [1.0, 2.0, 1.0, 2.0, 4.0],
        }
    )

    # Expected candle metrics for the single 1m candle
    o = 10.0
    h = 14.0
    low = 10.0
    c = 14.0
    v = sum(trades["amount"].tolist())

    price_range_ratio = (h - low) / o
    candle_body_ratio = (c - o) / (h - low)

    vwap = float((trades["price"] * trades["amount"]).sum() / v)
    vwap_deviation = (c - vwap) / vwap

    # volume_spike_ratio uses rolling 20 avg excluding current candle.
    # Since this is the first candle, the rolling average excluding it is NaN,
    # so the ratio should be NaN.

    feats = compute_ohlcv_features(trades, resolutions=["1m"])

    assert feats["price_range_ratio_1m"] == pytest.approx(price_range_ratio)
    # high (14) != low (10) here, so denom != 0 and this is a normal ratio,
    # not the high==low NaN edge case.
    assert feats["candle_body_ratio_1m"] == pytest.approx(candle_body_ratio)
    assert feats["vwap_deviation_1m"] == pytest.approx(vwap_deviation)
    assert math.isnan(feats["volume_spike_ratio_1m"])


def test_compute_ohlcv_features_single_trade_window_nan_for_candle_metrics():
    ts0 = pd.Timestamp("2024-01-01T00:00:05Z")
    trades = pd.DataFrame(
        {
            "ledger_close_time": [ts0],
            "price": [10.0],
            "amount": [5.0],
        }
    )

    feats = compute_ohlcv_features(trades, resolutions=["1m"])
    assert math.isnan(feats["price_range_ratio_1m"])
    assert math.isnan(feats["candle_body_ratio_1m"])
    assert math.isnan(feats["vwap_deviation_1m"])
    assert math.isnan(feats["volume_spike_ratio_1m"])


def test_compute_ohlcv_features_keys_include_resolution_suffixes():
    ts0 = pd.Timestamp("2024-01-01T00:00:05Z")
    trades = pd.DataFrame(
        {
            "ledger_close_time": [ts0 + pd.Timedelta(seconds=i * 10) for i in range(3)],
            "price": [10.0, 11.0, 12.0],
            "amount": [1.0, 1.0, 1.0],
        }
    )

    feats = compute_ohlcv_features(trades, resolutions=["1m", "5m", "1h"])
    for res in ["1m", "5m", "1h"]:
        for base in [
            "price_range_ratio",
            "vwap_deviation",
            "candle_body_ratio",
            "volume_spike_ratio",
        ]:
            assert f"{base}_{res}" in feats


def test_resolution_validation_allowlist():
    ts0 = pd.Timestamp("2024-01-01T00:00:05Z")
    trades = pd.DataFrame(
        {
            "ledger_close_time": [ts0],
            "price": [10.0],
            "amount": [5.0],
        }
    )

    with pytest.raises(ValueError):
        compute_ohlcv_features(trades, resolutions=["2m"])
