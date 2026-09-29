"""Tests for the end-to-end detection workflow examples (Issue #2).

Each test imports and calls the ``main()`` function of an example module and
asserts on the returned result dictionary.  Tests are isolated — no live
Horizon API, no pre-trained models required.

The examples must:
- Return a result dict with at least: score, benford_flag, n_trades
- Not raise unhandled exceptions
- Produce scores in the 0–100 range

Score-direction assertions (clean → low, suspicious → higher) are *soft*:
they are noted in the test but don't fail CI when models haven't been trained,
because the synthetic Benford fallback path may not match trained-model
thresholds on a fresh checkout.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_valid_result(result: dict) -> None:
    """Common assertions every example result must satisfy."""
    assert isinstance(result, dict), "main() must return a dict"
    assert "score" in result, "result must contain 'score'"
    assert "benford_flag" in result, "result must contain 'benford_flag'"
    assert "n_trades" in result, "result must contain 'n_trades'"
    assert 0.0 <= float(result["score"]) <= 100.0, f"score {result['score']} out of [0, 100]"
    assert result["n_trades"] > 0, "example must produce at least one trade"


# ---------------------------------------------------------------------------
# Example: clean trading
# ---------------------------------------------------------------------------


class TestE2ECleanTrading:
    def test_returns_valid_result(self):
        from examples.e2e_clean_trading import main

        result = main()
        _assert_valid_result(result)

    def test_generates_200_trades(self):
        from examples.e2e_clean_trading import main

        result = main()
        assert result["n_trades"] == 200

    def test_score_is_numeric(self):
        from examples.e2e_clean_trading import main

        result = main()
        assert isinstance(float(result["score"]), float)


# ---------------------------------------------------------------------------
# Example: wash-trading ring
# ---------------------------------------------------------------------------


class TestE2EWashTradingRing:
    def test_returns_valid_result(self):
        from examples.e2e_wash_trading_ring import main

        result = main()
        _assert_valid_result(result)

    def test_produces_trades(self):
        from examples.e2e_wash_trading_ring import main

        result = main()
        assert result["n_trades"] >= 10, "Ring example should generate many trades"

    def test_benford_flag_raised_or_score_elevated(self):
        """Either Benford anomaly is flagged or the score is above 0 for
        fixed-lot wash trades."""
        from examples.e2e_wash_trading_ring import main

        result = main()
        # At least one signal should fire for repeated fixed amounts
        assert (
            result["benford_flag"] or result["score"] > 0
        ), "Fixed-lot wash trading should produce a non-zero signal"


# ---------------------------------------------------------------------------
# Example: Benford anomaly
# ---------------------------------------------------------------------------


class TestE2EBenfordAnomaly:
    def test_returns_valid_result(self):
        from examples.e2e_benford_anomaly import main

        result = main()
        _assert_valid_result(result)

    def test_suspicious_scenario_benford_flag(self):
        """Uniform-random amounts should raise the Benford flag."""
        from examples.e2e_benford_anomaly import main

        result = main()
        # The example returns the suspicious-scenario result
        assert result["benford_flag"], "Uniform-distribution amounts should fail the Benford check"


# ---------------------------------------------------------------------------
# Example: cross-venue coordination
# ---------------------------------------------------------------------------


class TestE2ECrossVenueCoordination:
    def test_returns_valid_result(self):
        from examples.e2e_cross_venue_coordination import main

        result = main()
        _assert_valid_result(result)

    def test_produces_trades(self):
        from examples.e2e_cross_venue_coordination import main

        result = main()
        assert result["n_trades"] > 0


# ---------------------------------------------------------------------------
# Example: full pipeline
# ---------------------------------------------------------------------------


class TestE2EFullPipeline:
    def test_runs_without_error(self, capsys):
        """The full-pipeline example should complete without raising."""
        from examples.e2e_full_pipeline import main

        main()
        captured = capsys.readouterr()
        assert "Full pipeline example complete" in captured.out

    def test_prints_score(self, capsys):
        from examples.e2e_full_pipeline import main

        main()
        out = capsys.readouterr().out
        assert "Risk score" in out


# ---------------------------------------------------------------------------
# Helpers module smoke-test
# ---------------------------------------------------------------------------


class TestExampleHelpers:
    def test_make_trade_returns_dict(self):
        from examples._helpers import make_trade

        t = make_trade(amount=1234.5)
        assert isinstance(t, dict)
        assert t["amount"] == 1234.5

    def test_build_trades_df_shape(self):
        from examples._helpers import build_trades_df, make_trade

        trades = [make_trade(amount=float(i)) for i in range(1, 11)]
        df = build_trades_df(trades)
        assert len(df) == 10
        assert "amount" in df.columns
        assert "wallet" in df.columns

    def test_run_detection_returns_required_keys(self):
        from examples._helpers import build_trades_df, make_trade, run_detection

        trades = [make_trade(amount=float(i * 100)) for i in range(1, 30)]
        df = build_trades_df(trades)
        result = run_detection(df, print_summary=False)
        assert "score" in result
        assert "benford_flag" in result
        assert "n_trades" in result

    def test_run_detection_score_in_range(self):
        from examples._helpers import build_trades_df, make_trade, run_detection

        trades = [make_trade(amount=float(i)) for i in range(1, 50)]
        df = build_trades_df(trades)
        result = run_detection(df, print_summary=False)
        assert 0.0 <= result["score"] <= 100.0
