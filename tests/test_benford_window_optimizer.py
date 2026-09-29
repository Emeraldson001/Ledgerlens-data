"""Tests for adaptive Benford window selection (Issue #693).

Focus: the insufficient-data path, where a wallet/pair has too few trades for
any candidate window to hold a statistically meaningful sample.
"""

import pytest

from detection.benford_window_optimizer import select_optimal_window

PAIR = "USDC:GA5Z/XLM:native"
WINDOWS = [1, 4, 24, 168, 720]


class TestInsufficientData:
    """A pair with only one or two trades must not pick a window on noise."""

    def test_two_trades_total_falls_back_to_longest_window(self):
        # Both trades land inside the 24h window, so every window is far below
        # the threshold.
        counts = {1: 0, 4: 1, 24: 2, 168: 2, 720: 2}

        assert select_optimal_window(PAIR, counts, min_sample_size=50) == 720

    def test_single_trade_falls_back_to_longest_window(self):
        counts = {1: 0, 4: 0, 24: 1, 168: 1, 720: 1}

        assert select_optimal_window(PAIR, counts, min_sample_size=50) == 720

    def test_new_pair_with_zero_trades_falls_back_to_longest_window(self):
        counts = dict.fromkeys(WINDOWS, 0)

        assert select_optimal_window(PAIR, counts, min_sample_size=50) == 720

    def test_fallback_respects_supplied_candidate_windows(self):
        """The fallback is the longest *candidate*, not a hardcoded 720."""
        counts = {1: 0, 4: 1, 24: 2, 168: 2, 720: 2}

        selected = select_optimal_window(
            PAIR, counts, min_sample_size=50, candidate_windows=[1, 4, 24]
        )

        assert selected == 24

    def test_empty_candidate_windows_falls_back_to_widest_default(self):
        assert select_optimal_window(PAIR, {}, min_sample_size=50) == 720

    def test_warning_is_logged_on_insufficient_data(self, caplog):
        counts = {1: 0, 4: 1, 24: 2, 168: 2, 720: 2}

        with caplog.at_level("WARNING", logger="detection.benford_window_optimizer"):
            select_optimal_window(PAIR, counts, min_sample_size=50)

        assert "no window meets minimum sample threshold" in caplog.text


class TestSufficientData:
    """Sanity checks that the fallback does not swallow the normal path."""

    def test_shortest_qualifying_window_is_selected(self):
        counts = {1: 5, 4: 20, 24: 100, 168: 500, 720: 2000}

        assert select_optimal_window(PAIR, counts, min_sample_size=50) == 24

    def test_min_sample_size_below_ten_is_rejected(self):
        with pytest.raises(ValueError, match="must be >= 10"):
            select_optimal_window(PAIR, {1: 5}, min_sample_size=2)
