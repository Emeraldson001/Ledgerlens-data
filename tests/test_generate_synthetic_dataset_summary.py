"""Tests for label-distribution summary in generate_synthetic_dataset."""

import io

import pandas as pd

from scripts.generate_synthetic_dataset import (
    generate_synthetic_dataset,
    print_dataset_summary,
)


def test_print_dataset_summary_shows_label_counts_and_features():
    df = generate_synthetic_dataset(n_wallets=100, seed=42)
    buf = io.StringIO()

    print_dataset_summary(df, profile="NaiveAttacker", file=buf)
    text = buf.getvalue()

    assert "Label distribution:" in text
    assert "wash_trade  (label=1):" in text
    assert "legitimate  (label=0):" in text
    assert "50.0%" in text
    assert "Feature summary (wash_trade rows):" in text
    assert "benford_chi_square_24h:" in text
    assert "counterparty_concentration_ratio:" in text
    assert "Profile breakdown:" not in text


def test_print_dataset_summary_shows_profile_breakdown_for_simulator_profile():
    df = pd.DataFrame(
        {
            "wallet": ["G1", "G2", "G3", "G4"],
            "label": [1, 1, 0, 0],
            "profile": ["RingAttacker"] * 4,
            "benford_chi_square_24h": [60.0, 55.0, 5.0, 4.0],
            "counterparty_concentration_ratio": [0.9, 0.8, 0.2, 0.3],
        }
    )
    buf = io.StringIO()

    print_dataset_summary(df, profile="RingAttacker", file=buf)
    text = buf.getvalue()

    assert "Profile breakdown:" in text
    assert "RingAttacker: 4 rows  (wash=2, legitimate=2)" in text


def test_print_dataset_summary_handles_empty_dataframe():
    buf = io.StringIO()
    print_dataset_summary(pd.DataFrame(), profile="NaiveAttacker", file=buf)
    assert buf.getvalue() == ""


def test_generate_synthetic_dataset_produces_correct_class_balance():
    """Test that generated dataset has expected 50/50 class balance."""
    df = generate_synthetic_dataset(n_wallets=100, seed=42)

    # Verify we have exactly 50 wash-trade and 50 legitimate
    wash_count = int((df["label"] == 1).sum())
    legit_count = int((df["label"] == 0).sum())

    assert wash_count == 50, f"Expected 50 wash-trade, got {wash_count}"
    assert legit_count == 50, f"Expected 50 legitimate, got {legit_count}"
    assert wash_count + legit_count == 100, "Total rows should be 100"


def test_print_dataset_summary_includes_exact_percentages():
    """Test that summary shows precise percentages for deterministic seed."""
    df = generate_synthetic_dataset(n_wallets=50, seed=99)
    buf = io.StringIO()

    print_dataset_summary(df, profile="NaiveAttacker", file=buf)
    text = buf.getvalue()

    # With n_wallets=50, seed=99, we should have 25 wash and 25 legitimate
    # (feature-level generation splits at n_wallets // 2)
    assert "(50.0%)" in text, "Should show 50% for balanced dataset"
