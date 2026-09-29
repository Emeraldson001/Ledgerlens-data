"""Tests for validation/reconciliation.py — Issue #554: Reconciliation checks."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from streaming.alert_ledger import AlertDeliveryRecord
from validation.reconciliation import (
    ReconciliationError,
    ReconciliationReport,
    merge_reports,
    reconcile_alert_delivery,
    reconcile_features,
    reconcile_trade_counts,
    reconcile_wallet_scores,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _raw_trades(wallets: list[str], n_trades_each: int = 5) -> pd.DataFrame:
    rows = []
    for w in wallets:
        for _i in range(n_trades_each):
            rows.append(
                {
                    "base_account": w,
                    "counter_account": "GCOUNTERPARTY123456789012345678901234567890",
                    "amount": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _feature_matrix(wallets: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wallet_id": wallets,
            "benford_chi_square_1h": [1.0] * len(wallets),
            "trade_count": [10] * len(wallets),
            "label": [0] * len(wallets),
        }
    )


def _scored_wallets(wallets: list[str], scores: list[float] | None = None) -> pd.DataFrame:
    if scores is None:
        scores = [50.0] * len(wallets)
    return pd.DataFrame({"wallet_id": wallets, "score": scores})


# ---------------------------------------------------------------------------
# ReconciliationReport helpers
# ---------------------------------------------------------------------------


class TestReconciliationReport:
    def test_ok_when_no_errors(self):
        r = ReconciliationReport()
        assert r.ok is True

    def test_not_ok_when_hard_error(self):
        r = ReconciliationReport()
        r.errors.append(ReconciliationError("check", "wallet1", "present", "absent", "error"))
        assert r.ok is False

    def test_ok_with_only_warnings(self):
        r = ReconciliationReport()
        r.errors.append(ReconciliationError("check", "wallet1", "x", "y", "warning"))
        assert r.ok is True

    def test_raise_if_errors_raises(self):
        r = ReconciliationReport()
        r.errors.append(ReconciliationError("check", "w", "expected", "observed", "error"))
        with pytest.raises(ValueError, match="Reconciliation failed"):
            r.raise_if_errors()

    def test_raise_if_errors_silent_on_warnings(self):
        r = ReconciliationReport()
        r.errors.append(ReconciliationError("check", "w", "x", "y", "warning"))
        r.raise_if_errors()  # should not raise

    def test_to_dict_structure(self):
        r = ReconciliationReport(checks_run=["trade_counts"])
        d = r.to_dict()
        assert "ok" in d
        assert "errors" in d
        assert "checks_run" in d

    def test_summary_string(self):
        r = ReconciliationReport()
        s = r.summary()
        assert "ReconciliationReport" in s


class TestMergeReports:
    def test_merges_checks_and_errors(self):
        r1 = ReconciliationReport(checks_run=["a"])
        r2 = ReconciliationReport(checks_run=["b"])
        r1.errors.append(ReconciliationError("a", "w1", "x", "y", "error"))
        merged = merge_reports(r1, r2)
        assert set(merged.checks_run) == {"a", "b"}
        assert merged.hard_error_count == 1


# ---------------------------------------------------------------------------
# reconcile_trade_counts
# ---------------------------------------------------------------------------


class TestReconcileTradeCounts:
    def test_passes_when_all_wallets_present(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        raw = _raw_trades(wallets)
        features = _feature_matrix(wallets)
        r = reconcile_trade_counts(raw, features)
        assert r.ok

    def test_flags_wallet_missing_from_raw(self):
        raw = _raw_trades(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        features = _feature_matrix(
            [
                "GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
                "GMISSINGWALLET1234567890ABCDEFGHIJKLMNOPQR",
            ]
        )
        r = reconcile_trade_counts(raw, features)
        assert not r.ok
        missing_entities = {e.entity for e in r.errors if e.severity == "error"}
        assert "GMISSINGWALLET1234567890ABCDEFGHIJKLMNOPQR" in missing_entities

    def test_tolerance_allows_some_missing(self):
        wallets = [f"GAAAA{str(i).zfill(51)}"[:56] for i in range(10)]
        raw = _raw_trades(wallets[:8])  # 2 missing
        features = _feature_matrix(wallets)
        # 20% tolerance should allow 2/10 missing
        r = reconcile_trade_counts(raw, features, tolerance=0.2)
        # With tolerance, missing wallets become warnings not errors
        hard_errors = [e for e in r.errors if e.severity == "error"]
        assert len(hard_errors) == 0

    def test_reports_metadata(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        raw = _raw_trades(wallets)
        features = _feature_matrix(wallets)
        r = reconcile_trade_counts(raw, features)
        assert "raw_wallet_count" in r.metadata
        assert "feature_wallet_count" in r.metadata

    def test_missing_raw_wallet_column_returns_error(self):
        raw = pd.DataFrame({"amount": [1.0, 2.0]})  # no account columns
        features = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        r = reconcile_trade_counts(raw, features)
        assert not r.ok

    def test_missing_feature_wallet_column_returns_error(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        raw = _raw_trades(wallets)
        features = pd.DataFrame({"some_feature": [1.0]})  # no wallet_id
        r = reconcile_trade_counts(raw, features)
        assert not r.ok


# ---------------------------------------------------------------------------
# reconcile_features
# ---------------------------------------------------------------------------


class TestReconcileFeatures:
    def test_passes_when_all_columns_present(self):
        df = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        r = reconcile_features(df, required_columns=["wallet_id", "label"])
        assert r.ok

    def test_flags_missing_required_column(self):
        df = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        r = reconcile_features(df, required_columns=["wallet_id", "nonexistent_col"])
        assert not r.ok
        col_errors = {e.entity for e in r.errors if e.severity == "error"}
        assert "nonexistent_col" in col_errors

    def test_flags_all_nan_column(self):
        df = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        df["all_nan_col"] = float("nan")
        r = reconcile_features(df)
        error_entities = {e.entity for e in r.errors if e.severity == "error"}
        assert "all_nan_col" in error_entities

    def test_range_check_warns_out_of_range(self, tmp_path):
        ranges = {"benford_chi_square_1h": {"min": 0.0, "max": 50.0, "mean": 5.0, "std": 2.0}}
        ranges_path = tmp_path / "feature_ranges.json"
        ranges_path.write_text(json.dumps(ranges), encoding="utf-8")

        df = pd.DataFrame(
            {
                "wallet_id": ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"],
                "benford_chi_square_1h": [999.0],  # way above max
            }
        )
        r = reconcile_features(df, feature_ranges_path=ranges_path)
        warning_entities = {e.entity for e in r.errors if e.severity == "warning"}
        assert "benford_chi_square_1h" in warning_entities

    def test_records_metadata(self):
        df = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        r = reconcile_features(df)
        assert r.metadata["row_count"] == 1
        assert r.metadata["column_count"] == len(df.columns)


# ---------------------------------------------------------------------------
# reconcile_wallet_scores
# ---------------------------------------------------------------------------


class TestReconcileWalletScores:
    def test_passes_when_all_wallets_scored(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix(wallets)
        scores = _scored_wallets(wallets)
        r = reconcile_wallet_scores(features, scores)
        assert r.ok

    def test_flags_unscored_wallet(self):
        wallets = [
            "GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "GBBBBBBCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
        ]
        features = _feature_matrix(wallets)
        scores = _scored_wallets(wallets[:1])  # only first wallet scored
        r = reconcile_wallet_scores(features, scores)
        assert not r.ok

    def test_unscored_is_warning_when_allow_unscored(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix(wallets)
        scores = _scored_wallets([])  # no scores at all
        r = reconcile_wallet_scores(features, scores, allow_unscored=True)
        # all unscored → only warnings, ok = True
        assert r.ok

    def test_flags_out_of_range_score(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix(wallets)
        scores = _scored_wallets(wallets, scores=[150.0])  # > 100
        r = reconcile_wallet_scores(features, scores)
        assert not r.ok

    def test_flags_nan_score(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix(wallets)
        scores = pd.DataFrame({"wallet_id": wallets, "score": [float("nan")]})
        r = reconcile_wallet_scores(features, scores)
        assert not r.ok

    def test_orphan_score_is_warning(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix([])  # empty feature matrix
        scores = _scored_wallets(wallets)
        r = reconcile_wallet_scores(features, scores)
        # orphan score = warning severity
        orphan_errors = [e for e in r.errors if "orphan" in e.observed]
        assert all(e.severity == "warning" for e in orphan_errors)

    def test_missing_column_returns_error(self):
        features = _feature_matrix(["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"])
        scores = pd.DataFrame({"wallet_id": ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]})
        # missing 'score' column
        r = reconcile_wallet_scores(features, scores)
        assert not r.ok

    def test_records_metadata(self):
        wallets = ["GAAAAAABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"]
        features = _feature_matrix(wallets)
        scores = _scored_wallets(wallets)
        r = reconcile_wallet_scores(features, scores)
        assert "feature_wallet_count" in r.metadata
        assert "scored_wallet_count" in r.metadata


# ---------------------------------------------------------------------------
# reconcile_alert_delivery (Issue #670, required scope E)
# ---------------------------------------------------------------------------


def _scores_df(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["wallet_id", "asset_pair", "score"])


class TestReconcileAlertDelivery:
    def test_delivered_alert_is_accounted_for(self):
        scores = _scores_df([("GABC", "pair-1", 85.0)])
        records = [
            AlertDeliveryRecord(
                wallet="GABC",
                pair_id="pair-1",
                score=85,
                outcome="delivered",
                channel="stdout",
                reason=None,
            )
        ]
        report = reconcile_alert_delivery(scores, records, threshold=70.0)
        assert report.ok
        assert report.metadata["missing_count"] == 0
        assert report.metadata["delivered_count"] == 1

    def test_dead_lettered_alert_is_accounted_for_not_missing(self):
        """The exact acceptance-criterion scenario: a dead-lettered webhook
        alert must be reported as accounted-for, not missing."""
        scores = _scores_df([("GABC", "pair-1", 90.0)])
        records = [
            AlertDeliveryRecord(
                wallet="GABC",
                pair_id="pair-1",
                score=90,
                outcome="dead_lettered",
                channel="webhook",
                reason="HTTP 500",
            )
        ]
        report = reconcile_alert_delivery(scores, records, threshold=70.0)
        assert report.ok  # dead-lettered is a warning, not a hard error
        assert report.metadata["missing_count"] == 0
        assert report.metadata["dead_lettered_count"] == 1
        assert any(
            "dead_lettered" in str(e.observed) and e.severity == "warning" for e in report.errors
        )

    def test_suppressed_cooldown_alert_is_accounted_for(self):
        scores = _scores_df([("GABC", "pair-1", 80.0)])
        records = [
            AlertDeliveryRecord(
                wallet="GABC",
                pair_id="pair-1",
                score=80,
                outcome="suppressed_cooldown",
                channel="stdout",
                reason="cooldown active",
            )
        ]
        report = reconcile_alert_delivery(scores, records, threshold=70.0)
        assert report.ok
        assert report.metadata["missing_count"] == 0
        assert report.metadata["suppressed_cooldown_count"] == 1

    def test_missing_outcome_is_hard_error(self):
        """A qualifying score with NO recorded outcome is exactly the
        silently-dropped-alert failure mode this check exists to catch."""
        scores = _scores_df([("GABC", "pair-1", 95.0)])
        report = reconcile_alert_delivery(scores, [], threshold=70.0)
        assert not report.ok
        assert report.hard_error_count == 1
        assert report.metadata["missing_count"] == 1

    def test_sub_threshold_score_requires_no_outcome(self):
        scores = _scores_df([("GABC", "pair-1", 10.0)])
        report = reconcile_alert_delivery(scores, [], threshold=70.0)
        assert report.ok
        assert report.metadata["qualifying_count"] == 0

    def test_missing_required_column_is_error(self):
        scores = pd.DataFrame({"wallet_id": ["GABC"]})
        report = reconcile_alert_delivery(scores, [], threshold=70.0)
        assert not report.ok
