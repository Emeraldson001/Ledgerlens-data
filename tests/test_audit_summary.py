"""Tests for reporting.audit_summary — audit-ready summaries for anomaly outputs.

Covers:
- Tamper-evident integrity hashing (SHA-256)
- Evidence extraction (Benford, SHAP, trade anomalies)
- Investigation notes generation
- Severity mapping from verdicts
- Serialisation (to_dict, to_json)
- Batch building
- Prior summary chaining
- Edge cases (empty reports, missing fields)
"""

from __future__ import annotations

import json

import pytest

from reporting.audit_summary import (
    AuditSummaryBuilder,
    EvidenceItem,
    _build_investigation_notes,
    _extract_benford_evidence,
    _extract_shap_evidence,
    _extract_trade_evidence,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_report(**overrides) -> dict:
    """Create a minimal forensic report dict for testing."""
    base = {
        "report_id": "rpt-001",
        "generated_at": "2025-01-15T12:00:00+00:00",
        "wallet": "GBLT2XJKNNB7DOYP3QOELK4WPU64BXFMFYXKGQP6K5FKZRZE6SYGNM",
        "asset_pair": "USDC:native/XLM:native",
        "risk_score": 82,
        "score_lower": 72,
        "score_upper": 92,
        "verdict": "wash_trade",
        "top_shap_features": [
            {
                "feature": "benford_mad_24h",
                "contribution": 0.35,
                "value": 0.042,
                "description": "Benford MAD 24h",
            },
            {
                "feature": "counterparty_concentration_ratio",
                "contribution": 0.22,
                "value": 0.91,
                "description": "Counterparty concentration",
            },
            {
                "feature": "round_trip_frequency",
                "contribution": 0.15,
                "value": 0.8,
                "description": "Round-trip frequency",
            },
        ],
        "benford_analysis": {
            "24": {
                "chi_square": 45.2,
                "mad": 0.042,
                "mad_nonconforming": True,
                "z_scores": {"1": 3.2, "2": 1.1},
                "sample_size": 150,
            },
            "168": {
                "chi_square": 12.1,
                "mad": 0.008,
                "mad_nonconforming": False,
                "sample_size": 500,
            },
        },
        "trade_evidence": [
            {
                "trade_id": "trade-001",
                "ledger": 12345,
                "base_account": "GACC1",
                "counter_account": "GACC2",
                "base_amount": 1000.0,
                "counter_amount": 999.5,
                "asset_pair": "USDC/XLM",
                "horizon_url": "https://horizon.stellar.org/trades/trade-001",
            },
        ],
        "model_metadata": {
            "name": "LedgerLens Ensemble",
            "version": "2.1.0",
            "training_dataset_sha256": "abc123",
        },
    }
    base.update(overrides)
    return base


@pytest.fixture()
def sample_report() -> dict:
    return _make_report()


@pytest.fixture()
def builder() -> AuditSummaryBuilder:
    return AuditSummaryBuilder()


# ---------------------------------------------------------------------------
# AuditSummary integrity
# ---------------------------------------------------------------------------


class TestAuditSummaryIntegrity:
    def test_integrity_valid(self, builder: AuditSummaryBuilder, sample_report: dict) -> None:
        summary = builder.build(sample_report)
        assert summary.verify_integrity()

    def test_integrity_fails_on_tamper(
        self, builder: AuditSummaryBuilder, sample_report: dict
    ) -> None:
        summary = builder.build(sample_report)
        summary.risk_score = 0  # Tamper
        assert not summary.verify_integrity()

    def test_hash_is_hex_64(self, builder: AuditSummaryBuilder, sample_report: dict) -> None:
        summary = builder.build(sample_report)
        assert len(summary.summary_sha256) == 64
        int(summary.summary_sha256, 16)  # Should not raise

    def test_different_reports_different_hashes(self, builder: AuditSummaryBuilder) -> None:
        s1 = builder.build(_make_report(risk_score=50))
        s2 = builder.build(_make_report(risk_score=90))
        assert s1.summary_sha256 != s2.summary_sha256


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


class TestBenfordEvidence:
    def test_extracts_nonconforming_windows(self, sample_report: dict) -> None:
        items = _extract_benford_evidence(sample_report)
        assert len(items) == 1  # Only 24h window is non-conforming
        assert items[0].category == "benford_violation"
        assert "24h" in items[0].description

    def test_no_benford_returns_empty(self) -> None:
        assert _extract_benford_evidence({}) == []
        assert _extract_benford_evidence({"benford_analysis": {}}) == []

    def test_all_conforming_returns_empty(self) -> None:
        report = _make_report(
            benford_analysis={
                "24": {"chi_square": 5.0, "mad": 0.005, "mad_nonconforming": False},
            }
        )
        assert _extract_benford_evidence(report) == []


class TestShapEvidence:
    def test_extracts_top_features(self, sample_report: dict) -> None:
        items = _extract_shap_evidence(sample_report, top_n=2)
        assert len(items) == 2
        assert items[0].category == "shap_feature"
        # Should be sorted by contribution magnitude
        assert "0.35" in items[0].description

    def test_no_shap_returns_empty(self) -> None:
        assert _extract_shap_evidence({}) == []
        assert _extract_shap_evidence({"top_shap_features": []}) == []


class TestTradeEvidence:
    def test_extracts_trades(self, sample_report: dict) -> None:
        items = _extract_trade_evidence(sample_report)
        assert len(items) == 1
        assert items[0].category == "trade_anomaly"
        assert "trade-001" in items[0].value

    def test_no_trades_returns_empty(self) -> None:
        assert _extract_trade_evidence({}) == []
        assert _extract_trade_evidence({"trade_evidence": []}) == []

    def test_caps_at_10_trades(self) -> None:
        trades = [{"trade_id": f"t{i}", "base_amount": i, "counter_amount": i} for i in range(20)]
        report = _make_report(trade_evidence=trades)
        items = _extract_trade_evidence(report)
        assert len(items) == 10


# ---------------------------------------------------------------------------
# Investigation notes
# ---------------------------------------------------------------------------


class TestInvestigationNotes:
    def test_basic_notes(self, sample_report: dict) -> None:
        notes = _build_investigation_notes(sample_report)
        assert "82/100" in notes
        assert "wash_trade" in notes

    def test_causal_attribution_included(self) -> None:
        report = _make_report(
            causal_attribution={
                "root_cause_wallet": "GMALICIOUS",
                "counterfactual_score": 25,
            }
        )
        notes = _build_investigation_notes(report)
        assert "GMALICIOUS" in notes
        assert "25" in notes

    def test_propagation_included(self) -> None:
        report = _make_report(propagation_path={"propagated_risk": 0.45})
        notes = _build_investigation_notes(report)
        assert "0.45" in notes

    def test_missing_fields_handled(self) -> None:
        notes = _build_investigation_notes({})
        assert "unknown" in notes


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestAuditSummaryBuilder:
    def test_build_basic(self, builder: AuditSummaryBuilder, sample_report: dict) -> None:
        summary = builder.build(sample_report)
        assert summary.report_id == "rpt-001"
        assert summary.risk_score == 82
        assert summary.verdict == "wash_trade"
        assert summary.severity == "high"
        assert len(summary.evidence_items) > 0
        assert summary.model_version == "2.1.0"
        assert summary.summary_id  # Non-empty UUID

    def test_severity_mapping(self, builder: AuditSummaryBuilder) -> None:
        for verdict, expected in [
            ("clean", "low"),
            ("suspicious", "medium"),
            ("wash_trade", "high"),
        ]:
            s = builder.build(_make_report(verdict=verdict))
            assert s.severity == expected

    def test_unknown_verdict_severity(self, builder: AuditSummaryBuilder) -> None:
        s = builder.build(_make_report(verdict="new_category"))
        assert s.severity == "unknown"

    def test_prior_summary_chaining(
        self, builder: AuditSummaryBuilder, sample_report: dict
    ) -> None:
        s1 = builder.build(sample_report)
        s2 = builder.build(sample_report, prior_summary_id=s1.summary_id)
        assert s2.prior_summary_id == s1.summary_id
        assert s2.verify_integrity()

    def test_build_batch(self, builder: AuditSummaryBuilder) -> None:
        reports = [_make_report(risk_score=i * 10) for i in range(5)]
        summaries = builder.build_batch(reports)
        assert len(summaries) == 5
        assert all(s.verify_integrity() for s in summaries)
        scores = [s.risk_score for s in summaries]
        assert scores == [0, 10, 20, 30, 40]

    def test_custom_top_shap(self) -> None:
        builder = AuditSummaryBuilder(top_shap_features=1)
        report = _make_report()
        summary = builder.build(report)
        shap_evidence = [e for e in summary.evidence_items if e.category == "shap_feature"]
        assert len(shap_evidence) == 1

    def test_empty_report(self, builder: AuditSummaryBuilder) -> None:
        summary = builder.build({})
        assert summary.risk_score == 0
        assert summary.verdict == "unknown"
        assert summary.severity == "unknown"
        assert summary.verify_integrity()


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_dict(self, builder: AuditSummaryBuilder, sample_report: dict) -> None:
        summary = builder.build(sample_report)
        d = summary.to_dict()
        assert d["summary_sha256"] == summary.summary_sha256
        assert d["report_id"] == "rpt-001"
        assert isinstance(d["evidence_items"], list)

    def test_to_json(self, builder: AuditSummaryBuilder, sample_report: dict) -> None:
        summary = builder.build(sample_report)
        j = summary.to_json()
        parsed = json.loads(j)
        assert parsed["report_id"] == "rpt-001"
        assert parsed["summary_sha256"] == summary.summary_sha256

    def test_roundtrip_json_integrity(
        self, builder: AuditSummaryBuilder, sample_report: dict
    ) -> None:
        summary = builder.build(sample_report)
        j = summary.to_json()
        parsed = json.loads(j)
        # Verify the hash in JSON matches the original
        assert parsed["summary_sha256"] == summary.summary_sha256


# ---------------------------------------------------------------------------
# EvidenceItem
# ---------------------------------------------------------------------------


class TestEvidenceItem:
    def test_basic_construction(self) -> None:
        item = EvidenceItem(
            category="test",
            description="A test item",
            value="42",
            source_reference="test_ref",
        )
        assert item.category == "test"
        assert item.value == "42"

    def test_default_source_reference(self) -> None:
        item = EvidenceItem(category="test", description="test", value="1")
        assert item.source_reference == ""
