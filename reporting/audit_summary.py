"""Audit-ready summaries for anomaly investigation outputs.

Produces structured, tamper-evident summary documents from anomaly
investigation results that are suitable for compliance review, regulatory
submission, and internal audit trails.

Each ``AuditSummary`` includes:
- A unique summary ID and generation timestamp.
- Investigation metadata (wallet, asset pair, verdict, risk score).
- Key evidence items extracted from the forensic report.
- A SHA-256 integrity hash over all fields (excluding the hash itself).
- An optional chain-of-custody record linking to prior summaries.

Usage::

    from reporting.audit_summary import AuditSummaryBuilder

    builder = AuditSummaryBuilder()
    summary = builder.build(forensic_report.to_dict())
    assert summary.verify_integrity()

    # Serialise for storage or transmission
    doc = summary.to_dict()
    json_str = summary.to_json()

    # Batch processing
    summaries = builder.build_batch([r.to_dict() for r in reports])

Security invariants
-------------------
- ``summary_sha256`` is computed over all other fields in ``__post_init__``.
- ``verify_integrity()`` recomputes and compares the hash.
- All timestamps are UTC ISO-8601.
- No user-supplied URLs are included; Horizon links are constructed from
  ``config.HORIZON_URL`` only.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from config import config
from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

_SEVERITY_MAP: dict[str, str] = {
    "clean": "low",
    "suspicious": "medium",
    "wash_trade": "high",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """A single piece of evidence supporting the investigation conclusion."""

    category: str  # e.g. "benford_violation", "shap_feature", "trade_anomaly"
    description: str
    value: str
    source_reference: str = ""  # e.g. Horizon URL or feature name


@dataclass
class AuditSummary:
    """Tamper-evident audit summary of an anomaly investigation."""

    summary_id: str
    generated_at: str
    report_id: str
    wallet: str
    asset_pair: str
    risk_score: int
    score_lower: int
    score_upper: int
    verdict: str
    severity: str
    evidence_items: list[EvidenceItem]
    investigation_notes: str
    model_version: str
    prior_summary_id: str | None = None
    summary_sha256: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.summary_sha256 = self._compute_sha256()

    def _to_dict_without_hash(self) -> dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "generated_at": self.generated_at,
            "report_id": self.report_id,
            "wallet": self.wallet,
            "asset_pair": self.asset_pair,
            "risk_score": self.risk_score,
            "score_lower": self.score_lower,
            "score_upper": self.score_upper,
            "verdict": self.verdict,
            "severity": self.severity,
            "evidence_items": [
                {
                    "category": e.category,
                    "description": e.description,
                    "value": e.value,
                    "source_reference": e.source_reference,
                }
                for e in self.evidence_items
            ],
            "investigation_notes": self.investigation_notes,
            "model_version": self.model_version,
            "prior_summary_id": self.prior_summary_id,
        }

    def _compute_sha256(self) -> str:
        payload = json.dumps(self._to_dict_without_hash(), sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify_integrity(self) -> bool:
        """Recompute the SHA-256 hash and verify it matches the stored value."""
        return self._compute_sha256() == self.summary_sha256

    def to_dict(self) -> dict[str, Any]:
        """Serialise the summary to a dictionary."""
        d = self._to_dict_without_hash()
        d["summary_sha256"] = self.summary_sha256
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialise the summary to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _extract_benford_evidence(report: dict[str, Any]) -> list[EvidenceItem]:
    """Extract Benford analysis violations as evidence items."""
    items: list[EvidenceItem] = []
    benford = report.get("benford_analysis")
    if not benford or not isinstance(benford, dict):
        return items

    for window, metrics in benford.items():
        if not isinstance(metrics, dict):
            continue
        if metrics.get("mad_nonconforming"):
            items.append(
                EvidenceItem(
                    category="benford_violation",
                    description=(
                        f"Benford's Law MAD non-conforming in {window}h window "
                        f"(chi2={metrics.get('chi_square', 'N/A')}, "
                        f"MAD={metrics.get('mad', 'N/A')})"
                    ),
                    value=f"chi2={metrics.get('chi_square')}, mad={metrics.get('mad')}",
                    source_reference=f"benford_analysis.{window}h",
                )
            )

    return items


def _extract_shap_evidence(report: dict[str, Any], top_n: int = 5) -> list[EvidenceItem]:
    """Extract top SHAP feature contributions as evidence items."""
    items: list[EvidenceItem] = []
    shap_features = report.get("top_shap_features") or []

    sorted_features = sorted(
        shap_features,
        key=lambda f: abs(f.get("contribution", 0) or 0),
        reverse=True,
    )

    for feat in sorted_features[:top_n]:
        name = feat.get("feature", "unknown")
        contribution = feat.get("contribution", 0)
        value = feat.get("value", "N/A")
        description = feat.get("description", name)
        items.append(
            EvidenceItem(
                category="shap_feature",
                description=f"{description} (contribution={contribution})",
                value=str(value),
                source_reference=name,
            )
        )

    return items


def _extract_trade_evidence(report: dict[str, Any]) -> list[EvidenceItem]:
    """Extract anomalous trade evidence items."""
    items: list[EvidenceItem] = []
    trades = report.get("trade_evidence") or []

    for trade in trades[:10]:  # Cap at 10 trades for summary
        trade_id = trade.get("trade_id", "unknown")
        horizon_base = config.HORIZON_URL.rstrip("/")
        items.append(
            EvidenceItem(
                category="trade_anomaly",
                description=(
                    f"Anomalous trade {trade_id}: "
                    f"base_amount={trade.get('base_amount', 'N/A')}, "
                    f"counter_amount={trade.get('counter_amount', 'N/A')}"
                ),
                value=trade_id,
                source_reference=f"{horizon_base}/trades/{trade_id}",
            )
        )

    return items


def _build_investigation_notes(report: dict[str, Any]) -> str:
    """Build a concise investigation summary from the report data."""
    wallet = report.get("wallet", "unknown")
    verdict = report.get("verdict", "unknown")
    score = report.get("risk_score", 0)
    asset_pair = report.get("asset_pair", "unknown")

    parts = [
        f"Wallet {wallet} scored {score}/100 for asset pair {asset_pair}.",
        f"Verdict: {verdict}.",
    ]

    # Add causal attribution note if present
    causal = report.get("causal_attribution")
    if causal and isinstance(causal, dict):
        root_cause = causal.get("root_cause_wallet")
        if root_cause:
            parts.append(f"Root cause traced to wallet {root_cause}.")
        cf_score = causal.get("counterfactual_score")
        if cf_score is not None:
            parts.append(f"Counterfactual score (without flagged trades): {cf_score}.")

    # Add propagation note if present
    propagation = report.get("propagation_path")
    if propagation and isinstance(propagation, dict):
        prop_risk = propagation.get("propagated_risk")
        if prop_risk is not None:
            parts.append(f"Propagated risk from network: {prop_risk}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class AuditSummaryBuilder:
    """Builds audit-ready summaries from forensic report dictionaries.

    Parameters
    ----------
    top_shap_features : int
        Number of top SHAP features to include in evidence (default 5).
    """

    def __init__(self, *, top_shap_features: int = 5) -> None:
        self._top_shap = top_shap_features

    def build(
        self,
        report: dict[str, Any],
        *,
        prior_summary_id: str | None = None,
    ) -> AuditSummary:
        """Build an audit summary from a forensic report dict.

        Parameters
        ----------
        report : dict
            A dict produced by ``ForensicReport.to_dict()``.
        prior_summary_id : str | None
            If this is a follow-up investigation, link to the prior summary.

        Returns
        -------
        AuditSummary
            A tamper-evident, serialisable audit summary.
        """
        verdict = report.get("verdict", "unknown")
        severity = _SEVERITY_MAP.get(verdict, "unknown")

        evidence: list[EvidenceItem] = []
        evidence.extend(_extract_benford_evidence(report))
        evidence.extend(_extract_shap_evidence(report, top_n=self._top_shap))
        evidence.extend(_extract_trade_evidence(report))

        model_meta = report.get("model_metadata") or {}
        model_version = model_meta.get("version", "unknown")

        return AuditSummary(
            summary_id=uuid.uuid4().hex,
            generated_at=datetime.now(UTC).isoformat(),
            report_id=report.get("report_id", ""),
            wallet=report.get("wallet", ""),
            asset_pair=report.get("asset_pair", ""),
            risk_score=int(report.get("risk_score", 0)),
            score_lower=int(report.get("score_lower", 0)),
            score_upper=int(report.get("score_upper", 0)),
            verdict=verdict,
            severity=severity,
            evidence_items=evidence,
            investigation_notes=_build_investigation_notes(report),
            model_version=model_version,
            prior_summary_id=prior_summary_id,
        )

    def build_batch(
        self,
        reports: list[dict[str, Any]],
    ) -> list[AuditSummary]:
        """Build audit summaries for a batch of forensic reports.

        Parameters
        ----------
        reports : list[dict]
            List of forensic report dicts.

        Returns
        -------
        list[AuditSummary]
            One summary per report.
        """
        return [self.build(r) for r in reports]
