"""Reconciliation checks between raw and derived records.

Issue #554 — Build reconciliation checks between raw and derived records
=========================================================================

This module provides *cross-layer consistency assertions* that verify the
derived artefacts produced by the detection and feature pipeline stay in sync
with the raw data they were computed from.

Why reconciliation?
-------------------
The LedgerLens pipeline has three distinct data layers:

1. **Raw** – Horizon trade records, order-book events, account activity.
2. **Derived (features)** – per-wallet feature vectors built by
   ``detection/feature_engineering.py``.
3. **Scored** – ``RiskScore`` records persisted by
   ``detection/risk_score_store.py``.

Silent discrepancies between these layers (e.g. a feature row referencing a
wallet that has no raw trades, or a risk score whose wallet is absent from the
feature matrix) signal pipeline bugs, partial failures, or data-corruption
events.  ``ReconciliationReport`` surfaces these discrepancies so they can be
caught in CI, nightly jobs, or contributor validation suites (Issue #558).

Public API
----------
::

    from validation.reconciliation import (
        reconcile_trade_counts,
        reconcile_features,
        reconcile_wallet_scores,
        reconcile_alert_delivery,
        ReconciliationReport,
        ReconciliationError,
    )

``reconcile_alert_delivery`` (Issue #670, required scope E) traces every
scored wallet at or above the alert threshold through to a terminal
delivery outcome recorded in ``streaming.alert_ledger.AlertDeliveryLedger``
— ``delivered``, ``dead_lettered`` (e.g. a webhook 500), or
``suppressed_cooldown`` — so a silently dropped alert is distinguishable
from an intentionally suppressed one, and a dead-lettered alert is reported
as accounted-for rather than as a missing trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from streaming.alert_ledger import AlertDeliveryRecord

# ---------------------------------------------------------------------------
# Result / error types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReconciliationError:
    """A single reconciliation discrepancy.

    Attributes
    ----------
    check:
        Short identifier for the check that failed (e.g. ``"trade_count"``).
    entity:
        The wallet, pair, or file that triggered the discrepancy.
    expected:
        Expected value or description.
    observed:
        Observed value or description.
    severity:
        ``"error"`` (data integrity failure) or ``"warning"`` (soft anomaly).
    """

    check: str
    entity: str
    expected: Any
    observed: Any
    severity: str = "error"

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.check} – {self.entity}: "
            f"expected {self.expected!r}, got {self.observed!r}"
        )


@dataclass
class ReconciliationReport:
    """Aggregated result of one or more reconciliation checks.

    Attributes
    ----------
    checks_run:
        Names of the checks that were executed.
    errors:
        All :class:`ReconciliationError` instances (both ``"error"`` and
        ``"warning"`` severity).
    metadata:
        Arbitrary key/value annotations (e.g. counts, timestamps).
    """

    checks_run: list[str] = field(default_factory=list)
    errors: list[ReconciliationError] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        """True when no ``"error"``-severity discrepancies were found."""
        return not any(e.severity == "error" for e in self.errors)

    @property
    def hard_error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == "warning")

    def raise_if_errors(self) -> None:
        """Raise :class:`ReconciliationError` (as a ValueError) if any hard errors exist."""
        if not self.ok:
            lines = "\n  ".join(str(e) for e in self.errors if e.severity == "error")
            raise ValueError(
                f"Reconciliation failed with {self.hard_error_count} error(s):\n  {lines}"
            )

    def summary(self) -> str:
        return (
            f"ReconciliationReport("
            f"checks={len(self.checks_run)}, "
            f"errors={self.hard_error_count}, "
            f"warnings={self.warning_count}, "
            f"ok={self.ok})"
        )

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks_run": self.checks_run,
            "hard_error_count": self.hard_error_count,
            "warning_count": self.warning_count,
            "errors": [
                {
                    "check": e.check,
                    "entity": e.entity,
                    "expected": e.expected,
                    "observed": e.observed,
                    "severity": e.severity,
                }
                for e in self.errors
            ],
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Helper: merge multiple sub-reports into one
# ---------------------------------------------------------------------------


def merge_reports(*reports: ReconciliationReport) -> ReconciliationReport:
    """Combine multiple :class:`ReconciliationReport` instances into one."""
    merged = ReconciliationReport()
    for r in reports:
        merged.checks_run.extend(r.checks_run)
        merged.errors.extend(r.errors)
        merged.metadata.update(r.metadata)
    return merged


# ---------------------------------------------------------------------------
# Check 1: trade-count reconciliation (raw trades ↔ feature-matrix rows)
# ---------------------------------------------------------------------------


def reconcile_trade_counts(
    raw_trades: pd.DataFrame,
    feature_matrix: pd.DataFrame,
    *,
    wallet_column: str = "wallet_id",
    raw_wallet_columns: tuple[str, str] = ("base_account", "counter_account"),
    min_trades_threshold: int = 1,
    tolerance: float = 0.0,
) -> ReconciliationReport:
    """Assert that every wallet in the feature matrix has corresponding raw trades.

    Checks
    ------
    - **Missing wallets**: wallets present in ``feature_matrix`` but absent from
      ``raw_trades`` (i.e. no trade references the wallet as base or counter
      account).
    - **Zero-trade wallets**: feature rows whose trade count is suspiciously low
      given the raw data (soft warning when count < *min_trades_threshold*).
    - **Coverage**: fraction of raw wallets that have a feature row.

    Parameters
    ----------
    raw_trades:
        DataFrame of raw trade records (e.g. from Horizon or CSV).  Must
        contain the columns named by *raw_wallet_columns*.
    feature_matrix:
        Per-wallet feature DataFrame.  Must contain *wallet_column*.
    wallet_column:
        Column in *feature_matrix* that holds the wallet identifier.
    raw_wallet_columns:
        The (base_account, counter_account) columns in *raw_trades* used to
        derive the set of wallets that appear in raw data.
    min_trades_threshold:
        Minimum number of raw trades expected per wallet in the feature matrix.
        Wallets below this threshold emit a ``"warning"``-severity entry.
    tolerance:
        Fraction of feature wallets allowed to be missing from raw data before
        the check fails (0.0 = strict, 0.1 = allow 10% missing).

    Returns
    -------
    ReconciliationReport
    """
    report = ReconciliationReport(checks_run=["trade_counts"])

    base_col, counter_col = raw_wallet_columns

    # Build the set of wallets that appear in raw trades.
    raw_wallet_sets: list[pd.Series] = []
    for col in (base_col, counter_col):
        if col in raw_trades.columns:
            raw_wallet_sets.append(raw_trades[col].dropna())
    if not raw_wallet_sets:
        report.errors.append(
            ReconciliationError(
                check="trade_counts",
                entity="<raw_trades>",
                expected=f"columns {raw_wallet_columns}",
                observed=list(raw_trades.columns),
                severity="error",
            )
        )
        return report

    raw_wallets: set[str] = set(pd.concat(raw_wallet_sets).unique())

    if wallet_column not in feature_matrix.columns:
        report.errors.append(
            ReconciliationError(
                check="trade_counts",
                entity="<feature_matrix>",
                expected=f"column '{wallet_column}'",
                observed=list(feature_matrix.columns),
                severity="error",
            )
        )
        return report

    feature_wallets: set[str] = set(feature_matrix[wallet_column].dropna().unique())

    missing_wallets = feature_wallets - raw_wallets
    missing_frac = len(missing_wallets) / max(len(feature_wallets), 1)

    report.metadata["raw_wallet_count"] = len(raw_wallets)
    report.metadata["feature_wallet_count"] = len(feature_wallets)
    report.metadata["missing_from_raw"] = len(missing_wallets)

    severity = "error" if missing_frac > tolerance else "warning"
    for wallet in sorted(missing_wallets):
        report.errors.append(
            ReconciliationError(
                check="trade_counts",
                entity=wallet,
                expected="present in raw trades",
                observed="absent from raw trades",
                severity=severity,
            )
        )

    # Per-wallet trade-count check
    raw_counts = (
        pd.concat(
            [
                raw_trades[col].value_counts().rename("count")
                for col in (base_col, counter_col)
                if col in raw_trades.columns
            ]
        )
        .groupby(level=0)
        .sum()
    )

    for wallet in feature_wallets & raw_wallets:
        count = int(raw_counts.get(wallet, 0))
        if count < min_trades_threshold:
            report.errors.append(
                ReconciliationError(
                    check="trade_counts",
                    entity=wallet,
                    expected=f">= {min_trades_threshold} raw trades",
                    observed=count,
                    severity="warning",
                )
            )

    return report


# ---------------------------------------------------------------------------
# Check 2: feature-matrix reconciliation (required columns + value ranges)
# ---------------------------------------------------------------------------


def reconcile_features(
    feature_matrix: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    feature_ranges_path: str | Path | None = None,
    range_violation_severity: str = "warning",
) -> ReconciliationReport:
    """Assert that the feature matrix is well-formed and values are within expected ranges.

    Checks
    ------
    - **Required columns**: every column in *required_columns* is present.
    - **No all-NaN columns**: columns that are entirely NaN indicate a broken
      upstream computation.
    - **Value range bounds**: when *feature_ranges_path* is provided, numeric
      feature values are checked against the ``{min, max}`` bounds in
      ``data/feature_ranges.json``.

    Parameters
    ----------
    feature_matrix:
        Per-wallet feature DataFrame.
    required_columns:
        Column names that must be present in *feature_matrix*.
    feature_ranges_path:
        Optional path to ``data/feature_ranges.json``.  When provided, out-of-
        range values produce entries in the report.
    range_violation_severity:
        Severity for range-bound violations.  Defaults to ``"warning"`` because
        legitimate market conditions can push features outside historical ranges.

    Returns
    -------
    ReconciliationReport
    """
    report = ReconciliationReport(checks_run=["feature_columns", "feature_ranges"])

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------
    if required_columns:
        missing = [c for c in required_columns if c not in feature_matrix.columns]
        for col in missing:
            report.errors.append(
                ReconciliationError(
                    check="feature_columns",
                    entity=col,
                    expected="present in feature matrix",
                    observed="missing",
                    severity="error",
                )
            )

    # ------------------------------------------------------------------
    # All-NaN columns
    # ------------------------------------------------------------------
    for col in feature_matrix.columns:
        if feature_matrix[col].isna().all():
            report.errors.append(
                ReconciliationError(
                    check="feature_columns",
                    entity=col,
                    expected="at least one non-NaN value",
                    observed="all-NaN column",
                    severity="error",
                )
            )

    report.metadata["row_count"] = len(feature_matrix)
    report.metadata["column_count"] = len(feature_matrix.columns)

    # ------------------------------------------------------------------
    # Value range checks
    # ------------------------------------------------------------------
    if feature_ranges_path is not None:
        path = Path(feature_ranges_path)
        try:
            raw_ranges: dict = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.errors.append(
                ReconciliationError(
                    check="feature_ranges",
                    entity=str(path),
                    expected="readable JSON feature ranges file",
                    observed=str(exc),
                    severity="error",
                )
            )
            raw_ranges = {}

        for feature, bounds in raw_ranges.items():
            if feature not in feature_matrix.columns:
                continue
            col_data = feature_matrix[feature].dropna()
            if col_data.empty:
                continue

            lo = bounds.get("min")
            hi = bounds.get("max")

            if lo is not None:
                n_below = int((col_data < lo).sum())
                if n_below:
                    report.errors.append(
                        ReconciliationError(
                            check="feature_ranges",
                            entity=feature,
                            expected=f">= {lo}",
                            observed=f"{n_below} row(s) below minimum",
                            severity=range_violation_severity,
                        )
                    )
            if hi is not None:
                n_above = int((col_data > hi).sum())
                if n_above:
                    report.errors.append(
                        ReconciliationError(
                            check="feature_ranges",
                            entity=feature,
                            expected=f"<= {hi}",
                            observed=f"{n_above} row(s) above maximum",
                            severity=range_violation_severity,
                        )
                    )

    return report


# ---------------------------------------------------------------------------
# Check 3: wallet-score reconciliation (feature rows ↔ risk score records)
# ---------------------------------------------------------------------------


def reconcile_wallet_scores(
    feature_matrix: pd.DataFrame,
    scored_wallets: pd.DataFrame,
    *,
    wallet_column: str = "wallet_id",
    score_column: str = "score",
    score_range: tuple[float, float] = (0.0, 100.0),
    allow_unscored: bool = False,
) -> ReconciliationReport:
    """Assert that every wallet in the feature matrix has a valid risk score.

    Checks
    ------
    - **Missing scores**: wallets in *feature_matrix* absent from
      *scored_wallets*.
    - **Orphan scores**: wallets in *scored_wallets* absent from
      *feature_matrix* (soft warning — scores may predate the current feature
      run).
    - **Score bounds**: all scores in *scored_wallets* lie within *score_range*
      ``[0, 100]`` by default.
    - **NaN scores**: null/NaN score values indicate a failed scoring step.

    Parameters
    ----------
    feature_matrix:
        Per-wallet feature DataFrame with a *wallet_column* column.
    scored_wallets:
        Risk score DataFrame with *wallet_column* and *score_column* columns.
    wallet_column:
        Shared wallet identifier column name.
    score_column:
        Column in *scored_wallets* holding the numeric risk score.
    score_range:
        ``(min, max)`` valid score range.  Scores outside this range are
        flagged as hard errors.
    allow_unscored:
        When ``True``, feature wallets with no corresponding score emit
        ``"warning"`` instead of ``"error"``.

    Returns
    -------
    ReconciliationReport
    """
    report = ReconciliationReport(checks_run=["wallet_scores"])

    # ------------------------------------------------------------------
    # Column presence
    # ------------------------------------------------------------------
    for df_name, df, col in [
        ("feature_matrix", feature_matrix, wallet_column),
        ("scored_wallets", scored_wallets, wallet_column),
        ("scored_wallets", scored_wallets, score_column),
    ]:
        if col not in df.columns:
            report.errors.append(
                ReconciliationError(
                    check="wallet_scores",
                    entity=f"{df_name}.{col}",
                    expected="column present",
                    observed="missing",
                    severity="error",
                )
            )
    if not report.ok:
        return report

    feature_wallets: set[str] = set(feature_matrix[wallet_column].dropna().unique())
    score_wallets: set[str] = set(scored_wallets[wallet_column].dropna().unique())

    # ------------------------------------------------------------------
    # Missing scores
    # ------------------------------------------------------------------
    unscored = feature_wallets - score_wallets
    missing_severity = "warning" if allow_unscored else "error"
    for wallet in sorted(unscored):
        report.errors.append(
            ReconciliationError(
                check="wallet_scores",
                entity=wallet,
                expected="score record present",
                observed="no score record",
                severity=missing_severity,
            )
        )

    # ------------------------------------------------------------------
    # Orphan scores (warning only)
    # ------------------------------------------------------------------
    orphan = score_wallets - feature_wallets
    for wallet in sorted(orphan):
        report.errors.append(
            ReconciliationError(
                check="wallet_scores",
                entity=wallet,
                expected="feature row present",
                observed="no feature row (orphan score)",
                severity="warning",
            )
        )

    # ------------------------------------------------------------------
    # Score value integrity
    # ------------------------------------------------------------------
    lo, hi = score_range
    for _, row in scored_wallets.iterrows():
        wallet = str(row[wallet_column])
        score_val = row[score_column]

        if pd.isna(score_val):
            report.errors.append(
                ReconciliationError(
                    check="wallet_scores",
                    entity=wallet,
                    expected="non-null score",
                    observed="NaN",
                    severity="error",
                )
            )
            continue

        score_f = float(score_val)
        if not (lo <= score_f <= hi):
            report.errors.append(
                ReconciliationError(
                    check="wallet_scores",
                    entity=wallet,
                    expected=f"score in [{lo}, {hi}]",
                    observed=score_f,
                    severity="error",
                )
            )

    report.metadata["feature_wallet_count"] = len(feature_wallets)
    report.metadata["scored_wallet_count"] = len(score_wallets)
    report.metadata["unscored_count"] = len(unscored)
    report.metadata["orphan_score_count"] = len(orphan)

    return report


# ---------------------------------------------------------------------------
# Check 4: alert-delivery reconciliation (score >= threshold <-> ledger outcome)
# ---------------------------------------------------------------------------


def reconcile_alert_delivery(
    scored_wallets: pd.DataFrame,
    delivery_records: list[AlertDeliveryRecord],
    *,
    wallet_column: str = "wallet_id",
    pair_column: str = "asset_pair",
    score_column: str = "score",
    threshold: float = 70.0,
) -> ReconciliationReport:
    """Assert every wallet scored at/above *threshold* has an accounted-for
    alert outcome (Issue #670, required scope E and invariant 7).

    A missing outcome is a hard error — it means a score crossed the alert
    threshold but the dispatcher never recorded what happened to it (neither
    delivered, dead-lettered, nor suppressed by cooldown), which is exactly
    the "silently dropped alert" failure mode this Grand is required to make
    observable. A ``dead_lettered`` outcome is reported as *accounted for*,
    not as missing — the point of the ledger is to distinguish "we know this
    failed and why" from "we have no idea what happened to this".

    Parameters
    ----------
    scored_wallets:
        DataFrame with *wallet_column*, *pair_column*, *score_column*.
    delivery_records:
        Every :class:`~streaming.alert_ledger.AlertDeliveryRecord` for the
        run being reconciled (e.g. from ``AlertDeliveryLedger.all_records()``).
    threshold:
        Score at/above which a delivery outcome is required.

    Returns
    -------
    ReconciliationReport
    """
    report = ReconciliationReport(checks_run=["alert_delivery"])

    for col in (wallet_column, pair_column, score_column):
        if col not in scored_wallets.columns:
            report.errors.append(
                ReconciliationError(
                    check="alert_delivery",
                    entity=f"scored_wallets.{col}",
                    expected="column present",
                    observed="missing",
                    severity="error",
                )
            )
    if not report.ok:
        return report

    accounted: dict[tuple[str, str], list[AlertDeliveryRecord]] = {}
    for rec in delivery_records:
        accounted.setdefault((rec.wallet, rec.pair_id), []).append(rec)

    qualifying = scored_wallets[scored_wallets[score_column] >= threshold]

    delivered_count = 0
    dead_lettered_count = 0
    suppressed_count = 0
    missing_count = 0

    for _, row in qualifying.iterrows():
        wallet = str(row[wallet_column])
        pair_id = str(row[pair_column])
        outcomes = accounted.get((wallet, pair_id), [])

        if not outcomes:
            missing_count += 1
            report.errors.append(
                ReconciliationError(
                    check="alert_delivery",
                    entity=f"{wallet}@{pair_id}",
                    expected="a recorded delivery outcome "
                    "(delivered, dead_lettered, or suppressed_cooldown)",
                    observed="no outcome recorded — alert may have been silently dropped",
                    severity="error",
                )
            )
            continue

        for outcome in outcomes:
            if outcome.outcome == "delivered":
                delivered_count += 1
            elif outcome.outcome == "dead_lettered":
                dead_lettered_count += 1
                # Accounted for, not an error — but surfaced as a warning so
                # operators can see dead-lettered alerts without them being
                # conflated with silently-missing ones.
                report.errors.append(
                    ReconciliationError(
                        check="alert_delivery",
                        entity=f"{wallet}@{pair_id}",
                        expected="delivered",
                        observed=f"dead_lettered ({outcome.reason or 'no reason recorded'})",
                        severity="warning",
                    )
                )
            elif outcome.outcome == "suppressed_cooldown":
                suppressed_count += 1

    report.metadata["threshold"] = threshold
    report.metadata["qualifying_count"] = len(qualifying)
    report.metadata["delivered_count"] = delivered_count
    report.metadata["dead_lettered_count"] = dead_lettered_count
    report.metadata["suppressed_cooldown_count"] = suppressed_count
    report.metadata["missing_count"] = missing_count

    return report
