"""Schema and range validators for model input feature vectors.

Issue #531 — Stellar Wave advanced build.

Provides a layered validation system that guards the ensemble ML models
against malformed, corrupted, or adversarially-crafted input rows **before**
they reach the scorer.  Validation operates at three levels:

1. **Schema validation** — checks that the expected feature columns are
   present and have numeric dtypes.
2. **Range validation** — checks per-feature [low, high] bounds loaded from
   ``data/feature_ranges.json`` (or overridden at construction time).
3. **Null / NaN handling** — detects NaN / infinite values and either
   rejects rows or imputes them according to the configured strategy.

The validator is deliberately decoupled from ``detection/model_training.py``'s
``validate_incremental_samples`` so it can be called at inference time without
importing the entire training stack.

Configuration:

.. code-block:: bash

    MODEL_INPUT_VALIDATOR_STRICTNESS=warn    # "raise" | "warn" | "coerce" | "ignore"
    MODEL_INPUT_VALIDATOR_RANGES_PATH=data/feature_ranges.json
    MODEL_INPUT_VALIDATOR_NAN_STRATEGY=drop  # "drop" | "impute_zero" | "impute_median" | "raise"

Usage (inference path)::

    from detection.model_input_validator import ModelInputValidator

    validator = ModelInputValidator(expected_columns=FEATURE_COLUMNS)
    X_clean, report = validator.validate(X_raw)

Usage (training path)::

    validator = ModelInputValidator.from_metadata(metadata_path="models/model_metadata.json")
    X_clean, report = validator.validate(X_train)

"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

_STRICTNESS: str = getattr(config, "MODEL_INPUT_VALIDATOR_STRICTNESS", "warn").lower()
_RANGES_PATH: str = getattr(
    config,
    "MODEL_INPUT_VALIDATOR_RANGES_PATH",
    "data/feature_ranges.json",
)
_NAN_STRATEGY: str = getattr(
    config,
    "MODEL_INPUT_VALIDATOR_NAN_STRATEGY",
    "drop",
).lower()

_VALID_STRICTNESS = frozenset({"raise", "warn", "coerce", "ignore"})
_VALID_NAN_STRATEGIES = frozenset({"drop", "impute_zero", "impute_median", "raise"})

# ---------------------------------------------------------------------------
# Built-in feature bounds (mirrored from model_training._FEATURE_BOUNDS and
# extended with full feature set).  These serve as the fallback when no
# feature_ranges.json is available.
# ---------------------------------------------------------------------------

_BUILTIN_BOUNDS: dict[str, tuple[float, float]] = {
    # Benford features — chi-square and MAD are non-negative
    "benford_chi_square_1h": (0.0, 1e6),
    "benford_chi_square_4h": (0.0, 1e6),
    "benford_chi_square_24h": (0.0, 1e6),
    "benford_chi_square_168h": (0.0, 1e6),
    "benford_chi_square_720h": (0.0, 1e6),
    "benford_mad_1h": (0.0, 1.0),
    "benford_mad_4h": (0.0, 1.0),
    "benford_mad_24h": (0.0, 1.0),
    "benford_mad_168h": (0.0, 1.0),
    "benford_mad_720h": (0.0, 1.0),
    # Z-score per digit — unbounded in principle but clamped for safety
    "benford_z_max_1h": (0.0, 1e4),
    "benford_z_max_4h": (0.0, 1e4),
    "benford_z_max_24h": (0.0, 1e4),
    "benford_z_max_168h": (0.0, 1e4),
    "benford_z_max_720h": (0.0, 1e4),
    # Ratio / proportion features [0, 1]
    "counterparty_concentration_ratio": (0.0, 1.0),
    "self_matching_rate": (0.0, 1.0),
    "order_cancellation_rate": (0.0, 1.0),
    "cross_pair_trade_synchrony": (0.0, 1.0),
    "cross_pair_counterparty_overlap": (0.0, 1.0),
    "pair_diversity_score": (0.0, 1.0),
    # Correlation — [-1, 1]
    "cross_pair_volume_correlation": (-1.0, 1.0),
    # Non-negative floats
    "net_asset_flow_deviation": (0.0, 1e9),
    "cross_pair_benford_mad_std": (0.0, 1.0),
    "round_trip_trade_frequency": (0.0, 1.0),
    "volume_to_unique_counterparty_ratio": (0.0, 1e9),
    "intra_minute_trade_clustering": (0.0, 1.0),
    "off_hours_activity_ratio": (0.0, 1.0),
    "volume_spike_frequency": (0.0, 1e6),
    "funding_source_similarity": (0.0, 1.0),
    "network_centrality": (0.0, 1.0),
    "account_age_days": (0.0, 36500.0),
    # Ring detection features
    "ring_size": (0.0, 1e6),
    "ring_internal_density": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Validation result types
# ---------------------------------------------------------------------------


@dataclass
class ColumnIssue:
    """Describes a schema problem with a single column."""

    column: str
    issue: str  # "missing" | "wrong_dtype" | "all_nan"

    def __str__(self) -> str:
        return f"ColumnIssue(column={self.column!r}, issue={self.issue!r})"


@dataclass
class RangeViolation:
    """Describes out-of-range values in a column."""

    column: str
    low: float
    high: float
    n_violations: int
    # Indices of violating rows (capped at 10 for brevity)
    sample_indices: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"RangeViolation(column={self.column!r}, "
            f"bounds=[{self.low}, {self.high}], "
            f"n_violations={self.n_violations})"
        )


@dataclass
class ValidationReport:
    """Summary of all issues found during validation.

    Attributes:
        input_rows: Total rows in the input DataFrame.
        output_rows: Rows retained after validation.
        rows_dropped: Rows removed due to NaN / range violations.
        column_issues: Schema / dtype problems detected.
        range_violations: Out-of-range value detections.
        nan_rows: Rows with at least one NaN in a required column.
        inf_rows: Rows with at least one ±Inf value.
        passed: True only when there are no column issues and no violations.
    """

    input_rows: int = 0
    output_rows: int = 0
    rows_dropped: int = 0
    column_issues: list[ColumnIssue] = field(default_factory=list)
    range_violations: list[RangeViolation] = field(default_factory=list)
    nan_rows: int = 0
    inf_rows: int = 0

    @property
    def passed(self) -> bool:
        """True only when no schema issues and no range violations were found."""
        return not self.column_issues and not self.range_violations

    @property
    def has_issues(self) -> bool:
        return bool(self.column_issues or self.range_violations or self.nan_rows or self.inf_rows)

    def summary(self) -> str:
        lines = [
            f"ValidationReport: input={self.input_rows} rows, "
            f"output={self.output_rows} rows, "
            f"dropped={self.rows_dropped}",
        ]
        if self.column_issues:
            lines.append(f"  Column issues ({len(self.column_issues)}):")
            for ci in self.column_issues:
                lines.append(f"    - {ci}")
        if self.range_violations:
            lines.append(f"  Range violations ({len(self.range_violations)}):")
            for rv in self.range_violations:
                lines.append(f"    - {rv}")
        if self.nan_rows:
            lines.append(f"  NaN rows: {self.nan_rows}")
        if self.inf_rows:
            lines.append(f"  Inf rows: {self.inf_rows}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_rows": self.input_rows,
            "output_rows": self.output_rows,
            "rows_dropped": self.rows_dropped,
            "nan_rows": self.nan_rows,
            "inf_rows": self.inf_rows,
            "column_issues": [
                {"column": ci.column, "issue": ci.issue} for ci in self.column_issues
            ],
            "range_violations": [
                {
                    "column": rv.column,
                    "low": rv.low,
                    "high": rv.high,
                    "n_violations": rv.n_violations,
                    "sample_indices": rv.sample_indices,
                }
                for rv in self.range_violations
            ],
            "passed": self.passed,
        }

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------


class ModelInputValidator:
    """Validates a feature DataFrame against a known schema and range bounds.

    Parameters
    ----------
    expected_columns:
        Ordered list of feature columns the model expects.  Only columns in
        this list are validated and returned; extra columns are silently
        dropped.
    bounds:
        Per-feature ``{column: (low, high)}`` bounds.  Columns absent from
        this mapping fall back to :data:`_BUILTIN_BOUNDS`, then to
        ``(-inf, +inf)`` (i.e., range check skipped).
    strictness:
        ``"raise"``  — raise ``ValueError`` on the first issue found.
        ``"warn"``   — log warnings, return cleaned data (default).
        ``"coerce"`` — silently drop/fix violations, no logging.
        ``"ignore"`` — skip all validation, pass data through unchanged.
    nan_strategy:
        ``"drop"``          — drop rows with NaN / Inf in required columns.
        ``"impute_zero"``   — replace NaN / Inf with 0.
        ``"impute_median"`` — replace NaN / Inf with column median.
        ``"raise"``         — raise ``ValueError`` on NaN / Inf.
    """

    def __init__(
        self,
        expected_columns: list[str] | None = None,
        *,
        bounds: dict[str, tuple[float, float]] | None = None,
        strictness: str | None = None,
        nan_strategy: str | None = None,
    ) -> None:
        self._expected_columns: list[str] | None = expected_columns
        self._bounds: dict[str, tuple[float, float]] = dict(_BUILTIN_BOUNDS)
        if bounds:
            self._bounds.update(bounds)

        self._strictness = (strictness or _STRICTNESS).lower()
        if self._strictness not in _VALID_STRICTNESS:
            raise ValueError(
                f"Unknown strictness {self._strictness!r}. "
                f"Choose from: {sorted(_VALID_STRICTNESS)}"
            )

        self._nan_strategy = (nan_strategy or _NAN_STRATEGY).lower()
        if self._nan_strategy not in _VALID_NAN_STRATEGIES:
            raise ValueError(
                f"Unknown nan_strategy {self._nan_strategy!r}. "
                f"Choose from: {sorted(_VALID_NAN_STRATEGIES)}"
            )

        # Load extra bounds from feature_ranges.json if it exists
        self._load_ranges_from_file()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_metadata(
        cls,
        metadata_path: str = "models/model_metadata.json",
        *,
        bounds_path: str | None = None,
        strictness: str | None = None,
        nan_strategy: str | None = None,
    ) -> ModelInputValidator:
        """Construct a validator from a ``model_metadata.json`` sidecar file.

        The ``feature_columns`` list in the metadata is used as the expected
        schema; existing ``data/feature_ranges.json`` provides range bounds.
        """
        path = Path(metadata_path)
        if not path.exists():
            logger.warning(
                "ModelInputValidator.from_metadata: %s not found, "
                "creating validator with no expected columns",
                metadata_path,
            )
            return cls(
                expected_columns=None,
                strictness=strictness,
                nan_strategy=nan_strategy,
            )

        with path.open() as f:
            meta = json.load(f)

        feature_columns: list[str] | None = meta.get("feature_columns")
        validator = cls(
            expected_columns=feature_columns,
            strictness=strictness,
            nan_strategy=nan_strategy,
        )
        # Optionally load a custom bounds file on top of the defaults
        if bounds_path is not None:
            validator._load_ranges_from_file(bounds_path)
        return validator

    def _load_ranges_from_file(self, path: str | None = None) -> None:
        """Merge bounds from ``feature_ranges.json`` into self._bounds."""
        ranges_path = Path(path or _RANGES_PATH)
        if not ranges_path.exists():
            return
        try:
            with ranges_path.open() as f:
                ranges_data: dict[str, dict] = json.load(f)
            loaded = 0
            for col, spec in ranges_data.items():
                low = float(spec.get("min", -math.inf))
                high = float(spec.get("max", math.inf))
                if col not in self._bounds:
                    self._bounds[col] = (low, high)
                    loaded += 1
            if loaded:
                logger.debug(
                    "ModelInputValidator: loaded %d additional bounds from %s",
                    loaded,
                    ranges_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ModelInputValidator: could not load feature ranges from %s: %s",
                ranges_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Core validation
    # ------------------------------------------------------------------

    def validate(
        self,
        X: pd.DataFrame,
        *,
        expected_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, ValidationReport]:
        """Validate *X* and return a cleaned copy plus a :class:`ValidationReport`.

        Steps:
        1. If ``expected_columns`` is set (instance or argument), check schema.
        2. Select only expected columns (drops extras).
        3. Detect and handle NaN / Inf values.
        4. Detect and handle out-of-range values.
        5. Return the cleaned DataFrame and the report.

        Parameters
        ----------
        X:
            Input feature DataFrame, one row per wallet.
        expected_columns:
            Override the instance-level expected columns for this call.

        Returns
        -------
        (X_clean, report)
            ``X_clean`` is a validated (possibly filtered) copy of *X*.
            ``report`` describes all issues found.

        Raises
        ------
        ValueError:
            When ``strictness="raise"`` and any issue is detected.
        """
        if self._strictness == "ignore":
            report = ValidationReport(
                input_rows=len(X),
                output_rows=len(X),
            )
            return X.copy(), report

        report = ValidationReport(input_rows=len(X))
        eff_cols = expected_columns or self._expected_columns

        # --- 1. Schema validation ---
        if eff_cols is not None:
            schema_issues = self._check_schema(X, eff_cols)
            report.column_issues.extend(schema_issues)

            if schema_issues and self._strictness == "raise":
                raise ValueError(
                    "Model input schema errors:\n" + "\n".join(str(ci) for ci in schema_issues)
                )
            if schema_issues and self._strictness == "warn":
                for ci in schema_issues:
                    logger.warning("ModelInputValidator schema issue: %s", ci)

        # --- 2. Select expected columns ---
        if eff_cols is not None:
            available = [c for c in eff_cols if c in X.columns]
            X_work = X[available].copy()
        else:
            # Validate all numeric columns
            X_work = X.select_dtypes(include="number").copy()

        # --- 3. NaN / Inf handling ---
        X_work, nan_dropped, nan_rows, inf_rows = self._handle_nans(X_work)
        report.nan_rows = nan_rows
        report.inf_rows = inf_rows
        report.rows_dropped += nan_dropped

        # --- 4. Range validation ---
        X_work, range_violations, range_dropped = self._check_ranges(X_work)
        report.range_violations.extend(range_violations)
        report.rows_dropped += range_dropped

        if range_violations:
            if self._strictness == "raise":
                raise ValueError(
                    "Model input range violations:\n"
                    + "\n".join(str(rv) for rv in range_violations)
                )
            elif self._strictness == "warn":
                for rv in range_violations:
                    logger.warning("ModelInputValidator range violation: %s", rv)

        report.output_rows = len(X_work)
        return X_work, report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_schema(self, X: pd.DataFrame, expected_columns: list[str]) -> list[ColumnIssue]:
        """Check that all expected columns are present and numeric."""
        issues: list[ColumnIssue] = []
        for col in expected_columns:
            if col not in X.columns:
                issues.append(ColumnIssue(column=col, issue="missing"))
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                issues.append(ColumnIssue(column=col, issue="wrong_dtype"))
            elif X[col].isna().all():
                issues.append(ColumnIssue(column=col, issue="all_nan"))
        return issues

    def _handle_nans(self, X: pd.DataFrame) -> tuple[pd.DataFrame, int, int, int]:
        """Handle NaN and ±Inf values according to the configured strategy.

        Returns
        -------
        (X_clean, rows_dropped, nan_count, inf_count)
        """
        # Replace ±inf with NaN first so all handling is uniform
        inf_mask = X.isin([float("inf"), float("-inf")]).any(axis=1)
        inf_count = int(inf_mask.sum())
        X = X.replace([float("inf"), float("-inf")], float("nan"))

        nan_mask = X.isnull().any(axis=1)
        nan_count = int(nan_mask.sum())
        rows_dropped = 0

        if not (inf_count or nan_count):
            return X, 0, 0, 0

        if self._nan_strategy == "raise":
            raise ValueError(
                f"Model input contains {nan_count} NaN / {inf_count} Inf value(s) "
                "and nan_strategy='raise'"
            )
        elif self._nan_strategy == "drop":
            before = len(X)
            X = X[~X.isnull().any(axis=1)]
            rows_dropped = before - len(X)
        elif self._nan_strategy == "impute_zero":
            X = X.fillna(0.0)
        elif self._nan_strategy == "impute_median":
            X = X.fillna(X.median(numeric_only=True))

        return X, rows_dropped, nan_count, inf_count

    def _check_ranges(self, X: pd.DataFrame) -> tuple[pd.DataFrame, list[RangeViolation], int]:
        """Validate per-feature ranges.  Returns the cleaned DF, violations, and
        number of rows dropped (``strictness='coerce'`` path drops violating rows).
        """
        violations: list[RangeViolation] = []
        all_oor_mask = pd.Series(False, index=X.index)

        for col in X.columns:
            if col not in self._bounds:
                continue
            low, high = self._bounds[col]
            col_oor = (X[col] < low) | (X[col] > high)
            n_oor = int(col_oor.sum())
            if n_oor == 0:
                continue

            sample_idx = list(X.index[col_oor][:10])
            violations.append(
                RangeViolation(
                    column=col,
                    low=low,
                    high=high,
                    n_violations=n_oor,
                    sample_indices=sample_idx,
                )
            )
            all_oor_mask |= col_oor

        rows_dropped = 0
        if violations and self._strictness == "coerce":
            before = len(X)
            X = X[~all_oor_mask].copy()
            rows_dropped = before - len(X)

        return X, violations, rows_dropped

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def add_bounds(self, bounds: dict[str, tuple[float, float]]) -> None:
        """Merge additional feature bounds into the validator at runtime.

        Existing entries are overwritten so callers can tighten or relax
        individual bounds without reconstructing the validator.
        """
        self._bounds.update(bounds)

    @property
    def expected_columns(self) -> list[str] | None:
        """The expected feature columns, or ``None`` if not configured."""
        return self._expected_columns

    @property
    def bounds(self) -> dict[str, tuple[float, float]]:
        """A copy of the current per-feature bounds mapping."""
        return dict(self._bounds)

    @property
    def strictness(self) -> str:
        """The configured strictness level."""
        return self._strictness

    @property
    def nan_strategy(self) -> str:
        """The configured NaN handling strategy."""
        return self._nan_strategy

    def __repr__(self) -> str:
        n_cols = len(self._expected_columns) if self._expected_columns else "?"
        return (
            f"ModelInputValidator("
            f"expected_columns={n_cols}, "
            f"strictness={self._strictness!r}, "
            f"nan_strategy={self._nan_strategy!r})"
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def validate_features(
    X: pd.DataFrame,
    expected_columns: list[str] | None = None,
    *,
    strictness: str = "warn",
    nan_strategy: str = "drop",
) -> tuple[pd.DataFrame, ValidationReport]:
    """One-shot feature validation helper.

    Creates a temporary :class:`ModelInputValidator`, validates *X*, and
    returns the cleaned DataFrame plus the report.

    Example::

        X_clean, report = validate_features(X_raw, FEATURE_COLUMNS)
        if not report.passed:
            logger.warning(report.summary())
    """
    validator = ModelInputValidator(
        expected_columns=expected_columns,
        strictness=strictness,
        nan_strategy=nan_strategy,
    )
    return validator.validate(X)
