# Issue #732 — Guard Against Negative `--lookback-days` in `retrain_if_drifted.py`

## Summary

`scripts/retrain_if_drifted.py` accepts a `--lookback-days` integer argument
that controls how far back in time the script reaches when building the
"current feature distribution" used for drift detection:

```python
since = datetime.now(UTC) - timedelta(days=lookback_days)
trades_df = load_watched_pairs_to_dataframe(start_time=since)
```

A value of `0` produces `since = now`, which requests zero days of data and
returns an empty trade set. A negative value (e.g. `--lookback-days -5`)
produces a `since` timestamp in the **future**, which the Horizon API will
return no data for. In both cases the feature matrix is empty and the drift
monitor reports no drift — a **false negative** that silently suppresses a
retraining run that may have been needed.

---

## Current Behaviour

### `parse_args` (lines 249–262 in `retrain_if_drifted.py`)

```python
parser.add_argument(
    "--lookback-days",
    type=int,
    default=30,
    help="Number of days to look back for current feature distribution (default: 30)",
)
```

`type=int` converts the string to an integer but performs no range check.
Values of `0`, `-1`, `-365`, or any other non-positive integer are accepted
silently.

### Downstream consequence in `get_feature_data`

```python
def get_feature_data(lookback_days: int) -> pd.DataFrame:
    since = datetime.now(UTC) - timedelta(days=lookback_days)
    trades_df = load_watched_pairs_to_dataframe(start_time=since)
    if trades_df.empty:
        logger.warning("No trades loaded; returning empty feature matrix")
        return pd.DataFrame()
    ...
```

When `lookback_days <= 0`, `since >= now`, the Horizon API returns no trades,
`trades_df` is empty, and the function returns an empty `DataFrame` with only a
warning log. Control then flows to:

```python
if current_data.empty:
    logger.warning("Current feature matrix is empty — cannot compute drift")
    return 0   # exit code 0 = no drift
```

The process exits with code `0`, identical to a legitimate "no drift detected"
result. There is nothing in the output to distinguish a misconfiguration from a
genuine clean-bill-of-health.

---

## Root Cause

`argparse` `type=int` validates only that the value is parseable as an integer.
There is no `choices=` constraint or custom validator function applied to
`--lookback-days`. The `get_feature_data` function does not pre-check
`lookback_days` before computing the date range.

---

## Risk

| Value | `since` computed as | Horizon API returns | Drift result | Real impact |
|---|---|---|---|---|
| `30` (default) | 30 days ago | Normal trade history | Accurate | — |
| `1` | Yesterday | Minimal data | May miss slow drift | Low |
| `0` | Now (exact) | Empty | **False no-drift** | Skipped retrain |
| `-1` | 1 day in the future | Empty | **False no-drift** | Skipped retrain |
| `-365` | 1 year in the future | Empty | **False no-drift** | Skipped retrain |

A **false negative** here has a direct operational consequence: a model that
has drifted is not retrained, continues scoring wallets with stale features,
and may produce systematically wrong risk scores until the next successful
drift-detection run. In an automated CI/CD pipeline where the script is
scheduled, a misconfigured cron argument could suppress retraining indefinitely.

---

## Acceptance Criteria

1. `--lookback-days` uses an argparse `type` validator (or `action`) that
   rejects values `<= 0` at argument-parse time with a clear message, before
   any Horizon API call or file I/O, such as:

   ```
   error: argument --lookback-days: must be a positive integer (> 0), got 0
   ```

2. The script exits with a **non-zero** code on invalid input (argparse
   default behaviour when a `type` validator raises `ArgumentTypeError`).

3. A test in `tests/test_retrain_trigger.py` covers:
   - `--lookback-days 0` → non-zero exit / `ArgumentTypeError`
   - `--lookback-days -1` → non-zero exit / `ArgumentTypeError`
   - `--lookback-days 1` → accepted (boundary value)
   - `--lookback-days 30` → accepted (default)

4. The upper bound is documented (see §Upper Bound Consideration below) even if
   not enforced programmatically.

---

## Proposed Implementation

### Custom argparse `type` validator

```python
def _positive_int(value: str) -> int:
    """Argparse type: reject non-positive integers for --lookback-days."""
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"must be an integer, got {value!r}"
        )
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"must be a positive integer (> 0), got {ivalue}"
        )
    return ivalue
```

Applied in `parse_args`:

```python
parser.add_argument(
    "--lookback-days",
    type=_positive_int,
    default=30,
    help=(
        "Number of days to look back for current feature distribution "
        "(must be > 0, default: 30, practical max: ~365 — see docs/issues/"
        "issue-732-lookback-days-guard.md for upper bound rationale)"
    ),
)
```

This is consistent with the approach recommended in issue #733 for
`--annotator-id`: prefer argparse `type=` validators over post-parse checks so
failures fire before any side-effecting code runs.

### Optional: secondary guard in `get_feature_data`

For defence-in-depth and to protect programmatic callers that bypass argparse:

```python
def get_feature_data(lookback_days: int) -> pd.DataFrame:
    if lookback_days <= 0:
        raise ValueError(
            f"lookback_days must be a positive integer, got {lookback_days}. "
            "A zero or negative value produces a future start date and an "
            "empty feature matrix, causing a false no-drift result."
        )
    since = datetime.now(UTC) - timedelta(days=lookback_days)
    ...
```

---

## Upper Bound Consideration

The issue asks that any sane upper bound be documented. The following factors
constrain how large `--lookback-days` can usefully be:

### Horizon API rate limits

Horizon (Stellar's public API) enforces request-per-second limits and paginates
trade history. Very long lookback windows require many paginated requests and
risk hitting rate limits, producing truncated data that looks like low-volume
trade history and potentially **understating** recent wash-trade activity.

### Practical data volume

| Lookback | Approx. trades for a watched pair set | Notes |
|---|---|---|
| ≤ 7 days | < 50 K | Fast; fine for incremental drift checks |
| 30 days (default) | ~200 K | Recommended for full drift detection |
| 90 days | ~600 K | Acceptable; increased latency |
| 180 days | ~1.2 M | Approaching practical limit |
| > 365 days | > 2.4 M | Not recommended; exceeds typical Horizon retention window |

### Recommended soft cap: 365 days

A lookback beyond 365 days is unlikely to improve drift detection quality
(models are retrained on a rolling basis) and may exceed Horizon's effective
data retention. A practical soft cap of `365` is recommended. It is **not
enforced** programmatically in this issue to avoid breaking existing long-range
backfill workflows, but the help text should document it.

If the team decides to enforce it in future, the `_positive_int` validator can
be extended:

```python
_MAX_LOOKBACK_DAYS = 365

def _bounded_positive_int(value: str) -> int:
    ivalue = _positive_int(value)   # reuses existing > 0 check
    if ivalue > _MAX_LOOKBACK_DAYS:
        raise argparse.ArgumentTypeError(
            f"--lookback-days exceeds recommended maximum of {_MAX_LOOKBACK_DAYS} "
            f"(got {ivalue}). See docs/issues/issue-732-lookback-days-guard.md."
        )
    return ivalue
```

---

## Suggested Tests

```python
# tests/test_retrain_trigger.py

import pytest
from scripts.retrain_if_drifted import parse_args

class TestLookbackDaysValidation:

    def test_zero_lookback_days_is_rejected(self):
        """Issue #732 — --lookback-days 0 must be rejected at parse time."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--lookback-days", "0"])
        assert exc_info.value.code != 0

    def test_negative_lookback_days_is_rejected(self):
        """Issue #732 — negative --lookback-days must be rejected at parse time."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--lookback-days", "-1"])
        assert exc_info.value.code != 0

    def test_large_negative_lookback_days_is_rejected(self):
        """Issue #732 — large negative --lookback-days must be rejected."""
        with pytest.raises(SystemExit) as exc_info:
            parse_args(["--lookback-days", "-365"])
        assert exc_info.value.code != 0

    def test_positive_lookback_days_is_accepted(self):
        """Issue #732 — boundary value 1 must be accepted."""
        args = parse_args(["--lookback-days", "1"])
        assert args.lookback_days == 1

    def test_default_lookback_days_is_accepted(self):
        """Issue #732 — default value 30 must be accepted."""
        args = parse_args([])
        assert args.lookback_days == 30

    def test_lookback_days_365_is_accepted(self):
        """Issue #732 — 365 day lookback is within the documented soft cap."""
        args = parse_args(["--lookback-days", "365"])
        assert args.lookback_days == 365
```

---

## False Negative vs False Positive Trade-off

This issue is about eliminating a class of **false negatives** (missed drift),
not false positives (spurious retrains). The existing promotion gate
(`should_promote`) already protects against spurious retraining degrading the
production model: even if drift is incorrectly detected, a retrained model must
pass the AUC-ROC and F1 gate before promotion. Therefore tightening the
`--lookback-days` input validation carries no risk of increasing false
promotions; it only ensures that the drift-detection step is operating on a
genuine data window.

---

## Affected Files

| File | Change type |
|---|---|
| `scripts/retrain_if_drifted.py` | Add `_positive_int` argparse `type` validator for `--lookback-days`; optionally add guard in `get_feature_data` |
| `tests/test_retrain_trigger.py` | Add `TestLookbackDaysValidation` covering 0, negative, boundary (1), default (30), and 365 |

---

## Related

- Issue #733 — `--annotator-id` validation (same argparse `type=` validator pattern)
- `scripts/retrain_if_drifted.py` — full script including shadow deployment, incremental training, and promotion gate
- `detection/drift_monitor.py` — `DriftMonitor.compute` (consumes the feature matrix built by `get_feature_data`)
- `docs/drift_detection.md` — drift detection architecture
- `docs/model_artifact_lifecycle.md` — retraining and promotion lifecycle
- Horizon API docs — https://developers.stellar.org/api/horizon

