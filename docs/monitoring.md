# LedgerLens Monitoring

## CUSUM Change-Point Detection (`monitoring/cusum_detector.py`)

### Theory

The **CUSUM (Cumulative Sum) control chart** (Page, 1954) detects sustained
shifts in the mean of a metric stream in O(1) time and O(1) space per update.

LedgerLens uses a **two-sided CUSUM** applied to the stream of risk scores
produced by the pipeline:

```
S_high[n] = max(0, S_high[n-1] + x_n - (μ₀ + k))
S_low[n]  = max(0, S_low[n-1]  - x_n + (μ₀ - k))
```

An **alarm** fires when either statistic exceeds the decision threshold **h**.
After acknowledgement both statistics are reset to zero.

| Parameter | Config variable | Default | Meaning |
|---|---|---|---|
| μ₀ (target mean) | `CUSUM_TARGET_MEAN` | `30.0` | Expected in-control mean risk score |
| k (allowable slack) | `CUSUM_ALLOWABLE_SLACK` | `5.0` | Half the minimum detectable shift |
| h (decision threshold) | `CUSUM_DECISION_THRESHOLD` | `25.0` | Alarm trigger level |

### Parameter selection guidance

For a risk score stream with **σ ≈ 15** and a target shift of **10 points**:

- **k = δ/2 = 5.0** — allowable slack set to half the minimum shift of interest
- **h = 25.0** — achieves:
  - In-control ARL (average run length before false alarm) ≈ **500 observations**
  - Out-of-control ARL for a 10-point shift ≈ **10 observations**

Use the formula `h ≈ (σ²/δ²) · 2 · ln(ARL_out / ARL_in)` for custom tuning,
or consult Montgomery (2009) Table 9-5.

Both `CUSUM_ALLOWABLE_SLACK` and `CUSUM_DECISION_THRESHOLD` must be
non-negative; `CUSUM_DECISION_THRESHOLD` must be strictly positive. A
`ConfigurationError` is raised on startup if these constraints are violated.

### Prometheus metric

`ledgerlens_cusum_alarm{metric="risk_score"}` — Gauge: **1** when alarming, **0** in-control.

The metric is emitted by `monitoring/cusum_detector.py` via `prometheus_client`.

### Alarm state persistence

Alarm state is persisted to Redis under the key
`ledgerlens:cusum:{metric_name}:alarm` so it survives worker restarts.
When Redis is unavailable, state is in-memory only (lost on restart).

### Alarm response procedure

1. **Triage**: Check whether the alarm is driven by a genuine new wave of wash
   trading (risk score distribution shift upward) or model degradation (scores
   drift upward due to feature drift — cross-reference with PSI drift monitor).
2. **Investigate**: Query the top-scoring wallets from the past 1h using the
   `/alerts/recent` API endpoint.
3. **Resolve**: After investigation, call `CUSUMDetector.acknowledge()` (or
   the corresponding API endpoint) to reset the statistics.
4. **Model action**: If the alarm is caused by feature drift, trigger a
   retraining run via `scripts/retrain_if_drifted.py`.

## End-to-End Latency Budgets

### Theory

LedgerLens operates with strict latency budgets from data ingestion (Horizon) to consumer decision (scoring and alerting). Exceeding this budget undermines the relevance of risk scores in fast-moving markets.

### Configuration

Latency budgets are governed by:
- `E2E_LATENCY_BUDGET_MS` (default `2000`): The target latency limit from Horizon ingestion through the scoring worker.
- `LATENCY_ANOMALY_RATE_THRESHOLD` (default `0.90`): The proportion of events in the rolling 60-second window that must exceed the budget to trigger an emergency pause.

### Prometheus Metrics

- `ledgerlens_e2e_latency_seconds` — Histogram tracking end-to-end latency for successfully scored events.

### Emergency Watchdog

The `EmergencyWatchdog` monitors the stream of e2e latency observations dynamically.
If the fraction of events exceeding `E2E_LATENCY_BUDGET_MS` surpasses `LATENCY_ANOMALY_RATE_THRESHOLD`, the watchdog proposes an on-chain emergency pause. 

### Latency Drill Procedures

An automated rehearsal drill is provided in `scripts/rehearsal_latency_pause.py`. Operators should routinely run this drill on Testnet to verify that partial failures (latency injection) properly trigger the watchdog and result in a valid on-chain proposal without disrupting state.
