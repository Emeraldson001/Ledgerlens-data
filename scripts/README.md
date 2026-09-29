# scripts/

## `sandbox.py` — Sandboxed execution checks

`sandboxed_execution()` wraps a maintenance script's entrypoint with CPU/memory
resource limits and optional network blocking, raising `SandboxViolation` with
an actionable message instead of letting the process die opaquely.
`dry_run_guard()` centralizes `--dry-run` logging for destructive actions
(see `migrate_add_ring_id.py` for a reference integration).

```python
from scripts.sandbox import sandboxed_execution, dry_run_guard

with sandboxed_execution(allow_network=False):
    dry_run_guard(args.dry_run, "ALTER TABLE ...", lambda: migrate(engine))
```

---

## `stream.py` — Real-time streaming pipeline

Streams trades from the Stellar Horizon SSE API, maintains a rolling feature
buffer per wallet, and dispatches risk alerts within one ledger close (~5 s)
of a wallet crossing the risk threshold.

### Usage

```bash
# Alert to stdout (local dev default)
python -m scripts.stream

# Webhook delivery
ALERT_WEBHOOK_URL=https://hooks.example.com/alert \
python -m scripts.stream --alert-channel webhook

# WebSocket broadcast (starts ws server on 127.0.0.1:8765)
python -m scripts.stream --alert-channel websocket

# Skip WebSocket server but still use websocket channel via custom ws_client
python -m scripts.stream --alert-channel websocket --no-ws

# Custom dedup window and warmup threshold
python -m scripts.stream --cooldown-seconds 1800 --min-trades 50
```

| Flag | Default | Description |
|---|---|---|
| `--alert-channel` | `stdout` | Alert delivery: `stdout`, `webhook`, or `websocket` |
| `--cooldown-seconds` | `3600` | Per-wallet alert dedup window (seconds) |
| `--min-trades` | `20` | Minimum buffered trades before a wallet is scored |
| `--no-ws` | off | Disable the WebSocket broadcast server |

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `WATCHED_ASSET_PAIRS` | — | **Required** — comma-separated `CODE:ISSUER` pairs |
| `ALERT_CHANNEL` | `stdout` | Overrides `--alert-channel` flag |
| `ALERT_WEBHOOK_URL` | — | HTTPS webhook endpoint (required for webhook channel) |
| `ALERT_COOLDOWN_SECONDS` | `3600` | Overrides `--cooldown-seconds` flag |
| `WS_PORT` | `8765` | WebSocket server port |
| `WS_BIND_HOST` | `127.0.0.1` | WebSocket bind address |
| `WS_ALLOW_EXTERNAL` | — | Set to `1` to bind to `0.0.0.0` |

### Stdout alert format

```
[ALERT] wallet=G… pair=USDC:…/XLM:native score=83 benford=True ml=True confidence=76
```

See [docs/streaming_architecture.md](../docs/streaming_architecture.md) for the
full pipeline diagram and threading model.

---

## `generate_synthetic_dataset.py`

Generates a synthetic labelled feature matrix for local training, demos,
and tests, without needing live Stellar Horizon data.

The output schema matches `detection/feature_engineering.py::build_feature_matrix`
(`wallet` + all Benford / trade-pattern / volume-timing / wallet-graph
feature columns), plus a `label` column (`1` = wash-trading-like, `0` =
legitimate). Roughly half the rows are generated with "legitimate"
distributions and half with "wash-trading-like" distributions, then
shuffled.

### Usage

```bash
python -m scripts.generate_synthetic_dataset \
    --n-wallets 500 \
    --seed 42 \
    --output data/synthetic_dataset.parquet

# Use a specific attacker profile
python -m scripts.generate_synthetic_dataset \
    --profile RingAttacker \
    --n-wallets 20 \
    --output data/ring_dataset.parquet

# Run the full adversarial training loop
python -m scripts.generate_synthetic_dataset \
    --profile AdaptiveAttacker \
    --gan-rounds 5
```

| Flag | Default | Description |
|---|---|---|
| `--n-wallets` | `500` | Number of synthetic wallet rows to generate |
| `--seed` | `42` | Random seed (controls both data generation and the final shuffle) |
| `--output` | `data/synthetic_dataset.parquet` | Output parquet path |
| `--profile` | `NaiveAttacker` | Attacker profile: `NaiveAttacker`, `TimingJitterAttacker`, `AmountConformanceAttacker`, `RingAttacker`, `LayeringAttacker`, `CrossPairAttacker`, `AdaptiveAttacker` |
| `--gan-rounds` | `0` | Run N rounds of adversarial training (0 = skip). Requires `--profile AdaptiveAttacker` |
| `--model-path` | — | Path to trained model `.joblib` file for `AdaptiveAttacker` |

---

## `wash_trade_simulator.py`

The Wash Trade Simulation Engine (WTSE) implements 7 attacker strategy
profiles for generating realistic trade-level data.

### Profiles

| Profile | Description |
|---|---|
| `NaiveAttacker` | Fixed amounts, regular intervals — baseline |
| `TimingJitterAttacker` | Poisson-distributed trade intervals |
| `AmountConformanceAttacker` | Benford-conforming amounts via log-uniform sampling |
| `RingAttacker` | N-wallet ring where each wallet trades with neighbours |
| `LayeringAttacker` | Interleaves wash trades with noise trades (3:1 ratio) |
| `CrossPairAttacker` | Rotates wash volume across K asset pairs |
| `AdaptiveAttacker` | Reads model feature importances and down-weights top features |

### Programmatic usage

```python
from scripts.wash_trade_simulator import NaiveAttacker, trades_to_feature_matrix

profile = NaiveAttacker(n_wallets=10, trades_per_wallet=50)
trades = profile.generate_trades()
features = trades_to_feature_matrix(trades)
```

---

## `adversarial_training_loop.py`

Runs a GAN-style adversarial training loop: Round 0 uses `NaiveAttacker`,
subsequent rounds use `AdaptiveAttacker` (which reads the previous round's
model feature importances). Per-round metrics are written to
`reports/adversarial_loop_{timestamp}.json`.

### Usage

```bash
python -m scripts.adversarial_training_loop \
    --gan-rounds 5 \
    --n-wallets 50
```

| Flag | Default | Description |
|---|---|---|
| `--gan-rounds` | `5` | Number of adversarial rounds |
| `--n-wallets` | `50` | Wallets per generated dataset |
| `--trades-per-wallet` | `100` | Trades per wallet |
| `--output-dir` | `reports` | Directory for output JSON |
| `--seed` | `42` | Random seed |

---

## `evaluate_simulator_realism.py`

Computes realism metrics for the simulator: Fréchet Feature Distance (FFD)
and discriminator accuracy between simulated and real labelled data.

### Usage

```bash
python -m scripts.evaluate_simulator_realism \
    --simulated data/synthetic_dataset.parquet \
    --real data/labelled_dataset.parquet
```

| Flag | Default | Description |
|---|---|---|
| `--simulated` | `data/synthetic_dataset.parquet` | Path to simulated feature matrix |
| `--real` | `data/labelled_dataset.parquet` | Path to real labelled dataset |
| `--output-dir` | `reports` | Directory for output JSON |
| `--seed` | `42` | Random seed |

---

### Training on the generated dataset

```bash
python -m detection.model_training --data-path data/synthetic_dataset.parquet
```

This trains every model in `MODEL_REGISTRY` (Random Forest, XGBoost,
LightGBM) with SMOTE-balanced training data, writes the fitted models to
`config.MODEL_DIR`, and writes `metrics.json` (AUC-ROC / PR-AUC / F1 per
model) alongside them.

## `run_adversarial_eval.py`

Generates an adversarial-robustness report for a trained ensemble. It runs
FGSM and PGD evasion attacks (`detection/adversarial/`) against the
high-scoring wash wallets in a labelled feature matrix and writes a JSON
report covering:

- PGD / FGSM evasion success rate (fraction of `80+` wash wallets pushed
  below the alert threshold within the L-inf budget),
- per-feature minimum epsilon and the most vulnerable features, and
- the AUC-ROC gain from adversarial-augmentation retraining.

### Usage

```bash
python -m scripts.run_adversarial_eval \
    --data-path data/synthetic_dataset.parquet \
    --model-dir ./models \
    --output reports/adversarial_robustness.json
```

| Flag | Default | Description |
|---|---|---|
| `--data-path` | *(required)* | Labelled feature matrix (parquet) with a `label` column |
| `--model-dir` | `MODEL_DIR` | Directory of trained model artifacts |
| `--output` | `reports/adversarial_robustness.json` | Output JSON report path |
| `--epsilon` | `3.0` | L-inf perturbation budget (per-feature std units) |
| `--steps` | `40` | PGD iterations |
| `--target-score` | `40` | Evasion succeeds when the score drops below this |
| `--high-score` | `80` | Minimum score for a wallet to enter the attacked cohort |
| `--skip-augmentation` | off | Skip the slower adversarial-augmentation retraining comparison |

Requires trained models (run `model_training.py` first).

---

## `trace_feature.py`

Trace which Horizon trade IDs contributed to a specific wallet feature score.
This is useful when debugging a suspicious score or verifying that the feature
value came from the expected trade window.

### Usage

```bash
# Seed a synthetic risk-score record in a local SQLite database.
RISK_SCORE_DB_URL=sqlite:////tmp/ledgerlens-risk.db \
python - <<'PY'
from detection.persistence import Base, RiskScoreRecord, get_engine
from sqlalchemy.orm import Session

engine = get_engine()
Base.metadata.create_all(engine)
with Session(engine) as session:
    session.add(
        RiskScoreRecord(
            wallet="GCRN5Q6QK6SP3U2PB5UD2JJJ5K3T2EC2KSN3FV7RO4DVK2G4MYA6V3E",
            asset_pair="XLM/native",
            score=82,
            benford_flag=True,
            ml_flag=False,
            confidence=79,
            provenance_json='{"benford_chi_square_24h": ["trade_001", "trade_002", "trade_003"]}',
        )
    )
    session.commit()
PY

# Trace one feature for that wallet.
RISK_SCORE_DB_URL=sqlite:////tmp/ledgerlens-risk.db \
python -m scripts.trace_feature \
    --wallet GCRN5Q6QK6SP3U2PB5UD2JJJ5K3T2EC2KSN3FV7RO4DVK2G4MYA6V3E \
    --feature benford_chi_square_24h \
    --asset-pair XLM/native
```

### Example output

```text
Feature:    benford_chi_square_24h
Wallet:     GCRN5Q6QK6SP3U2PB5UD2JJJ5K3T2EC2KSN3FV7RO4DVK2G4MYA6V3E
Asset pair: XLM/native
Trade IDs (3):
  trade_001  https://horizon.stellar.org/trades/trade_001
  trade_002  https://horizon.stellar.org/trades/trade_002
  trade_003  https://horizon.stellar.org/trades/trade_003
```

The output prints the feature name, wallet, asset pair, and each trade ID that
contributed to the score, plus a Horizon explorer link for each trade. If the
wallet is missing, the provenance JSON is empty, or the requested feature is not
tracked, the command exits non-zero with an explanatory message.
