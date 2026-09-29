# Wash-Trade Simulators and Realism Evaluation

LedgerLens ships three related tools for producing and grading synthetic
wash-trading data:

| Script | Role |
|---|---|
| [`scripts/wash_trade_simulator.py`](../scripts/wash_trade_simulator.py) | Library of hand-written attacker *strategy profiles* (Wash Trade Simulation Engine). |
| [`scripts/adversarial_wash_trade_simulator.py`](../scripts/adversarial_wash_trade_simulator.py) | Genetic algorithm that *evolves* a strategy to minimise its LedgerLens risk score. |
| [`scripts/evaluate_simulator_realism.py`](../scripts/evaluate_simulator_realism.py) | Measures how close simulated data is to real (Testnet) wash-trade data. |

The synthetic data these tools produce is used for local training, demos,
regression tests, and adversarial-robustness evaluation — it is **not** a
substitute for the labelled Testnet dataset built by
[`scripts/build_labelled_dataset.py`](../scripts/build_labelled_dataset.py).

---

## What the simulators generate

Both simulators emit a **trade DataFrame** whose schema matches
`ingestion.historical_loader.trades_to_dataframe`
(`trade_id`, `ledger_close_time`, `base_account`, `counter_account`,
`base_asset`, `counter_asset`, `amount`, `price`). That schema is what
`detection.feature_engineering.build_feature_matrix` consumes, so simulated
trades flow through the exact same feature pipeline as real Horizon data.

### Strategy profiles (`wash_trade_simulator.py`)

Each profile is a dataclass subclassing `BaseAttackerProfile` and implementing
`generate_trades()`. Instantiate them by name with `create_profile(name, **kwargs)`.

| Profile | What it models |
|---|---|
| `NaiveAttacker` | Fixed amount, perfectly regular intervals, one counterparty — the baseline wash trader. |
| `TimingJitterAttacker` | Same trades, but inter-arrival gaps drawn from an exponential (Poisson-process) distribution. |
| `AmountConformanceAttacker` | Amounts drawn log-uniformly so leading digits conform to Benford's Law, defeating the Benford signal. |
| `RingAttacker` | An N-wallet cycle where wallet `i` only trades with wallet `(i+1) % N` — high centrality, low funding-source similarity. |
| `LayeringAttacker` | Interleaves wash trades with legitimate-looking noise trades at a 3:1 ratio; output carries an `is_wash` column. |
| `CrossPairAttacker` | Rotates wash volume across K asset pairs to dilute the per-pair signal. |
| `AdaptiveAttacker` | Loads a trained model, reads its feature importances, and down-weights the top-K most discriminative features. |

### Adversarial simulator (`adversarial_wash_trade_simulator.py`)

`AdversarialWashTradeSimulator` runs a genetic algorithm over a strategy genome
(`n_trades`, `amount_mean`, `amount_std`, `inter_trade_seconds`,
`n_counterparties`, `jitter_fraction`, `use_round_numbers`). Fitness is
`1 / (risk_score + 1)`, so lower LedgerLens risk scores are selected for, subject
to an economic-plausibility constraint (total volume 1,000–10,000,000 XLM). If no
trained model is found under `--model-dir`, it falls back to a heuristic scorer.
The best (lowest) score reached is exported as the Prometheus gauge
`ledgerlens_adversarial_lowest_score`.

---

## What "realism" means

A simulator is *realistic* when a detector — or a statistician — **cannot tell
its output apart from genuine wash-trading activity** observed on Testnet. That
is deliberately a strong bar: it is not enough for the synthetic trades to "look
plausible", their per-wallet **feature distributions** must overlap those of real
wash-trade wallets. `evaluate_simulator_realism.py` operationalises this with two
complementary metrics computed on the feature matrices (not the raw trades).

### Metrics computed by `evaluate_simulator_realism.py`

1. **Fréchet Feature Distance (FFD)** — `compute_frechet_feature_distance()`.
   Treats the real and simulated feature sets as two multivariate Gaussians and
   measures the Fréchet (Wasserstein-2) distance between them:

   ```
   FFD = ||mu_real - mu_sim||^2
       + Tr(Sigma_real + Sigma_sim - 2 * sqrt(Sigma_real @ Sigma_sim))
   ```

   Lower is better. `0.0` means the first two moments of the two distributions
   are identical. It is returned as `null` when either side has fewer than two
   usable rows.

2. **Discriminator accuracy** — `compute_discriminator_accuracy()`.
   Balances the two classes, trains a held-out `RandomForestClassifier`
   (real = 0, simulated = 1), and reports:

   | Key | Meaning |
   |---|---|
   | `accuracy` | Hold-out (30%) accuracy of the real-vs-sim classifier. |
   | `auc_roc` | Hold-out ROC AUC. |
   | `cross_val_mean_accuracy` / `cross_val_std` | 5-fold cross-validated accuracy and its spread. |
   | `n_real_samples` / `n_sim_samples` | Rows used per class after balancing. |

   If either class has fewer than five usable rows the classifier is skipped and
   `{"error": "Insufficient samples", "accuracy": 1.0, "auc_roc": 1.0}` is
   returned (treated as "maximally unrealistic").

Columns listed in `FEATURE_COLUMNS_EXCLUDE` (identifiers, labels, provenance
metadata) and the raw `benford_residual_*` columns are dropped before either
metric is computed; only numeric feature columns present in **both** frames are
used.

### How to interpret the scores

| Signal | Realistic simulator | Unrealistic simulator |
|---|---|---|
| `discriminator_accuracy.accuracy` | ≈ 0.50, and no more than ~0.55–0.60 (near chance — the classifier cannot separate the two) | → 1.0 (trivially separable) |
| `discriminator_accuracy.auc_roc` | ≈ 0.5 | → 1.0 |
| `frechet_feature_distance` | close to 0 | large / unbounded |

The near-chance thresholds (`<= 55%` in the module docstring, `<= 60%` in
`compute_discriminator_accuracy`) are guidance, not a hard gate — read them
together with the FFD and the cross-validation spread. A high `cross_val_std`
next to a borderline accuracy usually means there are too few real samples to
draw a firm conclusion rather than a genuinely good simulator.

The evaluation writes a JSON report to
`reports/simulator_realism_<UTC-timestamp>.json` with keys `timestamp`,
`simulated_data`, `real_data`, `frechet_feature_distance`, and
`discriminator_accuracy`.

---

## Command examples

### 1. Generate a simulated dataset

`scripts/generate_synthetic_dataset.py` is the entry point — it drives the
Wash Trade Simulation Engine and runs the output through
`build_feature_matrix`, producing a labelled feature Parquet.

```bash
# Default profile (NaiveAttacker), 500 wallets
python -m scripts.generate_synthetic_dataset \
    --n-wallets 500 \
    --output data/synthetic_dataset.parquet

# A specific attacker profile
python -m scripts.generate_synthetic_dataset \
    --profile RingAttacker \
    --n-wallets 50 \
    --output data/ring_dataset.parquet

# Adaptive attacker against a trained model
python -m scripts.generate_synthetic_dataset \
    --profile AdaptiveAttacker \
    --model-path models/random_forest.joblib \
    --n-wallets 200 \
    --output data/adaptive_dataset.parquet
```

To evolve an evasive strategy with the genetic algorithm instead:

```bash
python -m scripts.adversarial_wash_trade_simulator \
    --generations 100 \
    --population 50 \
    --model-dir ./models \
    --output data/adversarial_trades.parquet
```

Using a profile directly from Python:

```python
from scripts.wash_trade_simulator import create_profile, trades_to_feature_matrix

profile = create_profile("RingAttacker", n_wallets=20, trades_per_wallet=100)
trades = profile.generate_trades()
features = trades_to_feature_matrix(trades)  # per-wallet feature matrix with `label`
```

### 2. Evaluate its realism

Both `--simulated` and `--real` must point at **feature matrices** (Parquet),
not raw trade frames. Build the real side once with
`scripts/build_labelled_dataset.py`.

```bash
# One-off real dataset (skip if data/labelled_dataset.parquet already exists)
python -m scripts.build_labelled_dataset \
    --trades data/testnet_trades.parquet \
    --output data/labelled_dataset.parquet

# Score the simulator
python -m scripts.evaluate_simulator_realism \
    --simulated data/synthetic_dataset.parquet \
    --real data/labelled_dataset.parquet \
    --output-dir reports \
    --seed 42
```

The run prints the sample counts, the FFD, and the discriminator accuracy, and
writes the full JSON report under `reports/`.
