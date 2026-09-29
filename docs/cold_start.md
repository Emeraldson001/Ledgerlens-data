# Cold-Start Scoring with Neural Process Meta-Learning

> Last verified against code: 2026-08-28. `NP_COLD_START_THRESHOLD` and the
> blend formula below were checked against `detection/neural_process.py`.

## Overview

When a new asset pair is first listed on the Stellar DEX, the system has too few
trades to compute reliable Benford statistics or ML features.  The standard
ensemble would fall back to the global prior (average statistics across all
pairs), producing poorly calibrated scores for that pair.

The Neural Process (NP) meta-learning layer addresses this by learning **how to
adapt** from a small context set rather than relying on a fixed global fallback.

## Architecture

The implementation in `detection/neural_process.py` uses a **Conditional Neural
Process (CNP)**:

- **Encoder** — a two-layer MLP that maps each `(features, label)` context
  trade to a fixed-dimensional latent vector, then aggregates variable-size
  context sets via **mean pooling**.  This makes the encoder permutation-
  invariant and compatible with any context size from 1 to 50 trades.
- **Decoder** — a two-layer MLP that concatenates the pooled context embedding
  with a query feature vector and outputs a wash-trade probability.

The CNP design was chosen over a Latent NP (which adds a stochastic latent
variable) because calibration accuracy — not uncertainty quantification — is the
primary goal in the cold-start path.

## Cold-Start Threshold and Blending

```
NP_COLD_START_THRESHOLD = 50  # trades
```

When a pair has `trade_count < 50` labelled trades, the scorer blends the NP
score with the ensemble score **linearly**:

```
blend_weight = 1.0 - trade_count / threshold
blended_score = blend_weight * np_score + (1 - blend_weight) * ensemble_score
```

- At `trade_count = 0` → pure NP score (blend_weight = 1.0)
- At `trade_count = 25` → 50 / 50 mix
- At `trade_count ≥ 50` → pure ensemble score (blend_weight = 0.0)

This transition avoids a hard cutover and produces smooth score evolution as
trade history accumulates.

## Usage

`detection/neural_process.py` is not currently wired into
`detection.model_inference.RiskScorer` — there is no `score_cold_start`
method on the main scorer. The module is used directly today (see
`detection/certified_robustness.py` and `scripts/run_adversarial_eval.py`).
The public API is the `NeuralProcess` class plus the blending helpers:

```python
from detection.neural_process import NeuralProcess, cold_start_blend_weight, blend_scores
import numpy as np

np_model = NeuralProcess(feature_dim=32)

# context_features: (n_context, feature_dim) array of seed trades
# context_labels:   binary wash-trade labels for context trades
np_score = np_model.predict_score(
    context_features=np.array([...]),
    context_labels=[0, 1, 0, 1, 0],
    query_feature_row=feature_row,
)

blend_weight = cold_start_blend_weight(trade_count=5)  # == 0.9
blended = blend_scores(np_score, ensemble_score, trade_count=5)
```

Wiring this into `RiskScorer` so the ensemble path picks it up automatically
is tracked as a separate follow-up, not covered by this doc.

## Testing

There is no dedicated `tests/test_neural_process.py` yet; `NeuralProcess` is
currently exercised indirectly through `detection/certified_robustness.py`
and `scripts/run_adversarial_eval.py`. Adding direct unit tests (consistency
of predictions for identical context/query sets, cold-start regression on
known wash-trade pairs) is tracked as a separate follow-up.
