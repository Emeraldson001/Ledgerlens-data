# Scoring Architecture

LedgerLens normalises anomaly scores per asset pair to enable fair comparison across different trading patterns.

## Per-Pair Normalisation

Raw anomaly scores are normalised against a rolling historical percentile distribution for each asset pair:

- **Window size**: `SCORE_NORM_WINDOW_SIZE` (default 1000) scores per pair
- **Min samples**: `SCORE_NORM_MIN_SAMPLES` (default 50) samples required before normalisation
- **Algorithm**: Linear interpolation of percentile rank within the rolling window

## Redis Storage

Rolling windows are stored in Redis sorted sets:

- Key: `score_window:{asset_pair}`
- Members: Score values as strings
- Scores: The score value itself (for sorting)
- Eviction: `ZREMRANGEBYRANK` maintains exactly `SCORE_NORM_WINDOW_SIZE` entries

## Degradation

When fewer than `SCORE_NORM_MIN_SAMPLES` exist for a pair, normalisation is skipped and the raw score is returned with a `normalisation_skipped` flag.

## Interpreting Scores

- **Raw score**: Absolute anomaly score from the detection engine
- **Normalised score**: Percentile rank (0-1) relative to the pair's historical distribution
- **Normalised scores > 0.99**: Highly anomalous relative to the pair's baseline
- **Normalised scores ~ 0.5**: Typical behavior for the pair

## Explainability: SHAP vs Counterfactual

Every LedgerLens score is accompanied by explainability outputs that help auditors,
compliance teams, and regulators understand *why* a wallet was scored as risky.

### SHAP Attribution (Diagnostic)

The `detection/shap_explainer.py` module produces SHAP values — per-feature contributions
that explain the historical score. Each SHAP value represents how much a single feature
pushed the score up (positive value) or down (negative value) relative to the model's
baseline expectation.

**Use SHAP for:**
- Forensic reports explaining past risk behavior
- Regulatory filings requiring transparent audit trails
- Understanding which features most influenced a decision

**Example:** "This wallet's 24-hour Benford MAD score of 0.084 contributed +12.34 points
to the 82-point wash-trade verdict, indicating abnormally manipulated trade amounts."

### Counterfactual Explanations (Prescriptive)

The `detection/counterfactual_explainer.py` module generates counterfactuals — plausible
scenarios of what *would need to change* for a flagged wallet to drop below the
risk threshold.

**Use counterfactuals for:**
- Remediation workflows guiding flagged wallet operators toward legitimate behavior
- Assessing whether a score can realistically be reduced
- Wallet owner appeals to explain what changes would clear their flag

**Example:** "To drop below the flag threshold, this wallet would need to reduce
counterparty concentration from 0.89 to < 0.70, which would lower the score from 82 to 68."

## Security

Asset pair names used as Redis key suffixes are validated against an allowlist to prevent key injection.
