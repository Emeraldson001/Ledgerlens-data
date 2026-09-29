# Model Card — xgboost v0.2.0

**Training date:** 2026-08-28T00:00:00Z  
**Dataset version:** synthetic-v1  
**Dataset fingerprint (SHA-256):** `8eab53ef6a55d699407767e637f74b6c63bc5ea73b628a4f60bc3a4c3f35ff32`  

## Intended Use

Wash-trade risk scoring for Stellar DEX asset pairs, as an input to the LedgerLens ensemble scorer.

## Out-of-Scope Uses

Not intended for automated enforcement action or as sole evidence in a regulatory filing without human review.

## Known Limitations

Trained on synthetic trade data only; has not been validated against real Stellar DEX activity. Cold-start pairs (fewer than 50 labelled trades) fall back to the Neural Process blend described in docs/cold_start.md.

## Hyperparameters

| Parameter | Value |
| --- | --- |
| `n_estimators` | `300` |
| `max_depth` | `6` |
| `learning_rate` | `0.05` |

## Performance Metrics

| Asset Pair | Precision | Recall | F1 |
| --- | --- | --- | --- |
| USDC:GA5Z.../XLM:native | 0.91 | 0.87 | 0.89 |

## Data Provenance

Training dataset fingerprint (SHA-256): `8eab53ef6a55d699407767e637f74b6c63bc5ea73b628a4f60bc3a4c3f35ff32`

_This fingerprint is computed from the actual training Parquet file at training time and cannot be supplied by the caller._
