# Code Review Checklist — Model Training/Inference Changes

> **When to use this checklist:**  
> A PR touches `detection/model_training.py`, `detection/model_inference.py`,
> `detection/ensemble_calibrator.py`, `detection/drift_monitor.py`, or the
> `models/` artifact directory.

---

## Before approving, verify ALL items below

### 1. Model artifact integrity

- [ ] `models/metrics.json` present and signed (`models/metrics.json.sig` exists)
- [ ] `models/metrics.json` includes AUC-ROC, F1, precision, recall for all three models
- [ ] If model files changed, SHA-256 hashes in `metrics.json` updated
- [ ] `model_metadata.json` has correct `feature_columns` list and `feature_schema_hash`
- [ ] All `.joblib` model files have corresponding `.joblib.sig` signatures
- [ ] Every `joblib.load()` is immediately followed by `verify_chain()` call
- [ ] `TRUSTED_SIGNING_KEY_FINGERPRINT` matches signing authority

### 2. Ensemble behavior

- [ ] BFT voting (`_combine_probabilities`) not bypassed or weakened
- [ ] `BFT_SCORE_DIVERGENCE_THRESHOLD` (default 30 points) not increased
- [ ] Outlier trimming still uses median when divergence exceeds threshold
- [ ] `bft_divergence_detected_total` Prometheus counter incremented on divergence
- [ ] `models/pareto_front.json` exists if calibration was run
- [ ] If ensemble weights changed, documented rationale in PR and CHANGELOG.md

### 3. Model performance

- [ ] CHANGELOG.md includes before/after AUC-ROC and F1 for all three models
- [ ] No performance regression >1% AUC-ROC unless justified
- [ ] Confusion matrix or precision-recall curve included in PR or linked report
- [ ] Cross-validation results reported if training set changed
- [ ] SMOTE or class weighting used if wash-trade ratio changed significantly

### 4. Training reproducibility

- [ ] Random seeds set in `model_training.py` (`RANDOM_SEED`, `np.random.seed`, etc.)
- [ ] Training dataset SHA-256 recorded in `model_metadata.json`
- [ ] Training data lineage: source and date of `data/synthetic_dataset.parquet` or labelled data documented
- [ ] Retraining instructions: `python -m detection.model_training --data-path <path>` tested

### 5. Drift monitoring

- [ ] If drift detection changed, PSI thresholds not weakened (PSI ≥ 0.25 triggers retrain)
- [ ] Reference distribution stored in `model_metadata.json` for each feature
- [ ] Retraining trigger logged and alert dispatched on drift detection
- [ ] `scripts/retrain_if_drifted.py` still functional with new drift monitor code

### 6. Inference correctness

- [ ] `RiskScorer.score()` returns a value in [0, 100]
- [ ] `confidence` field reflects inter-model agreement (not just score copy)
- [ ] Conformal prediction intervals (`score_lower`, `score_upper`) present if calibrated
- [ ] `coverage_guarantee` field defaults to maximally conservative (0.0, 100.0, 1.0) if uncalibrated
- [ ] SHAP explainer works on new model (`ShapExplainer.explain()` doesn't crash)
- [ ] Feature columns coerced to numeric dtypes before model input

### 7. Test coverage

- [ ] `tests/test_model_training.py` passes
- [ ] `tests/test_model_inference.py` covers new inference path
- [ ] `tests/test_ensemble_calibrator.py` covers new calibration logic (if changed)
- [ ] BFT divergence scenario tested (one model returns anomalous score)

---

## Performance regression gate

| Model | Before AUC | After AUC | Δ | Pass? |
|-------|-----------|-----------|---|-------|
| Random Forest | — | — | — | ☐ |
| XGBoost | — | — | — | ☐ |
| LightGBM | — | — | — | ☐ |

✅ Pass if all Δ ≥ -0.01 (tolerate up to 1% drop)  
❌ Fail if any Δ < -0.01 (regressions must be justified or reverted)

---

## Reviewer notes

_Document any performance trade-offs, hyperparameter tuning rationale, or downstream impact here._
