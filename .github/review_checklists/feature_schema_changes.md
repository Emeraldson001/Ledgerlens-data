# Code Review Checklist — Feature Schema Changes

> **When to use this checklist:**  
> A PR touches `detection/feature_engineering.py`, `data/feature_dictionary.md`,
> `data/feature_ranges.json`, or the number/names of feature columns produced by
> `build_feature_matrix()`.

---

## Before approving, verify ALL items below

### 1. Feature column additions

- [ ] New feature column is documented in `data/feature_dictionary.md` with:
  - Description of what it measures
  - Value range (min/max or enumerated values)
  - Which detection category it belongs to (Benford / Trade Pattern / Wallet Graph / Cross-Asset / etc.)
  - Formula or computation reference
- [ ] `data/feature_ranges.json` updated (`python scripts/compute_feature_ranges.py`)
- [ ] New feature is listed in `README.md` under the relevant feature category
- [ ] SHAP interpretability works for the new feature (test via `detect/shap_explainer.py`)
- [ ] Feature is included in `FEATURE_COLUMNS_EXCLUDE` allowlist or scorer scoring path
- [ ] If cross-asset feature: `all_pairs_df` parameter path is tested
- [ ] Default/fallback value is defined for wallets with insufficient data

### 2. Feature column removals or renames ⚠️ BREAKING

- [ ] Old feature name still aliased or gracefully ignored (backward compat)
- [ ] `feature_schema_hash` in `model_metadata.json` will change — acknowledged
- [ ] Models must be retrained — scheduled retrain confirmed
- [ ] `ledgerlens-core` feature type definitions updated
- [ ] Linked PRs opened in `ledgerlens-api` and `ledgerlens-dashboard`
- [ ] Version bump included (minor for addition, major for removal/rename)
- [ ] CHANGELOG.md entry explicitly states which columns changed

### 3. Feature computation correctness

- [ ] Feature values are in the expected range (0–1, raw counts, etc.)
- [ ] NaN / None handling: fallback values or imputation defined
- [ ] No data leakage: feature uses only data available at scoring time
- [ ] Rolling window behavior is correct (no off-by-one in time windows)
- [ ] Performance: feature doesn't introduce O(n²) or worse complexity per wallet

### 4. Test coverage

- [ ] New unit tests in `tests/test_feature_engineering.py` or `tests/test_features.py`
- [ ] Edge cases covered: empty wallet history, single trade, missing counterparty
- [ ] Feature values deterministic under same input (no randomness without seeding)

---

## Reviewer notes

_Add any observations about correctness, performance, or downstream impact here._
