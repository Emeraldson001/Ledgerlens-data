# Code Review Checklist — Privacy & Security Changes

> **When to use this checklist:**  
> A PR touches `detection/differential_privacy.py`, `detection/shap_explainer.py`,
> `detection/privacy/`, `detection/adversarial/`, `detection/audit_trail.py`,
> or `utils/field_encryption.py`.

---

## Before approving, verify ALL items below

### 1. Differential privacy (DP)

- [ ] `DP_EPSILON` ≤ 1.0 (strong privacy standard)
- [ ] `DP_DELTA` ≤ 1e-5 (standard: δ << 1/n where n is dataset size)
- [ ] `DP_RENYI_QUERY_THRESHOLD` not increased (default: 100 queries)
- [ ] Per-wallet query count tracked in `ShapQueryCount` table
- [ ] Noise multiplier (`DP_RENYI_NOISE_MULTIPLIER`) applied after threshold
- [ ] `explain_private()` default is `private=True` (noise applied)
- [ ] `private=False` (audit mode) is logged and access-controlled
- [ ] SHAP sensitivity estimates in `models/shap_sensitivity.json` updated

**Rationale for any ε or δ increase:**

_If DP_EPSILON or DP_DELTA was relaxed, document why here._

---

### 2. Field encryption

- [ ] Encryption scheme not weakened (Fernet / AES-GCM or stronger)
- [ ] Key rotation procedure documented and tested
- [ ] Encrypted fields never logged or exposed in error messages
- [ ] Decryption only occurs in secure compute context (not client-side)
- [ ] Key management: keys stored in env vars or KMS, never hardcoded

---

### 3. Adversarial robustness

- [ ] `reports/adversarial_benchmark.json` updated with new attack results
- [ ] FGSM / PGD evasion success rate documented
- [ ] Minimum L-inf perturbation per feature recorded
- [ ] Adversarial training augmentation: AUC-ROC on perturbed test set ≥ baseline - 0.02
- [ ] Certified robustness bounds computed if randomized smoothing applied

**Attack success rate threshold:**  
✅ Pass if FGSM evasion <15% and PGD evasion <30%  
❌ Fail if either exceeds threshold — revert change or apply adversarial training

---

### 4. Membership inference defence

- [ ] Overfitting mitigated: train/test split maintains 80/20 ratio
- [ ] Model confidence calibration (conformal prediction or Platt scaling) applied
- [ ] Membership inference attack success rate tested (`scripts/audit_membership_inference.py`)
- [ ] MI attack success ≤ 55% (within 5% of random guessing for balanced classes)
- [ ] Privacy-accuracy tradeoff documented in PR if accuracy dropped

---

### 5. Audit trail

- [ ] `detection/audit_trail.py` NDJSON log writes are append-only
- [ ] Each audit entry signed with Ed25519 detached signature
- [ ] `AUDIT_LOG_PATH` and `AUDIT_VERIFY_PUBLIC_KEY_PATH` configured
- [ ] `scripts/verify_audit_trail.py` passes on existing log
- [ ] Log rotation: old logs moved to immutable storage (S3 Glacier / IPFS / etc.)
- [ ] Audit log integrity checked on every scheduled run (GitHub Actions / cron)

---

### 6. Secret management

- [ ] No hardcoded secrets (API keys, private keys, passwords) in code
- [ ] `.env.example` updated with any new secret names (values masked)
- [ ] `DP_SHAP_SENSITIVITY` and `ANNOTATION_HMAC_SECRET` stored in env, not config.py
- [ ] Secrets loaded from environment or secrets manager (AWS Secrets Manager / Vault)
- [ ] Secret rotation: documented procedure and tested path

---

### 7. Test coverage

- [ ] `tests/test_dp_shap.py` covers new DP noise application logic
- [ ] `tests/test_adversarial.py` covers new attack/defence code
- [ ] `tests/test_audit_trail.py` covers signature verification
- [ ] `tests/test_field_encryption.py` covers encryption round-trip
- [ ] `tests/test_membership_inference_defence.py` measures MI attack success

---

## Security review sign-off

- [ ] Reviewed by `@Ledger-Lenz/security` team
- [ ] No secrets detected by `git grep` pre-commit hook
- [ ] All secrets rotated if any were accidentally committed in history

---

## Reviewer notes

_Document any risk assessment, threat model considerations, or outstanding concerns here._
