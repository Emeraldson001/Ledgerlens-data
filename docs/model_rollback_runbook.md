# Model Rollback Runbook

This runbook covers the automated shadow-deployment/promotion/rollback flow
in `scripts/retrain_if_drifted.py` and `detection/model_governance.py`, and
manual intervention when it fails or is triggered repeatedly.

Every flag and command documented below is exercised by an automated test
(`tests/test_docs_cli_consistency.py`, `tests/test_retrain_trigger.py`,
`tests/test_model_governance.py`) so this document cannot silently drift from
the shipped CLI the way it did before Grand 2 (issue #671) — see
[ADR: Single source of truth for model artifact trust and promotion](model_artifact_trust_and_promotion_adr.md).

---

## Background

After concept drift is detected, `retrain_if_drifted.py` trains a new
ensemble candidate. What happens next depends on `--no-shadow`:

- **Default (no `--no-shadow`)**: the candidate starts a **shadow
  deployment** rather than being promoted immediately. During the shadow
  period (`SHADOW_PERIOD_HOURS`, default 24h), a configurable fraction of
  live scoring requests (`SHADOW_TRAFFIC_PERCENT`, default 20%) are scored
  by both the production and candidate models. Disagreements larger than
  `SHADOW_DRIFT_THRESHOLD_POINTS` (default 15 points) are counted as
  **shadow drift events**. A `ModelVersionRecord` row with `status="shadow"`
  is created so the shadow period is queryable, not just a JSON file.
- **`--no-shadow`**: the candidate is submitted for **immediate gated
  promotion** — it still goes through every check below, it just skips the
  live shadow-traffic observation period.

At the end of the shadow period, re-run with `--check-shadow`. It:

1. Rolls back (discards the candidate, production untouched) if the shadow
   drift rate ≥ `SHADOW_DRIFT_MAX_RATE` (default 5%), or the candidate's FP
   rate exceeds production's by more than `SHADOW_FP_RATE_MAX_EXCESS`
   (default 10%, only checked when `--retrain-data-path` is supplied).
2. Otherwise calls `detection.model_governance.promote_candidate` — the
   **single gated path** to production, which re-runs the offline AUC/F1
   regression gate and the full Ed25519 + transparency-log trust chain
   before touching `MODEL_DIR`. A shadow candidate that passed the live
   checks above can still be rejected here (e.g. it also regressed on the
   offline test set, or fails to sign/verify) — in which case it is treated
   as a rollback, not a partial promotion.

**Every promotion/rollback, automated or manual, is authenticated and
audited.** The automated pipeline authenticates as the actor named by
`MODEL_PROMOTION_SYSTEM_ACTOR` (default `retrain-pipeline`), which must be
listed in `MODEL_PROMOTION_AUTHORIZED_ACTORS`; its credential is an
HMAC-SHA256 of its own name keyed by `MODEL_PROMOTION_SECRET`, so no
interactive secret is needed in CI. Every attempt — successful, denied, or
failed — is written to the `promotion_audit_log` table
(`detection.persistence.PromotionAuditRecord`), and the actor who approved
each promotion/rollback is recorded on `ModelVersionRecord.promoted_by` /
`rolled_back_by`.

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | No drift detected, or no pending shadow candidate (`--check-shadow`) |
| 1 | Fatal error (missing metadata, promotion denied by authorization/trust-chain failure with no drift-report context, etc.) |
| 2 | Drift detected, retrained, promoted (via `--no-shadow` or `--check-shadow`) |
| 3 | Drift detected, retrained, **not** promoted (regression, compatibility, or authorization/trust failure) |
| 4 | Shadow deployment started, or shadow period still running (`--check-shadow`) |
| 5 | Shadow candidate promoted (`--check-shadow`) |
| 6 | Shadow candidate rolled back — drift, FP regression, or a gated-promotion failure (`--check-shadow`) |

---

## When Automatic Rollback Is Triggered Repeatedly

Repeated rollbacks indicate that the retrained model consistently disagrees
with production or produces more false positives. Common causes:

| Symptom | Likely cause |
|---------|--------------|
| High shadow drift rate (> 5%) | Distribution shift in training data; corrupted batch |
| FP rate regression > 10% | Mislabelled training examples; class imbalance spike |
| Rollback on every cycle | Upstream data pipeline feeding corrupt/synthetic rows |
| Rollback with an authorization/trust-chain reason in the audit log | Misconfigured `MODEL_SIGNING_PRIVATE_KEY_PATH`/`TRUSTED_SIGNING_PUBLIC_KEY_PATH`, or a rotated `MODEL_PROMOTION_SECRET` the pipeline wasn't updated with |

---

## Step-by-Step Manual Intervention

### 1. Check current shadow state

```bash
cat "${MODEL_DIR}/shadow_deployment_state.json"
```

Key fields: `version_id`, `candidate_dir`, `shadow_start`, `drift_rate`. Cross-reference with its `ModelVersionRecord` row:

```bash
python -c "
from detection.persistence import ModelVersionRecord, get_engine, get_session_factory
from config import config
sf = get_session_factory(get_engine(config.RISK_SCORE_DB_URL))
with sf() as s:
    row = s.query(ModelVersionRecord).filter_by(version_id='<version_id>').one()
    print(row.status, row.shadow_drift_rate, row.promotion_blocked_reason)
"
```

### 2. Inspect the drift report and promotion audit log

```bash
ls reports/retrain_report_*.json | sort | tail -3
python -c 'import json, pathlib; print(json.dumps(json.loads(pathlib.Path("reports/retrain_report_<latest>.json").read_text()), indent=2))'

python -c "
from detection.persistence import PromotionAuditLog, get_engine, get_session_factory
from config import config
log = PromotionAuditLog(get_session_factory(get_engine(config.RISK_SCORE_DB_URL)))
for row in log.recent(20):
    print(row.created_at, row.actor, row.action, row.success, row.reason)
"
```

### 3. Audit the training data

```bash
python -m scripts.build_labelled_dataset --output data/audit_$(date +%Y%m%d).parquet
```

Check label distribution and feature statistics against the archive:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/audit_<date>.parquet')
print(df['label'].value_counts())
print(df.describe())
"
```

### 4. Roll back to the previous production version (single command)

Rollback is a single authenticated, audited, trust-chain-verified operation
— it is **not** a manual `cp` of archived files. It restores from the
archive directory recorded on the prior `ModelVersionRecord`, re-verifies
that archive's Ed25519 signature and transparency-log membership before
publishing it (an archive on disk could in principle have been tampered
with at rest — rollback does not weaken the trust chain), and records both
the rolled-back and the newly-restored version.

```bash
python -c "
from detection import model_governance as mg
from config import config

actor, credential = 'oncall-<name>', '<hmac-issued-by-security-team>'
# Or, if you are the automated pipeline's identity:
# actor, credential = mg.system_actor_credential()

record = mg.rollback_production(
    model_dir=config.MODEL_DIR,
    actor=actor,
    credential=credential,
    reason='<why you are rolling back>',
)
print('Restored version_id:', record.version_id, 'parent:', record.parent_version_id)
"
```

If it raises `NoRollbackTargetError`, there is no prior `ModelVersionRecord`
to restore (e.g. this is the first-ever promoted version, or its archive
directory is missing) — see step 5 for last-resort recovery. If it raises
`ArtifactTrustError`, the archived candidate itself failed verification;
treat this as a security incident (an archive should never fail its own
recorded signature) and escalate rather than forcing the restore.

`--credential` for a human operator is issued by whoever administers
`MODEL_PROMOTION_SECRET` (e.g. `python -c "from detection import
model_governance as mg; print(mg.expected_credential('<your-actor-name>'))"`
run wherever the secret is available) — it is not a personal password, it's
an HMAC keyed by the shared promotion secret, rotated the same way any other
shared secret is.

### 5. Last-resort recovery when no ModelVersionRecord/archive exists

Only if step 4 raised `NoRollbackTargetError` and the environment predates
Grand 2's `ModelVersionRecord` bookkeeping (very old deployments) or the
archive directory (`${MODEL_DIR}/archive/<timestamp>/`) was deleted:

```bash
ls -lt ${MODEL_DIR}/archive/ | head -5
ARCHIVE=${MODEL_DIR}/archive/<timestamp>

cp ${ARCHIVE}/*.joblib ${MODEL_DIR}/
cp ${ARCHIVE}/model_metadata.json ${MODEL_DIR}/
cp ${ARCHIVE}/metrics.json ${MODEL_DIR}/
rm -f ${MODEL_DIR}/shadow_deployment_state.json
```

Then verify the restored artifacts before assuming production is healthy —
this bypasses the gated rollback path, so nothing has re-checked the trust
chain for you:

```bash
python -c "
from detection.model_inference import verify_model_artifact_signature
ok = verify_model_artifact_signature('${MODEL_DIR}', 'manual-restore')
print('Signatures OK:', ok)
"
```

If this returns `False`, do **not** restart services pointed at `MODEL_DIR`
— `RiskScorer` will hard-block on load (see "Emergency integrity override"
below only if you understand and accept its tradeoff).

### 6. Promote a candidate manually (bypass shadow, not the trust/regression gate)

If you are confident the candidate is correct (e.g., you have reviewed the
training data and confirmed the drift is real):

```bash
python -m scripts.retrain_if_drifted \
    --retrain-data-path data/verified_dataset.parquet \
    --no-shadow
```

`--no-shadow` only skips the live shadow-traffic observation period — the
regression gate and trust chain still run. Exit code 2 means the model was
promoted; exit code 3 means the gate rejected it (check the audit log from
step 2 for the reason).

### 7. Adjust thresholds to reduce rollback sensitivity

If legitimate model updates are being blocked by overly tight thresholds,
adjust via environment variables before re-running:

```bash
export SHADOW_DRIFT_MAX_RATE=0.10       # allow up to 10% drift (default 5%)
export SHADOW_FP_RATE_MAX_EXCESS=0.15  # allow up to 15% FP excess (default 10%)
export SHADOW_PERIOD_HOURS=12          # shorten shadow period (default 24)
export MODEL_PROMOTION_REGRESSION_TOLERANCE=0.02  # allow up to 2pp AUC/F1 regression (default 0.01)
python -m scripts.retrain_if_drifted --check-shadow
```

---

## Emergency integrity override (RiskScorer)

If `RiskScorer` construction is hard-blocking on a model that operators have
independently confirmed is safe to skip (e.g. a known-corrupted secondary
model in a 3-model ensemble, while a fix is rolled out), set:

```bash
export MODEL_INTEGRITY_OVERRIDE_ACTOR="<your-name-or-oncall-id>"
export MODEL_INTEGRITY_OVERRIDE_REASON="<why — required for the audit trail>"
```

This does **not** bypass verification — a model that fails the trust chain
is still never loaded unverified. It changes construction from "raise and
refuse to start" to "skip the failing model, continue if at least one model
still passes." Every use is written to `promotion_audit_log` with
`action="integrity_override"`. Never set these in a persisted `.env` file;
export them for a single incident-response shell session only, and unset
them once the underlying artifact is fixed.

---

## Alert: Repeated Rollback

If rollback is triggered more than 3 times in 7 days:

1. Open a P2 incident — the production model may be degrading while retraining
   is blocked.
2. Page the ML-Ops on-call rotation.
3. Consider temporarily pinning the model version by setting `MODEL_DIR` to the
   last known-good archive path (read-only mount, not a write target).

## Alert: Drift-monitor health {#drift-monitor-health}

`monitoring/drift_detector.CovarianceShiftDetector` records a heartbeat on
every `detect()` call — success or failure. If no successful call has been
recorded within `DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS` (default 3600s),
the `DriftMonitorHeartbeatStale` alert fires (see
`monitoring/alert_rules.yml`) — this means feature-drift monitoring itself
has stopped running or is failing on every invocation, which is distinct
from (and more serious than) "no drift detected": no retraining trigger can
fire until this is resolved, even if drift is actually occurring. Check
`ledgerlens_drift_monitor_check_failures_total` and the process logs for the
periodic drift-check job for the underlying exception.

## Alert: Unauthorized promotion attempts {#authorization-failures}

`ModelPromotionUnauthorizedAttempts` fires on a burst of denied
promotion/rollback attempts (see `monitoring/alert_rules.yml`). Query
`promotion_audit_log` (step 2 above) for the `actor` and `reason` on each
denied row. A misconfigured automation actor after a `MODEL_PROMOTION_SECRET`
rotation looks like repeated denials for `MODEL_PROMOTION_SYSTEM_ACTOR`; an
unrecognized actor name is a candidate security incident.

---

## Key Files

| File | Purpose |
|------|---------|
| `${MODEL_DIR}/shadow_deployment_state.json` | Persisted shadow state (version ID, start time, drift rate) — mirrored in the `model_versions` table |
| `${MODEL_DIR}/archive/<timestamp>/` | Point-in-time snapshots taken by `detection.model_governance.archive_production` before every promotion/rollback |
| `reports/retrain_report_<timestamp>.json` | Per-retraining audit report |
| `detection/model_governance.py` | The single gated promotion/rollback path — `promote_candidate`, `rollback_production`, `authorize_actor`, `evaluate_regression_gate` |
| `detection/persistence.py` | `ModelArtifactVerifier`, `ModelVersionRecord`, `PromotionAuditRecord` |
| `detection/model_inference.py` | `RiskScorer` (hard-blocking load path), `ShadowScorer`, `verify_model_artifact_signature` |
| `scripts/retrain_if_drifted.py` | Drift detection + shadow orchestration CLI |
| `scripts/manage_artifact_lifecycle.py` | Authenticated CLI for the per-artifact-name lifecycle registry |

---

## Related Documentation

- [ADR: Single source of truth for model artifact trust and promotion](model_artifact_trust_and_promotion_adr.md)
- `docs/model_artifact_lifecycle.md` — `ModelArtifactRegistry` state machine
- `docs/drift_detection.md` — Feature drift monitoring (PSI thresholds, MMD)
- `docs/gnn_architecture.md` — Incremental graph update strategy
