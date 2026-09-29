# ADR: Single source of truth for model artifact trust and promotion

**Status:** Accepted
**Related:** Grand 2 — issue #671. Grand 5's consolidated CI release gate is
expected to add this ADR's docs-vs-CLI and promotion-authorization tests to
its checklist.

## Context

An audit of the model-lifecycle system (issue #671) found a real
cryptographic trust chain that production inference silently bypassed, a
separate lifecycle state machine that could reach `PROMOTED` without ever
consulting that trust chain, at least two additional code paths that could
overwrite the production model directory with no regression check at all,
and a documented shadow-deployment/rollback flow whose CLI flags did not
exist, making the entire script uninvokable:

- `detection/persistence.py` implements `ModelArtifact.verify_chain()` and
  `ModelArtifactVerifier.verify()` — SHA-256 + Ed25519 signature +
  transparency-log membership. This is real and correct.
- `detection/artifact_lifecycle.py`'s `ModelArtifactRegistry` (a
  STAGED→VALIDATED→PROMOTED JSON-manifest state machine) never called the
  trust chain. `promote()` only flipped a stage string.
- `detection/model_inference.py`'s `RiskScorer._load_models` called
  `verify_chain()` with no public key configured anywhere (so it always
  raised `ModelIntegrityError`) and caught the exception with
  `logger.warning(...)`, still adding the model to the active `models` dict.
  Since `joblib.load` deserializes pickle, this was a silent,
  code-execution-capable load path for a tampered or unsigned artifact.
- `scripts/retrain_if_drifted.py` referenced `args.check_shadow` /
  `args.no_shadow` without either flag existing in `parse_args()` — every
  invocation of `main()` raised `AttributeError` before reaching any drift
  logic. The shadow-deployment code block also sat after unconditional
  `return` statements in the full-retrain path, making it dead code even
  once the argparse bug was fixed.
- `training/train.py` and `detection/model_training.py`'s `main()` could
  both write directly into `config.MODEL_DIR`, and
  `detection/active_learning/incremental_trainer.py`'s warm-start path
  overwrote production `.joblib` files in place — none of these ran the
  offline AUC/F1 regression gate that `retrain_if_drifted.py`'s own
  (disconnected) `should_promote()` implemented.
- `ModelVersionRecord` (shadow → production → rolled_back) was defined in
  the ORM but never written or read anywhere; actual rollback was a manual
  `cp` runbook.

## Decision

**`detection/model_governance.py` is the single gated path from a trained
candidate directory to production, for every caller.** No other function
writes model artifacts to `config.MODEL_DIR`, and no other function reaches
the lifecycle registry's `PROMOTED` stage.

1. **Trust chain and lifecycle state machine cannot disagree.**
   `ModelArtifactRegistry.promote()` now requires a `trust_verifier`
   callback and calls it before any state transition; a registry
   constructed without one refuses to promote at all
   (`TrustVerificationRequiredError`) rather than silently skipping the
   check. `detection.model_governance.make_trust_verifier()` builds the
   real one, backed by the same `ModelArtifactVerifier` used at load time.

2. **`RiskScorer._load_models` hard-blocks by default.** A model that fails
   Ed25519 signature verification, transparency-log membership, or
   compatibility validation raises — the original typed exception
   (`ModelIntegrityError` vs `ArtifactCompatibilityError`), never wrapped —
   and is never added to `self.models`. The only way to construct a
   `RiskScorer` when a model fails verification is an explicit
   `integrity_override_actor` (or `MODEL_INTEGRITY_OVERRIDE_ACTOR` env var):
   this still never loads the failing model unverified, it only changes
   "raise and refuse to construct" into "skip that model, continue if at
   least one other model still passes" — and every use is written to the
   promotion audit log. `RiskScorer(..., require_trust_chain=False)` exists
   *only* for offline research tooling that trains disposable models never
   served to traffic (the adversarial-robustness evaluation loops); it must
   never be the default and is never used on `config.MODEL_DIR`.

3. **One write gate.** `detection.model_training.save_models` and
   `save_training_artifacts` — the only two functions elsewhere in the
   codebase that write `.joblib`/`metrics.json` files — call
   `detection.model_governance.guard_production_write(model_dir)` before
   touching disk. It raises `UngatedProductionWriteError` if `model_dir`
   resolves to `config.MODEL_DIR` and that directory already holds a
   promoted artifact (`metrics.json` present); a fresh/bootstrap
   environment is allowed to write once. `IncrementalTrainer.update()`
   calls the same guard before its warm-start path, which writes via raw
   `joblib.dump` rather than `save_models`. `tests/test_no_ungated_model_dir_writes.py`
   is a static check requiring every `joblib.dump(` call site in production
   code to carry a `# GUARDED: ...` or `# UNGUARDED-OK: ...` comment, so a
   new ungated write path cannot be added silently.

4. **`promote_candidate(candidate_dir, model_dir, actor, credential, ...)`**
   is the only way a candidate becomes production:
   1. `authorize_actor(actor, credential)` — HMAC-SHA256 credential checked
      against `MODEL_PROMOTION_AUTHORIZED_ACTORS`/`MODEL_PROMOTION_SECRET`.
      Fails closed: an unconfigured secret authorizes nothing.
   2. `evaluate_regression_gate(old_metrics, new_metrics)` — AUC-ROC and F1
      must not regress beyond `MODEL_PROMOTION_REGRESSION_TOLERANCE`
      (default 0.01) for any model family present in both. No prior metrics
      (first-ever promotion) always approves.
   3. `sign_and_verify_candidate` — signs the candidate with
      `MODEL_SIGNING_PRIVATE_KEY_PATH`, appends to the transparency log,
      then re-verifies through `ModelArtifactVerifier` — the *identical*
      code path `RiskScorer` runs at load time, so a successfully-published
      artifact is guaranteed loadable.
   4. `verify_candidate_compatibility` — the existing
      `ArtifactCompatibilityGate` contract.

   Only once all four pass: archive the current production directory,
   publish the candidate's files, update `ModelArtifactRegistry` (per
   artifact name) and insert a `ModelVersionRecord` row
   (`status="production"`, `promoted_by=actor`,
   `parent_version_id=<superseded version>`). Every attempt — successful,
   denied, or failed — is written to `promotion_audit_log`
   (`PromotionAuditRecord`), which is append-only and never updated in
   place. Two concurrent promotions targeting the same `model_dir` are
   serialized by an advisory file lock (`_promotion_lock`) around the
   publish/registry/DB-write critical section, so their file writes and
   `ModelVersionRecord.parent_version_id` chains cannot interleave.

5. **`rollback_production(model_dir, actor, credential, reason, ...)`** is
   the single rollback operation. It resolves the target version (the
   immediate parent of the current production row by default, or an
   explicit `target_version`), locates the archive snapshot that captured
   that version's content (a `ModelVersionRecord`'s own `archive_path` is a
   snapshot of what was live *before* it was promoted — i.e. of its
   *parent* — so restoring version X means using the archive recorded on
   X's child), **re-verifies that archive's trust chain before publishing
   it** (an archive at rest is not exempt from tampering), and records both
   the rolled-back and newly-restored `ModelVersionRecord` rows. Same
   authorization and audit-log requirements as promotion, same file lock.

6. **`scripts/retrain_if_drifted.py`** is rewired end-to-end: the missing
   `--check-shadow`/`--no-shadow` argparse flags are added (fixing the
   `AttributeError` that made every invocation crash); the full-retrain
   path no longer has an unconditional `return` before the shadow block;
   `--no-shadow` and `--check-shadow`'s promotion path both call
   `promote_candidate`; shadow start/outcome are persisted via
   `record_shadow_start`/`record_shadow_outcome`; the incremental
   (warm-start) retraining path now writes its updated LightGBM model into
   a **candidate directory**, never in place into `model_dir`, and is
   promoted through the same gate (`model_names=["lightgbm"]`, since only
   that model changed).

7. **`scripts/manage_artifact_lifecycle.py promote`/`rollback`** require
   `--actor`/`--credential` (or the `MODEL_PROMOTION_ACTOR`/
   `MODEL_PROMOTION_CREDENTIAL` env vars), authorize through the same
   `authorize_actor`, and audit through the same `PromotionAuditLog`.

8. **`scripts/publish_model_artifact.py`** and `promote_candidate` share one
   signing implementation, `detection.persistence.sign_and_register_artifact`
   — a manually-published artifact and an automatically-promoted one go
   through byte-identical signing logic.

9. **Drift-monitor health.** `CovarianceShiftDetector` records a heartbeat
   (`DriftMonitorHealth`) on every `detect()` call, success or failure, and
   still re-raises on failure — it does not swallow exceptions.
   `check_drift_monitor_health()` reports the monitor stale (a distinct
   "drift-check failed" state, separate from "no drift detected") when no
   successful call has landed within `DRIFT_MONITOR_HEARTBEAT_MAX_AGE_SECONDS`
   (default 3600s); `monitoring/alert_rules.yml`'s `DriftMonitorHeartbeatStale`
   alert fires on that condition.

## Consequences

- **Trust is enforced, not advisory.** A tampered or unsigned model can no
  longer reach a scoring `RiskScorer` instance; a compromised training job
  cannot silently overwrite production without also forging a valid Ed25519
  signature and a transparency-log entry.
- **Test/dev environments must sign fixtures.** Every test that constructs
  a `RiskScorer` from freshly-trained models now signs them (see
  `tests/conftest.py`'s `sign_and_trust_models`/`build_signed_model_dir`)
  and injects `public_key`/`transparency_log` explicitly, rather than
  relying on the previous silent-fallback behavior. This is deliberate
  friction: it is the same signing step a real deployment must perform.
- **`require_trust_chain=False` is a narrow, explicit escape hatch** for
  offline research tooling only (`detection/adversarial/augmentation.py`,
  `detection/adversarial/robustness.py`), each call site commented with why
  it's safe (disposable models, never served).
- **Automated retraining needs signing-key access.** Unlike the prior
  design note in `docs/model_artifact_lifecycle.md` ("signing is done by
  `scripts/publish_model_artifact.py` in a controlled environment"),
  `promote_candidate` signs on the automated pipeline's behalf using
  `MODEL_SIGNING_PRIVATE_KEY_PATH`. Operators running fully automated
  drift-triggered retraining must provision that key to the retraining
  job's environment (HSM-backed or a secrets-manager-mounted path, per the
  existing security note in `detection/persistence.py`); a deployment that
  wants a human release step in between should not automate `--no-shadow`
  or `--check-shadow` and should instead sign/publish manually via
  `scripts/publish_model_artifact.py`.
- **SQLite `:memory:` engines are not shared across independently-created
  connections.** Every `detection.model_governance` function accepts a
  `session_factory`/`transparency_log`/`audit_log` parameter and resolves a
  *single* instance once per call, threading it through every DB write in
  that call — building a second, independently-configured session factory
  from `config.RISK_SCORE_DB_URL` partway through a call would silently
  point part of the operation at an unrelated in-memory database. Callers
  that need read-your-writes consistency across multiple governance calls
  (as `scripts/retrain_if_drifted.py main()` does) must construct one
  `session_factory` and pass it to every call themselves.
- **Out of scope (per the issue):** federated-learning/MPC aggregation
  trust, zero-knowledge attestation of model outputs (Grand 3), training
  methodology/model architecture. Also out of scope for this PR:
  `scripts/quantize_models.py`'s compressed edge-deployment variants (they
  write under distinct filenames the primary trust chain never checks —
  see `docs/edge_deployment.md`), and adding governance-gating to
  `detection/gnn_encoder.py`/other `torch.save`-based artifacts (the static
  write-guard check in `tests/test_no_ungated_model_dir_writes.py`
  currently covers `joblib.dump` call sites only; a follow-up should extend
  it to `torch.save` if/when those artifacts join the trust chain).

## Alternatives considered

- **Merge `ModelArtifactRegistry` and `TransparencyLog`/`ModelArtifactVerifier`
  into one class.** Rejected, for the same reason `docs/model_artifact_lifecycle.md`
  originally gave: the transparency log is a security/audit primitive
  (signed, append-only, tamper-evident) and the lifecycle registry is an
  operational primitive (mutable "what's active now" state). Conflating
  them would force the audit log to support mutation. Instead, the registry
  depends on the verifier through a single injected `trust_verifier`
  callback — composition, not merger — which is enough to make the two
  structurally incapable of disagreeing about `PROMOTED`.
- **Have `RiskScorer` fail construction entirely on any verification
  failure, with no override.** Rejected: a genuine incident where one of
  three ensemble models is corrupted at rest but the other two are fine
  should not force a full outage while the artifact is being fixed, given
  the BFT-voting ensemble is explicitly designed to tolerate a minority of
  models being unavailable. The audited override exists for exactly this,
  and only this.
