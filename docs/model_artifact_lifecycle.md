# Versioned Model Artifact Lifecycle

## Why

Model artifacts under `models/` previously had two disconnected mechanisms:

- `scripts/publish_model_artifact.py` signs an artifact and appends its hash
  to the append-only `TransparencyLog` (see `detection/persistence.py`).
- `scripts/list_model_versions.py` reads that log back out.

Neither tracked *operational state*: which version is currently serving
traffic, which versions were staged but never promoted, or why a version
was rolled back. Answering "what's live right now, and what was live before
the last rollback" required reconstructing it from the transparency log by
hand.

## What this adds

`detection/artifact_lifecycle.py` introduces a small, file-backed registry
(`ModelArtifactRegistry`) with a typed `ArtifactRecord` contract and an
explicit state machine:

```
STAGED -> VALIDATED -> PROMOTED -> DEPRECATED
                           |
                           v
                      ROLLED_BACK -> STAGED (re-registration)
```

Illegal transitions (e.g. `STAGED -> PROMOTED`, skipping validation) raise
`InvalidTransitionError` with the current stage and the allowed next stages
in the message. `promote()` automatically demotes any previously-promoted
version of the same artifact family to `DEPRECATED`, so `get_active()`
always resolves to exactly one version. `rollback()` reactivates the parent
version that was active before the rolled-back one was promoted.

**Trust-chain-gated promotion (Grand 2 / issue #671).** `promote()` requires
the registry to be constructed with a `trust_verifier` callback — a function
invoked with the target `ArtifactRecord` before any stage transition, which
must raise (typically `ArtifactTrustError`, wrapping the underlying
`ModelIntegrityError`/`ArtifactCompatibilityError`) on failure. A registry
built with `trust_verifier=None` raises `TrustVerificationRequiredError` on
every `promote()` call rather than silently skipping the check — the trust
chain and this state machine must never be able to disagree about
`PROMOTED`. Use `detection.model_governance.make_trust_verifier(model_dir)`
to build the real one, backed by the same `ModelArtifactVerifier` the
production `RiskScorer` load path uses. See
[the ADR](model_artifact_trust_and_promotion_adr.md) for the full design.

Every record stores the artifact's `sha256` at registration time;
`verify_integrity()` re-hashes the on-disk file and raises
`IntegrityCheckError` (with expected vs. actual hash) if it no longer
matches — this catches artifacts that were modified or replaced outside the
registry after being registered.

The manifest (`models/artifact_manifest.json` by default) is written with a
temp-file-plus-`os.replace` pattern so a crash mid-write cannot corrupt it.

## Contract surface

| Method | Purpose |
|---|---|
| `register(name, artifact_path, metrics, tags)` | Hash + record a new artifact in `STAGED` |
| `validate(name, version)` | `STAGED -> VALIDATED` |
| `promote(name, version)` | `VALIDATED -> PROMOTED`, demotes prior active version |
| `deprecate(name, version, reason)` | Retire a version |
| `rollback(name, version=None, reason=None)` | Roll back active (or given) version, reactivate parent |
| `verify_integrity(name, version)` | Re-hash and compare against manifest |
| `get_active(name)` | Currently promoted version, or raises `ArtifactNotFoundError` |
| `list_versions(name)` / `list_names()` | Enumeration for tooling/CLI |

## Developer commands

`promote` and `rollback` are authenticated, audited actions — see
`--actor`/`--credential` below and
[the rollback runbook](model_rollback_runbook.md).

```
python -m scripts.manage_artifact_lifecycle register --name rf --artifact-path models/rf.joblib
python -m scripts.manage_artifact_lifecycle validate  --name rf --version <version>
python -m scripts.manage_artifact_lifecycle promote   --name rf --version <version> \
    --actor alice --credential <hmac-issued-by-security-team>
python -m scripts.manage_artifact_lifecycle status     --name rf
python -m scripts.manage_artifact_lifecycle rollback   --name rf --reason "AUC regression on canary" \
    --actor alice --credential <hmac-issued-by-security-team>
python -m scripts.manage_artifact_lifecycle verify     --name rf --version <version>
```

Tests: `tests/test_artifact_lifecycle.py` (happy path, illegal transitions,
promotion supersession, rollback/reactivation, tamper detection via
`verify_integrity`, manifest persistence across process instances, and the
`trust_verifier` requirement/failure paths). `tests/test_model_governance.py`
covers the end-to-end gated promotion/rollback path this registry is now
wired into. Run with `pytest tests/test_artifact_lifecycle.py tests/test_model_governance.py -v`.

## Design tradeoffs

- **JSON manifest instead of a DB table.** Keeps this dependency-free and
  inspectable/editable in an incident, matching the existing SQLite/JSON
  mix already used by `TransparencyLog` and `models/metrics.json`. Tradeoff:
  no concurrent-writer locking; single-writer (CI/release process) is the
  assumed usage pattern, same as the existing `metrics.json` update in
  `publish_model_artifact.py`.
- **Separate from `TransparencyLog`.** The transparency log is a
  security/audit primitive (signed, append-only, tamper-evident). The
  lifecycle registry is an operational primitive (mutable state, "what's
  active now"). Conflating them would force the audit log to support
  mutation, weakening its guarantees. They are connected instead through
  the `trust_verifier` callback described above — composition, not merger.
- **Wired into the retrain/promote flows (Grand 2 / issue #671).**
  `detection.model_governance.promote_candidate` constructs a
  `ModelArtifactRegistry` internally (with a real `trust_verifier`) and
  calls `register()`/`validate()`/`promote()` for every model in a
  candidate as part of the single gated promotion path used by
  `scripts/retrain_if_drifted.py`'s `--no-shadow` and `--check-shadow`
  flows. This registry remains available standalone (via
  `scripts/manage_artifact_lifecycle.py`) for operational
  inspection/manual intervention, but production promotions no longer go
  through it directly — see [the ADR](model_artifact_trust_and_promotion_adr.md).
