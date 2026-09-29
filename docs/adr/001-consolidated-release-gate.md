# ADR 001: Consolidated Production Release Gate

**Status:** Accepted  
**Date:** 2026-08-31  
**Author:** LedgerLens Team

## Context

Prior to this ADR, the codebase had three separate, partially overlapping release-readiness mechanisms:

1. **`scripts/release_gate.py`** — Claimed (in docstring) to be CI-enforced but was not invoked by any workflow
2. **`scripts/check_release_readiness.py`** — Another ad-hoc Python script, never invoked
3. **`.github/workflows/release-readiness.yml`** — A separate GitHub Actions workflow with a different set of checks

This fragmentation created several risks:

- **Conflicting signals**: Different mechanisms checked different gates (tests, CHANGELOG, TODOs, schema, but not coverage)
- **Hidden failures**: Scripts claimed to be "CI-enforced" in their own docstrings but were not actually invoked anywhere
- **No single source of truth**: Engineers had no clear authority to consult
- **Incomplete test coverage**: None validated migrations against populated data or checked concurrent migration lock enforcement

## Decision

Consolidate into a **single authoritative release gate** with clear levels:

1. **Single Python script** (`scripts/release_gate.py`) as the canonical release-readiness logic
2. **CI-wired job** (`release-gate` in `.github/workflows/ci.yml`) that actually invokes it as a **blocking gate**
3. **Clear gate levels**:
   - **CRITICAL** gates block the pipeline entirely
   - **HIGH** gates warn but do not block (with timeline for elevation to CRITICAL)
   - **INFO** gates log only, no blocking

4. **No misleading docstrings**: Every gate states whether it is blocking or advisory with concrete reasoning

## Rationale

- **Single source of truth**: Engineers know to look at `scripts/release_gate.py`, not three separate mechanisms
- **Actually enforced**: The gate is wired into CI and will fail the pipeline if CRITICAL gates fail
- **Transparency**: Each gate reports its level and pass/fail status clearly in CI logs
- **Extensibility**: New Grands (Grand 1–4) can register their tests as CRITICAL gates; this Grand registers migration, backup, config, and security gates
- **Auditability**: A single script makes it auditable who checks what and why; three mechanisms would not

## CRITICAL Gates (Block Release)

1. **Test Suite**: All pytest tests must pass
2. **Migration Framework**: `migrations/base.py`, `migrations/runner.py`, and `tests/test_migrations.py` must exist
3. **Schema Review Gate**: `.github/review-gates.yml` must include `migrations/` glob patterns
4. **Concurrent Lock Enforcement**: Migration runner must prevent double-application via advisory locks

## HIGH Gates (Warn, Timeline to Critical)

1. **Security Baseline**: `ruff`, `mypy`, `bandit` checks (currently non-blocking; timeline: 1 release cycle to make blocking)
2. **Backup/Restore Readiness**: `scripts/backup.py` and `scripts/restore.py` must exist and be executable
3. **Configuration Completeness**: `production.yaml` and `staging.yaml` must exist in `config/environments/`

## INFO Gates (Logging Only)

1. **Threat Model Documentation**: `docs/security_threat_model.md` must exist

## Deprecations

- `scripts/check_release_readiness.py` is superseded by `scripts/release_gate.py` and should be marked for removal in the next release
- `.github/workflows/release-readiness.yml` (if it exists) is superseded by the `release-gate` job in `ci.yml`
- Any docstrings claiming "CI-enforced" without actual invocation are corrected to state the gate's actual status

## Consequences

**Positive:**
- Clear, enforceable production-readiness criteria
- Single, auditable source of truth
- Early detection of migration, backup, or security issues
- No more contradictory signals from overlapping mechanisms

**Negative:**
- Developers must update their understanding of which gate(s) to consult
- Integration with Grands 1–4 tests will be driven by this gate's registration mechanism (not the other way around)

## Future Extensions

As Grands 1–4 add their own test suites, they register as gates in the release-gate mechanism:

- Grand 1 (Fault Injection): Registers fault-injection tests as CRITICAL gate
- Grand 2 (Model Promotion): Registers model validation as HIGH gate
- Grand 3 (Contract Authorization): Registers authorization tests as CRITICAL gate
- Grand 4 (Tenant Isolation): Registers isolation tests as CRITICAL gate

This way, all release criteria flow through a single, CI-wired, authoritative gate.
