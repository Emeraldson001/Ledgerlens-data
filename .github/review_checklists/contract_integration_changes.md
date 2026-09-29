# Code Review Checklist — Soroban Contract Integration Changes

> **When to use this checklist:**  
> A PR touches `integrations/contract_client.py`, `integrations/zk_attestor.py`,
> `integrations/governance_contract.rs`, `integrations/emergency_pause_contract.rs`,
> or `integrations/soroban/`.

---

## Before approving, verify ALL items below

### 1. RiskScore struct alignment

- [ ] `contract_client.py` encodes all eight on-chain `RiskScore` fields:
  - `score` (u32, 0–100)
  - `benford_flag` (bool)
  - `ml_flag` (bool)
  - `timestamp` (u64 — Stellar ledger timestamp)
  - `confidence` (u32, 0–100)
  - `score_lower` (u32, ×100 scaled)
  - `score_upper` (u32, ×100 scaled)
  - `coverage_guarantee` (u32 — percentage, e.g. 90)
- [ ] Field types and encodings match `ledgerlens-contract`'s Rust definition
- [ ] No new on-chain fields added without linked PR in `ledgerlens-contract`
- [ ] `ring_id` is **not** submitted on-chain (API/storage only)

---

### 2. `submit_score` function

- [ ] `submit_score(wallet, asset_pair, score, timestamp)` signature unchanged
- [ ] Asset pair string uses canonical format: `CODE:ISSUER/CODE:ISSUER`
- [ ] Signed by `LEDGERLENS_SUBMITTER_SECRET` (never hardcoded, loaded from env)
- [ ] Submission fee estimated and within budget (LEDGERLENS_MAX_FEE_STROOPS)
- [ ] Retry with exponential backoff on transient Horizon errors
- [ ] Idempotent: re-submitting same score for same wallet/pair doesn't create duplicates

---

### 3. `get_score` function

- [ ] Returns maximally conservative defaults if wallet/pair not on-chain
- [ ] Handles contract `NOT_FOUND` response gracefully
- [ ] Response deserialized back into `RiskScore` Python dataclass
- [ ] Cache or rate limiting applied (avoid hammering Horizon on burst traffic)

---

### 4. Testnet verification

> ⚠️ Contract changes MUST be verified on Stellar Testnet before merge.

- [ ] `python -m scripts.testnet_setup` completed without error
- [ ] `python run_pipeline.py --submit-onchain --dry-run` passes
- [ ] At least one real `submit_score` + `get_score` round-trip tested on testnet
- [ ] Transaction hash and testnet explorer link added to PR description
- [ ] Emergency pause contract still functional after changes

---

### 5. ZK attestation (`integrations/zk_attestor.py`)

- [ ] Proof generation and verification still pass (`tests/test_zk_attestor.py`)
- [ ] If Soroban proof contract updated, linked PR in `ledgerlens-contract`
- [ ] ZK proof schema version bump documented if format changed

---

### 6. Governance contract (`integrations/governance_contract.rs`)

- [ ] Voting quorum threshold unchanged unless explicitly approved
- [ ] Proposal and vote encoding matches on-chain contract ABI
- [ ] Timelock duration not shortened (security invariant)
- [ ] Emergency pause mechanism (`emergency_pause_contract.rs`) still functional

---

### 7. Security

- [ ] `LEDGERLENS_SUBMITTER_SECRET` never logged, never in error messages
- [ ] All secret keys in `.env.example` (values masked), not in `config.py`
- [ ] `STELLAR_NETWORK_PASSPHRASE` correct for target network (Testnet vs Mainnet)
- [ ] No hardcoded contract IDs (use `LEDGERLENS_CONTRACT_ID` env var)

---

### 8. Test coverage

- [ ] `tests/test_contract_client.py` mocks pass for new submit/get logic
- [ ] `tests/integration/test_contract_client_live.py` verified (testnet)
- [ ] `tests/integration/test_pipeline_submit_onchain.py` passes with mocks

---

## Cross-repo impact

This change may require linked PRs in:

| Repo | Impact | Linked PR |
|------|--------|-----------|
| `ledgerlens-contract` | Rust RiskScore struct or ABI changed | ☐ |
| `ledgerlens-core` | Shared Python/TS types updated | ☐ |
| `ledgerlens-api` | API reads on-chain scores | ☐ |
| `ledgerlens-dashboard` | Dashboard displays on-chain scores | ☐ |

---

## Reviewer notes

_Document testnet transaction hashes, ABI diffs, and cross-repo co-ordination plan here._
