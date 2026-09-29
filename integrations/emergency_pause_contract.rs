// Soroban contract: 2-of-3 emergency pause for the ledgerlens-score oracle.
//
// When paused the contract rejects all submit_score / get_score calls and
// emits a `contract_paused` event consumed by the Python event listener to
// halt the local scoring pipeline.
//
// Entry points
// ─────────────
// init(deployer: Address, keyholders: Vec<Address>)
// initiate_pause(initiator: Address, reason: String) → u64 (proposal_id)
// approve_pause(approver: Address, proposal_id: u64) → bool (paused?)
// initiate_unpause(initiator: Address, pause_proposal_id: u64) → u64
// approve_unpause(approver: Address, unpause_proposal_id: u64) → bool
// is_paused() → bool
// get_admin() → Address
//
// Security
// ─────────
// - init requires the deployer's signature, so the keyholder set cannot be
//   claimed by whoever front-runs the deployment.
// - Only the 3 human-operated emergency keys may call initiate_pause /
//   approve_pause / initiate_unpause / approve_unpause.  The scoring pipeline
//   keys are NOT in this set.
// - Both pausing and unpausing require 2-of-3.  Unpause is as safety-critical
//   as pause, so it is not a single-signer operation.
// - Pause and unpause proposals expire after PAUSE_PROPOSAL_TTL_LEDGERS
//   (≈15 minutes).

#![no_std]
use soroban_sdk::{
    contract, contractimpl, contracttype, symbol_short,
    Address, Env, Map, String, Vec,
};

const PAUSE_PROPOSAL_TTL_LEDGERS: u32 = 180; // ≈15 min @ ~5 s/ledger

#[contracttype]
#[derive(Clone)]
pub struct PauseProposal {
    pub reason: String,
    pub initiator: Address,
    pub approvals: Vec<Address>,
    pub expiry_ledger: u32,
    pub applied: bool,
}

#[contract]
pub struct EmergencyPauseContract;

#[contractimpl]
impl EmergencyPauseContract {
    /// One-time initialisation: register the 3 emergency keyholders.
    ///
    /// `deployer` must authorise the call. Without this the initialiser was
    /// guarded only against *re*-initialisation, so whoever landed `init`
    /// first became all three keyholders -- a front-run of the deployment
    /// itself, handing an attacker the emergency stop.
    pub fn init(env: Env, deployer: Address, keyholders: Vec<Address>) {
        deployer.require_auth();

        if env.storage().instance().has(&symbol_short!("init")) {
            panic!("already initialised");
        }
        assert!(keyholders.len() == 3, "exactly 3 keyholders required");
        env.storage().instance().set(&symbol_short!("admin"), &deployer);
        env.storage().instance().set(&symbol_short!("keys"), &keyholders);
        env.storage().instance().set(&symbol_short!("paused"), &false);
        env.storage().instance().set(&symbol_short!("next_id"), &0u64);
        env.storage().instance().set(&symbol_short!("init"), &true);
    }

    /// The address that initialised the contract.
    pub fn get_admin(env: Env) -> Address {
        env.storage()
            .instance()
            .get(&symbol_short!("admin"))
            .expect("not initialised")
    }

    /// Propose an emergency pause (counts as the first approval).
    /// Returns proposal_id.
    pub fn initiate_pause(env: Env, initiator: Address, reason: String) -> u64 {
        initiator.require_auth();
        Self::require_keyholder(&env, &initiator);
        assert!(
            !env.storage().instance().get::<_, bool>(&symbol_short!("paused")).unwrap_or(false),
            "contract already paused"
        );

        let proposal_id: u64 = env.storage().instance().get(&symbol_short!("next_id")).unwrap();
        let proposal = PauseProposal {
            reason,
            initiator: initiator.clone(),
            approvals: Vec::from_array(&env, [initiator]),
            expiry_ledger: env.ledger().sequence() + PAUSE_PROPOSAL_TTL_LEDGERS,
            applied: false,
        };

        let mut proposals: Map<u64, PauseProposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("pprops"))
            .unwrap_or(Map::new(&env));
        proposals.set(proposal_id, proposal);
        env.storage().instance().set(&symbol_short!("pprops"), &proposals);
        env.storage().instance().set(&symbol_short!("next_id"), &(proposal_id + 1));
        proposal_id
    }

    /// Cast the second approval; if quorum (2-of-3) is reached the contract
    /// is paused and a `contract_paused` event is emitted.
    pub fn approve_pause(env: Env, approver: Address, proposal_id: u64) -> bool {
        approver.require_auth();
        Self::require_keyholder(&env, &approver);

        let mut proposals: Map<u64, PauseProposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("pprops"))
            .unwrap();
        let mut p = proposals.get(proposal_id).expect("pause proposal not found");

        assert!(!p.applied, "proposal already applied");
        assert!(
            env.ledger().sequence() <= p.expiry_ledger,
            "pause proposal expired"
        );
        assert!(!p.approvals.contains(&approver), "already approved");

        p.approvals.push_back(approver);
        let paused = (p.approvals.len() as u32) >= 2;

        if paused {
            p.applied = true;
            env.storage().instance().set(&symbol_short!("paused"), &true);
            env.storage().instance().set(&symbol_short!("pause_id"), &proposal_id);
            env.events().publish(
                (symbol_short!("c_paused"), proposal_id),
                p.reason.clone(),
            );
        }

        proposals.set(proposal_id, p);
        env.storage().instance().set(&symbol_short!("pprops"), &proposals);
        paused
    }

    /// Propose lifting the pause (counts as the first approval).
    /// Returns the unpause proposal_id; call `approve_unpause` for the second.
    ///
    /// Unpausing is as safety-critical as pausing -- it is what puts a system
    /// halted for a suspected incident back into service -- so it runs the
    /// same 2-of-3 cycle. The previous single-step version let one keyholder
    /// restore a paused system unilaterally, contradicting this file's own
    /// header.
    pub fn initiate_unpause(env: Env, initiator: Address, pause_proposal_id: u64) -> u64 {
        initiator.require_auth();
        Self::require_keyholder(&env, &initiator);

        let stored_id: u64 = env
            .storage()
            .instance()
            .get(&symbol_short!("pause_id"))
            .unwrap_or(u64::MAX);
        assert!(stored_id == pause_proposal_id, "invalid pause_proposal_id");
        assert!(
            env.storage().instance().get::<_, bool>(&symbol_short!("paused")).unwrap_or(false),
            "not paused"
        );

        let proposal_id: u64 = env.storage().instance().get(&symbol_short!("next_id")).unwrap();
        let proposal = PauseProposal {
            reason: String::from_str(&env, "unpause"),
            initiator: initiator.clone(),
            approvals: Vec::from_array(&env, [initiator]),
            expiry_ledger: env.ledger().sequence() + PAUSE_PROPOSAL_TTL_LEDGERS,
            applied: false,
        };

        let mut proposals: Map<u64, PauseProposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("uprops"))
            .unwrap_or(Map::new(&env));
        proposals.set(proposal_id, proposal);
        env.storage().instance().set(&symbol_short!("uprops"), &proposals);
        env.storage().instance().set(&symbol_short!("next_id"), &(proposal_id + 1));
        proposal_id
    }

    /// Cast the second approval on an unpause proposal; if quorum (2-of-3) is
    /// reached the contract is unpaused and `c_unpaused` is emitted.
    pub fn approve_unpause(env: Env, approver: Address, unpause_proposal_id: u64) -> bool {
        approver.require_auth();
        Self::require_keyholder(&env, &approver);

        let mut proposals: Map<u64, PauseProposal> = env
            .storage()
            .instance()
            .get(&symbol_short!("uprops"))
            .expect("no unpause proposals");
        let mut p = proposals.get(unpause_proposal_id).expect("unpause proposal not found");

        assert!(!p.applied, "proposal already applied");
        assert!(
            env.ledger().sequence() <= p.expiry_ledger,
            "unpause proposal expired"
        );
        assert!(!p.approvals.contains(&approver), "already approved");

        p.approvals.push_back(approver);
        let unpaused = (p.approvals.len() as u32) >= 2;

        if unpaused {
            p.applied = true;
            env.storage().instance().set(&symbol_short!("paused"), &false);
            env.events().publish(
                (symbol_short!("c_unpaused"), unpause_proposal_id),
                true,
            );
        }

        proposals.set(unpause_proposal_id, p);
        env.storage().instance().set(&symbol_short!("uprops"), &proposals);
        unpaused
    }

    pub fn is_paused(env: Env) -> bool {
        env.storage().instance().get(&symbol_short!("paused")).unwrap_or(false)
    }

    fn require_keyholder(env: &Env, addr: &Address) {
        let keys: Vec<Address> = env.storage().instance().get(&symbol_short!("keys")).unwrap();
        assert!(keys.contains(addr), "not an emergency keyholder");
    }
}
