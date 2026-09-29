"""Offline stubs for integration development workflows.

Developing against `LedgerLensContractClient` normally requires a funded
Testnet keypair and a deployed `ledgerlens-score` contract (see
`scripts/testnet_setup.py` and `tests/integration/README.md`) — high friction
for iterating on code that merely *calls* the contract client (new scripts,
alert dispatch, manual smoke-testing). `StubContractClient` is a drop-in,
in-memory stand-in with the same public method surface as
`LedgerLensContractClient` — no network, no `stellar_sdk`, no Testnet setup.

Usage:
    from integrations.offline_stubs import get_contract_client

    # Real client if LEDGERLENS_CONTRACT_ID/LEDGERLENS_SUBMITTER_SECRET are
    # set, StubContractClient if LEDGERLENS_OFFLINE=1 is set.
    client = get_contract_client()

    # Or force one explicitly:
    client = get_contract_client(offline=True)
"""

from __future__ import annotations

import os
from typing import Any

_STUB_SECRET = "SSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUBSTUB"


class StubContractClient:
    """In-memory stand-in for `LedgerLensContractClient`. No network calls.

    Mirrors `submit_score` / `submit_score_with_commitment` /
    `submit_score_with_uncertainty` / `get_score` /
    `propose_threshold_change` / `approve_threshold_change`. Scores submitted
    are readable back via `get_score` within the same process; nothing is
    persisted across runs. `approve_threshold_change` is a single-approval
    stub — it does not model the on-chain multi-sig quorum in
    `governance_contract.rs`, since offline development rarely needs that.
    """

    def __init__(
        self,
        contract_id: str = "STUB_CONTRACT",
        rpc_url: str = "offline://stub",
        network_passphrase: str = "Offline Stub Network",
        submitter_secret: str | None = _STUB_SECRET,
    ):
        self.contract_id = contract_id
        self.rpc_url = rpc_url
        self.network_passphrase = network_passphrase
        self.submitter_secret = submitter_secret
        self._scores: dict[tuple[str, str], dict] = {}
        self._next_proposal_id = 1
        self._proposals: dict[int, int] = {}
        self.calls: list[tuple[str, dict]] = []

    def submit_score(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        *,
        commitment: str | None = None,
        trade_data_hash: str | None = None,
        model_version_hash: str | None = None,
    ) -> dict:
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")
        if commitment is not None and (trade_data_hash is None or model_version_hash is None):
            raise ValueError(
                "trade_data_hash and model_version_hash are required when commitment is set"
            )

        self.calls.append(("submit_score", {"wallet": wallet, "asset_pair": asset_pair}))
        stored = dict(risk_score)
        if commitment is not None:
            stored.update(
                commitment=commitment,
                trade_data_hash=trade_data_hash,
                model_version_hash=model_version_hash,
            )
        self._scores[(wallet, asset_pair)] = stored
        return stored

    def submit_score_with_commitment(
        self,
        wallet: str,
        asset_pair: str,
        risk_score: dict,
        commitment: str,
        trade_data_hash: str,
        model_version_hash: str,
    ) -> dict:
        return self.submit_score(
            wallet,
            asset_pair,
            risk_score,
            commitment=commitment,
            trade_data_hash=trade_data_hash,
            model_version_hash=model_version_hash,
        )

    def submit_score_with_uncertainty(
        self, wallet: str, asset_pair: str, risk_score_dict: dict
    ) -> dict:
        if not self.submitter_secret:
            raise ValueError("LEDGERLENS_SUBMITTER_SECRET is not configured")
        self.calls.append(
            ("submit_score_with_uncertainty", {"wallet": wallet, "asset_pair": asset_pair})
        )
        self._scores[(wallet, asset_pair)] = dict(risk_score_dict)
        return risk_score_dict

    def get_score(self, wallet: str, asset_pair: str) -> dict:
        try:
            return self._scores[(wallet, asset_pair)]
        except KeyError:
            raise LookupError(
                f"StubContractClient has no score for wallet={wallet!r} "
                f"asset_pair={asset_pair!r} — call submit_score first"
            ) from None

    def propose_threshold_change(
        self, governance_contract_id: str, new_threshold: int, proposer_secret: str
    ) -> int:
        proposal_id = self._next_proposal_id
        self._next_proposal_id += 1
        self._proposals[proposal_id] = new_threshold
        self.calls.append(("propose_threshold_change", {"proposal_id": proposal_id}))
        return proposal_id

    def approve_threshold_change(
        self, governance_contract_id: str, proposal_id: int, approver_secret: str
    ) -> bool:
        if proposal_id not in self._proposals:
            raise LookupError(f"StubContractClient has no open proposal {proposal_id}")
        self.calls.append(("approve_threshold_change", {"proposal_id": proposal_id}))
        return True


def get_contract_client(*, offline: bool | None = None, **kwargs: Any) -> Any:
    """Return a real `LedgerLensContractClient`, or `StubContractClient` when offline.

    `offline` defaults to the `LEDGERLENS_OFFLINE` env var (true for "1" or
    "true", case-insensitive). Extra `kwargs` are forwarded to whichever
    client is constructed.
    """
    if offline is None:
        offline = os.getenv("LEDGERLENS_OFFLINE", "").strip().lower() in ("1", "true")

    if offline:
        return StubContractClient(**kwargs)

    from integrations.contract_client import LedgerLensContractClient

    return LedgerLensContractClient(**kwargs)
