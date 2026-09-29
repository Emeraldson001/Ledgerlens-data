"""Adapter for EVM-style chain event logs (e.g. an ERC-20 bridge counterpart).

Demonstrates that :class:`~ingestion.adapters.base.ChainAdapter` supports a
second, structurally different chain without touching the Stellar pipeline
or the normalized contract itself. Intended as the template for a future
real EVM RPC/log-subscription ingestion source (analogous to
``ingestion/horizon_streamer.py`` for Stellar) -- this adapter takes
already-decoded log dicts (the shape a ``web3.py`` event filter or a
Graph-node subgraph response would hand back) rather than making its own
RPC calls, keeping the adapter itself free of network/transport concerns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ingestion.adapters.base import (
    AdapterValidationError,
    ChainAdapter,
    EventType,
    NormalizedAsset,
    NormalizedEvent,
)

_REQUIRED_TRANSFER_FIELDS = (
    "tx_hash",
    "log_index",
    "from",
    "to",
    "token_address",
    "value",
    "block_time",
)

#: ERC-20 tokens are typically transferred in atomic units; without on-chain
#: decimals metadata we assume 18 (the ERC-20 default) unless the raw event
#: supplies its own `decimals` field.
_DEFAULT_DECIMALS = 18


class EvmAdapter(ChainAdapter):
    """Normalizes decoded EVM ``Transfer`` log events.

    Expects a dict shaped like a decoded ``Transfer(address,address,uint256)``
    log, e.g.::

        {
            "tx_hash": "0xabc...",
            "log_index": 3,
            "from": "0x1111...",
            "to": "0x2222...",
            "token_address": "0xa0b8...",
            "token_symbol": "USDC",
            "value": 1000000,          # atomic units
            "decimals": 6,             # optional, defaults to 18
            "block_time": 1732900000,  # unix epoch seconds
        }
    """

    chain = "ethereum"

    def normalize(self, raw_event: Any) -> NormalizedEvent:
        if not isinstance(raw_event, dict):
            raise AdapterValidationError(
                self.chain, f"expected dict, got {type(raw_event).__name__}", raw_event
            )

        missing = [f for f in _REQUIRED_TRANSFER_FIELDS if f not in raw_event]
        if missing:
            raise AdapterValidationError(
                self.chain, f"missing required fields: {missing}", raw_event
            )

        try:
            decimals = int(raw_event.get("decimals", _DEFAULT_DECIMALS))
            amount = float(raw_event["value"]) / (10**decimals)
            occurred_at = datetime.fromtimestamp(int(raw_event["block_time"]), tz=UTC)
            symbol = str(raw_event.get("token_symbol") or raw_event["token_address"][:10])

            return NormalizedEvent(
                event_id=f"{raw_event['tx_hash']}:{raw_event['log_index']}",
                chain=self.chain,
                event_type=EventType.TRANSFER,
                occurred_at=occurred_at,
                account=raw_event["from"],
                counterparty=raw_event["to"],
                asset=NormalizedAsset(
                    symbol=symbol, issuer=raw_event["token_address"], native=False
                ),
                amount=amount,
                raw=raw_event,
            )
        except AdapterValidationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AdapterValidationError(self.chain, str(exc), raw_event) from exc
