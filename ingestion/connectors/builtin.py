"""Built-in connectors — thin adapters over the existing Horizon loaders.

These wrap `ingestion.historical_loader`, `ingestion.amm_pool_loader`,
`ingestion.orderbook_loader`, and `ingestion.account_activity_loader`
behind the `DataConnector` contract. None of those modules' public
functions change behavior — this file only adds a normalized, discoverable
way to reach them (and proves the plugin boundary fits the sources that
already exist before any third party adds a new one).

Importing this module registers all four connectors as a side effect.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

from stellar_sdk import Asset as SdkAsset

from config import config
from ingestion.account_activity_loader import load_accounts_activity
from ingestion.amm_pool_loader import iter_amm_pool_trades
from ingestion.connectors.base import ConnectorError, ConnectorMetadata, DataConnector
from ingestion.connectors.registry import register_connector
from ingestion.data_models import AccountActivity, OrderBookEvent, Trade
from ingestion.historical_loader import load_trades, trades_to_dataframe
from ingestion.orderbook_loader import load_orderbook_events, orderbook_events_to_dataframe


@register_connector
class SdexTradeConnector(DataConnector[Trade]):
    """Bulk historical SDEX trade ingestion via Horizon's paginated trades endpoint.

    Wraps `ingestion.historical_loader.load_trades`.
    """

    metadata = ConnectorMetadata(
        connector_id="stellar-sdex-trades",
        record_type=Trade,
        source="horizon-rest",
        description="Historical SDEX trades for watched asset pairs against XLM.",
    )

    def load(
        self,
        *,
        since: datetime | None = None,
        pairs: list[tuple[str, str]] | None = None,
        **_: object,
    ) -> Iterator[Trade]:
        """Yield trades for each configured (or explicitly given) asset pair.

        Args:
            since: Skip trades before this time (default: all available).
            pairs: `(code, issuer)` tuples to load; defaults to
                `config.WATCHED_ASSET_PAIRS`.
        """
        xlm = SdkAsset.native()
        watched = pairs if pairs is not None else config.WATCHED_ASSET_PAIRS
        for code, issuer in watched:
            asset = xlm if issuer == "native" else SdkAsset(code, issuer)
            if asset == xlm:
                continue
            yield from load_trades(asset, xlm, start_time=since)

    def to_dataframe(self, records):
        return trades_to_dataframe(records)


@register_connector
class AmmPoolTradeConnector(DataConnector[Trade]):
    """Historical AMM liquidity-pool trade ingestion via Horizon.

    Wraps `ingestion.amm_pool_loader.iter_amm_pool_trades`.
    """

    metadata = ConnectorMetadata(
        connector_id="stellar-amm-pool-trades",
        record_type=Trade,
        source="horizon-rest",
        description="Historical trades for watched (or explicitly given) AMM pools.",
    )

    def load(
        self,
        *,
        since: datetime,
        until: datetime,
        pool_ids: list[str] | None = None,
        **_: object,
    ) -> Iterator[Trade]:
        """Yield trades for each configured (or explicitly given) pool id.

        Args:
            since: Start of the time range (required — no default; a global
                AMM pool history load without bounds would be unbounded).
            until: End of the time range (required).
            pool_ids: Pool ids to load; defaults to `config.WATCHED_AMM_POOLS`.

        Raises:
            ConnectorError: No pool ids were configured or supplied.
        """
        pools = pool_ids if pool_ids is not None else config.WATCHED_AMM_POOLS
        if not pools:
            raise ConnectorError(
                f"Connector '{self.metadata.connector_id}' has no pool ids to load — "
                "pass pool_ids=[...] to load(), or set WATCHED_AMM_POOLS."
            )
        for pool_id in pools:
            yield from iter_amm_pool_trades(pool_id, since, until)

    def to_dataframe(self, records):
        return trades_to_dataframe(records)


@register_connector
class OrderBookConnector(DataConnector[OrderBookEvent]):
    """Order-book create/cancel/update events via Horizon's operations endpoint.

    Wraps `ingestion.orderbook_loader.load_orderbook_events`.
    """

    metadata = ConnectorMetadata(
        connector_id="stellar-orderbook-events",
        record_type=OrderBookEvent,
        source="horizon-rest",
        description="Manage-offer operation history for a set of accounts.",
    )

    def load(
        self, *, account_ids: list[str] | None = None, **_: object
    ) -> Iterator[OrderBookEvent]:
        """Yield order-book events for the given accounts.

        Raises:
            ConnectorError: `account_ids` wasn't supplied — unlike trade
                connectors there is no repo-wide default account set.
        """
        if not account_ids:
            raise ConnectorError(
                f"Connector '{self.metadata.connector_id}' requires "
                "load(account_ids=[...]) — there is no default account set."
            )
        for account_id in account_ids:
            yield from load_orderbook_events(account_id)

    def to_dataframe(self, records):
        return orderbook_events_to_dataframe(records)


@register_connector
class AccountActivityConnector(DataConnector[AccountActivity]):
    """Account-creation / funding-source activity via Horizon's effects endpoint.

    Wraps `ingestion.account_activity_loader.load_accounts_activity`.
    """

    metadata = ConnectorMetadata(
        connector_id="stellar-account-activity",
        record_type=AccountActivity,
        source="horizon-rest",
        description="Funding-account lookups feeding the wallet funding graph.",
    )

    def load(
        self, *, account_ids: list[str] | None = None, **_: object
    ) -> Iterator[AccountActivity]:
        """Yield activity records for the given accounts.

        Per-account lookup failures are logged and skipped (see
        `load_accounts_activity`) rather than aborting the whole batch.

        Raises:
            ConnectorError: `account_ids` wasn't supplied.
        """
        if not account_ids:
            raise ConnectorError(
                f"Connector '{self.metadata.connector_id}' requires "
                "load(account_ids=[...]) — there is no default account set."
            )
        yield from load_accounts_activity(account_ids)
