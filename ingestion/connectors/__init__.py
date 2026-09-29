"""Plugin boundary for LedgerLens data connectors.

Importing this package registers every built-in connector (SDEX trades, AMM
pool trades, order-book events, account activity) and exposes the shared
`registry` used to look connectors up by id. See `docs/connectors.md` for
the full contract and a guide to adding a new connector — in-tree or as an
installable out-of-tree plugin.

Typical usage::

    from ingestion.connectors import registry

    connector = registry.create("stellar-sdex-trades")
    trades = list(connector.load(since=some_datetime))
    df = connector.to_dataframe(trades)
"""

# Imported for registration side effects: each built-in connector class
# registers itself with `registry` at import time via @register_connector.
from ingestion.connectors import builtin  # noqa: E402, F401
from ingestion.connectors.base import (
    ConnectorConfigError,
    ConnectorError,
    ConnectorHealth,
    ConnectorMetadata,
    ConnectorNotFoundError,
    DataConnector,
    DuplicateConnectorError,
)
from ingestion.connectors.registry import ConnectorRegistry, register_connector, registry

__all__ = [
    "ConnectorConfigError",
    "ConnectorError",
    "ConnectorHealth",
    "ConnectorMetadata",
    "ConnectorNotFoundError",
    "ConnectorRegistry",
    "DataConnector",
    "DuplicateConnectorError",
    "register_connector",
    "registry",
]
