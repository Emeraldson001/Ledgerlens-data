"""Registry for LedgerLens data connectors — the plugin lookup boundary.

Two ways to make a connector available through the registry:

1. **In-tree**: define the class anywhere under ``ingestion/`` and decorate
   it with ``@register_connector`` (see ``ingestion/connectors/builtin.py``
   for the reference examples), then import that module from
   ``ingestion/connectors/__init__.py`` so the decorator runs.

2. **Out-of-tree plugin**: ship a separate installable Python package that
   declares an entry point in the ``ledgerlens.connectors`` group, pointing
   at a ``DataConnector`` subclass:

   .. code-block:: toml

       # a third-party package's pyproject.toml
       [project.entry-points."ledgerlens.connectors"]
       my-exchange-trades = "my_package.connectors:MyExchangeTradeConnector"

   No change to ``ledgerlens-data`` source is required — installing the
   plugin package into the same environment is enough. ``registry.get()``
   / ``registry.list_metadata()`` discover it automatically the first time
   they're called. See ``docs/connectors.md`` for a full worked example.
"""

from __future__ import annotations

import inspect
import logging
from importlib import metadata as importlib_metadata

from ingestion.connectors.base import (
    ConnectorMetadata,
    ConnectorNotFoundError,
    DataConnector,
    DuplicateConnectorError,
)

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "ledgerlens.connectors"


class ConnectorRegistry:
    """Holds connector classes keyed by ``ConnectorMetadata.connector_id``."""

    def __init__(self) -> None:
        self._connectors: dict[str, type[DataConnector]] = {}
        self._plugins_discovered = False

    def register(
        self, connector_cls: type[DataConnector], *, replace: bool = False
    ) -> type[DataConnector]:
        """Register a connector class. Returns it unchanged (decorator-friendly).

        Raises:
            TypeError: ``connector_cls`` isn't a concrete ``DataConnector``
                subclass with a valid ``metadata`` attribute.
            DuplicateConnectorError: another class is already registered
                under the same ``connector_id`` and ``replace`` is False.
        """
        if not (inspect.isclass(connector_cls) and issubclass(connector_cls, DataConnector)):
            raise TypeError(f"{connector_cls!r} must be a subclass of DataConnector")
        if inspect.isabstract(connector_cls):
            raise TypeError(
                f"{connector_cls.__name__} has unimplemented abstract methods "
                "(did you forget to implement load()?) and cannot be registered"
            )

        metadata = getattr(connector_cls, "metadata", None)
        if not isinstance(metadata, ConnectorMetadata):
            raise TypeError(
                f"{connector_cls.__name__} must define a class-level "
                "`metadata = ConnectorMetadata(...)`"
            )

        connector_id = metadata.connector_id
        existing = self._connectors.get(connector_id)
        if existing is not None and existing is not connector_cls and not replace:
            raise DuplicateConnectorError(
                f"Connector id {connector_id!r} is already registered to "
                f"{existing.__module__}.{existing.__qualname__} — cannot also "
                f"register it to {connector_cls.__module__}.{connector_cls.__qualname__}. "
                "Give the new connector a unique connector_id, or pass "
                "replace=True if overriding the existing one is intentional."
            )

        self._connectors[connector_id] = connector_cls
        return connector_cls

    def unregister(self, connector_id: str) -> None:
        """Remove a connector id. Mainly useful for test isolation."""
        self._connectors.pop(connector_id, None)

    def discover_plugins(self) -> list[str]:
        """Load connectors published via the ``ledgerlens.connectors`` entry
        point group. Returns the connector ids that were newly registered.

        A single misbehaving plugin (import error, wrong base class, id
        collision) is logged and skipped so it can't take down discovery
        for every other installed plugin.
        """
        loaded: list[str] = []
        try:
            entry_points = importlib_metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
        except (
            Exception
        ) as exc:  # pragma: no cover - defensive, environment-dependent  # noqa: BLE001
            # Broad catch justified: entry_points() call can fail on import machinery
            # issues or corrupted environment. Plugin discovery failure is not fatal;
            # return empty list so built-in connectors still work.
            logger.warning("Connector plugin discovery failed: %s", exc)
            return loaded

        for entry_point in entry_points:
            try:
                connector_cls = entry_point.load()
                self.register(connector_cls)
            except Exception as exc:  # noqa: BLE001
                # Broad catch justified: plugin loading can fail on import errors,
                # missing dependencies, or malformed metadata. One bad plugin must not
                # prevent discovery of other plugins; log and skip.
                logger.warning(
                    "Skipping connector plugin entry point %r (%s): %s",
                    entry_point.name,
                    getattr(entry_point, "value", "?"),
                    exc,
                )
                continue
            loaded.append(connector_cls.metadata.connector_id)
            logger.info(
                "Loaded connector plugin %r from entry point %r",
                connector_cls.metadata.connector_id,
                entry_point.name,
            )
        return loaded

    def _ensure_plugins_discovered(self) -> None:
        if self._plugins_discovered:
            return
        self._plugins_discovered = True
        self.discover_plugins()

    def get(self, connector_id: str) -> type[DataConnector]:
        """Look up a registered connector class by id.

        Raises:
            ConnectorNotFoundError: no connector is registered under
                ``connector_id``. The message lists every id that *is*
                available so the fix is obvious from the error alone.
        """
        self._ensure_plugins_discovered()
        try:
            return self._connectors[connector_id]
        except KeyError:
            available = ", ".join(sorted(self._connectors)) or "(none registered)"
            raise ConnectorNotFoundError(
                f"No connector registered as {connector_id!r}. Available: {available}"
            ) from None

    def create(self, connector_id: str, **init_kwargs: object) -> DataConnector:
        """Look up, instantiate, and config-validate a connector in one call."""
        connector_cls = self.get(connector_id)
        instance = connector_cls(**init_kwargs)
        instance.validate_config()
        return instance

    def list_metadata(self) -> list[ConnectorMetadata]:
        """Return metadata for every registered connector (builtin + plugins)."""
        self._ensure_plugins_discovered()
        return [cls.metadata for cls in self._connectors.values()]


registry = ConnectorRegistry()


def register_connector(connector_cls: type[DataConnector]) -> type[DataConnector]:
    """Class decorator: registers ``connector_cls`` with the global ``registry``.

    Usage::

        @register_connector
        class MyConnector(DataConnector[Trade]):
            metadata = ConnectorMetadata(...)
            def load(self, **kwargs): ...
    """
    return registry.register(connector_cls)
