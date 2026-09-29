"""Storage abstraction boundaries for generated datasets.

Issue #529 — Stellar Wave advanced build.

Provides a ``DatasetStore`` protocol plus two concrete implementations:

``LocalDatasetStore``
    Reads / writes Parquet (or CSV / JSON) files on the local filesystem.
    Suitable for development, CI, and single-node deployments.

``ObjectStoreDatasetStore``
    Thin wrapper around ``fsspec`` (which supports S3, GCS, Azure Blob,
    SFTP, …).  Falls back gracefully to ``LocalDatasetStore`` when
    ``DATASET_STORE_BACKEND=local`` or when ``fsspec`` is not installed,
    so development environments need no extra dependencies.

All implementations share the same interface so callers are fully
decoupled from the underlying storage engine.

Configuration:

.. code-block:: bash

    DATASET_STORE_BACKEND=local            # "local" (default) | "object"
    DATASET_STORE_BASE_PATH=./data         # root dir / bucket prefix
    DATASET_STORE_FORMAT=parquet           # "parquet" | "csv" | "json"
    DATASET_STORE_OBJECT_STORE_URL=s3://my-bucket/ledgerlens   # for "object" backend

Usage::

    from ingestion.dataset_store import build_dataset_store

    store = build_dataset_store()

    # Persist a labelled feature matrix
    store.save(df, "synthetic_dataset")

    # Load it back
    df2 = store.load("synthetic_dataset")

    # Check existence
    if store.exists("synthetic_dataset"):
        ...

    # List all stored datasets
    names = store.list_datasets()

    # Delete
    store.delete("synthetic_dataset")
"""

from __future__ import annotations

import io
import logging
import os
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from config import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_BACKEND: str = getattr(config, "DATASET_STORE_BACKEND", "local").lower()
_BASE_PATH: str = getattr(config, "DATASET_STORE_BASE_PATH", "./data")
_FORMAT: str = getattr(config, "DATASET_STORE_FORMAT", "parquet").lower()
_OBJECT_STORE_URL: str = getattr(config, "DATASET_STORE_OBJECT_STORE_URL", "")

_VALID_FORMATS = frozenset({"parquet", "csv", "json"})
_VALID_BACKENDS = frozenset({"local", "object"})

# Sanitise dataset names to filesystem-safe strings
_NAME_RE = re.compile(r"[^a-zA-Z0-9_\-\.]")


def _sanitise_name(name: str) -> str:
    return _NAME_RE.sub("_", name)


# ---------------------------------------------------------------------------
# Protocol (structural typing contract)
# ---------------------------------------------------------------------------


@runtime_checkable
class DatasetStore(Protocol):
    """Abstract storage interface for labelled / raw Parquet datasets.

    All concrete implementations must satisfy this protocol so callers can
    type-annotate against ``DatasetStore`` without importing a specific backend.
    """

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        fmt: str | None = None,
        overwrite: bool = True,
    ) -> str:
        """Persist *df* under *name*.

        Parameters
        ----------
        df:
            DataFrame to persist.
        name:
            Logical dataset name (no path separators or extension needed).
        fmt:
            Override the configured storage format (``"parquet"`` / ``"csv"``
            / ``"json"``).  Defaults to :data:`DATASET_STORE_FORMAT`.
        overwrite:
            If ``False`` and the dataset already exists, raise
            ``FileExistsError``.

        Returns
        -------
        str
            The canonical path / URI where the dataset was written.
        """
        ...

    def load(
        self,
        name: str,
        *,
        fmt: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Load a previously-saved dataset by name.

        Parameters
        ----------
        name:
            Logical dataset name.
        fmt:
            Override the configured format for reading.
        columns:
            Subset of columns to load (Parquet predicate pushdown).

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        FileNotFoundError:
            If *name* does not exist in the store.
        """
        ...

    def exists(self, name: str, *, fmt: str | None = None) -> bool:
        """Return ``True`` if a dataset with *name* exists in the store."""
        ...

    def delete(self, name: str, *, fmt: str | None = None) -> None:
        """Remove a dataset from the store.  No-op if not found."""
        ...

    def list_datasets(self) -> list[str]:
        """Return the logical names of all datasets in the store."""
        ...


# ---------------------------------------------------------------------------
# Shared format helpers
# ---------------------------------------------------------------------------


def _extension(fmt: str) -> str:
    return f".{fmt}"


def _write_bytes(df: pd.DataFrame, fmt: str) -> bytes:
    """Serialise *df* to bytes in the given format."""
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    elif fmt == "csv":
        df.to_csv(buf, index=False)
    elif fmt == "json":
        df.to_json(buf, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")
    return buf.getvalue()


def _read_bytes(data: bytes, fmt: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Deserialise *data* in the given format."""
    buf = io.BytesIO(data)
    if fmt == "parquet":
        return pd.read_parquet(buf, columns=columns)
    elif fmt == "csv":
        df = pd.read_csv(buf)
        return df[columns] if columns else df
    elif fmt == "json":
        df = pd.read_json(buf, orient="records", lines=True)
        return df[columns] if columns else df
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalDatasetStore:
    """Reads / writes datasets as files on the local filesystem.

    Parameters
    ----------
    base_path:
        Root directory for all datasets.  Created automatically if absent.
    fmt:
        Default file format (``"parquet"`` / ``"csv"`` / ``"json"``).
    """

    def __init__(
        self,
        base_path: str | None = None,
        *,
        fmt: str | None = None,
    ) -> None:
        self._base = Path(base_path or _BASE_PATH)
        self._fmt = (fmt or _FORMAT).lower()
        if self._fmt not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported format {self._fmt!r}. Choose from: {sorted(_VALID_FORMATS)}"
            )
        self._base.mkdir(parents=True, exist_ok=True)
        logger.debug("LocalDatasetStore initialised at %s (fmt=%s)", self._base, self._fmt)

    def _path_for(self, name: str, fmt: str) -> Path:
        return self._base / f"{_sanitise_name(name)}{_extension(fmt)}"

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        fmt: str | None = None,
        overwrite: bool = True,
    ) -> str:
        eff_fmt = (fmt or self._fmt).lower()
        if eff_fmt not in _VALID_FORMATS:
            raise ValueError(f"Unsupported format: {eff_fmt!r}")

        dest = self._path_for(name, eff_fmt)
        if not overwrite and dest.exists():
            raise FileExistsError(f"Dataset {name!r} already exists at {dest}")

        data = _write_bytes(df, eff_fmt)
        dest.write_bytes(data)
        logger.info(
            "LocalDatasetStore.save: %d rows → %s (%d bytes)",
            len(df),
            dest,
            len(data),
        )
        return str(dest)

    def load(
        self,
        name: str,
        *,
        fmt: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        eff_fmt = (fmt or self._fmt).lower()
        path = self._path_for(name, eff_fmt)
        if not path.exists():
            raise FileNotFoundError(f"Dataset {name!r} not found at {path}")
        df = _read_bytes(path.read_bytes(), eff_fmt, columns)
        logger.debug("LocalDatasetStore.load: %s → %d rows", path, len(df))
        return df

    def exists(self, name: str, *, fmt: str | None = None) -> bool:
        eff_fmt = (fmt or self._fmt).lower()
        return self._path_for(name, eff_fmt).exists()

    def delete(self, name: str, *, fmt: str | None = None) -> None:
        eff_fmt = (fmt or self._fmt).lower()
        path = self._path_for(name, eff_fmt)
        if path.exists():
            path.unlink()
            logger.info("LocalDatasetStore.delete: removed %s", path)

    def list_datasets(self) -> list[str]:
        names: list[str] = []
        for fmt in _VALID_FORMATS:
            for p in self._base.glob(f"*{_extension(fmt)}"):
                names.append(p.stem)
        return sorted(set(names))

    @property
    def base_path(self) -> str:
        """The absolute base path where datasets are stored."""
        return str(self._base.resolve())

    def __repr__(self) -> str:
        return f"LocalDatasetStore(base_path={self._base!r}, fmt={self._fmt!r})"


# ---------------------------------------------------------------------------
# Object store backend (fsspec)
# ---------------------------------------------------------------------------


class ObjectStoreDatasetStore:
    """Reads / writes datasets via ``fsspec`` (S3, GCS, Azure Blob, SFTP, …).

    Falls back to ``LocalDatasetStore`` if ``fsspec`` is not installed or the
    URL is empty, logging a warning so deployments without cloud storage work
    out of the box.

    Parameters
    ----------
    url:
        Object store root URI, e.g. ``s3://my-bucket/ledgerlens`` or
        ``gcs://my-bucket/datasets``.  When empty, falls back to local.
    fmt:
        Default file format.
    storage_options:
        Extra keyword arguments forwarded to ``fsspec.open`` (credentials,
        endpoint overrides, etc.).
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        fmt: str | None = None,
        storage_options: dict | None = None,
    ) -> None:
        self._url = (url or _OBJECT_STORE_URL).rstrip("/")
        self._fmt = (fmt or _FORMAT).lower()
        self._storage_options: dict = storage_options or {}

        if not self._url:
            logger.warning(
                "ObjectStoreDatasetStore: no URL configured; falling back to LocalDatasetStore"
            )
            self._fallback: LocalDatasetStore | None = LocalDatasetStore(fmt=self._fmt)
        else:
            self._fallback = None
            try:
                import fsspec  # noqa: F401  # check import only

                self._fsspec_available = True
            except ImportError:
                logger.warning(
                    "ObjectStoreDatasetStore: fsspec not installed; "
                    "falling back to LocalDatasetStore"
                )
                self._fallback = LocalDatasetStore(fmt=self._fmt)
                self._fsspec_available = False

        if self._fmt not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported format {self._fmt!r}. Choose from: {sorted(_VALID_FORMATS)}"
            )

    def _uri_for(self, name: str, fmt: str) -> str:
        return f"{self._url}/{_sanitise_name(name)}{_extension(fmt)}"

    def save(
        self,
        df: pd.DataFrame,
        name: str,
        *,
        fmt: str | None = None,
        overwrite: bool = True,
    ) -> str:
        if self._fallback is not None:
            return self._fallback.save(df, name, fmt=fmt, overwrite=overwrite)

        import fsspec

        eff_fmt = (fmt or self._fmt).lower()
        uri = self._uri_for(name, eff_fmt)

        if not overwrite:
            fs, path = fsspec.core.url_to_fs(uri, **self._storage_options)
            if fs.exists(path):
                raise FileExistsError(f"Dataset {name!r} already exists at {uri}")

        data = _write_bytes(df, eff_fmt)
        with fsspec.open(uri, "wb", **self._storage_options) as f:
            f.write(data)
        logger.info(
            "ObjectStoreDatasetStore.save: %d rows → %s (%d bytes)",
            len(df),
            uri,
            len(data),
        )
        return uri

    def load(
        self,
        name: str,
        *,
        fmt: str | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        if self._fallback is not None:
            return self._fallback.load(name, fmt=fmt, columns=columns)

        import fsspec

        eff_fmt = (fmt or self._fmt).lower()
        uri = self._uri_for(name, eff_fmt)
        try:
            with fsspec.open(uri, "rb", **self._storage_options) as f:
                data = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset {name!r} not found at {uri}") from None
        df = _read_bytes(data, eff_fmt, columns)
        logger.debug("ObjectStoreDatasetStore.load: %s → %d rows", uri, len(df))
        return df

    def exists(self, name: str, *, fmt: str | None = None) -> bool:
        if self._fallback is not None:
            return self._fallback.exists(name, fmt=fmt)

        import fsspec

        eff_fmt = (fmt or self._fmt).lower()
        uri = self._uri_for(name, eff_fmt)
        fs, path = fsspec.core.url_to_fs(uri, **self._storage_options)
        return fs.exists(path)

    def delete(self, name: str, *, fmt: str | None = None) -> None:
        if self._fallback is not None:
            return self._fallback.delete(name, fmt=fmt)

        import fsspec

        eff_fmt = (fmt or self._fmt).lower()
        uri = self._uri_for(name, eff_fmt)
        fs, path = fsspec.core.url_to_fs(uri, **self._storage_options)
        if fs.exists(path):
            fs.rm(path)
            logger.info("ObjectStoreDatasetStore.delete: removed %s", uri)

    def list_datasets(self) -> list[str]:
        if self._fallback is not None:
            return self._fallback.list_datasets()

        import fsspec

        fs, base_path = fsspec.core.url_to_fs(self._url, **self._storage_options)
        try:
            entries = fs.ls(base_path, detail=False)
        except FileNotFoundError:
            return []

        names: set[str] = set()
        for entry in entries:
            stem = os.path.basename(entry)
            for fmt in _VALID_FORMATS:
                if stem.endswith(_extension(fmt)):
                    names.add(stem[: -len(_extension(fmt))])
        return sorted(names)

    def __repr__(self) -> str:
        if self._fallback:
            return f"ObjectStoreDatasetStore(fallback={self._fallback!r})"
        return f"ObjectStoreDatasetStore(url={self._url!r}, fmt={self._fmt!r})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_dataset_store(
    backend: str | None = None,
    base_path: str | None = None,
    fmt: str | None = None,
    object_store_url: str | None = None,
    storage_options: dict | None = None,
) -> LocalDatasetStore | ObjectStoreDatasetStore:
    """Instantiate and return the configured dataset store.

    Parameters
    ----------
    backend:
        ``"local"`` or ``"object"``.  Defaults to ``DATASET_STORE_BACKEND``
        env var (``"local"`` if unset).
    base_path:
        Local root dir.  Used by ``LocalDatasetStore`` and as fallback for
        ``ObjectStoreDatasetStore``.
    fmt:
        Storage format override.
    object_store_url:
        Object store URI for the ``"object"`` backend.
    storage_options:
        fsspec storage options for the ``"object"`` backend.

    Returns
    -------
    LocalDatasetStore or ObjectStoreDatasetStore

    Example::

        store = build_dataset_store()
        store.save(df, "synthetic_dataset")
        df2 = store.load("synthetic_dataset")
    """
    eff_backend = (backend or _BACKEND).lower()
    if eff_backend not in _VALID_BACKENDS:
        raise ValueError(f"Unknown backend {eff_backend!r}. Choose from: {sorted(_VALID_BACKENDS)}")

    if eff_backend == "local":
        return LocalDatasetStore(base_path=base_path, fmt=fmt)
    else:
        return ObjectStoreDatasetStore(
            url=object_store_url,
            fmt=fmt,
            storage_options=storage_options,
        )
