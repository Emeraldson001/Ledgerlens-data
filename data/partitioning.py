"""Partitioning strategies for large historical datasets — Issue #527.

Provides a reusable, strategy-based API for writing and reading partitioned
Parquet datasets from the LedgerLens data pipeline.  Three built-in strategies
are included:

* ``TimePartitionStrategy``   — partition by calendar period (year/month/day/hour)
* ``PairPartitionStrategy``   — partition by asset-pair identifier
* ``WalletPartitionStrategy`` — partition by wallet address prefix (bucket width
                                 configurable to avoid creating O(N) partitions)

Usage example
-------------
>>> writer = PartitionedDatasetWriter(
...     root=Path("data/partitioned"),
...     strategy=TimePartitionStrategy(period="month"),
... )
>>> writer.write(df)           # appends to correct partition directory
>>> reader = PartitionedDatasetReader(root=Path("data/partitioned"))
>>> df = reader.read(filters=[("year", "==", "2024"), ("month", "==", "06")])

Design notes
------------
* Partition directories are written atomically: data is written to a
  ``<partition>._tmp`` directory first and then renamed to avoid readers
  seeing partial writes.
* Every partition directory carries a ``_metadata.json`` sidecar with the
  strategy name, parameters, schema hash, creation/update timestamps, and row
  count — useful for validation and drift detection.
* Pruning removes partition directories whose last-updated timestamp is older
  than a caller-supplied ``max_age_days`` value; a dry-run mode lists candidates
  without deleting.
* All public classes and functions are fully type-annotated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "PartitionStrategy",
    "TimePartitionStrategy",
    "PairPartitionStrategy",
    "WalletPartitionStrategy",
    "PartitionedDatasetWriter",
    "PartitionedDatasetReader",
    "PartitionMetadata",
    "prune_partitions",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNSAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def _safe_path_component(value: str) -> str:
    """Sanitise *value* so it can be used as a filesystem path component."""
    return _UNSAFE_PATH_RE.sub("_", value)


def _schema_hash(df: pd.DataFrame) -> str:
    """Return a short SHA-256 hex digest of the sorted column list."""
    columns_str = ",".join(sorted(df.columns.tolist()))
    return hashlib.sha256(columns_str.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Partition strategies
# ---------------------------------------------------------------------------


class PartitionStrategy(ABC):
    """Abstract base for all partitioning strategies.

    Sub-classes must implement :meth:`partition_key` which maps a DataFrame
    row to a *relative* partition path such as ``"2024/06"`` or
    ``"USDC_XLM/2024"``.  They may also implement :meth:`add_partition_columns`
    to inject the derived key columns into the DataFrame before writing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable identifier used in metadata and partition directory names."""

    @abstractmethod
    def partition_key(self, row: pd.Series) -> str:
        """Return the relative partition path for *row*."""

    def add_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return *df* enriched with any strategy-specific columns.

        Default implementation is a no-op; override to inject columns such as
        ``year``, ``month``, ``pair_key``, etc.
        """
        return df

    def to_dict(self) -> dict[str, Any]:
        """Serialise strategy parameters for metadata storage."""
        return {"name": self.name}


class TimePartitionStrategy(PartitionStrategy):
    """Partition by calendar period derived from a timestamp column.

    Parameters
    ----------
    period:
        Granularity of the partition.  One of ``"year"``, ``"month"``
        (default), ``"day"``, or ``"hour"``.
    timestamp_column:
        Name of the timestamp column in the DataFrame.  Defaults to
        ``"timestamp"``.  The column is expected to be parseable by
        ``pd.to_datetime``.
    """

    _VALID_PERIODS = frozenset({"year", "month", "day", "hour"})

    def __init__(
        self,
        period: str = "month",
        timestamp_column: str = "timestamp",
    ) -> None:
        if period not in self._VALID_PERIODS:
            raise ValueError(
                f"Invalid period {period!r}; choose one of {sorted(self._VALID_PERIODS)}"
            )
        self._period = period
        self._timestamp_column = timestamp_column

    @property
    def name(self) -> str:
        return "time"

    def partition_key(self, row: pd.Series) -> str:
        ts = pd.to_datetime(row[self._timestamp_column], utc=True)
        if self._period == "year":
            return f"{ts.year}"
        if self._period == "month":
            return f"{ts.year}/{ts.month:02d}"
        if self._period == "day":
            return f"{ts.year}/{ts.month:02d}/{ts.day:02d}"
        # hour
        return f"{ts.year}/{ts.month:02d}/{ts.day:02d}/{ts.hour:02d}"

    def add_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ts = pd.to_datetime(df[self._timestamp_column], utc=True)
        df["year"] = ts.dt.year.astype(str)
        df["month"] = ts.dt.month.map(lambda m: f"{m:02d}")
        if self._period in {"day", "hour"}:
            df["day"] = ts.dt.day.map(lambda d: f"{d:02d}")
        if self._period == "hour":
            df["hour"] = ts.dt.hour.map(lambda h: f"{h:02d}")
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "period": self._period,
            "timestamp_column": self._timestamp_column,
        }


class PairPartitionStrategy(PartitionStrategy):
    """Partition by asset-pair identifier.

    The pair column value is sanitised so special characters (``/``, ``:``)
    become underscores, making the value safe as a directory name.

    Parameters
    ----------
    pair_column:
        Name of the column holding the asset-pair string.  Defaults to
        ``"pair_id"``.
    include_time:
        If ``True``, appends a year/month sub-path to each pair partition so
        large pairs don't accumulate unbounded files.  Defaults to ``False``.
    timestamp_column:
        Only used when *include_time* is ``True``.
    """

    def __init__(
        self,
        pair_column: str = "pair_id",
        include_time: bool = False,
        timestamp_column: str = "timestamp",
    ) -> None:
        self._pair_column = pair_column
        self._include_time = include_time
        self._timestamp_column = timestamp_column

    @property
    def name(self) -> str:
        return "pair"

    def partition_key(self, row: pd.Series) -> str:
        pair = _safe_path_component(str(row[self._pair_column]))
        if self._include_time:
            ts = pd.to_datetime(row[self._timestamp_column], utc=True)
            return f"{pair}/{ts.year}/{ts.month:02d}"
        return pair

    def add_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["pair_key"] = df[self._pair_column].apply(_safe_path_component)
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pair_column": self._pair_column,
            "include_time": self._include_time,
            "timestamp_column": self._timestamp_column,
        }


class WalletPartitionStrategy(PartitionStrategy):
    """Partition by wallet-address prefix bucket.

    Rather than creating one directory per wallet (which could be millions),
    wallets are bucketed by the first ``prefix_length`` characters of their
    address, optionally combined with a time sub-partition.

    Parameters
    ----------
    wallet_column:
        Column containing the wallet address.  Defaults to ``"wallet"``.
    prefix_length:
        Number of leading characters used as the bucket key (default 2).
    include_time:
        Append a ``year/month`` sub-path under the wallet bucket.
    timestamp_column:
        Only used when *include_time* is ``True``.
    """

    def __init__(
        self,
        wallet_column: str = "wallet",
        prefix_length: int = 2,
        include_time: bool = False,
        timestamp_column: str = "timestamp",
    ) -> None:
        if prefix_length < 1 or prefix_length > 8:
            raise ValueError("prefix_length must be between 1 and 8")
        self._wallet_column = wallet_column
        self._prefix_length = prefix_length
        self._include_time = include_time
        self._timestamp_column = timestamp_column

    @property
    def name(self) -> str:
        return "wallet"

    def partition_key(self, row: pd.Series) -> str:
        wallet = str(row[self._wallet_column])
        bucket = _safe_path_component(wallet[: self._prefix_length].upper())
        if self._include_time:
            ts = pd.to_datetime(row[self._timestamp_column], utc=True)
            return f"{bucket}/{ts.year}/{ts.month:02d}"
        return bucket

    def add_partition_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["wallet_bucket"] = df[self._wallet_column].apply(
            lambda w: _safe_path_component(str(w)[: self._prefix_length].upper())
        )
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "wallet_column": self._wallet_column,
            "prefix_length": self._prefix_length,
            "include_time": self._include_time,
            "timestamp_column": self._timestamp_column,
        }


# ---------------------------------------------------------------------------
# Partition metadata sidecar
# ---------------------------------------------------------------------------


@dataclass
class PartitionMetadata:
    """Sidecar metadata stored alongside each partition directory."""

    strategy_name: str
    strategy_params: dict[str, Any]
    partition_key: str
    schema_hash: str
    row_count: int
    created_at: str
    updated_at: str
    extra: dict[str, Any] = field(default_factory=dict)

    # file name written inside every partition directory
    FILENAME: str = "_metadata.json"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "strategy_params": self.strategy_params,
            "partition_key": self.partition_key,
            "schema_hash": self.schema_hash,
            "row_count": self.row_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PartitionMetadata:
        extra = {
            k: v
            for k, v in data.items()
            if k
            not in {
                "strategy_name",
                "strategy_params",
                "partition_key",
                "schema_hash",
                "row_count",
                "created_at",
                "updated_at",
            }
        }
        return cls(
            strategy_name=data["strategy_name"],
            strategy_params=data.get("strategy_params", {}),
            partition_key=data["partition_key"],
            schema_hash=data.get("schema_hash", ""),
            row_count=data.get("row_count", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            extra=extra,
        )

    @classmethod
    def load(cls, partition_dir: Path) -> PartitionMetadata | None:
        path = partition_dir / cls.FILENAME
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, partition_dir: Path) -> None:
        partition_dir.mkdir(parents=True, exist_ok=True)
        path = partition_dir / self.FILENAME
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class PartitionedDatasetWriter:
    """Write DataFrames to a partitioned Parquet dataset.

    Each call to :meth:`write` groups *df* by partition key, writes each
    group to ``<root>/<partition_key>/data_<shard>.parquet`` atomically via a
    temp-file rename, and updates the partition's ``_metadata.json`` sidecar.

    Parameters
    ----------
    root:
        Root directory under which partitions are created.
    strategy:
        Partitioning strategy instance that determines the partition key for
        each row.
    compression:
        Parquet compression codec.  Defaults to ``"snappy"``.
    add_partition_columns:
        If ``True``, the strategy-derived key columns (``year``, ``month``,
        ``pair_key``, etc.) are included in the written Parquet files.
        Defaults to ``True``.
    """

    def __init__(
        self,
        root: Path | str,
        strategy: PartitionStrategy,
        compression: str = "snappy",
        add_partition_columns: bool = True,
    ) -> None:
        self.root = Path(root)
        self.strategy = strategy
        self.compression = compression
        self.add_partition_columns = add_partition_columns
        self.root.mkdir(parents=True, exist_ok=True)
        self._write_root_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, df: pd.DataFrame) -> dict[str, int]:
        """Partition *df* and write each group to the appropriate directory.

        Returns a mapping of ``{partition_key: rows_written}``.

        Raises
        ------
        ValueError
            If *df* is empty.
        """
        if df.empty:
            raise ValueError("Cannot write an empty DataFrame.")

        if self.add_partition_columns:
            df = self.strategy.add_partition_columns(df)

        # Compute per-row partition keys
        keys = df.apply(self.strategy.partition_key, axis=1)
        result: dict[str, int] = {}

        for key, group in df.groupby(keys, sort=False):
            key = str(key)
            self._write_group(key, group)
            result[key] = len(group)
            logger.debug("wrote %d rows to partition %r", len(group), key)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partition_dir(self, key: str) -> Path:
        return self.root / key

    def _write_group(self, key: str, group: pd.DataFrame) -> None:
        partition_dir = self._partition_dir(key)
        partition_dir.mkdir(parents=True, exist_ok=True)

        shard_index = self._next_shard_index(partition_dir)
        target = partition_dir / f"data_{shard_index:05d}.parquet"

        # Atomic write via temp file + rename
        tmp_dir = partition_dir.parent
        with tempfile.NamedTemporaryFile(dir=tmp_dir, suffix=".parquet.tmp", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            group.to_parquet(tmp_path, index=False, compression=self.compression)
            tmp_path.rename(target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        self._update_metadata(key, partition_dir, group)

    def _next_shard_index(self, partition_dir: Path) -> int:
        existing = list(partition_dir.glob("data_*.parquet"))
        if not existing:
            return 0
        indices = []
        for p in existing:
            m = re.match(r"data_(\d+)\.parquet$", p.name)
            if m:
                indices.append(int(m.group(1)))
        return max(indices) + 1 if indices else 0

    def _update_metadata(self, key: str, partition_dir: Path, group: pd.DataFrame) -> None:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        existing = PartitionMetadata.load(partition_dir)
        if existing is None:
            meta = PartitionMetadata(
                strategy_name=self.strategy.name,
                strategy_params=self.strategy.to_dict(),
                partition_key=key,
                schema_hash=_schema_hash(group),
                row_count=len(group),
                created_at=now,
                updated_at=now,
            )
        else:
            meta = existing
            meta.row_count += len(group)
            meta.updated_at = now
            new_hash = _schema_hash(group)
            if meta.schema_hash != new_hash:
                logger.warning(
                    "schema hash changed for partition %r: %s -> %s",
                    key,
                    meta.schema_hash,
                    new_hash,
                )
                meta.schema_hash = new_hash
        meta.save(partition_dir)

    def _write_root_manifest(self) -> None:
        manifest_path = self.root / "_strategy.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.strategy.to_dict(), f, indent=2)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class PartitionedDatasetReader:
    """Read partitioned Parquet datasets written by :class:`PartitionedDatasetWriter`.

    Parameters
    ----------
    root:
        Root directory of the partitioned dataset.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Partition root does not exist: {self.root}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(
        self,
        filters: list[tuple[str, str, str]] | None = None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Read all matching partitions and return a concatenated DataFrame.

        Parameters
        ----------
        filters:
            Optional list of ``(column, op, value)`` triples used to select
            partition directories.  The *column* must be one of the key-column
            names injected by the strategy (``"year"``, ``"month"``, ``"pair_key"``,
            ``"wallet_bucket"``).  Supported *op* values: ``"=="``, ``"!="``,
            ``"in"``, ``"not in"``.

            Note: filters are applied by matching the *partition directory path*
            or the ``_metadata.json`` sidecar — they are **not** pushed into the
            Parquet file itself.  This is intentionally simple; callers that need
            predicate push-down should use PyArrow directly.
        columns:
            If provided, only these columns are read from each Parquet file.

        Returns
        -------
        pd.DataFrame
            Concatenated DataFrame of all matching rows.  Empty DataFrame if no
            partitions match.

        Raises
        ------
        FileNotFoundError
            If the root directory does not exist.
        """
        partition_dirs = self._discover_partitions()

        if filters:
            partition_dirs = self._apply_filters(partition_dirs, filters)

        if not partition_dirs:
            logger.warning("No matching partitions found under %s", self.root)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for p_dir in sorted(partition_dirs):
            parquet_files = sorted(p_dir.glob("data_*.parquet"))
            for pf in parquet_files:
                try:
                    chunk = pd.read_parquet(pf, columns=columns)
                    frames.append(chunk)
                except Exception as exc:
                    logger.error("Failed to read %s: %s", pf, exc)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def list_partitions(self) -> list[str]:
        """Return relative keys of all discovered partitions."""
        return [str(p.relative_to(self.root)) for p in self._discover_partitions()]

    def get_metadata(self, partition_key: str) -> PartitionMetadata | None:
        """Return sidecar metadata for a given partition key, or ``None``."""
        return PartitionMetadata.load(self.root / partition_key)

    def validate_schema_consistency(self) -> dict[str, list[str]]:
        """Check that all partitions share the same schema hash.

        Returns a dict mapping each observed schema hash to the list of
        partition keys carrying that hash.  A result with more than one key
        indicates schema drift.
        """
        hash_to_keys: dict[str, list[str]] = {}
        for p_dir in self._discover_partitions():
            meta = PartitionMetadata.load(p_dir)
            if meta is None:
                continue
            h = meta.schema_hash
            hash_to_keys.setdefault(h, []).append(str(p_dir.relative_to(self.root)))
        return hash_to_keys

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_partitions(self) -> list[Path]:
        """Return all leaf directories that contain at least one Parquet shard."""
        result: list[Path] = []
        for dirpath, _dirnames, filenames in os.walk(self.root):
            p = Path(dirpath)
            if any(f.startswith("data_") and f.endswith(".parquet") for f in filenames):
                result.append(p)
        return result

    def _apply_filters(
        self,
        partition_dirs: list[Path],
        filters: list[tuple[str, str, str]],
    ) -> list[Path]:
        """Filter *partition_dirs* by matching sidecar metadata."""
        matched: list[Path] = []
        for p_dir in partition_dirs:
            meta = PartitionMetadata.load(p_dir)
            if meta is None:
                # No metadata — fall back to path-based matching
                meta_dict: dict[str, Any] = {}
            else:
                meta_dict = meta.to_dict()

            # Also expose path components as pseudo-columns for matching
            rel = str(p_dir.relative_to(self.root))
            parts = rel.replace("\\", "/").split("/")
            # Add strategy-specific columns from meta where available
            # plus the raw path parts indexed by position
            combined: dict[str, Any] = {
                **meta_dict,
                "_partition_key": rel,
                "_parts": parts,
            }

            if self._row_matches_filters(combined, filters):
                matched.append(p_dir)
        return matched

    @staticmethod
    def _row_matches_filters(
        row: dict[str, Any],
        filters: list[tuple[str, str, str]],
    ) -> bool:
        for col, op, value in filters:
            cell = row.get(col)
            if cell is None:
                # Try matching against the partition_key string
                cell = row.get("_partition_key", "")
            cell_str = str(cell)
            if op == "==":
                if cell_str != str(value):
                    return False
            elif op == "!=":
                if cell_str == str(value):
                    return False
            elif op == "in":
                vals = value if isinstance(value, (list, tuple, set)) else [value]
                if cell_str not in [str(v) for v in vals]:
                    return False
            elif op == "not in":
                vals = value if isinstance(value, (list, tuple, set)) else [value]
                if cell_str in [str(v) for v in vals]:
                    return False
            else:
                raise ValueError(f"Unsupported filter operator: {op!r}")
        return True


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def prune_partitions(
    root: Path | str,
    max_age_days: float,
    dry_run: bool = False,
) -> list[str]:
    """Remove partition directories that have not been updated recently.

    Parameters
    ----------
    root:
        Root of the partitioned dataset.
    max_age_days:
        Partitions whose ``updated_at`` timestamp is older than this many
        days (relative to now) will be deleted.
    dry_run:
        If ``True``, return the list of candidates without deleting anything.

    Returns
    -------
    list[str]
        Relative paths of partitions that were pruned (or would be pruned in
        dry-run mode).

    Raises
    ------
    ValueError
        If *max_age_days* is not positive.
    """
    if max_age_days <= 0:
        raise ValueError("max_age_days must be positive")

    root = Path(root)
    reader = PartitionedDatasetReader(root)
    cutoff = datetime.now(UTC) - timedelta(days=max_age_days)
    pruned: list[str] = []

    for p_dir in reader._discover_partitions():
        meta = PartitionMetadata.load(p_dir)
        if meta is None:
            logger.debug("Skipping %s — no metadata", p_dir)
            continue

        try:
            updated_at = datetime.fromisoformat(meta.updated_at.replace("Z", "+00:00"))
        except ValueError:
            logger.warning("Could not parse updated_at for %s; skipping", p_dir)
            continue

        if updated_at < cutoff:
            rel = str(p_dir.relative_to(root))
            pruned.append(rel)
            if dry_run:
                logger.info("[dry-run] would prune partition %r", rel)
            else:
                logger.info("Pruning partition %r (last updated %s)", rel, updated_at.isoformat())
                shutil.rmtree(p_dir)

    return pruned
