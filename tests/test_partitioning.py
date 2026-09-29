"""Tests for data/partitioning.py — Issue #527.

Covers:
* Write and read round-trips for all three strategies (time, pair, wallet)
* Multi-shard append behaviour
* Pruning (dry-run and live)
* Schema-consistency validation
* Filter-based reads
* Edge cases: empty DataFrame, unknown filter operator
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from data.partitioning import (
    PairPartitionStrategy,
    PartitionedDatasetReader,
    PartitionedDatasetWriter,
    PartitionMetadata,
    TimePartitionStrategy,
    WalletPartitionStrategy,
    prune_partitions,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_trade_df(n: int = 10, month: int = 6) -> pd.DataFrame:
    """Return a minimal trade-like DataFrame for testing."""
    base = datetime(2024, month, 1, tzinfo=UTC)
    return pd.DataFrame(
        {
            "timestamp": [base + timedelta(hours=i) for i in range(n)],
            "wallet": [f"G{'A' * (i % 4 + 1)}{i:04d}" for i in range(n)],
            "pair_id": [
                "USDC:issuer/XLM:native" if i % 2 == 0 else "BTC:issuer/XLM:native"
                for i in range(n)
            ],
            "amount": [100.0 + i for i in range(n)],
            "label": [0] * n,
        }
    )


# ---------------------------------------------------------------------------
# TimePartitionStrategy
# ---------------------------------------------------------------------------


class TestTimePartitionStrategy:
    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="Invalid period"):
            TimePartitionStrategy(period="week")

    def test_year_partition_key(self):
        strat = TimePartitionStrategy(period="year")
        row = pd.Series({"timestamp": "2024-03-15T10:00:00Z"})
        assert strat.partition_key(row) == "2024"

    def test_month_partition_key(self):
        strat = TimePartitionStrategy(period="month")
        row = pd.Series({"timestamp": "2024-03-15T10:00:00Z"})
        assert strat.partition_key(row) == "2024/03"

    def test_day_partition_key(self):
        strat = TimePartitionStrategy(period="day")
        row = pd.Series({"timestamp": "2024-03-05T10:00:00Z"})
        assert strat.partition_key(row) == "2024/03/05"

    def test_hour_partition_key(self):
        strat = TimePartitionStrategy(period="hour")
        row = pd.Series({"timestamp": "2024-03-05T09:00:00Z"})
        assert strat.partition_key(row) == "2024/03/05/09"

    def test_add_partition_columns_month(self):
        strat = TimePartitionStrategy(period="month")
        df = _make_trade_df(4)
        enriched = strat.add_partition_columns(df)
        assert "year" in enriched.columns
        assert "month" in enriched.columns
        assert "day" not in enriched.columns

    def test_add_partition_columns_day(self):
        strat = TimePartitionStrategy(period="day")
        df = _make_trade_df(4)
        enriched = strat.add_partition_columns(df)
        assert "day" in enriched.columns

    def test_to_dict(self):
        strat = TimePartitionStrategy(period="day")
        d = strat.to_dict()
        assert d["name"] == "time"
        assert d["period"] == "day"


class TestPairPartitionStrategy:
    def test_partition_key_sanitises_special_chars(self):
        strat = PairPartitionStrategy()
        row = pd.Series({"pair_id": "USDC:GA5Z/XLM:native"})
        key = strat.partition_key(row)
        assert "/" not in key
        assert ":" not in key

    def test_include_time_adds_subpath(self):
        strat = PairPartitionStrategy(include_time=True)
        row = pd.Series({"pair_id": "USDC:issuer/XLM:native", "timestamp": "2024-06-01"})
        key = strat.partition_key(row)
        assert "2024" in key

    def test_add_partition_columns_injects_pair_key(self):
        strat = PairPartitionStrategy()
        df = _make_trade_df(4)
        enriched = strat.add_partition_columns(df)
        assert "pair_key" in enriched.columns


class TestWalletPartitionStrategy:
    def test_invalid_prefix_length_raises(self):
        with pytest.raises(ValueError, match="prefix_length"):
            WalletPartitionStrategy(prefix_length=0)

    def test_prefix_bucket(self):
        strat = WalletPartitionStrategy(prefix_length=2)
        row = pd.Series({"wallet": "GABCDEF"})
        key = strat.partition_key(row)
        assert key == "GA"

    def test_include_time(self):
        strat = WalletPartitionStrategy(prefix_length=1, include_time=True)
        row = pd.Series({"wallet": "GABCDEF", "timestamp": "2024-06-01"})
        key = strat.partition_key(row)
        assert "2024" in key


# ---------------------------------------------------------------------------
# PartitionedDatasetWriter
# ---------------------------------------------------------------------------


class TestPartitionedDatasetWriter:
    def test_write_creates_parquet_files(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=TimePartitionStrategy(period="month"),
        )
        df = _make_trade_df(10)
        result = writer.write(df)
        assert len(result) == 1  # all rows in same month (June 2024)
        assert sum(result.values()) == 10

        parquet_files = list((tmp_path / "ds").rglob("data_*.parquet"))
        assert len(parquet_files) >= 1

    def test_write_empty_raises(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=TimePartitionStrategy(period="month"),
        )
        with pytest.raises(ValueError, match="empty"):
            writer.write(pd.DataFrame())

    def test_write_multiple_shards_increments_index(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=TimePartitionStrategy(period="month"),
        )
        df = _make_trade_df(5, month=6)
        writer.write(df)
        writer.write(df)
        parquet_files = list((tmp_path / "ds").rglob("data_*.parquet"))
        assert len(parquet_files) == 2

    def test_metadata_sidecar_created(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=TimePartitionStrategy(period="month"),
        )
        df = _make_trade_df(5)
        writer.write(df)
        metadata_files = list((tmp_path / "ds").rglob("_metadata.json"))
        assert len(metadata_files) == 1

    def test_metadata_row_count_updated_on_append(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=TimePartitionStrategy(period="month"),
        )
        df = _make_trade_df(5)
        writer.write(df)
        writer.write(df)

        reader = PartitionedDatasetReader(tmp_path / "ds")
        for key in reader.list_partitions():
            meta = reader.get_metadata(key)
            assert meta is not None
            assert meta.row_count == 10

    def test_pair_strategy_splits_partitions(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=PairPartitionStrategy(),
        )
        df = _make_trade_df(10)  # alternates between 2 pairs
        result = writer.write(df)
        assert len(result) == 2

    def test_wallet_strategy_buckets_by_prefix(self, tmp_path):
        writer = PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=WalletPartitionStrategy(prefix_length=1),
        )
        df = _make_trade_df(10)
        writer.write(df)
        parquet_files = list((tmp_path / "ds").rglob("data_*.parquet"))
        assert len(parquet_files) >= 1

    def test_root_strategy_manifest_written(self, tmp_path):
        PartitionedDatasetWriter(
            root=tmp_path / "ds",
            strategy=PairPartitionStrategy(),
        )
        assert (tmp_path / "ds" / "_strategy.json").exists()


# ---------------------------------------------------------------------------
# PartitionedDatasetReader
# ---------------------------------------------------------------------------


class TestPartitionedDatasetReader:
    def _write(self, root: Path, n: int = 10, month: int = 6) -> PartitionedDatasetWriter:
        writer = PartitionedDatasetWriter(
            root=root,
            strategy=TimePartitionStrategy(period="month"),
        )
        writer.write(_make_trade_df(n=n, month=month))
        return writer

    def test_read_returns_all_rows(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=10)
        reader = PartitionedDatasetReader(root)
        df = reader.read()
        assert len(df) == 10

    def test_read_nonexistent_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PartitionedDatasetReader(tmp_path / "does_not_exist")

    def test_read_specific_columns(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=5)
        reader = PartitionedDatasetReader(root)
        df = reader.read(columns=["wallet", "amount"])
        assert list(df.columns) == ["wallet", "amount"]

    def test_list_partitions(self, tmp_path):
        root = tmp_path / "ds"
        # Write two months
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        writer.write(_make_trade_df(n=5, month=6))
        writer.write(_make_trade_df(n=5, month=7))
        reader = PartitionedDatasetReader(root)
        parts = reader.list_partitions()
        assert len(parts) == 2

    def test_no_matching_partitions_returns_empty(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=5)
        reader = PartitionedDatasetReader(root)
        # Filter on a non-existent partition_key
        df = reader.read(filters=[("_partition_key", "==", "9999/99")])
        assert df.empty

    def test_schema_consistency_single_hash(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=5)
        reader = PartitionedDatasetReader(root)
        result = reader.validate_schema_consistency()
        assert len(result) == 1  # all partitions share one schema hash

    def test_get_metadata_returns_metadata(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=5)
        reader = PartitionedDatasetReader(root)
        parts = reader.list_partitions()
        meta = reader.get_metadata(parts[0])
        assert meta is not None
        assert meta.row_count == 5
        assert meta.strategy_name == "time"

    def test_filter_unsupported_op_raises(self, tmp_path):
        root = tmp_path / "ds"
        self._write(root, n=5)
        reader = PartitionedDatasetReader(root)
        with pytest.raises(ValueError, match="Unsupported filter operator"):
            reader.read(filters=[("year", ">", "2024")])

    def test_filter_in_operator(self, tmp_path):
        root = tmp_path / "ds"
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        writer.write(_make_trade_df(n=5, month=6))
        writer.write(_make_trade_df(n=5, month=7))
        reader = PartitionedDatasetReader(root)
        # All matching partitions — both months
        df = reader.read(filters=[("_partition_key", "in", ["2024/06", "2024/07"])])
        assert len(df) == 10

    def test_concat_multiple_months(self, tmp_path):
        root = tmp_path / "ds"
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        for month in [6, 7, 8]:
            writer.write(_make_trade_df(n=4, month=month))
        reader = PartitionedDatasetReader(root)
        df = reader.read()
        assert len(df) == 12


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


class TestPrunePartitions:
    def _write_and_age(self, root: Path, mtime_offset_seconds: int = 0) -> None:
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        writer.write(_make_trade_df(n=5))
        if mtime_offset_seconds:
            # Age the metadata file so pruning considers it old
            for meta_file in root.rglob("_metadata.json"):
                old_time = meta_file.stat().st_mtime + mtime_offset_seconds
                import os

                os.utime(meta_file, (old_time, old_time))
            for pq_file in root.rglob("data_*.parquet"):
                old_time = pq_file.stat().st_mtime + mtime_offset_seconds
                import os

                os.utime(pq_file, (old_time, old_time))

    def test_prune_dry_run_returns_candidates(self, tmp_path):
        root = tmp_path / "ds"
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        df = _make_trade_df(n=5)
        writer.write(df)
        # Manually backdate the metadata updated_at field
        for meta_file in root.rglob("_metadata.json"):
            import json

            data = json.loads(meta_file.read_text())
            data["updated_at"] = "2020-01-01T00:00:00Z"
            meta_file.write_text(json.dumps(data))
        candidates = prune_partitions(root, max_age_days=1, dry_run=True)
        assert len(candidates) == 1
        # Parquet still exists
        assert len(list(root.rglob("data_*.parquet"))) == 1

    def test_prune_deletes_old_partitions(self, tmp_path):
        root = tmp_path / "ds"
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        writer.write(_make_trade_df(n=5))
        for meta_file in root.rglob("_metadata.json"):
            import json

            data = json.loads(meta_file.read_text())
            data["updated_at"] = "2020-01-01T00:00:00Z"
            meta_file.write_text(json.dumps(data))
        pruned = prune_partitions(root, max_age_days=1, dry_run=False)
        assert len(pruned) == 1
        assert len(list(root.rglob("data_*.parquet"))) == 0

    def test_prune_keeps_recent_partitions(self, tmp_path):
        root = tmp_path / "ds"
        writer = PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        writer.write(_make_trade_df(n=5))
        # updated_at is now; max_age_days=30 → should be kept
        pruned = prune_partitions(root, max_age_days=30, dry_run=False)
        assert len(pruned) == 0
        assert len(list(root.rglob("data_*.parquet"))) == 1

    def test_prune_negative_max_age_raises(self, tmp_path):
        root = tmp_path / "ds"
        PartitionedDatasetWriter(root=root, strategy=TimePartitionStrategy(period="month"))
        with pytest.raises(ValueError, match="positive"):
            prune_partitions(root, max_age_days=-1)


# ---------------------------------------------------------------------------
# PartitionMetadata
# ---------------------------------------------------------------------------


class TestPartitionMetadata:
    def test_round_trip(self, tmp_path):
        meta = PartitionMetadata(
            strategy_name="time",
            strategy_params={"period": "month"},
            partition_key="2024/06",
            schema_hash="abc123",
            row_count=100,
            created_at="2024-06-01T00:00:00Z",
            updated_at="2024-06-15T00:00:00Z",
        )
        meta.save(tmp_path)
        loaded = PartitionMetadata.load(tmp_path)
        assert loaded is not None
        assert loaded.schema_hash == "abc123"
        assert loaded.row_count == 100

    def test_load_returns_none_for_missing(self, tmp_path):
        result = PartitionMetadata.load(tmp_path)
        assert result is None
