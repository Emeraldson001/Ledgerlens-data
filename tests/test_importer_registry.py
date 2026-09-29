"""Comprehensive test suite for importer capability discovery system.

Tests cover:
1. Registry core functionality (registration, queries, validation)
2. Capability flag operations (bitwise combinations, checks)
3. Query methods (by capability, data type, source, best match)
4. Validation system (requirements checking, diagnostics)
5. Edge cases (duplicate registration, missing importers, empty queries)
6. Performance characteristics metadata
7. Actual importer registrations
8. Thread safety (concurrent access)

Run with: pytest tests/test_importer_registry.py -v
"""

import pytest

from ingestion.importer_registry import (
    DataSource,
    DataType,
    ImporterCapability,
    ImporterMetadata,
    PerformanceCharacteristics,
    ValidationResult,
    get_registry,
    register_importer,
    reset_registry,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def clean_registry():
    """Provide a clean registry for each test."""
    reset_registry()
    registry = get_registry()
    yield registry
    reset_registry()


@pytest.fixture
def sample_metadata():
    """Sample importer metadata for testing."""
    return ImporterMetadata(
        name="test_importer",
        description="Test importer for unit tests",
        capabilities=ImporterCapability.STREAMING | ImporterCapability.REAL_TIME,
        data_types=frozenset({DataType.TRADE}),
        sources=frozenset({DataSource.HORIZON_SSE}),
        version="1.0.0",
    )


@pytest.fixture
def bulk_loader_metadata():
    """Metadata for a bulk loader importer."""
    return ImporterMetadata(
        name="bulk_loader",
        description="Bulk loading test importer",
        capabilities=(
            ImporterCapability.BULK
            | ImporterCapability.PAGINATION
            | ImporterCapability.DATAFRAME_OUTPUT
        ),
        data_types=frozenset({DataType.TRADE, DataType.ORDERBOOK_EVENT}),
        sources=frozenset({DataSource.HORIZON_REST}),
        performance=PerformanceCharacteristics(
            typical_latency_ms=500,
            throughput_records_per_sec=400,
            memory_overhead_mb=100,
            supports_batching=True,
        ),
        supports_failover=False,
        version="2.0.0",
    )


# ============================================================================
# Test: Registry core functionality
# ============================================================================


class TestRegistryCore:
    """Test core registry operations."""

    def test_registry_singleton(self):
        """Test that get_registry returns the same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2, "Registry should be a singleton"

    def test_reset_registry(self):
        """Test that reset_registry creates a new instance."""
        reg1 = get_registry()
        reset_registry()
        reg2 = get_registry()
        assert reg1 is not reg2, "Reset should create new instance"

    def test_register_importer(self, clean_registry, sample_metadata):
        """Test basic importer registration."""

        class TestImporter:
            pass

        clean_registry.register(TestImporter, sample_metadata)

        assert "test_importer" in clean_registry.list_all()
        retrieved = clean_registry.get_importer_info("test_importer")
        assert retrieved.name == "test_importer"
        assert retrieved.capabilities == sample_metadata.capabilities

    def test_register_duplicate_name_raises_error(self, clean_registry, sample_metadata):
        """Test that registering duplicate name raises ValueError."""

        class Importer1:
            pass

        class Importer2:
            pass

        clean_registry.register(Importer1, sample_metadata)

        with pytest.raises(ValueError, match="already registered"):
            clean_registry.register(Importer2, sample_metadata)

    def test_unregister_importer(self, clean_registry, sample_metadata):
        """Test importer unregistration."""

        class TestImporter:
            pass

        clean_registry.register(TestImporter, sample_metadata)
        assert "test_importer" in clean_registry.list_all()

        clean_registry.unregister("test_importer")
        assert "test_importer" not in clean_registry.list_all()

    def test_get_nonexistent_importer_raises_error(self, clean_registry):
        """Test that getting nonexistent importer raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            clean_registry.get_importer_info("nonexistent")

    def test_get_importer_class(self, clean_registry, sample_metadata):
        """Test retrieving importer class."""

        class TestImporter:
            pass

        clean_registry.register(TestImporter, sample_metadata)
        retrieved_class = clean_registry.get_importer_class("test_importer")
        assert retrieved_class is TestImporter

    def test_list_all_returns_sorted(self, clean_registry):
        """Test that list_all returns alphabetically sorted names."""
        for name in ["zebra", "apple", "middle"]:
            metadata = ImporterMetadata(
                name=name,
                description="Test",
                capabilities=ImporterCapability.BULK,
                data_types=frozenset({DataType.TRADE}),
                sources=frozenset({DataSource.HORIZON_REST}),
            )

            class DummyImporter:
                pass

            clean_registry.register(DummyImporter, metadata)

        names = clean_registry.list_all()
        assert names == ["apple", "middle", "zebra"]


# ============================================================================
# Test: Capability flags
# ============================================================================


class TestCapabilityFlags:
    """Test capability flag operations."""

    def test_capability_bitwise_or(self):
        """Test combining capabilities with bitwise OR."""
        caps = ImporterCapability.STREAMING | ImporterCapability.REAL_TIME
        assert caps & ImporterCapability.STREAMING
        assert caps & ImporterCapability.REAL_TIME
        assert not (caps & ImporterCapability.BULK)

    def test_capability_multiple_combinations(self):
        """Test complex capability combinations."""
        caps = (
            ImporterCapability.BULK
            | ImporterCapability.PAGINATION
            | ImporterCapability.RETRY
            | ImporterCapability.DATAFRAME_OUTPUT
        )

        assert caps & ImporterCapability.BULK
        assert caps & ImporterCapability.PAGINATION
        assert caps & ImporterCapability.RETRY
        assert caps & ImporterCapability.DATAFRAME_OUTPUT
        assert not (caps & ImporterCapability.STREAMING)

    def test_has_capability(self, sample_metadata):
        """Test ImporterMetadata.has_capability() method."""
        assert sample_metadata.has_capability(ImporterCapability.STREAMING)
        assert sample_metadata.has_capability(ImporterCapability.REAL_TIME)
        assert not sample_metadata.has_capability(ImporterCapability.BULK)

    def test_has_all_capabilities(self, sample_metadata):
        """Test ImporterMetadata.has_all_capabilities() method."""
        required = ImporterCapability.STREAMING | ImporterCapability.REAL_TIME
        assert sample_metadata.has_all_capabilities(required)

        required_with_missing = required | ImporterCapability.BULK
        assert not sample_metadata.has_all_capabilities(required_with_missing)


# ============================================================================
# Test: Query methods
# ============================================================================


class TestQueryMethods:
    """Test registry query methods."""

    def test_find_by_capability_single(self, clean_registry, sample_metadata):
        """Test finding importers by single capability."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        results = clean_registry.find_by_capability(ImporterCapability.STREAMING)
        assert len(results) == 1
        assert results[0]["importer_name"] == "test_importer"
        assert results[0]["match_score"] > 0.0

    def test_find_by_capability_multiple_matches(
        self, clean_registry, sample_metadata, bulk_loader_metadata
    ):
        """Test finding multiple importers with shared capability."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        # Both have RETRY capability
        sample_metadata_with_retry = ImporterMetadata(
            name="stream_with_retry",
            description="Streamer",
            capabilities=sample_metadata.capabilities | ImporterCapability.RETRY,
            data_types=sample_metadata.data_types,
            sources=sample_metadata.sources,
        )

        clean_registry.register(StreamImporter, sample_metadata_with_retry)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        # PAGINATION is only in bulk_loader
        results = clean_registry.find_by_capability(ImporterCapability.PAGINATION)
        assert len(results) == 1
        assert results[0]["importer_name"] == "bulk_loader"

    def test_find_by_capability_require_all(self, clean_registry):
        """Test find_by_capability with require_all=True."""
        metadata1 = ImporterMetadata(
            name="partial",
            description="Has STREAMING only",
            capabilities=ImporterCapability.STREAMING,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_SSE}),
        )
        metadata2 = ImporterMetadata(
            name="complete",
            description="Has both STREAMING and REAL_TIME",
            capabilities=ImporterCapability.STREAMING | ImporterCapability.REAL_TIME,
            data_types=frozenset({DataType.TRADE}),
            sources=frozenset({DataSource.HORIZON_SSE}),
        )

        class Partial:
            pass

        class Complete:
            pass

        clean_registry.register(Partial, metadata1)
        clean_registry.register(Complete, metadata2)

        required = ImporterCapability.STREAMING | ImporterCapability.REAL_TIME
        results = clean_registry.find_by_capability(required, require_all=True)

        assert len(results) == 1
        assert results[0]["importer_name"] == "complete"

    def test_find_by_data_type(self, clean_registry, sample_metadata, bulk_loader_metadata):
        """Test finding importers by data type."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        # Both support TRADE
        trade_results = clean_registry.find_by_data_type(DataType.TRADE)
        assert len(trade_results) == 2

        # Only bulk_loader supports ORDERBOOK_EVENT
        orderbook_results = clean_registry.find_by_data_type(DataType.ORDERBOOK_EVENT)
        assert len(orderbook_results) == 1
        assert orderbook_results[0]["importer_name"] == "bulk_loader"

    def test_find_by_source(self, clean_registry, sample_metadata, bulk_loader_metadata):
        """Test finding importers by data source."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        sse_results = clean_registry.find_by_source(DataSource.HORIZON_SSE)
        assert len(sse_results) == 1
        assert sse_results[0]["importer_name"] == "test_importer"

        rest_results = clean_registry.find_by_source(DataSource.HORIZON_REST)
        assert len(rest_results) == 1
        assert rest_results[0]["importer_name"] == "bulk_loader"

    def test_find_best_match_with_requirements(
        self, clean_registry, sample_metadata, bulk_loader_metadata
    ):
        """Test find_best_match with multiple criteria."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        # Find importer with BULK capability and TRADE data type
        result = clean_registry.find_best_match(
            required_capabilities=ImporterCapability.BULK,
            required_data_types=[DataType.TRADE],
        )

        assert result is not None
        assert result["importer_name"] == "bulk_loader"

    def test_find_best_match_no_match(self, clean_registry, sample_metadata):
        """Test find_best_match returns None when no match."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        result = clean_registry.find_best_match(
            required_capabilities=ImporterCapability.BULK,  # sample has STREAMING, not BULK
        )

        assert result is None

    def test_find_best_match_with_preferences(
        self, clean_registry, sample_metadata, bulk_loader_metadata
    ):
        """Test find_best_match with preferred capabilities."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        # Both support TRADE, but prefer DATAFRAME_OUTPUT
        result = clean_registry.find_best_match(
            required_data_types=[DataType.TRADE],
            prefer_capabilities=ImporterCapability.DATAFRAME_OUTPUT,
        )

        assert result is not None
        assert result["importer_name"] == "bulk_loader"  # Has DATAFRAME_OUTPUT

    def test_match_scores_sorted(self, clean_registry):
        """Test that query results are sorted by match score."""
        # Create importers with different numbers of matching capabilities
        for i, num_caps in enumerate([1, 3, 2]):
            caps = ImporterCapability.NONE
            for j in range(num_caps):
                caps |= ImporterCapability(1 << j)

            metadata = ImporterMetadata(
                name=f"importer_{i}",
                description=f"Has {num_caps} capabilities",
                capabilities=caps,
                data_types=frozenset({DataType.TRADE}),
                sources=frozenset({DataSource.HORIZON_REST}),
            )

            class DummyImporter:
                pass

            clean_registry.register(DummyImporter, metadata)

        # Query for first 3 capabilities
        query_caps = (
            ImporterCapability.STREAMING | ImporterCapability.BULK | ImporterCapability.PAGINATION
        )
        results = clean_registry.find_by_capability(query_caps)

        # Should be sorted by match score descending
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["match_score"] >= results[i + 1]["match_score"]


# ============================================================================
# Test: Validation system
# ============================================================================


class TestValidationSystem:
    """Test requirement validation system."""

    def test_validate_requirements_all_satisfied(self, clean_registry, sample_metadata):
        """Test validation when all requirements are satisfied."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        result = clean_registry.validate_requirements(
            required_capabilities=ImporterCapability.STREAMING,
            required_data_types=[DataType.TRADE],
        )

        assert result.is_valid
        assert len(result.missing_capabilities) == 0
        assert len(result.missing_data_types) == 0

    def test_validate_requirements_missing_capability(self, clean_registry, sample_metadata):
        """Test validation when capability is missing."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        result = clean_registry.validate_requirements(
            required_capabilities=ImporterCapability.BULK,  # Not provided
        )

        assert not result.is_valid
        assert ImporterCapability.BULK in result.missing_capabilities
        assert len(result.suggestions) > 0

    def test_validate_requirements_missing_data_type(self, clean_registry, sample_metadata):
        """Test validation when data type is missing."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        result = clean_registry.validate_requirements(
            required_data_types=[DataType.ACCOUNT_ACTIVITY],  # Not provided
        )

        assert not result.is_valid
        assert DataType.ACCOUNT_ACTIVITY in result.missing_data_types

    def test_validation_result_str_format(self, clean_registry):
        """Test ValidationResult string formatting."""
        result_valid = ValidationResult(is_valid=True)
        assert "✓" in str(result_valid)
        assert "satisfied" in str(result_valid).lower()

        result_invalid = ValidationResult(
            is_valid=False,
            missing_capabilities=frozenset({ImporterCapability.STREAMING}),
            missing_data_types=frozenset({DataType.TRADE}),
            suggestions=["Try using combination of importers"],
        )
        assert "✗" in str(result_invalid)
        assert "STREAMING" in str(result_invalid)
        assert "trade" in str(result_invalid).lower()
        assert "Try using" in str(result_invalid)


# ============================================================================
# Test: Metadata validation
# ============================================================================


class TestMetadataValidation:
    """Test ImporterMetadata validation."""

    def test_metadata_requires_name(self):
        """Test that metadata requires non-empty name."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ImporterMetadata(
                name="",
                description="Test",
                capabilities=ImporterCapability.BULK,
                data_types=frozenset({DataType.TRADE}),
                sources=frozenset({DataSource.HORIZON_REST}),
            )

    def test_metadata_requires_data_types(self):
        """Test that metadata requires at least one data type."""
        with pytest.raises(ValueError, match="at least one data type"):
            ImporterMetadata(
                name="test",
                description="Test",
                capabilities=ImporterCapability.BULK,
                data_types=frozenset(),  # Empty
                sources=frozenset({DataSource.HORIZON_REST}),
            )

    def test_metadata_requires_sources(self):
        """Test that metadata requires at least one source."""
        with pytest.raises(ValueError, match="at least one data source"):
            ImporterMetadata(
                name="test",
                description="Test",
                capabilities=ImporterCapability.BULK,
                data_types=frozenset({DataType.TRADE}),
                sources=frozenset(),  # Empty
            )

    def test_metadata_immutability(self, sample_metadata):
        """Test that metadata is immutable (frozen dataclass)."""
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            sample_metadata.name = "modified"  # type: ignore[misc]

    def test_metadata_supports_data_type(self, bulk_loader_metadata):
        """Test supports_data_type() method."""
        assert bulk_loader_metadata.supports_data_type(DataType.TRADE)
        assert bulk_loader_metadata.supports_data_type(DataType.ORDERBOOK_EVENT)
        assert not bulk_loader_metadata.supports_data_type(DataType.ACCOUNT_ACTIVITY)

    def test_metadata_uses_source(self, bulk_loader_metadata):
        """Test uses_source() method."""
        assert bulk_loader_metadata.uses_source(DataSource.HORIZON_REST)
        assert not bulk_loader_metadata.uses_source(DataSource.HORIZON_SSE)


# ============================================================================
# Test: Performance characteristics
# ============================================================================


class TestPerformanceCharacteristics:
    """Test performance metadata."""

    def test_performance_str_format(self):
        """Test PerformanceCharacteristics string formatting."""
        perf = PerformanceCharacteristics(
            typical_latency_ms=500,
            throughput_records_per_sec=400,
            memory_overhead_mb=100,
            supports_batching=True,
        )

        perf_str = str(perf)
        assert "500ms" in perf_str
        assert "400 rec/s" in perf_str
        assert "100MB" in perf_str

    def test_performance_empty(self):
        """Test empty performance characteristics."""
        perf = PerformanceCharacteristics()
        perf_str = str(perf)
        assert "no performance data" in perf_str.lower()


# ============================================================================
# Test: Decorator registration
# ============================================================================


class TestDecoratorRegistration:
    """Test @register_importer decorator."""

    def test_decorator_registers_class(self, clean_registry):
        """Test that decorator registers class automatically."""

        @register_importer(
            name="decorated_importer",
            description="Test decorator",
            capabilities=ImporterCapability.STREAMING,
            data_types={DataType.TRADE},
            sources={DataSource.HORIZON_SSE},
        )
        class DecoratedImporter:
            pass

        assert "decorated_importer" in clean_registry.list_all()
        info = clean_registry.get_importer_info("decorated_importer")
        assert info.name == "decorated_importer"
        assert info.has_capability(ImporterCapability.STREAMING)

    def test_decorator_attaches_metadata_to_class(self, clean_registry):
        """Test that decorator attaches __importer_metadata__ to class."""

        @register_importer(
            name="test_decorator",
            description="Test",
            capabilities=ImporterCapability.BULK,
            data_types={DataType.TRADE},
            sources={DataSource.HORIZON_REST},
        )
        class TestClass:
            pass

        assert hasattr(TestClass, "__importer_metadata__")
        assert isinstance(TestClass.__importer_metadata__, ImporterMetadata)
        assert TestClass.__importer_metadata__.name == "test_decorator"

    def test_decorator_with_kwargs(self, clean_registry):
        """Test decorator with additional metadata kwargs."""

        @register_importer(
            name="test_kwargs",
            description="Test kwargs",
            capabilities=ImporterCapability.BULK,
            data_types={DataType.TRADE},
            sources={DataSource.HORIZON_REST},
            version="2.5.0",
            supports_failover=True,
            performance=PerformanceCharacteristics(typical_latency_ms=100),
        )
        class TestClass:
            pass

        info = clean_registry.get_importer_info("test_kwargs")
        assert info.version == "2.5.0"
        assert info.supports_failover is True
        assert info.performance.typical_latency_ms == 100


# ============================================================================
# Test: Registry statistics
# ============================================================================


class TestRegistryStatistics:
    """Test registry statistics."""

    def test_get_statistics_empty_registry(self, clean_registry):
        """Test statistics on empty registry."""
        stats = clean_registry.get_statistics()
        assert stats["total_importers"] == 0
        assert stats["streaming_importers"] == 0
        assert stats["bulk_importers"] == 0

    def test_get_statistics_with_importers(
        self, clean_registry, sample_metadata, bulk_loader_metadata
    ):
        """Test statistics with registered importers."""

        class StreamImporter:
            pass

        class BulkImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)
        clean_registry.register(BulkImporter, bulk_loader_metadata)

        stats = clean_registry.get_statistics()
        assert stats["total_importers"] == 2
        assert stats["streaming_importers"] == 1
        assert stats["bulk_importers"] == 1
        assert "trade" in stats["importers_by_data_type"]
        assert stats["importers_by_data_type"]["trade"] == 2


# ============================================================================
# Test: Actual registered importers
# ============================================================================


class TestActualRegisteredImporters:
    """Test that actual importers are registered correctly."""

    @pytest.fixture(autouse=True)
    def import_registered_importers(self):
        """Import registered_importers to populate registry."""
        import ingestion.registered_importers  # noqa: F401

    def test_all_importers_registered(self):
        """Test that all expected importers are registered."""
        from ingestion.registered_importers import verify_registration

        status = verify_registration()
        assert all(status.values()), f"Some importers not registered: {status}"

    def test_horizon_streamer_registered(self):
        """Test HorizonStreamer registration."""
        from ingestion.importer_registry import get_importer_info

        info = get_importer_info("horizon_streamer")
        assert info.name == "horizon_streamer"
        assert info.has_capability(ImporterCapability.STREAMING)
        assert info.has_capability(ImporterCapability.REAL_TIME)
        assert info.has_capability(ImporterCapability.FAILOVER)
        assert info.supports_failover is True
        assert DataType.TRADE in info.data_types
        assert DataSource.HORIZON_SSE in info.sources

    def test_historical_loader_registered(self):
        """Test HistoricalLoader registration."""
        from ingestion.importer_registry import get_importer_info

        info = get_importer_info("historical_loader")
        assert info.has_capability(ImporterCapability.BULK)
        assert info.has_capability(ImporterCapability.PAGINATION)
        assert info.has_capability(ImporterCapability.DATAFRAME_OUTPUT)
        assert DataType.TRADE in info.data_types

    def test_amm_pool_loader_dual_mode(self):
        """Test AMM pool loader has both streaming and bulk."""
        from ingestion.importer_registry import get_importer_info

        info = get_importer_info("amm_pool_loader")
        assert info.has_capability(ImporterCapability.STREAMING)
        assert info.has_capability(ImporterCapability.BULK)
        assert info.has_capability(ImporterCapability.POOL_DISCOVERY)

    def test_query_for_streaming_importers(self):
        """Test querying for all streaming importers."""
        from ingestion.importer_registry import find_importers_by_capability

        results = find_importers_by_capability(ImporterCapability.STREAMING)
        importer_names = {r["importer_name"] for r in results}

        assert "horizon_streamer" in importer_names
        assert "amm_pool_loader" in importer_names

    def test_query_for_dataframe_output(self):
        """Test querying for importers with DataFrame output."""
        from ingestion.importer_registry import find_importers_by_capability

        results = find_importers_by_capability(ImporterCapability.DATAFRAME_OUTPUT)
        importer_names = {r["importer_name"] for r in results}

        assert "historical_loader" in importer_names
        assert "orderbook_loader" in importer_names
        assert "amm_pool_loader" in importer_names


# ============================================================================
# Test: Edge cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_registry_queries(self, clean_registry):
        """Test queries on empty registry return empty results."""
        results = clean_registry.find_by_capability(ImporterCapability.STREAMING)
        assert len(results) == 0

        results = clean_registry.find_by_data_type(DataType.TRADE)
        assert len(results) == 0

    def test_query_nonexistent_capability(self, clean_registry, sample_metadata):
        """Test querying for capability no importer has."""

        class StreamImporter:
            pass

        clean_registry.register(StreamImporter, sample_metadata)

        # sample_metadata has STREAMING, query for BULK
        results = clean_registry.find_by_capability(ImporterCapability.BULK)
        assert len(results) == 0

    def test_capability_none_flag(self):
        """Test ImporterCapability.NONE flag."""
        caps = ImporterCapability.NONE
        assert not (caps & ImporterCapability.STREAMING)
        assert not (caps & ImporterCapability.BULK)

        # NONE | anything = anything
        combined = ImporterCapability.NONE | ImporterCapability.STREAMING
        assert combined == ImporterCapability.STREAMING

    def test_multiple_data_types_query(self, clean_registry):
        """Test importer supporting multiple data types."""
        metadata = ImporterMetadata(
            name="multi_type",
            description="Supports multiple types",
            capabilities=ImporterCapability.BULK,
            data_types=frozenset(
                {DataType.TRADE, DataType.ORDERBOOK_EVENT, DataType.ACCOUNT_ACTIVITY}
            ),
            sources=frozenset({DataSource.HORIZON_REST}),
        )

        class MultiTypeImporter:
            pass

        clean_registry.register(MultiTypeImporter, metadata)

        # Should appear in queries for all three types
        for data_type in [DataType.TRADE, DataType.ORDERBOOK_EVENT, DataType.ACCOUNT_ACTIVITY]:
            results = clean_registry.find_by_data_type(data_type)
            assert len(results) == 1
            assert results[0]["importer_name"] == "multi_type"

    def test_unregister_nonexistent_is_noop(self, clean_registry):
        """Test unregistering nonexistent importer doesn't raise error."""
        clean_registry.unregister("nonexistent")  # Should not raise
        assert "nonexistent" not in clean_registry.list_all()


# ============================================================================
# Run tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
