"""Tests for detection/score_normaliser.py (Issue #779) and
scripts/list_model_versions.py (Issue #780)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import fakeredis
import pytest

from detection.score_normaliser import (
    SCORE_NORM_MIN_SAMPLES,
    PerPairScoreNormaliser,
)
from scripts.list_model_versions import list_versions, parse_trained_at

ASSET_PAIR = "USDC:GA5ZSEJYBY3RJRC5AVCIA5MOP4RHTM335X2KGX3IHOJAPP5RE34K4KZVN"


# ---------------------------------------------------------------------------
# score_normaliser.py — Issue #779
# ---------------------------------------------------------------------------


@pytest.fixture()
def normaliser() -> PerPairScoreNormaliser:
    client = fakeredis.FakeRedis(server=fakeredis.FakeServer(), decode_responses=False)
    return PerPairScoreNormaliser(client)


class TestScoreNormaliserContract:
    def test_calibrated_score_respects_the_documented_bound(
        self, normaliser: PerPairScoreNormaliser
    ):
        """Once calibrated, ``normalised_risk_score`` is strictly positive and
        never exceeds ``(n + 0.5) / n``; scores inside the observed sample
        range land in the ``(0, 1)`` unit interval."""
        n = SCORE_NORM_MIN_SAMPLES + 20
        for raw in range(n):
            normaliser.add_score(ASSET_PAIR, float(raw))
        upper_bound = (n + 0.5) / n

        for probe in (-1e9, -5.0, 0.0, 12.5, float(n - 1), 1e9):
            result = normaliser.normalise(ASSET_PAIR, probe)
            assert result.normalisation_skipped is False
            assert 0.0 < result.normalised_risk_score <= upper_bound

        for in_range_probe in (0.0, 12.5, float(n - 1)):
            result = normaliser.normalise(ASSET_PAIR, in_range_probe)
            assert 0.0 < result.normalised_risk_score < 1.0

    def test_normalisation_skipped_passes_raw_score_through(
        self, normaliser: PerPairScoreNormaliser
    ):
        """Below the sample threshold the raw score is returned unchanged."""
        for raw in range(SCORE_NORM_MIN_SAMPLES - 1):
            normaliser.add_score(ASSET_PAIR, float(raw))

        result = normaliser.normalise(ASSET_PAIR, 42.0)
        assert result.normalisation_skipped is True
        assert result.normalised_risk_score == 42.0

    def test_invalid_asset_pair_rejected(self, normaliser: PerPairScoreNormaliser):
        with pytest.raises(ValueError):
            normaliser.normalise("NOT:ALLOWLISTED", 1.0)


# ---------------------------------------------------------------------------
# list_model_versions.py — Issue #780
# ---------------------------------------------------------------------------


def _write_version(archive_dir, name: str, trained_at: str | None) -> None:
    version_dir = archive_dir / name
    version_dir.mkdir(parents=True)
    if trained_at is not None:
        (version_dir / "model_metadata.json").write_text(
            json.dumps({"trained_at": trained_at, "n_training_rows": 1, "n_test_rows": 1})
        )


class TestParseTrainedAt:
    def test_parses_iso_with_trailing_z(self):
        assert parse_trained_at("2026-06-16T12:00:00Z") == datetime(
            2026, 6, 16, 12, 0, 0, tzinfo=UTC
        )

    def test_assumes_utc_for_naive_timestamp(self):
        assert parse_trained_at("2026-06-16T12:00:00").tzinfo is UTC

    @pytest.mark.parametrize("value", [None, "unknown", "not-a-date", 12345, ""])
    def test_unknown_values_sort_last(self, value):
        assert parse_trained_at(value) == datetime.min.replace(tzinfo=UTC)


class TestListVersionsOrdering:
    def test_sorted_by_training_date_not_directory_name(self, tmp_path):
        """Regression: directory names that sort opposite to their real dates.

        ``20260101_000000`` was trained in December; ``20260901_000000`` in
        February. A lexicographic ``reverse=True`` sort would put the
        September-named directory first; a date-based sort must not.
        """
        archive = tmp_path / "archive"
        _write_version(archive, "20260101_000000", "2026-12-01T00:00:00Z")
        _write_version(archive, "20260901_000000", "2026-02-01T00:00:00Z")

        ordering = [v["version"] for v in list_versions(str(archive))]
        assert ordering == ["20260101_000000", "20260901_000000"]

    def test_versions_without_metadata_sort_last(self, tmp_path):
        archive = tmp_path / "archive"
        _write_version(archive, "20260301_000000", "2026-03-01T00:00:00Z")
        _write_version(archive, "99999999_999999", None)

        ordering = [v["version"] for v in list_versions(str(archive))]
        assert ordering == ["20260301_000000", "99999999_999999"]

    def test_missing_archive_dir_returns_empty(self, tmp_path):
        assert list_versions(str(tmp_path / "does-not-exist")) == []
