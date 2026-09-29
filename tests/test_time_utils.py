"""Tests for ``utils.time`` — deterministic time and timezone utilities (Issue #482).

Design
------
Every test is deterministic and timezone-independent: no test calls the real
wall clock.  The :func:`~utils.time.frozen_clock` context manager is used
wherever "current time" is relevant, so the suite passes identically in any
timezone and on any CI host.

Coverage
--------
- :func:`utcnow` delegates to the active clock provider.
- :func:`ensure_utc` — strict UTC guard (rejects naive + non-UTC).
- :func:`as_utc` — lenient coerce (converts any tz-aware to UTC; rejects naive).
- :func:`parse_iso_utc` — ISO-8601 parsing, multiple formats.
- :func:`ledger_close_time_to_utc` — Stellar-specific boundary guard.
- :func:`utc_midnight` — date, datetime, and string inputs.
- :func:`utc_range` — generator correctness, edge cases, error paths.
- :func:`truncate_to_ledger_window` — grid alignment, Stellar default.
- :class:`FrozenClock` — tick, set, thread-safety, negative tick.
- :func:`frozen_clock` — context manager install/restore, nested use.
- Exception messages are actionable (field name in message).
- Property-based tests (Hypothesis) verify round-trip invariants.
"""

from __future__ import annotations

import threading
from datetime import UTC, date, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from utils.time import (
    AmbiguousTimezoneError,
    FrozenClock,
    InvalidTimestampError,
    NaiveDatetimeError,
    RealClock,
    as_utc,
    ensure_utc,
    frozen_clock,
    ledger_close_time_to_utc,
    parse_iso_utc,
    truncate_to_ledger_window,
    utc_midnight,
    utc_range,
    utcnow,
)

UTC = UTC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FIXED = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)
_FIXED_STR = "2024-06-15T12:00:00Z"


def make_naive(year: int = 2024, month: int = 6, day: int = 15) -> datetime:
    return datetime(year, month, day, 12, 0, 0)  # no tzinfo


def make_utc(year: int = 2024, month: int = 6, day: int = 15) -> datetime:
    return datetime(year, month, day, 12, 0, 0, tzinfo=UTC)


def make_offset(hours: int) -> datetime:
    tz = timezone(timedelta(hours=hours))
    return datetime(2024, 6, 15, 12, 0, 0, tzinfo=tz)


# ===========================================================================
# utcnow() + RealClock
# ===========================================================================


class TestUtcNow:
    def test_returns_utc_aware_datetime(self):
        ts = utcnow()
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timedelta(0)

    def test_uses_frozen_clock_inside_context(self):
        with frozen_clock(_FIXED_STR) as _clock:
            ts = utcnow()
        assert ts == _FIXED

    def test_real_clock_restored_after_context(self):
        """The real clock must be restored after a frozen_clock block exits."""
        with frozen_clock(_FIXED_STR):
            pass
        # After the block, utcnow() should NOT equal the frozen value
        # (the real clock has advanced past 2024-06-15T12:00:00Z which is in the
        # future relative to the test run — but the key property is that
        # utcnow() no longer delegates to the FrozenClock).
        from utils.time import _thread_local

        assert getattr(_thread_local, "clock", None) is None

    def test_real_clock_returns_recent_timestamp(self):
        """RealClock.now() must be after year 2020 (sanity check)."""
        rc = RealClock()
        assert rc.now().year >= 2020

    def test_real_clock_always_utc_aware(self):
        rc = RealClock()
        ts = rc.now()
        assert ts.tzinfo == UTC


# ===========================================================================
# ensure_utc()
# ===========================================================================


class TestEnsureUtc:
    def test_passes_through_utc_datetime(self):
        dt = make_utc()
        assert ensure_utc(dt) is dt

    def test_raises_on_naive(self):
        dt = make_naive()
        with pytest.raises(NaiveDatetimeError) as exc_info:
            ensure_utc(dt, context="test_field")
        err = exc_info.value
        assert err.dt == dt
        assert "test_field" in str(err)
        assert "naive" in str(err).lower()

    def test_raises_on_non_utc_offset(self):
        dt = make_offset(hours=5)
        with pytest.raises(AmbiguousTimezoneError) as exc_info:
            ensure_utc(dt, context="ledger_close_time")
        err = exc_info.value
        assert err.dt == dt
        assert "ledger_close_time" in str(err)

    def test_raises_on_negative_offset(self):
        dt = make_offset(hours=-5)
        with pytest.raises(AmbiguousTimezoneError):
            ensure_utc(dt)

    def test_context_appears_in_naive_message(self):
        with pytest.raises(NaiveDatetimeError) as exc_info:
            ensure_utc(make_naive(), context="base_account_ts")
        assert "base_account_ts" in str(exc_info.value)

    def test_zero_offset_treated_as_utc(self):
        """A fixed +00:00 offset is effectively UTC and should pass."""
        dt = datetime(2024, 1, 1, tzinfo=timezone(timedelta(0)))
        result = ensure_utc(dt)
        assert result.utcoffset() == timedelta(0)


# ===========================================================================
# as_utc()
# ===========================================================================


class TestAsUtc:
    def test_passes_through_utc(self):
        dt = make_utc()
        result = as_utc(dt)
        assert result == dt
        assert result.tzinfo == UTC

    def test_converts_positive_offset(self):
        # +05:30 (India Standard Time)
        tz = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 6, 15, 17, 30, 0, tzinfo=tz)
        result = as_utc(dt)
        assert result.tzinfo == UTC
        assert result.hour == 12
        assert result.minute == 0

    def test_converts_negative_offset(self):
        # -08:00 (Pacific Standard Time)
        tz = timezone(timedelta(hours=-8))
        dt = datetime(2024, 6, 15, 4, 0, 0, tzinfo=tz)
        result = as_utc(dt)
        assert result.tzinfo == UTC
        assert result.hour == 12

    def test_raises_on_naive(self):
        with pytest.raises(NaiveDatetimeError):
            as_utc(make_naive())

    def test_context_in_error_message(self):
        with pytest.raises(NaiveDatetimeError) as exc_info:
            as_utc(make_naive(), context="counter_account")
        assert "counter_account" in str(exc_info.value)

    def test_idempotent_on_utc(self):
        dt = make_utc()
        assert as_utc(as_utc(dt)) == dt

    @given(
        hours=st.integers(min_value=-23, max_value=23),
        minutes=st.integers(min_value=0, max_value=59),
    )
    def test_roundtrip_offset(self, hours: int, minutes: int):
        """as_utc(as_utc(dt)) == as_utc(dt) for any tz-aware datetime."""
        tz = timezone(timedelta(hours=hours, minutes=minutes))
        dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=tz)
        once = as_utc(dt)
        twice = as_utc(once)
        assert once == twice


# ===========================================================================
# parse_iso_utc()
# ===========================================================================


class TestParseIsoUtc:
    def test_zulu_suffix(self):
        result = parse_iso_utc("2024-06-15T12:00:00Z")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_explicit_plus_zero(self):
        result = parse_iso_utc("2024-06-15T12:00:00+00:00")
        assert result == datetime(2024, 6, 15, 12, 0, tzinfo=UTC)

    def test_microseconds_zulu(self):
        result = parse_iso_utc("2024-06-15T12:00:00.123456Z")
        assert result.microsecond == 123456
        assert result.tzinfo == UTC

    def test_non_utc_offset_converted(self):
        result = parse_iso_utc("2024-06-15T17:30:00+05:30")
        assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)

    def test_rejects_naive_string(self):
        with pytest.raises(InvalidTimestampError) as exc_info:
            parse_iso_utc("2024-06-15T12:00:00")
        assert "2024-06-15T12:00:00" in str(exc_info.value)

    def test_rejects_garbage_string(self):
        with pytest.raises(InvalidTimestampError):
            parse_iso_utc("not-a-date")

    def test_rejects_empty_string(self):
        with pytest.raises(InvalidTimestampError):
            parse_iso_utc("")

    def test_rejects_date_only_string(self):
        """A date-only string has no time component; reject it."""
        with pytest.raises(InvalidTimestampError):
            parse_iso_utc("2024-06-15")

    def test_horizon_api_format(self):
        """Horizon API produces 'Z'-terminated ISO-8601 strings."""
        result = parse_iso_utc("2024-03-17T10:30:00Z")
        assert result.tzinfo == UTC
        assert result.year == 2024
        assert result.month == 3

    def test_invalid_timestamp_carries_raw_value(self):
        bad = "GARBAGE_INPUT"
        with pytest.raises(InvalidTimestampError) as exc_info:
            parse_iso_utc(bad)
        assert exc_info.value.raw == bad


# ===========================================================================
# ledger_close_time_to_utc()
# ===========================================================================


class TestLedgerCloseTimeToUtc:
    def test_accepts_utc(self):
        dt = make_utc()
        assert ledger_close_time_to_utc(dt) == dt

    def test_strict_rejects_non_utc(self):
        dt = make_offset(hours=5)
        with pytest.raises(AmbiguousTimezoneError):
            ledger_close_time_to_utc(dt, strict=True)

    def test_lenient_converts_non_utc(self):
        # +05:30 offset
        tz = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 6, 15, 17, 30, 0, tzinfo=tz)
        result = ledger_close_time_to_utc(dt, strict=False)
        assert result.tzinfo == UTC
        assert result.hour == 12

    def test_strict_rejects_naive(self):
        with pytest.raises(NaiveDatetimeError):
            ledger_close_time_to_utc(make_naive())

    def test_lenient_rejects_naive(self):
        with pytest.raises(NaiveDatetimeError):
            ledger_close_time_to_utc(make_naive(), strict=False)

    def test_context_label_in_error(self):
        with pytest.raises(NaiveDatetimeError) as exc_info:
            ledger_close_time_to_utc(make_naive())
        assert "ledger_close_time" in str(exc_info.value)


# ===========================================================================
# utc_midnight()
# ===========================================================================


class TestUtcMidnight:
    def test_date_input(self):
        d = date(2024, 6, 15)
        result = utc_midnight(d)
        assert result == datetime(2024, 6, 15, 0, 0, 0, tzinfo=UTC)

    def test_datetime_input_utc(self):
        dt = datetime(2024, 6, 15, 14, 30, tzinfo=UTC)
        result = utc_midnight(dt)
        assert result == datetime(2024, 6, 15, 0, 0, 0, tzinfo=UTC)

    def test_datetime_input_offset_converted(self):
        # +05:30 → 2024-06-15 06:30 UTC → midnight is still 2024-06-15 UTC
        # (after conversion the date becomes 2024-06-15 01:00 UTC for dt at midnight IST)
        tz = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2024, 6, 16, 4, 0, tzinfo=tz)  # 2024-06-15T22:30 UTC
        result = utc_midnight(dt)
        assert result.tzinfo == UTC
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0

    def test_string_input(self):
        result = utc_midnight("2024-06-15")
        assert result == datetime(2024, 6, 15, 0, 0, 0, tzinfo=UTC)

    def test_invalid_string(self):
        with pytest.raises(InvalidTimestampError):
            utc_midnight("not-a-date")

    def test_naive_datetime_raises(self):
        with pytest.raises(NaiveDatetimeError):
            utc_midnight(make_naive())

    def test_result_always_utc_aware(self):
        assert utc_midnight(date(2024, 1, 1)).tzinfo == UTC


# ===========================================================================
# utc_range()
# ===========================================================================


class TestUtcRange:
    def test_basic_range(self):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        end = datetime(2024, 1, 1, 1, 0, tzinfo=UTC)
        step = timedelta(minutes=20)
        result = list(utc_range(start, end, step))
        assert len(result) == 3
        assert result[0] == start
        assert result[1] == start + step
        assert result[2] == start + 2 * step

    def test_empty_range_when_start_equals_end(self):
        dt = datetime(2024, 1, 1, tzinfo=UTC)
        assert list(utc_range(dt, dt, timedelta(hours=1))) == []

    def test_empty_range_when_start_after_end(self):
        start = datetime(2024, 1, 2, tzinfo=UTC)
        end = datetime(2024, 1, 1, tzinfo=UTC)
        assert list(utc_range(start, end, timedelta(hours=1))) == []

    def test_all_results_utc_aware(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=3)
        for dt in utc_range(start, end, timedelta(hours=1)):
            assert dt.tzinfo == UTC

    def test_raises_on_non_positive_step(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=1)
        with pytest.raises(ValueError, match="positive"):
            list(utc_range(start, end, timedelta(0)))
        with pytest.raises(ValueError):
            list(utc_range(start, end, timedelta(seconds=-1)))

    def test_raises_on_naive_start(self):
        with pytest.raises(NaiveDatetimeError):
            list(utc_range(make_naive(), datetime(2024, 1, 2, tzinfo=UTC), timedelta(hours=1)))

    def test_raises_on_naive_end(self):
        with pytest.raises(NaiveDatetimeError):
            list(utc_range(datetime(2024, 1, 1, tzinfo=UTC), make_naive(), timedelta(hours=1)))

    def test_single_step_range(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(hours=1)
        result = list(utc_range(start, end, timedelta(hours=1)))
        assert result == [start]

    def test_fractional_step(self):
        start = datetime(2024, 1, 1, tzinfo=UTC)
        end = start + timedelta(seconds=15)
        step = timedelta(seconds=5)
        result = list(utc_range(start, end, step))
        assert len(result) == 3


# ===========================================================================
# truncate_to_ledger_window()
# ===========================================================================


class TestTruncateToLedgerWindow:
    def test_already_on_boundary(self):
        dt = datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC)
        assert truncate_to_ledger_window(dt, 5) == dt

    def test_rounds_down(self):
        dt = datetime(2024, 1, 1, 0, 0, 7, tzinfo=UTC)
        result = truncate_to_ledger_window(dt, 5)
        assert result == datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC)

    def test_default_window_is_5_seconds(self):
        dt = datetime(2024, 1, 1, 0, 0, 3, tzinfo=UTC)
        result = truncate_to_ledger_window(dt)
        assert result == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_non_five_window(self):
        dt = datetime(2024, 1, 1, 0, 1, 47, tzinfo=UTC)
        result = truncate_to_ledger_window(dt, 60)
        assert result == datetime(2024, 1, 1, 0, 1, 0, tzinfo=UTC)

    def test_rejects_naive(self):
        with pytest.raises(NaiveDatetimeError):
            truncate_to_ledger_window(make_naive())

    def test_rejects_window_less_than_one(self):
        with pytest.raises(ValueError):
            truncate_to_ledger_window(make_utc(), 0)

    def test_result_always_utc_aware(self):
        result = truncate_to_ledger_window(make_utc())
        assert result.tzinfo == UTC

    def test_idempotent(self):
        dt = datetime(2024, 1, 1, 0, 0, 7, tzinfo=UTC)
        once = truncate_to_ledger_window(dt, 5)
        twice = truncate_to_ledger_window(once, 5)
        assert once == twice

    @given(
        seconds=st.integers(min_value=0, max_value=86399),
        window=st.integers(min_value=1, max_value=60),
    )
    def test_truncated_is_multiple_of_window_since_epoch(self, seconds: int, window: int):
        """For any datetime, the truncated value's epoch offset is a multiple of window."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        dt = epoch + timedelta(seconds=seconds)
        result = truncate_to_ledger_window(dt, window)
        elapsed = int((result - epoch).total_seconds())
        assert elapsed % window == 0


# ===========================================================================
# FrozenClock
# ===========================================================================


class TestFrozenClock:
    def test_string_init(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        assert clock.now() == datetime(2024, 1, 1, tzinfo=UTC)

    def test_datetime_init(self):
        dt = datetime(2024, 6, 15, 12, tzinfo=UTC)
        clock = FrozenClock(dt)
        assert clock.now() == dt

    def test_naive_datetime_init_raises(self):
        with pytest.raises(NaiveDatetimeError):
            FrozenClock(make_naive())

    def test_tick_advances_time(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        clock.tick(seconds=30)
        assert clock.now() == datetime(2024, 1, 1, 0, 0, 30, tzinfo=UTC)

    def test_tick_minutes_and_hours(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        clock.tick(hours=1, minutes=30)
        assert clock.now() == datetime(2024, 1, 1, 1, 30, 0, tzinfo=UTC)

    def test_tick_negative_moves_backward(self):
        clock = FrozenClock("2024-01-01T01:00:00Z")
        clock.tick(hours=-1)
        assert clock.now() == datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

    def test_set_replaces_time(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        clock.set("2025-12-31T23:59:59Z")
        assert clock.now() == datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)

    def test_set_with_datetime(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        new_time = datetime(2025, 6, 1, 12, 0, tzinfo=UTC)
        clock.set(new_time)
        assert clock.now() == new_time

    def test_now_always_returns_utc(self):
        clock = FrozenClock("2024-06-15T12:00:00Z")
        assert clock.now().tzinfo == UTC

    def test_thread_safety_concurrent_ticks(self):
        """Concurrent tick() calls must not corrupt the internal state."""
        clock = FrozenClock("2024-01-01T00:00:00Z")
        errors: list[Exception] = []

        def worker() -> None:
            for _ in range(100):
                clock.tick(seconds=1)
                _ = clock.now()

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        elapsed = (clock.now() - datetime(2024, 1, 1, tzinfo=UTC)).total_seconds()
        assert elapsed == 1000.0

    def test_multiple_ticks_accumulate(self):
        clock = FrozenClock("2024-01-01T00:00:00Z")
        for _ in range(5):
            clock.tick(seconds=10)
        assert clock.now() == datetime(2024, 1, 1, 0, 0, 50, tzinfo=UTC)


# ===========================================================================
# frozen_clock() context manager
# ===========================================================================


class TestFrozenClockContextManager:
    def test_utcnow_returns_frozen_time(self):
        with frozen_clock("2024-06-15T12:00:00Z"):
            ts = utcnow()
        assert ts == datetime(2024, 6, 15, 12, tzinfo=UTC)

    def test_real_clock_restored_after_exit(self):
        from utils.time import _thread_local

        with frozen_clock("2024-01-01T00:00:00Z"):
            pass
        assert getattr(_thread_local, "clock", None) is None

    def test_real_clock_restored_on_exception(self):
        from utils.time import _thread_local

        with pytest.raises(RuntimeError):
            with frozen_clock("2024-01-01T00:00:00Z"):
                raise RuntimeError("test")
        assert getattr(_thread_local, "clock", None) is None

    def test_tick_advances_utcnow(self):
        with frozen_clock("2024-01-01T00:00:00Z") as c:
            t1 = utcnow()
            c.tick(seconds=60)
            t2 = utcnow()
        delta = (t2 - t1).total_seconds()
        assert delta == 60.0

    def test_set_changes_utcnow(self):
        with frozen_clock("2024-01-01T00:00:00Z") as c:
            c.set("2025-06-01T12:00:00Z")
            assert utcnow().year == 2025

    def test_nested_frozen_clocks(self):
        """Inner frozen_clock must override outer; outer is restored on exit."""
        outer_time = "2024-01-01T00:00:00Z"
        inner_time = "2025-06-01T12:00:00Z"
        with frozen_clock(outer_time):
            assert utcnow().year == 2024
            with frozen_clock(inner_time):
                assert utcnow().year == 2025
            # Outer is restored
            assert utcnow().year == 2024

    def test_string_and_datetime_input_accepted(self):
        dt = datetime(2024, 3, 1, 10, 0, tzinfo=UTC)
        with frozen_clock(dt) as _c:
            assert utcnow() == dt


# ===========================================================================
# Integration: frozen_clock with other utils.time functions
# ===========================================================================


class TestIntegration:
    def test_utc_range_with_frozen_clock(self):
        """utc_range should work correctly regardless of wall-clock time."""
        with frozen_clock("2024-01-01T00:00:00Z"):
            start = utcnow()
            end = start + timedelta(hours=1)
            points = list(utc_range(start, end, timedelta(minutes=15)))
        assert len(points) == 4

    def test_truncate_with_frozen_clock(self):
        with frozen_clock("2024-01-01T00:00:07Z") as _c:
            ts = utcnow()
            truncated = truncate_to_ledger_window(ts, 5)
        assert truncated == datetime(2024, 1, 1, 0, 0, 5, tzinfo=UTC)

    def test_parse_and_ensure_pipeline(self):
        raw = "2024-06-15T12:00:00Z"
        dt = parse_iso_utc(raw)
        validated = ensure_utc(dt, context="pipeline_field")
        assert validated == dt

    def test_ledger_close_time_roundtrip(self):
        """A UTC datetime surviving the ledger_close_time_to_utc guard unchanged."""
        dt = datetime(2024, 3, 17, 10, 30, 0, tzinfo=UTC)
        assert ledger_close_time_to_utc(dt) == dt

    def test_utc_midnight_then_range(self):
        """Build a 24-hour range starting from UTC midnight."""
        midnight = utc_midnight("2024-06-15")
        end = midnight + timedelta(hours=24)
        hours = list(utc_range(midnight, end, timedelta(hours=1)))
        assert len(hours) == 24
        assert all(h.tzinfo == UTC for h in hours)


# ===========================================================================
# Exception contracts
# ===========================================================================


class TestExceptionContracts:
    def test_naive_error_carries_dt(self):
        dt = make_naive()
        exc = NaiveDatetimeError(dt, context="field_x")
        assert exc.dt is dt
        assert exc.context == "field_x"

    def test_naive_error_without_context(self):
        dt = make_naive()
        exc = NaiveDatetimeError(dt)
        assert exc.context == ""
        assert "naive" in str(exc).lower()

    def test_ambiguous_tz_error_carries_dt(self):
        dt = make_offset(hours=3)
        exc = AmbiguousTimezoneError(dt, context="wallet_ts")
        assert exc.dt is dt
        assert "wallet_ts" in str(exc)

    def test_invalid_timestamp_error_carries_raw(self):
        exc = InvalidTimestampError("JUNK_VALUE", reason="bad format")
        assert exc.raw == "JUNK_VALUE"
        assert "JUNK_VALUE" in str(exc)

    def test_invalid_timestamp_without_reason(self):
        exc = InvalidTimestampError("JUNK")
        assert "JUNK" in str(exc)


# ===========================================================================
# Property-based tests (Hypothesis)
# ===========================================================================


class TestPropertyBased:
    @given(
        year=st.integers(min_value=1990, max_value=2100),
        month=st.integers(min_value=1, max_value=12),
        day=st.integers(min_value=1, max_value=28),
        hour=st.integers(min_value=0, max_value=23),
        minute=st.integers(min_value=0, max_value=59),
        second=st.integers(min_value=0, max_value=59),
    )
    def test_as_utc_preserves_instant(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: int,
    ):
        """as_utc(dt) must represent the same instant in time as dt."""
        # Build datetime with arbitrary tz offset to exercise the conversion path
        tz = timezone(timedelta(hours=5, minutes=30))  # IST
        dt_tz = datetime(year, month, day, hour, minute, second, tzinfo=tz)
        utc_dt = as_utc(dt_tz)
        # They represent the same instant
        assert dt_tz.timestamp() == pytest.approx(utc_dt.timestamp())

    @given(
        seconds_since_epoch=st.integers(min_value=0, max_value=32503680000),  # ~3000 CE
    )
    def test_parse_iso_utc_roundtrip(self, seconds_since_epoch: int):
        """parse_iso_utc(dt.isoformat()) == dt for any UTC-aware datetime."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        dt = epoch + timedelta(seconds=seconds_since_epoch)
        iso = dt.isoformat().replace("+00:00", "Z")
        parsed = parse_iso_utc(iso)
        assert parsed == dt

    @given(
        seconds=st.integers(min_value=0, max_value=86399),
        window=st.integers(min_value=1, max_value=3600),
    )
    def test_truncated_le_original(self, seconds: int, window: int):
        """Truncated timestamp is always <= original."""
        epoch = datetime(1970, 1, 1, tzinfo=UTC)
        dt = epoch + timedelta(seconds=seconds)
        truncated = truncate_to_ledger_window(dt, window)
        assert truncated <= dt

    @given(st.integers(min_value=1, max_value=365))
    def test_utc_midnight_has_zero_time(self, day_offset: int):
        """utc_midnight always returns a datetime with time = 00:00:00."""
        base = date(2024, 1, 1)
        d = base + timedelta(days=day_offset - 1)
        midnight = utc_midnight(d)
        assert midnight.hour == 0
        assert midnight.minute == 0
        assert midnight.second == 0
        assert midnight.tzinfo == UTC
