"""Deterministic time and timezone handling utilities (Issue #482).

All production code in LedgerLens must express timestamps as **timezone-aware
UTC datetimes**. Naive datetimes (no ``tzinfo``) are an endemic source of bugs:
they silently compare wrong across DST transitions, produce inconsistent ledger
sequences when replayed in a different local timezone, and are rejected by
PostgreSQL's ``TIMESTAMPTZ`` columns at runtime rather than at the point of
construction.

This module centralises every wall-clock read behind a single, replaceable
provider so that:

1. **Tests** can inject a frozen, deterministic clock without patching
   ``datetime.now`` or importing ``freezegun`` as a hard dependency.
2. **Production** always gets a UTC-aware datetime from a single call site,
   eliminating ``datetime.now()`` / ``datetime.utcnow()`` scatter.
3. **Diagnostics** are actionable: every function documents the exact error
   raised when a naive or non-UTC datetime is passed.

Quickstart::

    from utils.time import utcnow, as_utc, parse_iso_utc, frozen_clock

    # Production: current UTC time
    ts = utcnow()

    # Coerce any timezone-aware datetime to UTC
    ts_utc = as_utc(ts)

    # Parse an ISO-8601 string with timezone
    ts = parse_iso_utc("2024-06-15T12:00:00Z")

    # Tests: inject a frozen clock
    with frozen_clock("2024-01-01T00:00:00Z") as clock:
        assert utcnow().year == 2024
        clock.tick(seconds=30)
        assert (utcnow() - ts).total_seconds() == 30.0

Public API
----------
``utcnow()``
    Return the current UTC datetime from the active clock provider.

``as_utc(dt)``
    Coerce any timezone-aware datetime to UTC; raise on naive input.

``ensure_utc(dt)``
    Return *dt* unchanged if already UTC-aware; raise ``NaiveDatetimeError``
    otherwise.

``parse_iso_utc(s)``
    Parse an ISO-8601 string and return a UTC-aware datetime.

``ledger_close_time_to_utc(dt)``
    Validate and normalise a ``ledger_close_time`` value from a Stellar record.

``ClockProvider`` (Protocol)
    Contract for injectable clock implementations.

``FrozenClock``
    Test helper: a mutable, deterministic clock usable as a context manager.

``frozen_clock(at)``
    Context-manager factory that installs a ``FrozenClock`` for the duration
    of the ``with`` block, then restores the real clock.

``utc_midnight(date_or_str)``
    Return the UTC midnight of a given date.

``utc_range(start, end, step)``
    Generator yielding UTC datetimes from *start* to *end* with *step*.

``truncate_to_ledger_window(dt, window_seconds)``
    Truncate a UTC datetime to the nearest *window_seconds* boundary.
    Useful for aligning timestamps to Stellar's ~5-second ledger close cadence.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Protocol

__all__ = [
    # Core functions
    "utcnow",
    "as_utc",
    "ensure_utc",
    "parse_iso_utc",
    "ledger_close_time_to_utc",
    "utc_midnight",
    "utc_range",
    "truncate_to_ledger_window",
    # Providers / test helpers
    "ClockProvider",
    "RealClock",
    "FrozenClock",
    "frozen_clock",
    # Exceptions
    "NaiveDatetimeError",
    "AmbiguousTimezoneError",
    "InvalidTimestampError",
]

# ---------------------------------------------------------------------------
# UTC singleton — always use this, never `timezone(timedelta(0))` inline.
# ---------------------------------------------------------------------------
UTC = UTC


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class NaiveDatetimeError(ValueError):
    """Raised when a naive (timezone-unaware) datetime is encountered.

    Actionable message includes the offending value and the call site context
    so the developer knows exactly what to fix.

    >>> raise NaiveDatetimeError(datetime(2024, 1, 1), context="ledger_close_time")
    NaiveDatetimeError: Naive datetime 2024-01-01 00:00:00 rejected at \
'ledger_close_time'. Call .replace(tzinfo=UTC) or as_utc() after confirming \
the source timezone.
    """

    def __init__(self, dt: datetime, context: str = ""):
        loc = f" at {context!r}" if context else ""
        super().__init__(
            f"Naive datetime {dt!r} rejected{loc}. "
            "Call .replace(tzinfo=UTC) or as_utc() after confirming the source timezone."
        )
        self.dt = dt
        self.context = context


class AmbiguousTimezoneError(ValueError):
    """Raised when a datetime carries a non-UTC fixed-offset timezone and the
    caller requested strict UTC-only input.

    This is a separate exception from ``NaiveDatetimeError`` so callers can
    distinguish "missing timezone" from "wrong timezone" in error handlers.
    """

    def __init__(self, dt: datetime, context: str = ""):
        offset = dt.utcoffset()
        loc = f" at {context!r}" if context else ""
        super().__init__(
            f"Non-UTC timezone {offset!r} on datetime {dt!r} rejected{loc}. "
            "Use as_utc() to coerce to UTC or ensure the source produces UTC timestamps."
        )
        self.dt = dt
        self.context = context


class InvalidTimestampError(ValueError):
    """Raised when a string cannot be parsed as a valid ISO-8601 UTC timestamp.

    Carries the raw input so it can be included in structured log records.
    """

    def __init__(self, raw: str, reason: str = ""):
        detail = f": {reason}" if reason else ""
        super().__init__(f"Cannot parse {raw!r} as an ISO-8601 UTC timestamp{detail}.")
        self.raw = raw


# ---------------------------------------------------------------------------
# Clock provider protocol + implementations
# ---------------------------------------------------------------------------


class ClockProvider(Protocol):
    """Minimal protocol every clock implementation must satisfy.

    A production clock delegates to ``datetime.now(UTC)``.  A test clock
    returns a pre-configured, mutable value so that tests do not depend on
    wall-clock time and are therefore deterministic and timezone-independent.
    """

    def now(self) -> datetime:
        """Return the current UTC datetime (always timezone-aware)."""
        ...


class RealClock:
    """Production clock backed by the system wall clock.

    Always returns a UTC-aware datetime.  Import and use ``utcnow()`` from
    this module rather than constructing ``RealClock`` directly.
    """

    def now(self) -> datetime:  # noqa: D102
        return datetime.now(UTC)


class FrozenClock:
    """Deterministic, mutable test clock.

    ``FrozenClock`` is the primary way to make time-dependent tests
    deterministic and timezone-safe.  It is thread-safe (guarded by an
    internal ``threading.Lock``) so it can be used in async / threaded test
    suites.

    Attributes
    ----------
    _current : datetime
        The current frozen time (always UTC-aware).

    Examples
    --------
    Direct usage::

        clock = FrozenClock("2024-06-01T00:00:00Z")
        assert clock.now().year == 2024
        clock.tick(seconds=3600)
        assert clock.now().hour == 1

    As a context manager (preferred in tests)::

        with frozen_clock("2024-01-01T00:00:00Z") as c:
            ts1 = utcnow()
            c.tick(seconds=10)
            ts2 = utcnow()
            assert (ts2 - ts1).total_seconds() == 10.0
    """

    def __init__(self, at: str | datetime) -> None:
        self._current: datetime = _coerce_to_utc_datetime(at)
        self._lock = threading.Lock()

    # --- ClockProvider interface ---

    def now(self) -> datetime:
        """Return the frozen UTC datetime."""
        with self._lock:
            return self._current

    # --- Mutation helpers (test-only) ---

    def set(self, at: str | datetime) -> None:
        """Replace the frozen time with *at*.

        Args:
            at: New time as UTC-aware ``datetime`` or ISO-8601 string.
        """
        with self._lock:
            self._current = _coerce_to_utc_datetime(at)

    def tick(
        self,
        *,
        seconds: float = 0,
        minutes: float = 0,
        hours: float = 0,
        days: float = 0,
    ) -> None:
        """Advance the frozen clock by the given duration.

        All arguments are additive.  Negative values move the clock backward.

        Args:
            seconds: Seconds to add (may be fractional).
            minutes: Minutes to add.
            hours:   Hours to add.
            days:    Days to add.
        """
        delta = timedelta(seconds=seconds, minutes=minutes, hours=hours, days=days)
        with self._lock:
            self._current += delta


# ---------------------------------------------------------------------------
# Module-level clock state (thread-local override + global default)
# ---------------------------------------------------------------------------

_global_clock: ClockProvider = RealClock()
_thread_local = threading.local()


def _active_clock() -> ClockProvider:
    """Return the active clock, preferring a thread-local override if set."""
    return getattr(_thread_local, "clock", _global_clock)


# ---------------------------------------------------------------------------
# Core public functions
# ---------------------------------------------------------------------------


def utcnow() -> datetime:
    """Return the current UTC datetime from the active clock provider.

    In production this calls ``datetime.now(UTC)`` via ``RealClock``.
    In tests that use :func:`frozen_clock`, it returns the frozen value.

    Returns
    -------
    datetime
        A UTC-aware ``datetime`` (``tzinfo == timezone.utc``).

    Examples
    --------
    >>> from utils.time import utcnow
    >>> ts = utcnow()
    >>> assert ts.tzinfo is not None, "utcnow() never returns a naive datetime"
    """
    return _active_clock().now()


def ensure_utc(dt: datetime, *, context: str = "") -> datetime:
    """Return *dt* unchanged if it is UTC-aware; raise otherwise.

    This is the strict guard used at trust-boundary entry points (e.g. when
    ingesting a ``ledger_close_time`` from the Horizon API).

    Args:
        dt: Datetime to validate.
        context: Human-readable label for the call site (used in the exception
            message so the developer knows which field/argument failed).

    Returns
    -------
    datetime
        The original *dt* if it already carries ``timezone.utc``.

    Raises
    ------
    NaiveDatetimeError
        If *dt* has no timezone.
    AmbiguousTimezoneError
        If *dt* has a non-UTC timezone.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> ensure_utc(datetime(2024, 1, 1, tzinfo=timezone.utc))
    datetime.datetime(2024, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if dt.tzinfo is None:
        raise NaiveDatetimeError(dt, context=context)
    if dt.utcoffset() != timedelta(0):
        raise AmbiguousTimezoneError(dt, context=context)
    return dt


def as_utc(dt: datetime, *, context: str = "") -> datetime:
    """Coerce any timezone-aware datetime to UTC; raise on naive input.

    Unlike :func:`ensure_utc`, this accepts non-UTC timezones and converts
    them (e.g. ``America/New_York`` → UTC).  Naive datetimes are still
    rejected — the caller must supply a timezone, because assuming UTC for
    a naive datetime silently swallows offset bugs.

    Args:
        dt: Timezone-aware datetime to convert.
        context: Human-readable label used in the exception message.

    Returns
    -------
    datetime
        A UTC-aware datetime equivalent to *dt*.

    Raises
    ------
    NaiveDatetimeError
        If *dt* has no timezone.

    Examples
    --------
    >>> from datetime import datetime, timezone, timedelta
    >>> eastern = timezone(timedelta(hours=-5))
    >>> dt = datetime(2024, 1, 1, 12, 0, tzinfo=eastern)
    >>> as_utc(dt)
    datetime.datetime(2024, 1, 1, 17, 0, tzinfo=datetime.timezone.utc)
    """
    if dt.tzinfo is None:
        raise NaiveDatetimeError(dt, context=context)
    return dt.astimezone(UTC)


def parse_iso_utc(s: str) -> datetime:
    """Parse an ISO-8601 string and return a UTC-aware datetime.

    Accepts the following formats:
    - ``"2024-06-15T12:00:00Z"``          (Zulu / UTC marker)
    - ``"2024-06-15T12:00:00+00:00"``     (explicit +00:00 offset)
    - ``"2024-06-15T12:00:00.123456Z"``   (microseconds, Zulu)
    - ``"2024-06-15T12:00:00+05:30"``     (non-UTC offset, converted to UTC)

    Horizon API timestamps use the Zulu (``Z``) format; this function handles
    all common variants so the ingestion layer does not need per-caller parsing.

    Args:
        s: ISO-8601 datetime string.

    Returns
    -------
    datetime
        A UTC-aware datetime.

    Raises
    ------
    InvalidTimestampError
        If *s* cannot be parsed or is naive (no timezone).

    Examples
    --------
    >>> parse_iso_utc("2024-06-15T12:00:00Z")
    datetime.datetime(2024, 6, 15, 12, 0, tzinfo=datetime.timezone.utc)
    >>> parse_iso_utc("2024-06-15T12:00:00+05:30")
    datetime.datetime(2024, 6, 15, 6, 30, tzinfo=datetime.timezone.utc)
    """
    # Python 3.11+ fromisoformat handles "Z" suffix natively.
    # For 3.10 compatibility we normalise "Z" → "+00:00".
    normalized = s.rstrip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise InvalidTimestampError(s, reason=str(exc)) from exc
    if dt.tzinfo is None:
        raise InvalidTimestampError(s, reason="no timezone indicator found; expected 'Z' or +HH:MM")
    return dt.astimezone(UTC)


def ledger_close_time_to_utc(dt: datetime, *, strict: bool = True) -> datetime:
    """Validate and normalise a ``ledger_close_time`` value from a Stellar record.

    Stellar Horizon returns ``ledger_close_time`` as a UTC timestamp.  This
    function enforces that invariant so any accidental naive or non-UTC value
    fails at the ingestion boundary rather than silently propagating to the
    Benford engine or feature engineering pipeline.

    Args:
        dt: The raw ``ledger_close_time`` from a ``Trade`` or ``OrderBookEvent``.
        strict: If ``True`` (default), only accepts UTC-aware datetimes — any
            non-UTC offset raises ``AmbiguousTimezoneError``.  Set to ``False``
            to accept any timezone-aware datetime and convert to UTC.

    Returns
    -------
    datetime
        A UTC-aware datetime.

    Raises
    ------
    NaiveDatetimeError
        If *dt* is naive.
    AmbiguousTimezoneError
        If *dt* carries a non-UTC timezone and ``strict=True``.

    Examples
    --------
    >>> from datetime import datetime, timezone
    >>> ledger_close_time_to_utc(datetime(2024, 6, 1, 12, tzinfo=timezone.utc))
    datetime.datetime(2024, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if strict:
        return ensure_utc(dt, context="ledger_close_time")
    return as_utc(dt, context="ledger_close_time")


def utc_midnight(d: date | datetime | str) -> datetime:
    """Return UTC midnight (00:00:00+00:00) for *d*.

    Args:
        d: A ``date``, a UTC-aware ``datetime``, or an ISO-8601 date string
           (``"YYYY-MM-DD"``).

    Returns
    -------
    datetime
        UTC midnight of the given day.

    Raises
    ------
    InvalidTimestampError
        If *d* is a string that cannot be parsed as a date.
    NaiveDatetimeError
        If *d* is a naive ``datetime``.

    Examples
    --------
    >>> utc_midnight("2024-06-15")
    datetime.datetime(2024, 6, 15, 0, 0, tzinfo=datetime.timezone.utc)
    """
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d)
        except ValueError as exc:
            raise InvalidTimestampError(d, reason=str(exc)) from exc
    if isinstance(d, datetime):
        if d.tzinfo is None:
            raise NaiveDatetimeError(d, context="utc_midnight argument")
        d = d.astimezone(UTC).date()
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def utc_range(
    start: datetime,
    end: datetime,
    step: timedelta,
) -> Iterator[datetime]:
    """Yield UTC datetimes from *start* (inclusive) to *end* (exclusive) at *step* intervals.

    Useful for building time-bucketed feature matrices or replay windows that
    need deterministic alignment regardless of the local timezone.

    Args:
        start: Range start (UTC-aware).
        end:   Range end (UTC-aware, exclusive).
        step:  Positive timedelta between yielded values.

    Yields
    ------
    datetime
        UTC-aware datetimes at *step* intervals.

    Raises
    ------
    NaiveDatetimeError
        If *start* or *end* are naive.
    ValueError
        If *step* is not positive or *start* >= *end*.

    Examples
    --------
    >>> list(utc_range(
    ...     parse_iso_utc("2024-01-01T00:00:00Z"),
    ...     parse_iso_utc("2024-01-01T01:00:00Z"),
    ...     timedelta(minutes=20),
    ... ))
    [datetime(2024, 1, 1, 0, 0, tzinfo=utc),
     datetime(2024, 1, 1, 0, 20, tzinfo=utc),
     datetime(2024, 1, 1, 0, 40, tzinfo=utc)]
    """
    start = as_utc(start, context="utc_range start")
    end = as_utc(end, context="utc_range end")
    if step <= timedelta(0):
        raise ValueError(f"step must be positive, got {step!r}")
    if start >= end:
        return
    current = start
    while current < end:
        yield current
        current += step


def truncate_to_ledger_window(dt: datetime, window_seconds: int = 5) -> datetime:
    """Truncate a UTC datetime to the nearest *window_seconds* boundary.

    Stellar closes a ledger approximately every 5 seconds.  Aligning
    timestamps to that grid makes rolling-window Benford features and
    replay buffers reproducible across different ingestion latencies.

    Args:
        dt:             UTC-aware datetime to truncate.
        window_seconds: Grid size in seconds (default: 5, Stellar ledger cadence).

    Returns
    -------
    datetime
        *dt* truncated to the nearest multiple of *window_seconds* since epoch.

    Raises
    ------
    NaiveDatetimeError
        If *dt* is naive.
    ValueError
        If *window_seconds* < 1.

    Examples
    --------
    >>> truncate_to_ledger_window(parse_iso_utc("2024-01-01T00:00:07Z"), window_seconds=5)
    datetime.datetime(2024, 1, 1, 0, 0, 5, tzinfo=datetime.timezone.utc)
    """
    dt = as_utc(dt, context="truncate_to_ledger_window")
    if window_seconds < 1:
        raise ValueError(f"window_seconds must be >= 1, got {window_seconds}")
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    total_seconds = int((dt - epoch).total_seconds())
    truncated_seconds = (total_seconds // window_seconds) * window_seconds
    return epoch + timedelta(seconds=truncated_seconds)


# ---------------------------------------------------------------------------
# Context-manager for frozen clock injection
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def frozen_clock(
    at: str | datetime,
) -> Generator[FrozenClock, None, None]:
    """Context manager that installs a :class:`FrozenClock` for the block duration.

    Any call to :func:`utcnow` inside the block returns the clock's frozen time
    instead of the wall clock.  The real clock is restored on exit, even if the
    block raises.

    Args:
        at: Initial frozen time as a UTC-aware ``datetime`` or ISO-8601 string.

    Yields
    ------
    FrozenClock
        The installed clock instance, allowing the test to ``tick()`` or ``set()``
        the time during the block.

    Examples
    --------
    >>> with frozen_clock("2024-01-01T00:00:00Z") as c:
    ...     t1 = utcnow()
    ...     c.tick(hours=2)
    ...     t2 = utcnow()
    ...     assert (t2 - t1).total_seconds() == 7200.0
    >>> # Real clock is restored here.
    >>> utcnow() != t2   # Almost certainly True
    """
    clock = FrozenClock(at)
    previous = getattr(_thread_local, "clock", None)
    _thread_local.clock = clock
    try:
        yield clock
    finally:
        if previous is None:
            try:
                del _thread_local.clock
            except AttributeError:
                pass
        else:
            _thread_local.clock = previous


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_to_utc_datetime(at: str | datetime) -> datetime:
    """Convert *at* to a UTC-aware datetime (used internally by FrozenClock)."""
    if isinstance(at, str):
        return parse_iso_utc(at)
    if isinstance(at, datetime):
        if at.tzinfo is None:
            raise NaiveDatetimeError(at, context="FrozenClock initial time")
        return at.astimezone(UTC)
    raise TypeError(f"Expected str or datetime, got {type(at).__name__!r}")
