"""Tests for scripts/query_privacy_budget.py --since/--until filters."""

import json
import subprocess
from datetime import UTC, datetime, timedelta


def test_since_filter_excludes_earlier_events(tmp_path, monkeypatch):
    """Test that --since filter excludes events before the given date."""
    # We need to set up a temporary database for this test
    import tempfile
    from detection.privacy.budget_tracker import DPBudgetTracker, _get_engine
    from sqlalchemy.orm import sessionmaker

    # Use a temporary SQLite database
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RISK_SCORE_DB_URL", db_url)

    # Create engine and tables
    from sqlalchemy import create_engine
    engine = create_engine(db_url)
    from detection.privacy.budget_tracker import _Base
    _Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine, future=True)
    tracker = DPBudgetTracker(session_factory=session_factory)

    # Record events at different times
    now = datetime.now(UTC)
    old_time = now - timedelta(days=2)
    recent_time = now - timedelta(hours=1)

    # Manually insert events with specific timestamps
    from detection.privacy.budget_tracker import DPBudgetEvent
    with session_factory() as session:
        # Old event
        session.add(DPBudgetEvent(
            kind="training",
            epsilon=1.0,
            cumulative_epsilon=1.0,
            created_at=old_time,
            prev_log_hash="genesis",
            log_hash="hash1"
        ))
        # Recent event
        session.add(DPBudgetEvent(
            kind="training",
            epsilon=0.5,
            cumulative_epsilon=1.5,
            created_at=recent_time,
            prev_log_hash="hash1",
            log_hash="hash2"
        ))
        session.commit()

    # Get all events first
    status_all = tracker.status()
    assert len(status_all["events"]) == 2, "Should have 2 events total"

    # Now test via the CLI with --since filter
    since_date = (now - timedelta(hours=2)).isoformat()
    result = subprocess.run(
        [
            "python", "-m", "scripts.query_privacy_budget",
            "--since", since_date,
            "--json"
        ],
        capture_output=True,
        text=True,
        cwd="/home/ajidokwu/Desktop/Drips/Fred/Ledgerlens-data",
        env={**subprocess.os.environ, "RISK_SCORE_DB_URL": db_url}
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = json.loads(result.stdout)

    # Should only have the recent event
    assert len(output["events"]) == 1, "Should filter out old events"
    assert output["events"][0]["cumulative_epsilon"] == 1.5


def test_until_filter_excludes_later_events(tmp_path, monkeypatch):
    """Test that --until filter excludes events after the given date."""
    import tempfile
    from detection.privacy.budget_tracker import DPBudgetTracker
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from detection.privacy.budget_tracker import _Base, DPBudgetEvent

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RISK_SCORE_DB_URL", db_url)

    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, future=True)

    now = datetime.now(UTC)
    old_time = now - timedelta(days=1)
    future_time = now + timedelta(hours=1)

    with session_factory() as session:
        session.add(DPBudgetEvent(
            kind="training",
            epsilon=1.0,
            cumulative_epsilon=1.0,
            created_at=old_time,
            prev_log_hash="genesis",
            log_hash="hash1"
        ))
        session.add(DPBudgetEvent(
            kind="training",
            epsilon=0.5,
            cumulative_epsilon=1.5,
            created_at=future_time,
            prev_log_hash="hash1",
            log_hash="hash2"
        ))
        session.commit()

    until_date = (now + timedelta(minutes=30)).isoformat()
    result = subprocess.run(
        [
            "python", "-m", "scripts.query_privacy_budget",
            "--until", until_date,
            "--json"
        ],
        capture_output=True,
        text=True,
        cwd="/home/ajidokwu/Desktop/Drips/Fred/Ledgerlens-data",
        env={**subprocess.os.environ, "RISK_SCORE_DB_URL": db_url}
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = json.loads(result.stdout)

    # Should only have the old event
    assert len(output["events"]) == 1, "Should filter out future events"
    assert output["events"][0]["cumulative_epsilon"] == 1.0


def test_since_and_until_together(tmp_path, monkeypatch):
    """Test that --since and --until together create a date window."""
    from detection.privacy.budget_tracker import DPBudgetTracker
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from detection.privacy.budget_tracker import _Base, DPBudgetEvent

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RISK_SCORE_DB_URL", db_url)

    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, future=True)

    now = datetime.now(UTC)
    times = [
        now - timedelta(days=2),
        now - timedelta(hours=1),
        now + timedelta(hours=1),
    ]

    with session_factory() as session:
        for i, t in enumerate(times):
            session.add(DPBudgetEvent(
                kind="training",
                epsilon=0.5,
                cumulative_epsilon=0.5 * (i + 1),
                created_at=t,
                prev_log_hash=f"hash{i}" if i > 0 else "genesis",
                log_hash=f"hash{i+1}"
            ))
        session.commit()

    since_date = (now - timedelta(hours=2)).isoformat()
    until_date = (now + timedelta(minutes=30)).isoformat()

    result = subprocess.run(
        [
            "python", "-m", "scripts.query_privacy_budget",
            "--since", since_date,
            "--until", until_date,
            "--json"
        ],
        capture_output=True,
        text=True,
        cwd="/home/ajidokwu/Desktop/Drips/Fred/Ledgerlens-data",
        env={**subprocess.os.environ, "RISK_SCORE_DB_URL": db_url}
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = json.loads(result.stdout)

    # Should only have the middle event
    assert len(output["events"]) == 1, "Should filter to window with 1 event"
    assert output["events"][0]["cumulative_epsilon"] == 1.0


def test_omitting_filters_shows_all_events(tmp_path, monkeypatch):
    """Test that omitting --since/--until shows all events (backward compatible)."""
    from detection.privacy.budget_tracker import DPBudgetTracker
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from detection.privacy.budget_tracker import _Base, DPBudgetEvent

    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    monkeypatch.setenv("RISK_SCORE_DB_URL", db_url)

    engine = create_engine(db_url)
    _Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, future=True)

    now = datetime.now(UTC)

    with session_factory() as session:
        for i in range(3):
            session.add(DPBudgetEvent(
                kind="training",
                epsilon=0.5,
                cumulative_epsilon=0.5 * (i + 1),
                created_at=now - timedelta(days=i),
                prev_log_hash=f"hash{i}" if i > 0 else "genesis",
                log_hash=f"hash{i+1}"
            ))
        session.commit()

    result = subprocess.run(
        [
            "python", "-m", "scripts.query_privacy_budget",
            "--json"
        ],
        capture_output=True,
        text=True,
        cwd="/home/ajidokwu/Desktop/Drips/Fred/Ledgerlens-data",
        env={**subprocess.os.environ, "RISK_SCORE_DB_URL": db_url}
    )

    assert result.returncode == 0, f"Script failed: {result.stderr}"
    output = json.loads(result.stdout)

    # Should have all events
    assert len(output["events"]) == 3, "Should show all events when no filter applied"
