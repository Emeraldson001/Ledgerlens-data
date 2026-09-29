"""Tests for detection.audit_trail.AuditMerkleChain (Issue #670).

Covers the restart-rehydration fix: a routine process restart must not be
indistinguishable from real tampering, and a genuine tamper must still be
detected after rehydration.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from detection.audit_trail import (
    AuditChainIncompleteError,
    AuditMerkleChain,
    TamperDetectedError,
    _MerkleBase,
    _MerkleRootRecord,
)


def _session_factory_for(tmp_path, name="audit_merkle_test.db"):
    engine = create_engine(f"sqlite:///{tmp_path / name}")
    _MerkleBase.metadata.create_all(engine, checkfirst=True)
    return sessionmaker(bind=engine, future=True)


def test_append_and_verify_fresh_chain(tmp_path):
    factory = _session_factory_for(tmp_path)
    chain = AuditMerkleChain(session_factory=factory)

    for i in range(5):
        chain.append({"wallet": f"G{i}", "score": i * 10})

    assert chain.verify_chain() is True


def test_restart_rehydrates_and_verify_chain_still_succeeds(tmp_path):
    """The core fix: a fresh AuditMerkleChain() pointed at the same durable
    storage as a prior (now-discarded) instance must rehydrate its entries
    and verify successfully — simulating a process restart, not tampering.
    """
    factory = _session_factory_for(tmp_path)

    first_process_chain = AuditMerkleChain(session_factory=factory)
    for i in range(10):
        first_process_chain.append({"wallet": f"G{i}", "score": i})
    del first_process_chain  # simulate the process exiting

    second_process_chain = AuditMerkleChain(session_factory=factory)
    assert len(second_process_chain._entries) == 10
    assert second_process_chain.verify_chain() is True


def test_restart_then_append_more_entries_still_verifies(tmp_path):
    factory = _session_factory_for(tmp_path)

    chain_a = AuditMerkleChain(session_factory=factory)
    for i in range(3):
        chain_a.append({"i": i})
    del chain_a

    chain_b = AuditMerkleChain(session_factory=factory)
    for i in range(3, 6):
        chain_b.append({"i": i})

    assert len(chain_b._entries) == 6
    assert chain_b.verify_chain() is True

    chain_c = AuditMerkleChain(session_factory=factory)
    assert len(chain_c._entries) == 6
    assert chain_c.verify_chain() is True


def test_genuine_tamper_still_detected_after_restart(tmp_path):
    """A real tamper (row's stored merkle_root or content_hash altered
    directly in the DB) must still raise TamperDetectedError, not be
    confused with or masked by the rehydration fix.
    """
    factory = _session_factory_for(tmp_path)

    chain_a = AuditMerkleChain(session_factory=factory)
    for i in range(5):
        chain_a.append({"i": i})
    del chain_a

    # Tamper with entry index 2's content_hash directly in durable storage.
    with factory() as session:
        row = session.query(_MerkleRootRecord).filter(_MerkleRootRecord.entry_index == 2).one()
        row.content_hash = "0" * 64
        session.commit()

    chain_b = AuditMerkleChain(session_factory=factory)
    with pytest.raises(TamperDetectedError):
        chain_b.verify_chain()


def test_missing_row_raises_tamper_not_incomplete(tmp_path):
    """A deleted row (fewer persisted roots than expected) is data loss /
    tampering, not the benign "predates migration 0005" case."""
    factory = _session_factory_for(tmp_path)

    chain_a = AuditMerkleChain(session_factory=factory)
    for i in range(4):
        chain_a.append({"i": i})
    del chain_a

    with factory() as session:
        row = session.query(_MerkleRootRecord).filter(_MerkleRootRecord.entry_index == 3).one()
        session.delete(row)
        session.commit()

    chain_b = AuditMerkleChain(session_factory=factory)
    assert len(chain_b._entries) == 3  # index 3's row is gone
    with pytest.raises(TamperDetectedError):
        chain_b.verify_chain(end_index=4)


def test_legacy_gap_raises_incomplete_not_tamper(tmp_path):
    """A row written before migration 0005 (content_hash IS NULL) must
    surface AuditChainIncompleteError, not TamperDetectedError, when a
    verify_chain range crosses it — and append() must refuse to build on top
    of unknown history rather than silently computing a meaningless root.
    """
    factory = _session_factory_for(tmp_path)

    with factory() as session:
        session.add(
            _MerkleRootRecord(
                entry_index=0,
                merkle_root="a" * 64,
                content_hash=None,  # simulates a pre-migration-0005 row
                prev_merkle_root=None,
                created_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            )
        )
        session.commit()

    chain = AuditMerkleChain(session_factory=factory)
    assert len(chain._entries) == 1
    assert chain._entries[0].content_hash is None

    with pytest.raises(AuditChainIncompleteError):
        chain.verify_chain()

    with pytest.raises(AuditChainIncompleteError):
        chain.append({"wallet": "GNEW"})
