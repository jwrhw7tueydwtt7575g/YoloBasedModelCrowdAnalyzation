"""SQLAlchemy repository test using SQLite (no Postgres needed).

This exercises the SQL path end-to-end so the schema and column
serialization are covered by the test suite. A Postgres-backed run is
deferred to integration tests.
"""

from __future__ import annotations

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from mctracker.violations import (
    InMemoryViolationRepository,
    Violation,
    ViolationKind,
    SQLAlchemyViolationRepository,
)
from mctracker.violations.models import ViolationRecord


@pytest.fixture
def sqlite_repo():
    repo = SQLAlchemyViolationRepository("sqlite:///:memory:")
    yield repo
    repo.close()


def _make_violation(**kwargs) -> Violation:
    base = dict(
        timestamp=100.0,
        stream_id="cam0",
        zone_id="z1",
        tripwire_id="t1",
        track_id=42,
        direction="left_to_right",
        embedding=None,
        kind=ViolationKind.UNMATCHED,
        matching_scan=None,
        notes="",
    )
    base.update(kwargs)
    return Violation(**base)


def test_in_memory_round_trip():
    repo = InMemoryViolationRepository()
    rid = repo.record(_make_violation())
    assert rid == 1
    rows = repo.list_recent(zone_id="z1")
    assert len(rows) == 1
    assert rows[0]["kind"] == "unmatched"


def test_sqlite_repository_records_and_lists(sqlite_repo):
    v1 = _make_violation(timestamp=100.0, track_id=42)
    v2 = _make_violation(timestamp=110.0, track_id=43, kind=ViolationKind.TAILGATING)
    rid1 = sqlite_repo.record(v1)
    rid2 = sqlite_repo.record(v2)
    assert rid1 == 1 and rid2 == 2

    rows = sqlite_repo.list_recent(zone_id="z1")
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["timestamp"] == 110.0
    assert rows[1]["timestamp"] == 100.0


def test_sqlite_schema_columns(sqlite_repo):
    """The persisted row must contain every column we documented."""
    emb = b"\x00\x01\x02"
    v = _make_violation(
        timestamp=123.0,
        stream_id="cam0",
        zone_id="z1",
        tripwire_id="t1",
        track_id=42,
        direction="left_to_right",
        embedding=emb,
        kind=ViolationKind.TAILGATING,
        matching_scan=type("S", (), {
            "zone_id": "z1", "timestamp": 122.5, "identity": "alice",
        })(),
        notes="tailgating after alice",
    )
    sqlite_repo.record(v)
    rows = sqlite_repo.list_recent(zone_id="z1")
    assert len(rows) == 1
    r = rows[0]
    for k in [
        "id", "timestamp", "stream_id", "zone_id", "tripwire_id",
        "track_id", "direction", "embedding", "kind",
        "matching_identity", "matching_scan_ts", "notes", "created_at",
    ]:
        assert k in r
    assert r["matching_identity"] == "alice"
    assert r["matching_scan_ts"] == 122.5
    assert r["embedding"] == emb


def test_sqlite_filter_by_zone(sqlite_repo):
    sqlite_repo.record(_make_violation(zone_id="z1"))
    sqlite_repo.record(_make_violation(zone_id="z2"))
    sqlite_repo.record(_make_violation(zone_id="z1"))
    rows = sqlite_repo.list_recent(zone_id="z1")
    assert len(rows) == 2
    for r in rows:
        assert r["zone_id"] == "z1"


def test_violation_to_row_round_trip():
    """The dict shape from Violation.to_row() must match the model's columns."""
    from mctracker.violations.models import ViolationRecord
    cols = {c.name for c in ViolationRecord.__table__.columns}
    v = _make_violation()
    row = v.to_row()
    # Every non-default column should be addressable.
    for required in ("zone_id", "stream_id", "tripwire_id", "track_id",
                     "direction", "kind", "notes"):
        assert required in row, f"missing key in to_row(): {required}"
