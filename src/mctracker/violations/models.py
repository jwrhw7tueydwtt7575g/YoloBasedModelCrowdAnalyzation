"""SQLAlchemy ORM models for violation persistence.

The violation service writes to a ``ViolationRecord`` row whenever a
crossing is determined to be a violation. The schema is intentionally
narrow and append-only: each row is one crossing, annotated with the
violation kind and (optionally) the matching-scan identity. The
``embedding`` column is BYTEA — bytes from the tracker's appearance
embedding for BoT-SORT tracks.
"""

from __future__ import annotations

import datetime as _dt

try:
    _TZ_AWARE = _dt.timezone.utc
except AttributeError:  # pragma: no cover
    _TZ_AWARE = None

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


# SQLite's autoincrement only fires for `INTEGER PRIMARY KEY` (a rowid alias).
# Use BigInteger() with the dialect-specific variant so Postgres still gets
# a real BIGINT but SQLite gets INTEGER-with-autoincrement. We always test
# against SQLite in this repo, so we choose Integer-with-BigInteger-variant.
_IDType = BigInteger().with_variant(Integer(), "sqlite")


class ViolationRecord(Base):
    """A single persisted violation.

    Schema:
        id                 - primary key (auto-increment)
        timestamp          - crossing timestamp (epoch seconds)
        stream_id          - camera / stream identifier
        zone_id            - associated zone
        tripwire_id        - the tripwire that produced the crossing
        track_id           - tracker id of the subject
        direction          - "left_to_right" / "right_to_left" / etc.
        embedding          - optional appearance embedding (BYTEA)
        kind               - "unmatched" or "tailgating"
        matching_identity  - if tailgating, the identity that scanned
        matching_scan_ts   - timestamp of the matching scan
        notes              - human-readable context
        created_at         - DB insert time
    """

    __tablename__ = "violations"

    id = Column(_IDType, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    stream_id = Column(String(128), nullable=False)
    zone_id = Column(String(128), nullable=False)
    tripwire_id = Column(String(128), nullable=False)
    track_id = Column(Integer, nullable=False)
    direction = Column(String(32), nullable=False)
    embedding = Column(LargeBinary, nullable=True)
    kind = Column(String(32), nullable=False)
    matching_identity = Column(String(128), nullable=True)
    matching_scan_ts = Column(Float, nullable=True)
    notes = Column(String(1024), nullable=False, default="")
    clip_path = Column(String(1024), nullable=True)
    clip_url = Column(String(1024), nullable=True)
    created_at = Column(
        DateTime,
        default=lambda: _dt.datetime.now(tz=_dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_violations_zone_ts", "zone_id", "timestamp"),
        Index("ix_violations_stream_ts", "stream_id", "timestamp"),
        Index("ix_violations_kind", "kind"),
    )

    def __repr__(self) -> str:
        return (
            f"ViolationRecord(id={self.id}, kind={self.kind!r}, "
            f"zone={self.zone_id!r}, track={self.track_id}, "
            f"ts={self.timestamp})"
        )


class HighDensityViolationRecord(Base):
    """A persisted "high-density" / crowd alert.

    Distinct from ``ViolationRecord`` because it does not correspond to a
    crossing event — it's triggered when the live occupancy count in a
    zone stays above the configured threshold for longer than
    ``density_dwell_seconds``. The clip is the same evidence pipeline
    (pre+post ring buffer + post-event capture).

    Schema:
        id                  - primary key (auto-increment)
        timestamp           - alert fire timestamp (epoch seconds)
        stream_id           - camera / stream identifier
        zone_id             - the zone that crossed the threshold
        density_count       - peak count observed during the dwell
        threshold           - configured max_density_threshold
        dwell_seconds       - actual time the count stayed above threshold
        clip_path           - optional local path to the saved MP4
        clip_url            - optional public URL (S3 / file URL)
        notes               - operator-visible context
        created_at          - DB insert time
    """

    __tablename__ = "high_density_violations"

    id = Column(_IDType, primary_key=True, autoincrement=True)
    timestamp = Column(Float, nullable=False)
    stream_id = Column(String(128), nullable=False)
    zone_id = Column(String(128), nullable=False)
    density_count = Column(Integer, nullable=False)
    threshold = Column(Integer, nullable=False)
    dwell_seconds = Column(Float, nullable=False)
    clip_path = Column(String(1024), nullable=True)
    clip_url = Column(String(1024), nullable=True)
    notes = Column(String(1024), nullable=False, default="")
    created_at = Column(
        DateTime,
        default=lambda: _dt.datetime.now(tz=_dt.timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_high_density_zone_ts", "zone_id", "timestamp"),
        Index("ix_high_density_stream_ts", "stream_id", "timestamp"),
    )

    def __repr__(self) -> str:
        return (
            f"HighDensityViolationRecord(id={self.id}, zone={self.zone_id!r}, "
            f"count={self.density_count}, threshold={self.threshold})"
        )


def make_engine(url: str):
    """Create a SQLAlchemy engine for a Postgres URL.

    Imported lazily here so tests that don't need Postgres don't pull in
    the driver. Caller is responsible for installing psycopg/psycopg2.
    """
    return create_engine(url, future=True)


def make_session_factory(engine):
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)
