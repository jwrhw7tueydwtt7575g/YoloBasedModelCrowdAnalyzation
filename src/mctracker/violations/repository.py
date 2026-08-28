"""Repositories for violation persistence.

Two implementations share the same interface:

* ``InMemoryViolationRepository`` — used by tests and as the default when
  no Postgres URL is configured.
* ``ViolationRepository`` — SQLAlchemy-backed, works against any engine
  with the ``violations`` table (created via ``Base.metadata.create_all``).

The repository interface is small on purpose: ``record(violation)`` and
``list_recent(zone_id, limit)`` (read-only, useful for the dashboard and
tests).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, List, Optional, Protocol

from .models import Base, HighDensityViolationRecord, ViolationRecord, make_engine, make_session_factory
from .rules import HighDensityViolation, Violation


class ViolationRepository(Protocol):
    """Minimal interface a repository must implement."""

    def record(self, violation: Violation) -> int:
        """Persist ``violation``. Returns the row id."""

    def attach_clip(self, violation_id: int, clip_path: str, clip_url: str) -> None:
        """Attach the saved clip location to the persisted row.

        Implemented as a separate call (rather than folding into
        ``record``) so the recorder can run *after* the violation is
        persisted — the violation id is needed before the clip file
        can be written, but the clip file is large and slow to encode.
        """

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        """Return recent violations, newest first.

        Each returned dict has the shape produced by ``Violation.to_row()``
        plus ``id`` and ``created_at``.
        """

    def close(self) -> None:
        """Release any underlying resources."""


class InMemoryViolationRepository:
    """Pure-Python repository used by tests and as the no-DB default."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._next_id: int = 1

    def record(self, violation: Violation) -> int:
        row = {"id": self._next_id, "clip_path": None, "clip_url": None, **violation.to_row()}
        self._rows.append(row)
        self._next_id += 1
        return row["id"]

    def attach_clip(self, violation_id: int, clip_path: str, clip_url: str) -> None:
        for r in self._rows:
            if r["id"] == violation_id:
                r["clip_path"] = clip_path
                r["clip_url"] = clip_url
                return
        # If record wasn't called first, the row may not exist yet; tests
        # call attach_clip immediately and that's fine.

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        rows = self._rows
        if zone_id is not None:
            rows = [r for r in rows if r["zone_id"] == zone_id]
        rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)
        return rows[:limit]

    def close(self) -> None:
        return None


class SQLAlchemyViolationRepository:
    """Postgres-backed repository."""

    def __init__(self, engine_url: str, ensure_schema: bool = True) -> None:
        self._engine = make_engine(engine_url)
        self._Session = make_session_factory(self._engine)
        if ensure_schema:
            Base.metadata.create_all(self._engine)

    def record(self, violation: Violation) -> int:
        row = ViolationRecord(**_filter_model_kwargs(violation.to_row()))
        with self._Session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return int(row.id)

    def attach_clip(self, violation_id: int, clip_path: str, clip_url: str) -> None:
        from sqlalchemy import update

        with self._Session() as s:
            stmt = (
                update(ViolationRecord)
                .where(ViolationRecord.id == violation_id)
                .values(clip_path=clip_path, clip_url=clip_url)
            )
            s.execute(stmt)
            s.commit()

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        from sqlalchemy import select

        with self._Session() as s:
            stmt = select(ViolationRecord).order_by(
                ViolationRecord.timestamp.desc()
            ).limit(limit)
            if zone_id is not None:
                stmt = stmt.where(ViolationRecord.zone_id == zone_id)
            rows = s.execute(stmt).scalars().all()
        return [_row_to_dict(r) for r in rows]

    def close(self) -> None:
        self._engine.dispose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _filter_model_kwargs(d: dict) -> dict:
    """Keep only columns that exist on ViolationRecord (ignore extras)."""
    cols = {c.name for c in ViolationRecord.__table__.columns}
    return {k: v for k, v in d.items() if k in cols}


def _row_to_dict(row: ViolationRecord) -> dict:
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "stream_id": row.stream_id,
        "zone_id": row.zone_id,
        "tripwire_id": row.tripwire_id,
        "track_id": row.track_id,
        "direction": row.direction,
        "embedding": row.embedding,
        "kind": row.kind,
        "matching_identity": row.matching_identity,
        "matching_scan_ts": row.matching_scan_ts,
        "notes": row.notes,
        "clip_path": row.clip_path,
        "clip_url": row.clip_url,
        "created_at": row.created_at,
    }


# ---------------------------------------------------------------------------
# HighDensity repositories (Stage 6)
# ---------------------------------------------------------------------------


class HighDensityRepository(Protocol):
    """Persistence for high-density / crowd alerts.

    Mirror image of ``ViolationRepository`` but with different columns.
    """

    def record(self, alert: HighDensityViolation) -> int:
        """Persist ``alert``. Returns the row id."""

    def attach_clip(self, alert_id: int, clip_path: str, clip_url: str) -> None:
        """Attach the saved clip location to the persisted row."""

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        """Return recent alerts, newest first."""

    def close(self) -> None:
        """Release any underlying resources."""


class InMemoryHighDensityRepository:
    """Pure-Python high-density repository."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._next_id: int = 1

    def record(self, alert: HighDensityViolation) -> int:
        row = {"id": self._next_id, "clip_path": None, "clip_url": None, **alert.to_row()}
        self._rows.append(row)
        self._next_id += 1
        return row["id"]

    def attach_clip(self, alert_id: int, clip_path: str, clip_url: str) -> None:
        for r in self._rows:
            if r["id"] == alert_id:
                r["clip_path"] = clip_path
                r["clip_url"] = clip_url
                return

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        rows = self._rows
        if zone_id is not None:
            rows = [r for r in rows if r["zone_id"] == zone_id]
        rows = sorted(rows, key=lambda r: r["timestamp"], reverse=True)
        return rows[:limit]

    def close(self) -> None:
        return None


class SQLAlchemyHighDensityRepository:
    """Postgres / SQLite-backed high-density repository."""

    def __init__(self, engine_url: str, ensure_schema: bool = True) -> None:
        self._engine = make_engine(engine_url)
        self._Session = make_session_factory(self._engine)
        if ensure_schema:
            Base.metadata.create_all(self._engine)

    def record(self, alert: HighDensityViolation) -> int:
        row = HighDensityViolationRecord(**_filter_hd_kwargs(alert.to_row()))
        with self._Session() as s:
            s.add(row)
            s.commit()
            s.refresh(row)
            return int(row.id)

    def attach_clip(self, alert_id: int, clip_path: str, clip_url: str) -> None:
        from sqlalchemy import update

        with self._Session() as s:
            stmt = (
                update(HighDensityViolationRecord)
                .where(HighDensityViolationRecord.id == alert_id)
                .values(clip_path=clip_path, clip_url=clip_url)
            )
            s.execute(stmt)
            s.commit()

    def list_recent(
        self, zone_id: Optional[str] = None, limit: int = 100
    ) -> List[dict]:
        from sqlalchemy import select

        with self._Session() as s:
            stmt = select(HighDensityViolationRecord).order_by(
                HighDensityViolationRecord.timestamp.desc()
            ).limit(limit)
            if zone_id is not None:
                stmt = stmt.where(HighDensityViolationRecord.zone_id == zone_id)
            rows = s.execute(stmt).scalars().all()
        return [_hd_row_to_dict(r) for r in rows]

    def close(self) -> None:
        self._engine.dispose()


def _filter_hd_kwargs(d: dict) -> dict:
    cols = {c.name for c in HighDensityViolationRecord.__table__.columns}
    return {k: v for k, v in d.items() if k in cols}


def _hd_row_to_dict(row: HighDensityViolationRecord) -> dict:
    return {
        "id": row.id,
        "timestamp": row.timestamp,
        "stream_id": row.stream_id,
        "zone_id": row.zone_id,
        "density_count": row.density_count,
        "threshold": row.threshold,
        "dwell_seconds": row.dwell_seconds,
        "clip_path": row.clip_path,
        "clip_url": row.clip_url,
        "notes": row.notes,
        "created_at": row.created_at,
    }
