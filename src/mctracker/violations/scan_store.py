"""Time-indexed scan store.

Each scan is a structured record ``{zone_id, timestamp, identity}`` from
an external QR / checkpoint system. We index scans by ``(zone_id, timestamp)``
so a crossing event can ask "is there a scan in this zone within ±window
seconds of my crossing timestamp?" in O(log n + k) time.

We use ``sortedcontainers.SortedDict`` keyed by timestamp, with one
dict per zone. This is in-memory and process-local — for a production
deployment across multiple workers, replace this with a Postgres-backed
implementation that exposes the same `ScanStore` protocol.

For arrival-order jitter (a scan arriving *after* a crossing due to
network delay), the store does not delete scans as soon as they pass
out of the lookup window. A scan's expiry time is
``timestamp + max_window_seconds``; we drop it only after that. So a
crossing whose scan arrived late is still matchable for up to
``max_window_seconds`` after the scan's original timestamp.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Iterable, Optional

from sortedcontainers import SortedDict


@dataclass
class Scan:
    """A single QR / checkpoint scan event."""

    zone_id: str
    timestamp: float
    identity: str

    def __post_init__(self) -> None:
        if not self.zone_id:
            raise ValueError("scan zone_id is required")
        if not self.identity:
            raise ValueError("scan identity is required")


@dataclass
class ScansInWindow:
    """Result of a window query."""

    scans: list[Scan]

    @property
    def matched(self) -> bool:
        return len(self.scans) > 0

    def __len__(self) -> int:
        return len(self.scans)


class ScanStore:
    """In-memory time-indexed scan store.

    Thread-safe for one writer (the webhook handler) and any number of
    readers (the violation service).

    Parameters
    ----------
    max_window_seconds:
        Scans are retained for at least this long after their timestamp.
        This bounds how long a late-arriving scan can still match a
        crossing. The crossing's own ±window is bounded separately by
        the ViolationService.
    """

    def __init__(self, max_window_seconds: float = 60.0) -> None:
        self._max_window_seconds = float(max_window_seconds)
        # Per-zone SortedDict, keyed by timestamp. irange() gives O(log n + k)
        # range scans.
        self._by_zone: dict[str, SortedDict] = {}
        self._lock = threading.Lock()

    def add(self, scan: Scan) -> None:
        """Record one scan. Late-arriving scans are accepted (kept) so that
        a scan arriving after its crossing can still match when the crossing
        is reprocessed.

        No pruning here — pruning should be done explicitly via
        ``prune(before_ts)``. Doing pruning implicitly on every write would
        drop late-arriving scans whose timestamps are still inside any
        lookback window.
        """
        with self._lock:
            d = self._by_zone.setdefault(scan.zone_id, SortedDict())
            d[scan.timestamp] = scan

    def prune(self, before_ts: float) -> int:
        """Drop entries whose timestamp is strictly less than ``before_ts``.

        Returns the number of entries dropped. Safe to call from a
        maintenance thread; the violation service exposes the same
        contract.
        """
        with self._lock:
            dropped = 0
            for zone_id in list(self._by_zone.keys()):
                d = self._by_zone[zone_id]
                stale = [k for k in d.keys() if k < before_ts]
                for k in stale:
                    del d[k]
                    dropped += 1
                if not d:
                    self._by_zone.pop(zone_id, None)
            return dropped

    def query_window(
        self,
        zone_id: str,
        center_ts: float,
        window_seconds: float,
    ) -> ScansInWindow:
        """Return all scans in ``zone_id`` whose timestamp is in
        ``[center_ts - window_seconds, center_ts + window_seconds]``.

        Empty list if no scans are in that range. Half-open intervals on
        each side (so a scan and a crossing landing on exactly the same
        edge are matched — see exact-boundary test).
        """
        if window_seconds < 0:
            raise ValueError("window_seconds must be >= 0")
        with self._lock:
            d = self._by_zone.get(zone_id)
            if d is None:
                return ScansInWindow(scans=[])
            lo = center_ts - window_seconds
            hi = center_ts + window_seconds
            out: list[Scan] = []
            for k in d.irange(lo, hi, inclusive=(True, True)):
                out.append(d[k])
        return ScansInWindow(scans=out)

    def zone_count(self, zone_id: str, at_ts: float, window_seconds: float = 1.0) -> int:
        """Convenience: how many scans for ``zone_id`` happened within
        ``±window_seconds`` of ``at_ts``. Used for tailgating cross-checks
        and tests.
        """
        return len(self.query_window(zone_id, at_ts, window_seconds))

    def scan_count_by_zone(self, zone_id: str) -> int:
        with self._lock:
            d = self._by_zone.get(zone_id)
            return len(d) if d is not None else 0


# ---------------------------------------------------------------------------
# Webhook handler
# ---------------------------------------------------------------------------


class ScanWebhookHandler:
    """Receives external QR / checkpoint scans and writes them to a ScanStore.

    Uses FastAPI when available; in environments without FastAPI (e.g.,
    during pure unit tests), exposes a plain ``ingest(payload)`` method so
    tests can drive the same code path without spinning up an HTTP server.
    """

    def __init__(self, store: ScanStore) -> None:
        self._store = store

    def ingest(self, payload: dict) -> Scan:
        """Validate ``payload`` and add it to the store.

        Expected payload keys: ``zone_id``, ``timestamp``, ``identity``.
        Timestamps may be UNIX seconds (float) or ISO 8601 strings; we
        parse the ISO format lazily.
        """
        zone_id = str(payload.get("zone_id", ""))
        identity = str(payload.get("identity", ""))
        ts_raw = payload.get("timestamp")
        ts = _parse_timestamp(ts_raw)
        scan = Scan(zone_id=zone_id, timestamp=ts, identity=identity)
        self._store.add(scan)
        return scan

    def build_app(self):
        """Return a FastAPI app exposing ``POST /scans`` and ``GET /health``.

        Imports are lazy so that the violation service can be used
        without FastAPI installed (FastAPI is an optional dependency in
        pyproject).
        """
        try:
            from fastapi import FastAPI, HTTPException  # type: ignore
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "FastAPI is required to build the webhook HTTP app. "
                "Install with: pip install fastapi uvicorn"
            ) from e

        app = FastAPI(title="mctracker scan webhook")
        handler = self

        @app.get("/health")
        def health():
            return {"status": "ok"}

        @app.post("/scans")
        def post_scan(payload: dict):
            try:
                scan = handler.ingest(payload)
            except (ValueError, TypeError) as e:
                raise HTTPException(status_code=400, detail=str(e))
            return {
                "zone_id": scan.zone_id,
                "timestamp": scan.timestamp,
                "identity": scan.identity,
            }

        return app


def _parse_timestamp(raw) -> float:
    if raw is None:
        raise ValueError("scan payload missing 'timestamp'")
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        # Try ISO 8601 first; fall back to float string.
        try:
            from datetime import datetime
            # Python's fromisoformat handles "+00:00" but not "Z"; normalize.
            s = raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.timestamp()
        except Exception:
            pass
        try:
            return float(raw)
        except Exception as e:
            raise ValueError(f"could not parse timestamp {raw!r}") from e
    raise ValueError(f"timestamp must be a number or ISO string, got {type(raw).__name__}")
