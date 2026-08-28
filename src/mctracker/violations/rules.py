"""Violation rules: correlate crossings with QR/checkpoint scans and flag
tailgating; track high-density (crowd) conditions in zones.

The ViolationService is the heart of the violation detection stage. It
consumes one crossing at a time (from the tripwire event queue), asks the
``ScanStore`` for scans in the same zone within a ±window, and decides:

* **Authorized**: exactly one crossing is paired with a single scan within
  the window, and the zone occupancy at scan time is consistent with a
  single person crossing.
* **Unmatched (no scan)**: violation — no scan in window.
* **Tailgating**: scan exists in window, but a prior authorized crossing
  already consumed it during this window. Any crossing after the first
  within the scan's window is a violation, cross-checked against zone
  occupancy at scan time (if occupancy was N>1 at scan time, that itself
  suggests tailgating).

Out-of-order tolerance: scans arriving after the crossing still match, as
long as the lookup uses the crossing's own timestamp. The ``ScanStore``
already keeps scans alive for ``max_window_seconds`` after their timestamp
to absorb network jitter.

**High-density / crowd alert (Stage 6)**:

``DensityRule`` watches per-zone occupancy snapshots. When a zone's count
stays above ``max_density_threshold`` for at least ``density_dwell_seconds``
it emits a ``HighDensityViolation``. A per-zone cooldown prevents
spam: a fresh alert for the same zone cannot fire until the count drops
below the threshold AND ``cooldown_seconds`` have passed since the
previous alert.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Optional

from .scan_store import Scan, ScanStore


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ViolationKind(str, Enum):
    UNMATCHED = "unmatched"
    TAILGATING = "tailgating"


@dataclass
class CrossingRecord:
    """The minimal crossing info needed for violation correlation.

    The pipeline's ``CrossingEvent`` carries more (centroid, frame index),
    so this is what the rule layer needs from the consumer side.
    """

    timestamp: float
    stream_id: str
    zone_id: str
    tripwire_id: str
    track_id: int
    direction: str
    embedding: Optional[bytes] = None


@dataclass
class ZoneOccupancySnapshot:
    """Zone occupancy at a particular timestamp — cross-check for tailgating.

    The service receives fresh occupancy each time a crossing is processed;
    the consumer (pipeline or test) is expected to look up the most recent
    occupancy for the same zone and pass it back. We do not embed zone
    occupancy inside ScanStore because occupancy is camera-derived, not
    scan-derived.
    """

    timestamp: float
    zone_id: str
    count: int
    stream_id: str = ""  # only used by DensityRule; default empty for crossings


@dataclass
class ScanSnapshot:
    """Subset of a Scan used to keep tailgating state — captures identity
    so we can tell if the same or a different person scanned.
    """

    zone_id: str
    timestamp: float
    identity: str


# ---------------------------------------------------------------------------
# High-density / crowd alert
# ---------------------------------------------------------------------------


@dataclass
class HighDensityViolation:
    """A produced high-density alert ready for persistence + evidence capture.

    Distinct from ``Violation`` because it doesn't correspond to a crossing
    event — there's no ``track_id`` or ``tripwire_id``. The relevant info
    is the peak occupancy count, the threshold it crossed, and how long
    the count stayed above the threshold.
    """

    timestamp: float
    stream_id: str
    zone_id: str
    density_count: int
    threshold: int
    dwell_seconds: float
    notes: str = ""

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "stream_id": self.stream_id,
            "zone_id": self.zone_id,
            "density_count": self.density_count,
            "threshold": self.threshold,
            "dwell_seconds": self.dwell_seconds,
            "notes": self.notes,
        }


@dataclass
class _ZoneDensityState:
    """Per-zone dwell-tracking state."""

    threshold: int
    dwell_required: float
    cooldown_seconds: float
    # When did the count first go above the threshold? None if currently
    # below or right at the threshold (i.e. not accumulating dwell).
    above_since_ts: Optional[float] = None
    # Peak count observed during the current above-threshold run.
    peak_count: int = 0
    # When did we last fire an alert for this zone?
    last_fired_ts: Optional[float] = None
    # Has an alert already fired during the current above-threshold run?
    fired: bool = False


class DensityRule:
    """Per-zone occupancy → high-density alert evaluator.

    The rule is purely stateful: feed it ``ZoneOccupancySnapshot``s and it
    returns a ``HighDensityViolation`` when one fires, otherwise ``None``.
    Multiple zones can be tracked simultaneously; the per-zone state is
    keyed by ``(stream_id, zone_id)``.

    A fire-and-dismiss flow:

        rule = DensityRule(threshold=5, dwell_seconds=2.0, cooldown_seconds=10.0)
        for snap in snapshots:
            v = rule.observe(snap)
            if v is not None:
                recorder.record(v)   # capture pre+post evidence

    The cooldown is enforced *after* a fire: even if the count stays
    above the threshold, the rule won't fire again until the count drops
    below and ``cooldown_seconds`` have passed.
    """

    def __init__(
        self,
        threshold: int = 5,
        dwell_seconds: float = 2.0,
        cooldown_seconds: float = 10.0,
        on_violation: Optional[Callable[[HighDensityViolation], None]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be > 0")
        if dwell_seconds < 0:
            raise ValueError("dwell_seconds must be >= 0")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be >= 0")
        self._threshold = int(threshold)
        self._dwell_seconds = float(dwell_seconds)
        self._cooldown_seconds = float(cooldown_seconds)
        self._on_violation = on_violation
        self._clock = clock or (lambda: __import__("time").time())
        self._lock = threading.Lock()
        self._states: dict[tuple[str, str], _ZoneDensityState] = {}

    @property
    def threshold(self) -> int:
        return self._threshold

    @property
    def dwell_seconds(self) -> float:
        return self._dwell_seconds

    def observe(self, snap: ZoneOccupancySnapshot) -> Optional[HighDensityViolation]:
        """Update per-zone state from a single occupancy snapshot.

        Returns a ``HighDensityViolation`` if the rule fires; otherwise
        ``None``. If ``on_violation`` was configured, it is also invoked.
        """
        with self._lock:
            key = (snap.stream_id, snap.zone_id)
            state = self._states.get(key)
            if state is None:
                state = _ZoneDensityState(
                    threshold=self._threshold,
                    dwell_required=self._dwell_seconds,
                    cooldown_seconds=self._cooldown_seconds,
                )
                self._states[key] = state

            count = int(snap.count)
            now = float(snap.timestamp)

            # Above-threshold: start or extend the dwell.
            if count > state.threshold:
                if state.above_since_ts is None:
                    state.above_since_ts = now
                if count > state.peak_count:
                    state.peak_count = count
                if state.fired:
                    return None
                # Check dwell + cooldown.
                dwell = now - state.above_since_ts
                cooldown_ok = (
                    state.last_fired_ts is None
                    or (now - state.last_fired_ts) >= state.cooldown_seconds
                )
                if dwell >= state.dwell_required and cooldown_ok:
                    v = HighDensityViolation(
                        timestamp=now,
                        stream_id=snap.stream_id,
                        zone_id=snap.zone_id,
                        density_count=state.peak_count,
                        threshold=state.threshold,
                        dwell_seconds=dwell,
                        notes=(
                            f"count peaked at {state.peak_count} "
                            f"(threshold {state.threshold}) "
                            f"for {dwell:.2f}s"
                        ),
                    )
                    state.last_fired_ts = now
                    state.fired = True
                    self._invoke_callback(v)
                    return v
                return None

            # At or below threshold: reset dwell accumulation.
            state.above_since_ts = None
            state.peak_count = 0
            state.fired = False
            return None

    def reset_zone(self, stream_id: str, zone_id: str) -> None:
        """Clear dwell + cooldown state for a single zone."""
        with self._lock:
            self._states.pop((stream_id, zone_id), None)

    def reset(self) -> None:
        with self._lock:
            self._states.clear()

    def _invoke_callback(self, v: HighDensityViolation) -> None:
        if self._on_violation is None:
            return
        try:
            self._on_violation(v)
        except Exception:
            v.notes = (v.notes + " | on_violation callback raised").strip()


def density_snapshots_from_zone_counts(
    stream_id: str,
    zone_counts: Iterable,
    timestamp: float,
):
    """Convert a list of ``ZoneCount`` (from ``mctracker.zones``) into
    ``ZoneOccupancySnapshot``s with ``stream_id`` set.

    Returns snapshots ready for ``DensityRule.observe``.
    """
    from mctracker.zones import ZoneCount  # local import (avoid cycles)

    out = []
    for zc in zone_counts:
        if not isinstance(zc, ZoneCount):
            continue
        out.append(
            ZoneOccupancySnapshot(
                timestamp=timestamp,
                zone_id=zc.zone_id,
                count=int(zc.count),
                stream_id=stream_id,
            )
        )
    return out


@dataclass
class Violation:
    """A produced violation record ready for persistence."""

    timestamp: float
    stream_id: str
    zone_id: str
    tripwire_id: str
    track_id: int
    direction: str
    embedding: Optional[bytes]
    kind: ViolationKind
    matching_scan: Optional[ScanSnapshot] = None
    notes: str = ""

    def to_row(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "stream_id": self.stream_id,
            "zone_id": self.zone_id,
            "tripwire_id": self.tripwire_id,
            "track_id": self.track_id,
            "direction": self.direction,
            "embedding": self.embedding,
            "kind": self.kind.value,
            "matching_identity": (
                self.matching_scan.identity if self.matching_scan else None
            ),
            "matching_scan_ts": (
                self.matching_scan.timestamp if self.matching_scan else None
            ),
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ViolationService:
    """Stateful per-zone crossing-to-scan correlator.

    Per-zone, we keep a small ring of recent crossable scans that have
    *not yet* had a crossing paired with them (a "claimable" scan). When
    a crossing arrives in the window, we claim the first such scan. Any
    *subsequent* crossing in the same window against the *same* scan is
    flagged as tailgating.

    Threads: the service is intended to be called from a single consumer
    thread (e.g., one asyncio-loop or one worker), but the ScanStore is
    thread-safe. We add a small lock for safety.
    """

    def __init__(
        self,
        scan_store: ScanStore,
        window_seconds: float = 10.0,
        on_violation: Optional[Callable[[Violation], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        max_window_seconds: float = 60.0,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._scan_store = scan_store
        self._window_seconds = float(window_seconds)
        self._on_violation = on_violation
        self._clock = clock or (lambda: __import__("time").time())
        self._lock = threading.Lock()

        # Per-zone, the queue of recently-arrived claims still open.
        # Each entry is (expires_at, scan).
        self._open_scans: dict[str, deque] = {}

        # Set of (zone_id, scan_identity, scan_ts) already used to count
        # the *first* authorized crossing against a given scan. Used to
        # distinguish authorized crossings from tailgating.
        self._consumed: dict[tuple, list[tuple[float, int]]] = {}
        # value is list of (scan_consume_ts, track_id); we keep all the
        # crossings that "consumed" a scan so that occupied state stays
        # correct, but we only count the first one as authorized.

    # ------------------------------------------------------------------
    # Ingest scans: called when a new scan lands.
    # ------------------------------------------------------------------

    def register_scan(self, scan: Scan) -> None:
        """A new scan has been recorded. Add it to the open-claim pool.

        The store has already accepted it — this method tracks it for
        tailgating purposes so we can decide whether a crossing within
        the window is "first authorized" or "tailgating".
        """
        if scan.zone_id not in self._open_scans:
            self._open_scans[scan.zone_id] = deque()

        # Store its expiry as (timestamp + window, scan).
        expiry = scan.timestamp + self._window_seconds
        self._open_scans[scan.zone_id].append((expiry, scan))

    # ------------------------------------------------------------------
    # Process a crossing: the main entry point.
    # ------------------------------------------------------------------

    def process_crossing(
        self,
        crossing: CrossingRecord,
        occupancy: Optional[ZoneOccupancySnapshot] = None,
    ) -> Optional[Violation]:
        """Decide whether ``crossing`` is authorized or a violation.

        Returns the violation if one should be recorded, or None if the
        crossing is authorized. If ``on_violation`` was configured, also
        invokes it.
        """
        with self._lock:
            self._drop_expired_open_scans(crossing.zone_id, crossing.timestamp)

            scans = self._scan_store.query_window(
                crossing.zone_id, crossing.timestamp, self._window_seconds
            )

            if not scans.scans:
                return self._emit_violation(
                    crossing=crossing,
                    kind=ViolationKind.UNMATCHED,
                    matching_scan=None,
                    notes="no scan in window",
                    occupancy=occupancy,
                )

            # Pick the closest scan (smallest |Δt|). For tailgating detection
            # we then check if this scan has already been consumed.
            best = min(
                scans.scans, key=lambda s: abs(s.timestamp - crossing.timestamp)
            )

            consumed_key = (crossing.zone_id, best.timestamp, best.identity)
            existing = self._consumed.get(consumed_key, [])

            # If the scan exists but a previous crossing *already* consumed
            # it (i.e. the first crossing had this scan in its window), then
            # the current crossing is either:
            #   - the same person walking back-and-forth but with a different
            #     track_id (unusual), or
            #   - a second person crossing within the same window → TAILGATING.
            if existing:
                # Was this the same track as the authorized crossing? If so,
                # and the timestamps are far enough apart (>= window), it's
                # a re-crossing, authorized. But within the window, it's
                # tailgating.
                within_window = any(
                    abs(ts - crossing.timestamp) <= self._window_seconds
                    for ts, _tid in existing
                )
                if within_window:
                    self._consumed[consumed_key].append(
                        (crossing.timestamp, crossing.track_id)
                    )
                    return self._emit_violation(
                        crossing=crossing,
                        kind=ViolationKind.TAILGATING,
                        matching_scan=ScanSnapshot(
                            zone_id=best.zone_id,
                            timestamp=best.timestamp,
                            identity=best.identity,
                        ),
                        notes=(
                            f"scan already authorized for "
                            f"{len(existing)} prior crossing(s)"
                        ),
                        occupancy=occupancy,
                    )

            # First authorized crossing against this scan.
            self._consumed[consumed_key] = existing + [
                (crossing.timestamp, crossing.track_id)
            ]

            # Optionally cross-check occupancy: if more than one person was
            # in the zone at scan time, we still record this as authorized
            # (only the scan holder is, by definition — others tailgate),
            # but we record a note for the operator.
            note = ""
            if occupancy is not None and occupancy.count > 1:
                note = (
                    f"zone had {occupancy.count} people at scan time; "
                    f"treating this as authorized but flagging context"
                )
            self._register_open_scan_for_consumed(consumed_key, crossing.timestamp)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _register_open_scan_for_consumed(
        self, consumed_key: tuple, crossing_timestamp: float
    ) -> None:
        """When a scan has been consumed, mark its claim window for cleanup.

        The consumed record is enough on its own — ``_drop_expired_open_scans``
        in ``process_crossing`` handles expiration based on wall-clock-ish
        observations. This method exists so we can do time-based pruning
        asynchronously if needed.
        """
        # We don't strictly need to do anything here; the tuple key
        # includes the scan's own timestamp, and old entries can be
        # pruned by ``prune`` below.
        pass

    def _drop_expired_open_scans(self, zone_id: str, now: float) -> None:
        dq = self._open_scans.get(zone_id)
        if dq is None:
            return
        while dq and dq[0][0] < now - self._window_seconds:
            dq.popleft()

    def prune(self, before_ts: float) -> None:
        """Drop consumed-scan bookkeeping older than ``before_ts``.

        Bounded cleanup so the in-memory dicts don't grow forever. Safe to
        call periodically (e.g., on every Nth crossing).
        """
        with self._lock:
            for k in list(self._consumed.keys()):
                _zone, scan_ts, _id = k
                if scan_ts < before_ts - self._window_seconds:
                    self._consumed.pop(k, None)

            for zone_id in list(self._open_scans.keys()):
                dq = self._open_scans[zone_id]
                while dq and dq[0][0] < before_ts - self._window_seconds:
                    dq.popleft()
                if not dq:
                    self._open_scans.pop(zone_id, None)

    def _emit_violation(
        self,
        crossing: CrossingRecord,
        kind: ViolationKind,
        matching_scan: Optional[ScanSnapshot],
        notes: str,
        occupancy: Optional[ZoneOccupancySnapshot],
    ) -> Violation:
        v = Violation(
            timestamp=crossing.timestamp,
            stream_id=crossing.stream_id,
            zone_id=crossing.zone_id,
            tripwire_id=crossing.tripwire_id,
            track_id=crossing.track_id,
            direction=crossing.direction,
            embedding=crossing.embedding,
            kind=kind,
            matching_scan=matching_scan,
            notes=notes,
        )
        if self._on_violation is not None:
            try:
                self._on_violation(v)
            except Exception:
                # The callback must not break the violation pipeline.
                # Surface as a notes addition; in production you'd want
                # to log this somewhere.
                v.notes = (v.notes + " | on_violation callback raised").strip()
        return v


# ---------------------------------------------------------------------------
# Async / iterator wiring
# ---------------------------------------------------------------------------


def consume_crossing_queue(
    service: ViolationService,
    crossing_queue,
    occupancy_provider: Optional[
        Callable[[str, float], Optional[ZoneOccupancySnapshot]]
    ] = None,
    poll_timeout: float = 0.1,
) -> None:
    """Drain crossings from a queue-like (``queue.Queue`` or
    ``asyncio.Queue``-like) object and feed them to ``service``.

    Blocking. Designed to be run on a dedicated consumer thread or asyncio
    task. ``crossing_queue.get`` must accept a ``timeout`` argument.

    For each crossing, if ``occupancy_provider(zone_id, ts)`` returns a
    snapshot, it's passed to ``service.process_crossing``.
    """
    while True:
        try:
            item = crossing_queue.get(timeout=poll_timeout)
        except Exception:
            continue
        # Allow a sentinel to stop the loop.
        if item is None:
            return
        crossing = _coerce_crossing(item)
        if crossing is None:
            continue
        occ = None
        if occupancy_provider is not None:
            try:
                occ = occupancy_provider(crossing.zone_id, crossing.timestamp)
            except Exception:
                occ = None
        service.process_crossing(crossing, occupancy=occ)


def _coerce_crossing(item) -> Optional[CrossingRecord]:
    """Accept either a CrossingRecord, a CrossingEvent, or a tuple/dict."""
    if isinstance(item, CrossingRecord):
        return item
    if hasattr(item, "timestamp") and hasattr(item, "track_id") and hasattr(item, "stream_id"):
        return CrossingRecord(
            timestamp=float(item.timestamp),
            stream_id=str(item.stream_id),
            zone_id=str(getattr(item, "zone_id", "") or getattr(item, "tripwire_id", "default")),
            tripwire_id=str(getattr(item, "tripwire_id", "") or getattr(item, "zone_id", "")),
            track_id=int(item.track_id),
            direction=str(getattr(item, "direction", "unknown")),
            embedding=getattr(item, "embedding", None),
        )
    if isinstance(item, dict):
        return CrossingRecord(
            timestamp=float(item["timestamp"]),
            stream_id=str(item["stream_id"]),
            zone_id=str(item.get("zone_id") or item.get("tripwire_id") or "default"),
            tripwire_id=str(item.get("tripwire_id") or item.get("zone_id") or "default"),
            track_id=int(item["track_id"]),
            direction=str(item.get("direction", "unknown")),
            embedding=item.get("embedding"),
        )
    return None
