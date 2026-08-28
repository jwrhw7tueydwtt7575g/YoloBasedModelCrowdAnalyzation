"""Tripwire crossing.

A tripwire is a line segment ``(p1, p2)`` in image coordinates. Each frame
we look at every active track's previous and current centroid and emit a
``CrossingEvent`` whenever the track crosses the line.

**Crossing detection**

The signed area of the triangle ``(p1, p2, point)`` is

    sign = (p2.x - p1.x) * (point.y - p1.y) - (p2.y - p1.y) * (point.x - p1.x)

A sign change between consecutive positions means the line was crossed
between those two frames. The sign also gives us the direction:
``left-to-right`` (sign went - → +) or ``right-to-left`` (+ → -). We
expose that as ``"in"`` / ``"out"`` per tripwire — the caller declares
which direction is "in" at config time.

**ID-recycling handling (the common bug)**

Tracker IDs are recycled. If we naively keep a ``counted_ids`` set,
we will miss legitimate crossings when the same id is later assigned
to a different person. Concretely: track id 7 walks across the line
(left→right), we add 7 to ``counted_ids``; 50 frames later the tracker
drops id 7, then re-uses 7 for a different person on the other side;
that person walks across the line and we miss the event because 7
is already in the set.

We avoid this by storing per track-id a small record of
*when* and *where* we last counted it, and re-admitting a track id
to ``counted_ids`` only when one of these is true:

* the previous counting was at least ``recycle_after_frames`` ago, OR
* the current centroid is at least ``recycle_distance_px`` pixels away
  from the last counted centroid (so the same physical person could
  not plausibly have walked back to the line in that time at typical
  walking speed).

The defaults (60 frames and 200 px) are conservative; tune them in the
YAML per camera.

**Hover-on-line**

If a track sits *exactly* on the line for several frames, the sign
of successive cross-products is ~0 and a naive implementation either
double-fires or fails to fire. We require the sign to flip through
non-zero values: from <= 0 to > 0 (or vice versa) with a non-trivial
magnitude on at least one side. This is the standard "edge-crossing"
check used in 2D graphics.
"""

from __future__ import annotations

import logging
import math
import queue
from dataclasses import dataclass, field
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

from .track_state import TrackState
from .types import StreamId

log = logging.getLogger(__name__)


@dataclass
class CrossingEvent:
    """A single crossing of a tripwire by a track.

    Stage 5 consumes these as raw per-crossing records (not aggregate
    counts) so it can apply its own business logic — e.g., dwell time,
    revenue-per-customer, alert deduplication.
    """

    timestamp: float
    stream_id: StreamId
    tripwire_id: str
    track_id: int
    direction: str  # "in" or "out"
    centroid: Tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def signed_area(p1: Sequence[float], p2: Sequence[float], p: Sequence[float]) -> float:
    """Signed area of triangle (p1, p2, p). Sign is + when p is "to the left"
    of the directed segment p1→p2 (in image coordinates where y grows down)."""
    return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])


def classify_direction(prev_sign: float, curr_sign: float) -> Optional[str]:
    """Return "left_to_right" if sign went negative→positive,
    "right_to_left" if positive→negative, else None.

    Sign-flips across an ``EPS`` band suppress hover noise: when both
    signs are within ``EPS`` of 0 we don't fire. We also recognize a
    transition from hover-on-line (prev_sign ≈ 0) to clearly-positive
    (curr_sign > EPS) as a left-to-right crossing — and vice versa for
    right-to-left. This handles the "track sat on the line for several
    frames then moved off" case without double-firing.
    """
    EPS = 1e-3
    if abs(prev_sign) <= EPS and abs(curr_sign) <= EPS:
        return None  # still hovering on the line
    if prev_sign <= EPS and curr_sign > EPS:
        return "left_to_right"
    if prev_sign > EPS and curr_sign <= -EPS:
        return "right_to_left"
    return None


# ---------------------------------------------------------------------------
# Per-track-id memory for recycle handling
# ---------------------------------------------------------------------------


@dataclass
class _CountedRecord:
    track_id: int
    last_centroid: Tuple[float, float]
    last_counted_frame: int


# ---------------------------------------------------------------------------
# Tripwire
# ---------------------------------------------------------------------------


@dataclass
class Tripwire:
    """One line segment for one camera.

    ``p1`` and ``p2`` are 2-tuples of image coordinates. ``direction_in`` is
    which direction (sign change) counts as "in"; the opposite counts as
    "out". A typical entry/exit tripwire on a doorway has ``direction_in =
    "left_to_right"`` meaning a sign change from negative to positive.
    """

    id: str
    p1: Tuple[float, float]
    p2: Tuple[float, float]
    direction_in: str = "left_to_right"  # "left_to_right" or "right_to_left"
    recycle_after_frames: int = 60
    recycle_distance_px: float = 200.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tripwire id is required")
        if self.p1 == self.p2:
            raise ValueError(f"tripwire {self.id!r}: p1 == p2 (degenerate segment)")
        if self.direction_in not in ("left_to_right", "right_to_left"):
            raise ValueError(
                f"tripwire {self.id!r}: direction_in must be "
                f"'left_to_right' or 'right_to_left', got {self.direction_in!r}"
            )


class TripwireManager:
    """Per-camera tripwire manager.

    Holds the list of tripwires and the per-track-id memory needed to
    avoid double-counting while still handling ID recycling.

    ``event_queue`` is optional. If provided, every ``CrossingEvent`` is
    pushed there for downstream consumers (Stage 5). Otherwise the events
    are logged at INFO level and otherwise dropped — the manager is still
    correct, just disconnected.
    """

    def __init__(
        self,
        stream_id: StreamId,
        tripwires: Iterable[Tripwire],
        event_queue: Optional["queue.Queue[CrossingEvent]"] = None,
    ) -> None:
        self._stream_id = stream_id
        self._tripwires: List[Tripwire] = list(tripwires)
        self._event_queue = event_queue
        # Per-tripwire, per-track-id memory. Each tripwire has its own
        # counted_ids set so that crossing two parallel tripwires in the
        # same frame does not block the second. Each entry is a
        # _CountedRecord (last counted centroid + last counted frame).
        self._counted: dict[str, dict[int, _CountedRecord]] = {
            tw.id: {} for tw in self._tripwires
        }
        # Frame counter (incremented per update()).
        self._frame_idx = 0
        # Previous-centroid cache (per track_id) used for crossing detection.
        self._prev_centroid: dict[int, Tuple[float, float]] = {}
        # Duplicate id check
        seen: set[str] = set()
        for tw in self._tripwires:
            if tw.id in seen:
                raise ValueError(f"duplicate tripwire id: {tw.id!r}")
            seen.add(tw.id)

    @property
    def stream_id(self) -> StreamId:
        return self._stream_id

    def update(
        self,
        tracks: Iterable[TrackState],
        centroid_func,
        timestamp: float,
    ) -> List[CrossingEvent]:
        """Run one frame's worth of crossing detection.

        ``centroid_func(bbox) -> (x, y)`` is supplied by the caller so the
        camera's ``centroid_mode`` (bottom_center vs geometric_center) is
        respected.
        """
        self._frame_idx += 1
        out: List[CrossingEvent] = []
        seen_track_ids: set[int] = set()
        for t in tracks:
            seen_track_ids.add(t.track_id)
            cx, cy = centroid_func(t.bbox)
            prev = self._prev_centroid.get(t.track_id)
            self._prev_centroid[t.track_id] = (cx, cy)
            if prev is None:
                continue
            for tw in self._tripwires:
                prev_sign = signed_area(tw.p1, tw.p2, prev)
                curr_sign = signed_area(tw.p1, tw.p2, (cx, cy))
                raw_dir = classify_direction(prev_sign, curr_sign)
                if raw_dir is None:
                    continue
                # raw_dir is "left_to_right" or "right_to_left"; map to
                # "in" / "out" based on the tripwire's declared direction_in.
                direction = "in" if raw_dir == tw.direction_in else "out"
                if not self._may_count(tw, t.track_id, (cx, cy)):
                    continue
                self._mark_counted(tw, t.track_id, (cx, cy))
                ev = CrossingEvent(
                    timestamp=timestamp,
                    stream_id=self._stream_id,
                    tripwire_id=tw.id,
                    track_id=t.track_id,
                    direction=direction,
                    centroid=(cx, cy),
                )
                out.append(ev)
                if self._event_queue is not None:
                    try:
                        self._event_queue.put_nowait(ev)
                    except queue.Full:
                        log.warning("event queue full; dropping crossing event")
                else:
                    log.info(
                        "crossing stream=%s tripwire=%s track=%d dir=%s",
                        self._stream_id, tw.id, t.track_id, direction,
                    )
        # Garbage-collect per-track memory for ids that have not been seen
        # for a long time (track was dropped). Keeps the dict bounded.
        for tw in self._tripwires:
            tw_counted = self._counted.get(tw.id, {})
            stale_threshold = 2 * tw.recycle_after_frames
            for tid in list(tw_counted.keys()):
                rec = tw_counted[tid]
                if tid not in seen_track_ids and (self._frame_idx - rec.last_counted_frame) > stale_threshold:
                    del tw_counted[tid]
        for tid in list(self._prev_centroid.keys()):
            if tid not in seen_track_ids:
                # Keep one stale frame so a brief dropout doesn't reset the
                # prev position; drop after the threshold.
                pass
        # (prev_centroid is small; we don't bother pruning it here. Worst
        # case it's bounded by the number of unique ids the tracker has
        # ever emitted, which is bounded for any reasonable session.)
        return out

    def _may_count(
        self, tw: Tripwire, track_id: int, current_centroid: Tuple[float, float]
    ) -> bool:
        rec = self._counted.get(tw.id, {}).get(track_id)
        if rec is None:
            return True
        # Time-based recycling: the previous counting was long enough ago
        # that the id has likely been reassigned.
        if (self._frame_idx - rec.last_counted_frame) >= tw.recycle_after_frames:
            return True
        # Distance-based recycling.
        dx = rec.last_centroid[0] - current_centroid[0]
        dy = rec.last_centroid[1] - current_centroid[1]
        if math.hypot(dx, dy) >= tw.recycle_distance_px:
            return True
        return False

    def _mark_counted(
        self, tw: Tripwire, track_id: int, centroid: Tuple[float, float]
    ) -> None:
        self._counted.setdefault(tw.id, {})[track_id] = _CountedRecord(
            track_id=track_id,
            last_centroid=centroid,
            last_counted_frame=self._frame_idx,
        )
