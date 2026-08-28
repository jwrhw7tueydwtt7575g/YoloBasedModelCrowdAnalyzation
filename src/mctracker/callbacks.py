"""Default results callback.

The default callback just logs each track. The pipeline accepts a user
callback for anything heavier (display, recording, message publishing).
"""

from __future__ import annotations

import logging
import queue
from typing import Callable, List

from .track_state import TrackState
from .tripwire import CrossingEvent
from .types import StreamId
from .zones import ZoneCount

log = logging.getLogger(__name__)


def default_callback() -> Callable[[StreamId, List[TrackState], List[ZoneCount], List[CrossingEvent]], None]:
    """Return a callback that logs each track, zone count, and crossing."""
    def _cb(
        stream_id: StreamId,
        tracks: List[TrackState],
        zone_counts: List[ZoneCount],
        crossings: List[CrossingEvent],
    ) -> None:
        for zc in zone_counts:
            log.info(
                "stream=%s zone=%s count=%d",
                stream_id, zc.zone_id, zc.count,
            )
        for ev in crossings:
            log.info(
                "crossing stream=%s tripwire=%s track=%d dir=%s",
                ev.stream_id, ev.tripwire_id, ev.track_id, ev.direction,
            )
        for t in tracks:
            log.info(
                "stream=%s track_id=%d bbox=(%.0f,%.0f,%.0f,%.0f) conf=%.2f",
                stream_id, t.track_id, *t.bbox, t.confidence,
            )
    return _cb


def queue_callback(
    q: "queue.Queue",
) -> Callable[[StreamId, List[TrackState], List[ZoneCount], List[CrossingEvent]], None]:
    """Return a callback that pushes results into a queue.Queue.

    The consumer can pop from the queue without holding any lock. queue.Queue
    is thread-safe and the preferred way to hand data from the processor
    threads to a single downstream consumer. The pushed value is a tuple
    ``(stream_id, tracks, zone_counts, crossings)``.
    """
    def _cb(
        stream_id: StreamId,
        tracks: List[TrackState],
        zone_counts: List[ZoneCount],
        crossings: List[CrossingEvent],
    ) -> None:
        q.put((stream_id, list(tracks), list(zone_counts), list(crossings)))
    return _cb
