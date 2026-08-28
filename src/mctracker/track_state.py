"""Per-track state and a small active-track registry.

TrackState is the public, library-agnostic representation of a tracked person.
A Stream keeps a dict of these indexed by track_id so callers (and tests) can
inspect history.

The dict-of-deques design makes it easy to assert: "track X was at position P
at time T, then again at Q ten frames later" — exactly what the occlusion test
needs.
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

DEFAULT_HISTORY = 30  # frames of centroid history to retain


def _centroid_from_xyxy(xyxy: np.ndarray | tuple[float, float, float, float]) -> tuple[float, float]:
    arr = np.asarray(xyxy, dtype=np.float32)
    return float((arr[0] + arr[2]) / 2.0), float((arr[1] + arr[3]) / 2.0)


@dataclass
class TrackState:
    """State of a single track.

    The fields are kept small and primitive so the dataclass is easy to log,
    serialize, or send over a queue.
    """

    track_id: int
    bbox: tuple[float, float, float, float]
    centroid: tuple[float, float]
    confidence: float
    cls: int
    first_seen_ts: float
    last_seen_ts: float
    centroid_history: collections.deque = field(
        default_factory=lambda: collections.deque(maxlen=DEFAULT_HISTORY)
    )
    embedding: np.ndarray | None = None  # populated only when appearance tracking is on

    @property
    def age_seconds(self) -> float:
        return self.last_seen_ts - self.first_seen_ts

    @property
    def is_alive(self) -> bool:
        """True if the track was seen very recently. Streams maintain this externally."""
        # Without a stream context we don't know — callers should consult the
        # stream's active_ids set. This property exists for symmetry / future use.
        return True


def make_track_state(
    track_id: int,
    bbox_xyxy: Iterable[float],
    conf: float,
    cls: int,
    ts: float,
    embedding: np.ndarray | None = None,
    history_maxlen: int = DEFAULT_HISTORY,
) -> TrackState:
    """Build a TrackState, guarding against NaN / inf coming out of the tracker."""
    bbox = tuple(float(v) for v in bbox_xyxy)
    if not all(np.isfinite(bbox)):
        # boxmot can emit NaN on the first frame; clamp to a zero-sized box at origin
        bbox = (0.0, 0.0, 0.0, 0.0)
    cx, cy = _centroid_from_xyxy(bbox)
    history: collections.deque = collections.deque(maxlen=history_maxlen)
    history.append((cx, cy, ts))
    return TrackState(
        track_id=int(track_id),
        bbox=bbox,
        centroid=(cx, cy),
        confidence=float(conf),
        cls=int(cls),
        first_seen_ts=ts,
        last_seen_ts=ts,
        centroid_history=history,
        embedding=embedding,
    )
