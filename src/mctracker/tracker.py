"""Tracker interface and boxmot-backed implementations.

The Tracker ABC is what Stream talks to. Two production implementations —
``ByteTrackTracker`` and ``BoTSORTTracker`` — wrap boxmot, each holding its
own underlying tracker object so state is per-camera. A ``FakeTracker`` is
provided for tests and records every ``update`` call so we can assert
"low-confidence boxes actually reached the tracker".

boxmot has changed import paths across versions (``boxmot.tracker_zoo`` vs
``boxmot.trackers``, ``BotSort`` vs ``BOTSORT``). We try the modern names
first and fall back to the legacy ones, so the package works against a
range of releases.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .reid import OSNetEmbedder
from .track_state import TrackState, make_track_state
from .types import Detection, Frame

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------


class Tracker(ABC):
    """Per-camera tracker. Each Stream owns exactly one instance."""

    @abstractmethod
    def update(self, frame: Frame, detections: List[Detection]) -> List[TrackState]:
        """Update the tracker with the latest detections; return current tracks.

        Implementations MUST accept every detection regardless of confidence —
        the confidence cascade lives inside the tracker.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @property
    @abstractmethod
    def active_track_ids(self) -> set[int]:
        ...


# ---------------------------------------------------------------------------
# Detection / track conversion helpers
# ---------------------------------------------------------------------------


def detections_to_array(detections: List[Detection]) -> np.ndarray:
    """Convert a list of Detection to the (N, 6) array boxmot wants.

    Columns: ``[x1, y1, x2, y2, conf, cls]``. Empty list -> shape ``(0, 6)``.
    """
    if not detections:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.zeros((len(detections), 6), dtype=np.float32)
    for i, d in enumerate(detections):
        arr[i, 0:4] = d.xyxy
        arr[i, 4] = float(d.conf)
        arr[i, 5] = float(d.cls)
    return arr


def track_array_to_states(
    tracks: np.ndarray,
    ts: float,
    embedder: Optional[OSNetEmbedder] = None,
    frame: Optional[Frame] = None,
) -> List[TrackState]:
    """Convert boxmot's (M, 8) output ``[x1,y1,x2,y2,id,conf,cls,det_ind]``
    into a list of TrackState. Optionally runs OSNet on each returned bbox
    to attach an appearance embedding.
    """
    if tracks is None or len(tracks) == 0:
        return []
    states: List[TrackState] = []
    for row in tracks:
        x1, y1, x2, y2, tid, conf, cls, _det_ind = row[:8]
        emb: np.ndarray | None = None
        if embedder is not None and frame is not None and x2 > x1 and y2 > y1:
            emb = embedder(frame, (float(x1), float(y1), float(x2), float(y2)))
        states.append(
            make_track_state(
                track_id=int(tid),
                bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(conf),
                cls=int(cls),
                ts=ts,
                embedding=emb,
            )
        )
    return states


# ---------------------------------------------------------------------------
# boxmot loader
# ---------------------------------------------------------------------------


def _load_boxmot_tracker_class(prefer: str):
    """Find a usable ByteTrack / BoT-SORT class from boxmot.

    Returns the class. Raises ImportError with a helpful message if boxmot
    is not installed or the expected names can't be found.
    """
    try:
        from boxmot import trackers as _trackers_mod  # type: ignore
    except Exception as e:  # pragma: no cover - depends on env
        raise ImportError(
            "boxmot is required for ByteTrackTracker / BoTSORTTracker. "
            "Install with: pip install boxmot"
        ) from e

    # boxmot 23.x exposes tracker registry via get_tracker_class.
    try:
        from boxmot.trackers.registry import get_tracker_class  # type: ignore
        cls = get_tracker_class(prefer)
        if cls is not None:
            return cls
    except Exception:
        pass

    # boxmot 12.x exposes ByteTrack / BoTSORT (camelcase) at the top level.
    # Older releases had BYTETracker / BOTSORT or lived in boxmot.tracker_zoo.
    candidates_bt = ["ByteTrack", "BYTETracker", "ByteTrackTracker"]
    candidates_bot = ["BoTSORT", "BOTSORT", "BotSort", "BoTSORT"]
    candidates = candidates_bt if prefer == "bytetrack" else candidates_bot

    for name in candidates:
        cls = getattr(_trackers_mod, name, None)
        if cls is not None:
            return cls

    # Last-ditch: try the legacy tracker_zoo module.
    try:
        from boxmot import tracker_zoo as _zoo  # type: ignore
    except Exception:
        _zoo = None
    if _zoo is not None:
        for name in candidates:
            cls = getattr(_zoo, name, None)
            if cls is not None:
                return cls

    raise ImportError(
        f"Could not find a {prefer} class in boxmot. Tried: {candidates}. "
        "Check your boxmot version."
    )


# ---------------------------------------------------------------------------
# Production trackers
# ---------------------------------------------------------------------------


class _BoxmotBackedTracker(Tracker):
    """Common plumbing for ByteTrack / BoT-SORT boxmot wrappers."""

    _inner_cls = None  # set in subclass

    def __init__(self, frame_rate: int = 30, with_reid: bool = False) -> None:
        if self._inner_cls is None:
            self._inner_cls = _load_boxmot_tracker_class(self._prefer)  # type: ignore[arg-type]
        kwargs = self._build_kwargs(frame_rate, with_reid)
        self._inner = self._inner_cls(**kwargs)
        self._frame_rate = int(frame_rate)
        self._with_reid = bool(with_reid)
        self._embedder: Optional[OSNetEmbedder] = (
            OSNetEmbedder() if with_reid else None
        )

    def _build_kwargs(self, frame_rate: int, with_reid: bool) -> dict:
        return {
            "frame_rate": int(frame_rate),
            "with_reid": bool(with_reid),
        }

    def update(self, frame: Frame, detections: List[Detection]) -> List[TrackState]:
        dets = detections_to_array(detections)
        ts = float(_now())
        try:
            tracks = self._inner.update(dets, frame)
        except TypeError:
            # Some boxmot versions only accept (detections, img=None) — try that.
            tracks = self._inner.update(dets)
        if tracks is None:
            return []
        if isinstance(tracks, np.ndarray) and tracks.size == 0:
            return []
        return track_array_to_states(
            tracks, ts=ts, embedder=self._embedder if self._with_reid else None, frame=frame
        )

    def reset(self) -> None:
        # boxmot trackers don't expose a clean reset in all versions;
        # the simplest correct behavior is to replace the instance.
        self._inner = self._inner_cls(**self._build_kwargs(self._frame_rate, self._with_reid))

    @property
    def active_track_ids(self) -> set[int]:
        # boxmot trackers don't keep a public "active" set. We approximate
        # by returning the most recent track ids we produced — but those are
        # produced in update(). So the Stream keeps a registry (see stream.py).
        # For tests, the FakeTracker implements this directly from its recorded calls.
        return set()


class ByteTrackTracker(_BoxmotBackedTracker):
    _prefer = "bytetrack"

    def __init__(self, frame_rate: int = 30) -> None:
        super().__init__(frame_rate=frame_rate, with_reid=False)


class BoTSORTTracker(_BoxmotBackedTracker):
    _prefer = "botsort"

    def __init__(self, frame_rate: int = 30, with_reid: bool = True) -> None:
        super().__init__(frame_rate=frame_rate, with_reid=with_reid)


# ---------------------------------------------------------------------------
# Test fake
# ---------------------------------------------------------------------------


class FakeTracker(Tracker):
    """Records every (frame, detections) pair it sees; optionally delegates to
    a real inner tracker so we can exercise the real tracking logic while
    still observing what was passed in.

    The ``active_track_ids`` property is built from the last update's return
    value — sufficient for tests that ask "is *some* track X still alive?".
    """

    def __init__(self, inner: Optional[Tracker] = None) -> None:
        self._inner = inner
        self.calls: list[tuple[Frame, list[Detection]]] = []
        self._last_track_ids: set[int] = set()

    def update(self, frame: Frame, detections: List[Detection]) -> List[TrackState]:
        self.calls.append((frame, list(detections)))
        if self._inner is None:
            # No inner: synthesize a track id per detection (stable across frames
            # isn't expected from a fake; tests that need that use an inner).
            out = [
                make_track_state(
                    track_id=i,
                    bbox_xyxy=d.xyxy,
                    conf=d.conf,
                    cls=d.cls,
                    ts=_now(),
                )
                for i, d in enumerate(detections)
            ]
        else:
            out = self._inner.update(frame, detections)
        self._last_track_ids = {s.track_id for s in out}
        return out

    def reset(self) -> None:
        self.calls.clear()
        self._last_track_ids.clear()
        if self._inner is not None:
            self._inner.reset()

    @property
    def active_track_ids(self) -> set[int]:
        return set(self._last_track_ids)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now() -> float:
    import time
    return time.time()


def make_tracker(tracker_type: str, frame_rate: int = 30, with_reid: bool = False) -> Tracker:
    """Factory used by Pipeline.build().

    ``with_reid`` is ignored for ByteTrack (no appearance).
    """
    t = tracker_type.lower().strip()
    if t == "bytetrack":
        return ByteTrackTracker(frame_rate=frame_rate)
    if t == "botsort":
        return BoTSORTTracker(frame_rate=frame_rate, with_reid=with_reid)
    raise ValueError(f"unknown tracker_type: {tracker_type!r}")
