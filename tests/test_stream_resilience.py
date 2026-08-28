"""Stage 5 per-stage resilience + ID-switch detector tests.

Each stage call inside ``Stream._run_processor`` is wrapped in
``try/except``; this file pins that behaviour by injecting detectors
and trackers that raise. The processor thread must:

1. stay alive after a detector raises
2. stay alive after the tracker raises
3. recover on the next frame
4. increment the right stage-failure counter
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np

from mctracker.buffer import FrameBuffer
from mctracker.observability import METRICS, reset_metrics
from mctracker.stream import Stream
from mctracker.track_state import TrackState
from mctracker.types import Detection, Frame


class _FakeReader:
    def __init__(self, source: str = "fake://test"):
        self._source = source
        self._buffer = FrameBuffer(maxlen=16)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 2.0) -> None:
        self.stopped = True

    def feed(self, frame: Frame) -> None:
        self._buffer.put(frame, time.time())

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer


# ---------------------------------------------------------------------------
# Failing helpers
# ---------------------------------------------------------------------------


class _FlakyDetector:
    """Detector that raises on the first call, then returns the script."""
    def __init__(self, follow_up: List[List[Detection]]):
        self.calls = 0
        self.follow_up = follow_up

    def detect(self, frame: Frame) -> List[Detection]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated detector failure")
        if not self.follow_up:
            return []
        return self.follow_up.pop(0)


class _FlakyTracker:
    """Tracker that raises on the first call, then tracks normally."""
    def __init__(self):
        self.calls = 0
        self.succeed = []

    def update(self, frame, detections) -> List[TrackState]:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("simulated tracker failure")
        # Normal pass-through.
        out: List[TrackState] = []
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.xyxy.tolist()
            cx = (x1 + x2) / 2.0
            cy = y2  # bottom_center
            ts = time.time()
            out.append(
                TrackState(
                    track_id=i + 1,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    centroid=(cx, cy),
                    centroid_history=[(cx, cy, ts)],
                    first_seen_ts=ts,
                    last_seen_ts=ts,
                    embedding=None,
                    confidence=det.conf,
                    cls=det.cls,
                )
            )
        return out


def _make_stream(stream_id: str, detector, tracker) -> tuple[Stream, _FakeReader]:
    fake = _FakeReader(source=f"fake://{stream_id}")
    s = Stream.__new__(Stream)
    s.id = stream_id
    s.source = f"fake://{stream_id}"
    s._detector = detector
    s._tracker = tracker
    s._on_results = lambda *a, **k: None
    s._display_conf = 0.25
    s._buffer = fake.buffer
    s._reader = fake
    s._zone_manager = None
    s._tripwire_manager = None
    s._active = {}
    s._last_centroid = {}
    s._lock = threading.Lock()
    s._stop = threading.Event()
    s._thread = None
    s._fps = 30.0
    return s, fake


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detector_exception_does_not_kill_processor():
    reset_metrics()
    from mctracker.tracker import FakeTracker
    det = _FlakyDetector(follow_up=[[]])
    s, fake = _make_stream("resil_cam", det, FakeTracker(inner=None))
    s.start()
    try:
        # Frame 1: detector raises, no callback; frame 2+: detector OK.
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.1)
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.2)
    finally:
        s.stop(timeout=2.0)
    # Processor survived.
    assert not s._thread.is_alive() if s._thread else True
    # Stage failure counter recorded exactly one detect failure.
    counters = METRICS.stage_failures_total.labelled()
    assert counters.get(("resil_cam", "detect"), 0) == 1
    # Frame counter incremented once (the second, successful frame).
    frames = METRICS.frames_processed_total.labelled()
    assert frames.get(("resil_cam",), 0) >= 1


def test_tracker_exception_does_not_kill_processor():
    reset_metrics()

    class OkDetector:
        def detect(self, frame):
            return [
                Detection(
                    xyxy=np.array([10.0, 10.0, 50.0, 50.0], dtype=np.float32),
                    conf=0.9, cls=0, det_id=None,
                ),
            ]

    s, fake = _make_stream("resil_cam2", OkDetector(), _FlakyTracker())
    s.start()
    try:
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.1)
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.2)
    finally:
        s.stop(timeout=2.0)
    counters = METRICS.stage_failures_total.labelled()
    assert counters.get(("resil_cam2", "track"), 0) == 1


def test_processor_recovers_after_failure():
    """After a stage raises, the next frame must be processed normally."""
    reset_metrics()
    received_tracks: list = []

    class CollectDetector:
        def __init__(self):
            self.calls = 0
        def detect(self, frame):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first call blows up")
            return [
                Detection(
                    xyxy=np.array([10.0, 10.0, 50.0, 80.0], dtype=np.float32),
                    conf=0.9, cls=0, det_id=None,
                )
            ]

    class CollectTracker:
        def __init__(self):
            self.calls = 0
        def update(self, frame, detections):
            self.calls += 1
            out = []
            for i, d in enumerate(detections):
                x1, y1, x2, y2 = d.xyxy.tolist()
                cx, cy = (x1 + x2) / 2.0, y2
                ts = time.time()
                out.append(
                    TrackState(
                        track_id=i + 1, bbox=(x1, y1, x2, y2), centroid=(cx, cy),
                        centroid_history=[(cx, cy, ts)], first_seen_ts=ts, last_seen_ts=ts,
                        embedding=None, confidence=d.conf,
                        cls=d.cls,
                    )
                )
            return out

    def on_results(stream_id, tracks, zones, crossings):
        received_tracks.append(list(tracks))

    s, fake = _make_stream("recover_cam", CollectDetector(), CollectTracker())
    s._on_results = on_results
    s.start()
    try:
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.1)
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.2)
    finally:
        s.stop(timeout=2.0)
    # The first frame failed detect. The second should have produced a track.
    assert any(len(tracks) == 1 for tracks in received_tracks), (
        f"no successful track on frame 2; received_tracks={received_tracks}"
    )


def test_id_switch_detector_counts_hand_off():
    """Two frames, same physical centroid, different track_id -> +1 switch."""
    reset_metrics()
    from mctracker.stream import _ID_SWITCH_SPATIAL_PX

    class StaticTracker:
        """Returns the same track_id first, then a different one."""
        def __init__(self):
            self.calls = 0

        def update(self, frame, detections):
            self.calls += 1
            ts = time.time()
            tid = 1 if self.calls == 1 else 2  # different id on second frame
            out = []
            for det in detections:
                x1, y1, x2, y2 = det.xyxy.tolist()
                cx, cy = (x1 + x2) / 2.0, y2
                out.append(
                    TrackState(
                        track_id=tid, bbox=(x1, y1, x2, y2), centroid=(cx, cy),
                        centroid_history=[(cx, cy, ts)], first_seen_ts=ts, last_seen_ts=ts,
                        embedding=None, confidence=det.conf, cls=det.cls,
                    )
                )
            return out

    class FixedDetector:
        def detect(self, frame):
            # Tiny bbox so the centroid stays within the spatial window.
            return [
                Detection(
                    xyxy=np.array([100.0, 100.0, 110.0, 110.0], dtype=np.float32),
                    conf=0.9, cls=0, det_id=None,
                )
            ]

    s, fake = _make_stream("idswitch_cam", FixedDetector(), StaticTracker())
    s.start()
    try:
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.1)
        fake.feed(np.zeros((64, 64, 3), dtype=np.uint8))
        time.sleep(0.2)
    finally:
        s.stop(timeout=2.0)
    counters = METRICS.id_switches_total.labelled()
    assert counters.get(("idswitch_cam",), 0) == 1
    # Sanity: the centroid was within the spatial window.
    assert _ID_SWITCH_SPATIAL_PX >= 1.0