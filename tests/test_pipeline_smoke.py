"""End-to-end smoke test using only fakes (no YOLO, no boxmot)."""

from __future__ import annotations

import threading
import time

import numpy as np

from mctracker.config import AppConfig, StreamConfig
from mctracker.detector import DummyDetector
from mctracker.stream import Stream
from mctracker.tracker import FakeTracker
from mctracker.types import Detection


def _det(x, y, conf=0.9, det_id=0):
    return Detection(
        xyxy=np.array([x, y, x + 50, y + 80], dtype=np.float32),
        conf=conf,
        cls=0,
        det_id=det_id,
    )


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def build_fake_stream(stream_id, on_results):
    """Build a Stream with a fake reader, fake detector, fake tracker."""
    from ._helpers import build_stream_for_test
    detector = DummyDetector([[]])
    tracker = FakeTracker()
    stream, fake = build_stream_for_test(stream_id, detector, tracker, on_results)
    return stream, fake, detector, tracker


def test_callback_fires_per_frame():
    fired: list[tuple[str, int]] = []
    stream, fake, detector, tracker = build_fake_stream(
        "smoke1",
        lambda sid, tracks, *args, **kwargs: fired.append((sid, len(tracks))),
    )
    # Replace the detector's empty script with three frames of detections.
    detector.set_script([
        [_det(50, 50, conf=0.9), _det(200, 200, conf=0.6)],
        [_det(60, 60, conf=0.9)],
        [],
    ])
    stream.start()
    for _ in range(3):
        fake.feed(_frame())
        time.sleep(0.05)

    deadline = time.time() + 2.0
    while time.time() < deadline and len(fired) < 3:
        time.sleep(0.01)
    stream.stop()

    assert len(fired) >= 3
    assert fired[0] == ("smoke1", 2)
    assert fired[1] == ("smoke1", 1)
    assert fired[2] == ("smoke1", 0)


def test_two_streams_have_isolated_trackers():
    """Two streams built with fake detectors/trackers must not share state."""
    fired: list[tuple[str, list]] = []
    s1, f1, d1, t1 = build_fake_stream(
        "a", lambda sid, tracks, *a, **k: fired.append((sid, list(tracks)))
    )
    s2, f2, d2, t2 = build_fake_stream(
        "b", lambda sid, tracks, *a, **k: fired.append((sid, list(tracks)))
    )
    d1.set_script([[_det(10, 10, conf=0.9)]] * 3)
    d2.set_script([[_det(500, 500, conf=0.9)]] * 3)
    s1.start()
    s2.start()
    for _ in range(3):
        f1.feed(_frame())
        f2.feed(_frame())
        time.sleep(0.05)

    deadline = time.time() + 2.0
    while time.time() < deadline and len(fired) < 6:
        time.sleep(0.01)
    s1.stop()
    s2.stop()

    # Each stream's tracker should have been called with its own detections.
    assert t1.calls and t2.calls
    # The calls lists must be distinct objects.
    assert t1 is not t2
    # And the per-call detection centroids should not overlap.
    xs1 = [d.xyxy[0] for _f, dets in t1.calls for d in dets]
    xs2 = [d.xyxy[0] for _f, dets in t2.calls for d in dets]
    assert all(x < 200 for x in xs1)
    assert all(x > 400 for x in xs2)
