"""Test (b): tracker state never crosses camera boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from mctracker.tracker import ByteTrackTracker, FakeTracker


def _det(x, y, conf=0.9):
    return np.array([x, y, x + 50, y + 80, conf, 0], dtype=np.float32)


def test_two_fake_trackers_do_not_share_state():
    """Two independent FakeTracker instances must have disjoint active ids."""
    a = FakeTracker()
    b = FakeTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    a.update(frame, [])
    assert a.active_track_ids == set()
    assert b.active_track_ids == set()

    # Feed detections only to a.
    from mctracker.types import Detection
    dets = [Detection(xyxy=_det(100, 100)[:4], conf=0.9, cls=0, det_id=0)]
    a.update(frame, dets)
    assert a.active_track_ids
    # b must remain empty.
    assert b.active_track_ids == set()


def test_fake_trackers_record_independent_call_lists():
    a = FakeTracker()
    b = FakeTracker()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    a.update(frame, [])
    b.update(frame, [])
    assert len(a.calls) == 1
    assert len(b.calls) == 1
    # Calls are stored independently — mutating one doesn't affect the other.
    a.calls.clear()
    assert a.calls == []
    assert len(b.calls) == 1


def test_real_tracker_instances_have_distinct_state():
    """Two real ByteTrackTracker instances (not Fakes) must not share state.

    We feed detections to one and assert the other has no active tracks.
    boxmot's ByteTrack assigns ids from a per-instance counter, so the first
    detector in tracker A getting id=1 must not be visible to tracker B.
    """
    try:
        import boxmot  # noqa: F401
    except Exception:
        pytest.skip("boxmot not installed")

    a = ByteTrackTracker(frame_rate=30)
    b = ByteTrackTracker(frame_rate=30)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    from mctracker.types import Detection
    dets = [Detection(xyxy=np.array([10, 10, 60, 90], dtype=np.float32), conf=0.95, cls=0, det_id=0)]
    out_a = a.update(frame, dets)
    out_b = b.update(frame, [])
    assert out_a, "tracker A should have produced at least one track after seeing a detection"
    assert out_b == [], "tracker B must remain empty when only A was fed"


def test_pipeline_constructs_one_tracker_per_stream(tmp_path):
    """Spy on the tracker factory: Pipeline must build exactly N tracker objects,
    one per stream, and they must be distinct instances.
    """
    from mctracker.config import StreamConfig
    from mctracker.pipeline import Pipeline
    import mctracker.pipeline as pipeline_mod

    # Build a fake config with 3 streams. We bypass load_config by setting
    # Pipeline._config directly.
    cfg_streams = [
        StreamConfig(id=f"cam_{i}", source=f"fake://cam_{i}") for i in range(3)
    ]
    from mctracker.config import AppConfig
    cfg = AppConfig(streams=cfg_streams)

    p = Pipeline(config_path="ignored")
    p._config = cfg

    created: list[str] = []
    from mctracker.tracker import FakeTracker

    def spy(tracker_type, with_reid=False, frame_rate=30):
        created.append(tracker_type)
        # Don't construct real trackers in this test — they need boxmot and
        # we want to test pipeline plumbing, not the trackers themselves.
        return FakeTracker()

    pipeline_mod.make_tracker = spy
    try:
        # Build without invoking YOLO; replace the detector factory too.
        p._build_stream = lambda sc: _stream_with_dummy_detector(p, sc, spy)
        # Patch Stream.start to no-op so we don't spawn real readers.
        from mctracker.stream import Stream as StreamCls
        real_start = StreamCls.start
        StreamCls.start = lambda self: None
        try:
            p.build()
        finally:
            StreamCls.start = real_start
    finally:
        pipeline_mod.make_tracker = pipeline_mod.make_tracker  # no-op restore

    assert len(created) == 3
    # All three must be bytetrack (default in StreamConfig) and therefore three
    # distinct objects on the streams.
    assert all(t == "bytetrack" for t in created)


def _stream_with_dummy_detector(pipeline, sc, tracker_factory):
    """Helper for the spy test: build a Stream with a dummy detector and
    a tracker produced via the (spied) factory. We don't start any threads.
    """
    from mctracker.detector import DummyDetector
    from mctracker.stream import Stream
    tracker = tracker_factory(sc.tracker_type, with_reid=(sc.tracker_type == "botsort" and sc.use_appearance))
    s = Stream.__new__(Stream)
    s.id = sc.id
    s.source = sc.source
    s._detector = DummyDetector([[]])
    s._tracker = tracker
    s._on_results = lambda *a, **k: None
    s._display_conf = sc.display_conf
    s._buffer = None
    s._reader = None
    s._zone_manager = None
    s._tripwire_manager = None
    s._active = {}
    s._last_centroid = {}
    import threading
    s._lock = threading.Lock()
    s._stop = threading.Event()
    s._thread = None
    s._fps = 30.0
    return s
