"""Test (c): low-confidence boxes actually reach the tracker.

This is the critical correctness test for the requirement: "do not filter
before tracking; let the tracker's confidence-cascade logic handle it".
We instrument the tracker and assert that the detections passed to it
include boxes at conf=0.05 — well below any reasonable display threshold.
"""

from __future__ import annotations

import numpy as np

from mctracker.tracker import FakeTracker
from mctracker.types import Detection

from ._helpers import (
    ScriptedDetector,
    build_stream_for_test,
    make_detection,
    make_frame,
    wait_for,
)


def test_low_conf_detection_reaches_tracker():
    """A conf=0.05 box must be visible to the tracker.update() call, unchanged."""
    detector = ScriptedDetector()
    tracker = FakeTracker()
    seen: list[list[Detection]] = []
    tracker_calls = tracker.calls  # live reference

    def on_results(stream_id, tracks, zone_counts=None, crossings=None):
        # Pull a snapshot of the latest tracker call's detections
        if tracker_calls:
            seen.append(list(tracker_calls[-1][1]))

    stream, fake = build_stream_for_test("low_conf", detector, tracker, on_results)
    stream.start()

    # Push one frame with a single conf=0.05 detection.
    frame = make_frame()
    det = make_detection(conf=0.05, x1=50, y1=50, x2=200, y2=300)
    detector.set_script([[det]])
    fake.feed(frame)

    # Wait for the processor to handle the frame and invoke the callback.
    assert wait_for(lambda: len(seen) >= 1, timeout=2.0)

    stream.stop()

    # The tracker must have been called with the low-conf box intact.
    assert len(tracker.calls) >= 1, "tracker.update was never called"
    last_dets = tracker.calls[-1][1]
    assert len(last_dets) == 1
    assert last_dets[0].conf == 0.05, (
        f"low-conf detection was modified before reaching the tracker: "
        f"got conf={last_dets[0].conf}, expected 0.05"
    )


def test_multiple_low_conf_boxes_all_reach_tracker():
    """Several sub-threshold boxes in one frame all reach the tracker."""
    detector = ScriptedDetector()
    tracker = FakeTracker()
    stream, fake = build_stream_for_test("low_conf_multi", detector, tracker, lambda *a, **k: None)
    stream.start()

    dets = [
        make_detection(x1=10, y1=10, x2=60, y2=60, conf=0.05, det_id=0),
        make_detection(x1=70, y1=70, x2=120, y2=120, conf=0.02, det_id=1),
        make_detection(x1=130, y1=130, x2=200, y2=200, conf=0.10, det_id=2),
    ]
    detector.set_script([dets])
    fake.feed(make_frame())

    assert wait_for(lambda: len(tracker.calls) >= 1, timeout=2.0)
    stream.stop()

    last_dets = tracker.calls[-1][1]
    assert len(last_dets) == 3
    confs = sorted(d.conf for d in last_dets)
    assert confs == [0.02, 0.05, 0.10], (
        f"low-conf boxes were filtered or modified before the tracker; got {confs}"
    )


def test_low_conf_box_is_excluded_from_callback_output_only():
    """The display-stage conf filter (display_conf=0.25) must NOT affect what
    reaches the tracker, but it must remove low-conf boxes from the *callback*
    output. This proves the filter is applied after tracking, not before.
    """
    detector = ScriptedDetector()
    tracker = FakeTracker(inner=None)  # synthesize track ids 0..N-1
    callback_payload: list[tuple[str, list]] = []

    def on_results(stream_id, tracks, zone_counts=None, crossings=None):
        callback_payload.append((stream_id, list(tracks)))

    stream, fake = build_stream_for_test(
        "post_filter", detector, tracker, on_results, display_conf=0.25
    )
    stream.start()

    dets = [
        make_detection(x1=10, y1=10, x2=60, y2=60, conf=0.05, det_id=0),  # < display_conf
        make_detection(x1=70, y1=70, x2=120, y2=120, conf=0.80, det_id=1),  # >= display_conf
    ]
    detector.set_script([dets])
    fake.feed(make_frame())

    assert wait_for(lambda: len(callback_payload) >= 1, timeout=2.0)
    stream.stop()

    # Both detections reached the tracker.
    assert len(tracker.calls[-1][1]) == 2
    # But the callback saw only the high-conf one.
    _, callback_tracks = callback_payload[-1]
    assert len(callback_tracks) == 1
    assert callback_tracks[0].confidence == 0.80
