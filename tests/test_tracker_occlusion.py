"""Test (a): a track ID survives 10 frames of full occlusion when using BoT-SORT.

We construct a sequence of frames:
    3 frames with a person at (100,100)-(200,200)
    10 frames with no detections (full occlusion)
    3 frames with the person reappearing at the same location

and assert that the track_id produced in the first segment matches the
track_id produced in the last segment.

The test is gated on boxmot being installed; without it, BoT-SORT can't run
and we skip rather than fail. ByteTrack does NOT pass this test (it's
motion-only) — we mark that case xfail to document the expectation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mctracker.tracker import ByteTrackTracker, BoTSORTTracker, FakeTracker
from mctracker.types import Detection


def _det_at(x, y, conf=0.9, det_id=0):
    return Detection(
        xyxy=np.array([x, y, x + 100, y + 200], dtype=np.float32),
        conf=conf,
        cls=0,
        det_id=det_id,
    )


def _frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _run_sequence(tracker, person_frames: int, empty_frames: int):
    """Run a synthetic (visible, occluded, visible) sequence through tracker.

    Returns the set of track_ids seen in the first person segment and in the
    last person segment.
    """
    frame = _frame()
    first_segment_ids: set[int] = set()
    last_segment_ids: set[int] = set()

    # First visible segment.
    for _ in range(person_frames):
        tracks = tracker.update(frame, [_det_at(100, 100)])
        first_segment_ids.update(t.track_id for t in tracks)

    # Occlusion: no detections.
    for _ in range(empty_frames):
        tracks = tracker.update(frame, [])
        first_segment_ids.update(t.track_id for t in tracks)

    # Reappear at the same place.
    for _ in range(person_frames):
        tracks = tracker.update(frame, [_det_at(100, 100)])
        last_segment_ids.update(t.track_id for t in tracks)

    return first_segment_ids, last_segment_ids


def test_botsort_survives_10_frame_occlusion():
    try:
        import boxmot  # noqa: F401
    except Exception:
        pytest.skip("boxmot not installed")

    tracker = BoTSORTTracker(frame_rate=30, with_reid=False)
    first, last = _run_sequence(tracker, person_frames=3, empty_frames=10)
    assert first, "expected a track to be created in the first visible segment"
    assert last, "expected a track to be re-emitted in the reappear segment"
    assert first & last, (
        f"track IDs do not survive 10-frame occlusion: first={first}, last={last}"
    )


@pytest.mark.xfail(reason="ByteTrack is motion-only; full occlusion breaks it by design")
def test_bytetrack_does_not_survive_10_frame_occlusion():
    try:
        import boxmot  # noqa: F401
    except Exception:
        pytest.skip("boxmot not installed")

    tracker = ByteTrackTracker(frame_rate=30)
    first, last = _run_sequence(tracker, person_frames=3, empty_frames=10)
    # We expect this assertion to FAIL (the xfail is the point of the test):
    assert first & last, "ByteTrack accidentally survived full occlusion"


def test_fake_tracker_passes_occlusion_with_no_inner():
    """A FakeTracker without an inner tracker assigns stable synthetic ids per
    detection. The test verifies the framework's plumbing around occlusion
    works end-to-end, even without a real tracker.
    """
    tracker = FakeTracker(inner=None)
    first, last = _run_sequence(tracker, person_frames=3, empty_frames=10)
    # FakeTracker emits a fresh id per call when no inner is provided, so the
    # ids will not match across segments — but each segment will be non-empty.
    assert first and last
