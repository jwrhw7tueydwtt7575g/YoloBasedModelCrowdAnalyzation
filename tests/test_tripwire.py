"""Tripwire crossing tests."""

from __future__ import annotations

import queue

import numpy as np

from mctracker.track_state import make_track_state
from mctracker.tripwire import (
    CrossingEvent,
    Tripwire,
    TripwireManager,
    classify_direction,
    signed_area,
)
from mctracker.zones import bottom_center, geometric_center


def _track(track_id: int, bbox, ts: float = 0.0):
    return make_track_state(
        track_id=track_id,
        bbox_xyxy=bbox,
        conf=0.9,
        cls=0,
        ts=ts,
    )


def _bbox_from_centroid(cx: float, cy: float, w: int = 40, h: int = 80):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# Horizontal tripwire from (0, 200) to (400, 200). Crossing from above (y<200)
# to below (y>200) gives a sign change of - → + (left-to-right of the
# directed segment is irrelevant; we use the cross product sign).
HORIZONTAL = ((0, 200), (400, 200))


def test_signed_area_sign_changes_across_line():
    # p1=(0,200), p2=(400,200). Above the line: y<200 → negative sign.
    above = signed_area(HORIZONTAL[0], HORIZONTAL[1], (200, 100))
    below = signed_area(HORIZONTAL[0], HORIZONTAL[1], (200, 300))
    assert above < 0 and below > 0


def test_classify_direction_negative_to_positive():
    assert classify_direction(-10.0, 10.0) == "left_to_right"


def test_classify_direction_positive_to_negative():
    assert classify_direction(10.0, -10.0) == "right_to_left"


def test_classify_direction_no_change_returns_none():
    assert classify_direction(-10.0, -5.0) is None
    assert classify_direction(5.0, 10.0) is None


def test_classify_direction_hover_on_line_returns_none():
    """Sign of 0 on both sides means the track is still on (or just past)
    the line; we don't fire to avoid double-counting.

    The "crossing out of hover" case (0 → +5) is treated as a valid
    crossing — see test_tripwire_hover_on_line_does_not_double_fire.
    """
    assert classify_direction(0.0, 0.0) is None
    # 0 → +5: coming off hover, valid crossing.
    assert classify_direction(0.0, 5.0) == "left_to_right"
    # Both inside the eps band: hover / no crossing.
    assert classify_direction(-0.0005, 0.0005) is None
    assert classify_direction(-1e-6, 1e-6) is None


def test_tripwire_diagonal_crossing():
    """A track moving diagonally across the line fires exactly once."""
    tw = Tripwire(id="door", p1=HORIZONTAL[0], p2=HORIZONTAL[1], direction_in="left_to_right")
    mgr = TripwireManager("cam1", [tw])

    # Above the line first (y=100), then below (y=300).
    t1 = _track(1, _bbox_from_centroid(200, 100))
    events = mgr.update([t1], bottom_center, timestamp=0.0)
    assert events == []  # first frame: no prev

    t2 = _track(1, _bbox_from_centroid(200, 300))
    events = mgr.update([t2], bottom_center, timestamp=0.1)
    assert len(events) == 1
    assert events[0].direction == "in"
    assert events[0].track_id == 1


def test_tripwire_cross_then_immediately_re_cross_back():
    """Cross L→R, then R→L: two events, opposite directions."""
    tw = Tripwire(id="t", p1=HORIZONTAL[0], p2=HORIZONTAL[1])
    mgr = TripwireManager("cam1", [tw])

    # Frame 0: above line
    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    # Frame 1: cross to below
    evs1 = mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.1)
    assert len(evs1) == 1 and evs1[0].direction == "in"
    # Frame 2: cross back to above
    evs2 = mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.2)
    assert len(evs2) == 1 and evs2[0].direction == "out"


def test_tripwire_double_count_blocked_within_same_track():
    """A second frame where the track stays on the opposite side must NOT
    re-fire.
    """
    tw = Tripwire(id="t", p1=HORIZONTAL[0], p2=HORIZONTAL[1])
    mgr = TripwireManager("cam1", [tw])

    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.1)  # fires
    evs = mgr.update([_track(1, _bbox_from_centroid(200, 400))], bottom_center, timestamp=0.2)
    assert evs == [], "second crossing in the same direction should not re-fire"


def test_tripwire_hover_on_line_does_not_double_fire():
    """Track sitting exactly on the line for several frames must not produce
    one event per hover-frame.

    Expected behavior:

    * One event when the track enters the line (above → on-line).
    * Zero events while hovering (each consecutive hover-frame has prev≈0,
      curr≈0, so the sign-flip classifier returns None).
    * The frame where the track leaves the line (on-line → below) does
      NOT fire again — we already counted id=1 when it first crossed, and
      the track has been continuously tracked the whole time.

    Total: exactly 1 event for the whole scenario, regardless of how many
    frames the track sits on the line.
    """
    tw = Tripwire(id="t", p1=HORIZONTAL[0], p2=HORIZONTAL[1])
    mgr = TripwireManager("cam1", [tw])

    # Frame 0: above the line
    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    # Frames 1..5: track hovers exactly on the line.
    hover_events: list[list] = []
    for i in range(1, 6):
        evs = mgr.update([_track(1, _bbox_from_centroid(200, 200))], bottom_center, timestamp=i * 0.1)
        hover_events.append(evs)
    total_so_far = sum(len(evs) for evs in hover_events)
    # Hover-frame 1 (the transition into hover): fires "in" (above → on-line).
    # Hover-frames 2..5: still on-line → no fire.
    assert len(hover_events[0]) == 1, (
        f"first hover frame should fire one event (entering hover); got {hover_events[0]}"
    )
    assert all(len(evs) == 0 for evs in hover_events[1:]), (
        f"consecutive hover frames must not fire; got {hover_events[1:]}"
    )
    # Frame 6: track moves below the line. Already counted → no double-fire.
    evs_final = mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.6)
    assert evs_final == [], (
        f"frame 6 must not re-fire for a continuously-tracked id; got {evs_final}"
    )
    # Total event count across the entire scenario: exactly 1.
    assert total_so_far == 1, (
        f"expected exactly 1 event across all hover frames, got {total_so_far}"
    )


def test_tripwire_id_recycling_resets_counted():
    """The classic bug: id 1 crosses L→R, then id 1 is reassigned to a
    different person on the other side, who crosses back. Both events must
    fire — we must not block the second because id 1 is in counted_ids.
    """
    tw = Tripwire(
        id="t",
        p1=HORIZONTAL[0],
        p2=HORIZONTAL[1],
        recycle_after_frames=5,
        recycle_distance_px=50.0,
    )
    mgr = TripwireManager("cam1", [tw])

    # Person A (id=1): above → below. Fires "in".
    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    evs = mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.1)
    assert len(evs) == 1 and evs[0].direction == "in"

    # Track id 1 disappears for > recycle_after_frames. The manager must
    # forget it eventually.
    # We simulate the dropout by feeding many frames without id 1.
    for i in range(20):
        mgr.update([_track(99, _bbox_from_centroid(50, 50))], bottom_center, timestamp=0.2 + i * 0.01)

    # Now id 1 comes back as person B far away. Crosses back the other way.
    mgr.update([_track(1, _bbox_from_centroid(300, 300))], bottom_center, timestamp=10.0)
    evs2 = mgr.update([_track(1, _bbox_from_centroid(300, 100))], bottom_center, timestamp=10.1)
    assert len(evs2) == 1, (
        "after id 1 was reassigned to a far-away position and enough "
        "frames elapsed, a second crossing must fire — not be blocked "
        "by the stale counted_ids entry"
    )
    assert evs2[0].direction == "out"


def test_tripwire_id_recycling_distance_resets_counted():
    """Distance-based recycling: even if few frames have passed, if the
    new position is far from the old, we re-admit the id.
    """
    tw = Tripwire(
        id="t",
        p1=HORIZONTAL[0],
        p2=HORIZONTAL[1],
        recycle_after_frames=10000,  # effectively disabled
        recycle_distance_px=100.0,
    )
    mgr = TripwireManager("cam1", [tw])

    mgr.update([_track(1, _bbox_from_centroid(100, 100))], bottom_center, timestamp=0.0)
    evs = mgr.update([_track(1, _bbox_from_centroid(100, 300))], bottom_center, timestamp=0.1)
    assert len(evs) == 1

    # Same id, but 400px away — distance-based recycling should re-admit.
    mgr.update([_track(1, _bbox_from_centroid(500, 300))], bottom_center, timestamp=0.2)
    evs2 = mgr.update([_track(1, _bbox_from_centroid(500, 100))], bottom_center, timestamp=0.3)
    assert len(evs2) == 1, "distance-based recycling should re-admit the id"


def test_tripwire_event_queue_receives_events():
    q: queue.Queue = queue.Queue()
    tw = Tripwire(id="t", p1=HORIZONTAL[0], p2=HORIZONTAL[1])
    mgr = TripwireManager("cam1", [tw], event_queue=q)

    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.1)

    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CrossingEvent)
    assert ev.stream_id == "cam1"
    assert ev.tripwire_id == "t"
    assert ev.track_id == 1
    assert ev.direction == "in"
    # bottom-center of bbox(180,260,220,340) is (200, 340).
    assert ev.centroid == (200.0, 340.0)


def test_tripwire_multiple_tripwires_same_camera():
    """Two parallel tripwires in one camera: a crossing fires both events
    in a single frame.
    """
    tw1 = Tripwire(id="upper", p1=(0, 100), p2=(400, 100))
    tw2 = Tripwire(id="lower", p1=(0, 300), p2=(400, 300))
    mgr = TripwireManager("cam1", [tw1, tw2])

    mgr.update([_track(1, _bbox_from_centroid(200, 50))], bottom_center, timestamp=0.0)
    # One big jump that crosses both lines (50 → 350 in one frame).
    evs = mgr.update([_track(1, _bbox_from_centroid(200, 350))], bottom_center, timestamp=0.1)
    assert {e.tripwire_id for e in evs} == {"upper", "lower"}
    assert all(e.direction == "in" for e in evs)


def test_tripwire_direction_in_right_to_left():
    """A tripwire declared direction_in='right_to_left' should call a
    negative→positive sign change 'out' (because that motion is right-to-left).
    """
    tw = Tripwire(id="t", p1=HORIZONTAL[0], p2=HORIZONTAL[1], direction_in="right_to_left")
    mgr = TripwireManager("cam1", [tw])

    # Above → below: prev_sign negative, curr_sign positive → "left_to_right" raw.
    # direction_in is "right_to_left" → that motion should be "out".
    mgr.update([_track(1, _bbox_from_centroid(200, 100))], bottom_center, timestamp=0.0)
    evs = mgr.update([_track(1, _bbox_from_centroid(200, 300))], bottom_center, timestamp=0.1)
    assert len(evs) == 1
    assert evs[0].direction == "out"


def test_tripwire_rejects_invalid_direction_in():
    import pytest
    with pytest.raises(ValueError, match="direction_in"):
        Tripwire(id="t", p1=(0, 0), p2=(1, 1), direction_in="diagonal")
