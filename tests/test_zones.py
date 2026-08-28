"""Zone occupancy tests."""

from __future__ import annotations

import numpy as np

from mctracker.track_state import make_track_state
from mctracker.types import Detection
from mctracker.zones import (
    Zone,
    ZoneManager,
    bottom_center,
    geometric_center,
    point_in_polygon,
)


def _track(track_id: int, bbox) -> object:
    return make_track_state(
        track_id=track_id,
        bbox_xyxy=bbox,
        conf=0.9,
        cls=0,
        ts=0.0,
    )


# A simple rectangular zone from (100,100) to (300,300).
RECT = [[100, 100], [300, 100], [300, 300], [100, 300]]


def test_centroid_bottom_center_uses_y2():
    cx, cy = bottom_center((10, 20, 30, 50))
    assert cx == 20.0 and cy == 50.0


def test_centroid_geometric_uses_midpoint():
    cx, cy = geometric_center((10, 20, 30, 50))
    assert cx == 20.0 and cy == 35.0


def test_point_in_polygon_inside():
    assert point_in_polygon((150, 150), RECT) is True


def test_point_in_polygon_outside():
    assert point_in_polygon((50, 50), RECT) is False
    assert point_in_polygon((400, 400), RECT) is False


def test_zone_manager_counts_tracks_inside():
    zm = ZoneManager([Zone(id="lobby", polygon=RECT)])
    tracks = [
        _track(1, (110, 110, 130, 130)),  # inside
        _track(2, (200, 200, 220, 220)),  # inside
        _track(3, (500, 500, 520, 520)),  # outside
    ]
    counts = zm.update(tracks)
    assert len(counts) == 1
    assert counts[0].zone_id == "lobby"
    assert counts[0].count == 2
    assert sorted(counts[0].track_ids) == [1, 2]


def test_zone_straddling_boundary_counts_as_inside():
    """Centroid sitting exactly on a polygon edge must be counted as inside.

    This is the straddling case the requirement calls out. A naive
    ray-casting implementation will say "outside" for an exactly-on-edge
    point; our implementation falls back to an edge-boundary check.
    """
    zm = ZoneManager([Zone(id="rect", polygon=RECT)])
    # bbox bottom-center sits exactly on the rectangle's left edge (x=100).
    bbox = (100, 100, 130, 200)  # bottom-center = (115, 200) -> inside
    tracks = [_track(7, bbox)]
    counts = zm.update(tracks)
    assert counts[0].count == 1

    # Now bbox whose bottom-center sits exactly ON the left edge.
    bbox_on_edge = (100, 100, 100, 200)  # bottom-center = (100, 200) - on the edge x=100
    # Re-seed the manager with a fresh track so we can isolate this case.
    tracks_on_edge = [_track(9, bbox_on_edge)]
    counts2 = zm.update(tracks_on_edge)
    assert counts2[0].count == 1, (
        f"track with centroid on the polygon boundary should be counted as inside; "
        f"got {counts2[0]}"
    )


def test_zone_centroid_mode_geometric():
    """Geometric-center mode uses the bbox midpoint instead of bottom-center.

    We pick a tall bbox where bottom_center and geometric_center differ by
    enough that only one of them is inside the zone.
    """
    zm_bottom = ZoneManager([Zone(id="r", polygon=RECT)], centroid_mode="bottom_center")
    zm_geom = ZoneManager([Zone(id="r", polygon=RECT)], centroid_mode="geometric_center")

    # Tall bbox from y=80 to y=290: bottom_center is at y=290 (inside),
    # geometric_center is at y=185 (inside).
    tall_inside = (110, 80, 130, 290)
    assert zm_bottom.update([_track(1, tall_inside)])[0].count == 1
    assert zm_geom.update([_track(1, tall_inside)])[0].count == 1

    # bbox from y=80 to y=110: bottom_center at y=110 (inside top edge),
    # geometric_center at y=95 (outside, above the zone).
    top_strip = (110, 80, 130, 110)
    bottom_count = zm_bottom.update([_track(2, top_strip)])[0].count
    geom_count = zm_geom.update([_track(2, top_strip)])[0].count
    assert bottom_count == 1, "bottom_center should count this as inside (centroid on top edge)"
    assert geom_count == 0, "geometric_center should say outside (centroid at y=95 < 100)"


def test_zone_unknown_centroid_mode_rejected():
    import pytest
    with pytest.raises(ValueError):
        ZoneManager([Zone(id="r", polygon=RECT)], centroid_mode="diagonal_midpoint")


def test_duplicate_zone_id_rejected():
    import pytest
    with pytest.raises(ValueError, match="duplicate zone id"):
        ZoneManager([
            Zone(id="dup", polygon=RECT),
            Zone(id="dup", polygon=[[0, 0], [10, 0], [5, 10]]),
        ])
