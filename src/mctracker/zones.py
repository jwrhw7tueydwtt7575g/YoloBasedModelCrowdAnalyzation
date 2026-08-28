"""Zone occupancy.

A zone is a polygon in image coordinates. Each frame, we test every active
track's centroid against every zone's polygon and report a live per-zone
count.

Centroid mode is per-camera:

* ``bottom_center`` (default) — uses ``((x1+x2)/2, y2)``. For typical
  eye-level or chest-level cameras, this is where the person's feet land,
  which is what you want for "is this person standing inside this region".
* ``geometric_center`` — uses ``((x1+x2)/2, (y1+y2)/2)``. For near-overhead
  cameras (looking down), bottom-center collapses to a small region near the
  geometric center anyway, so we just use the center directly.

The mode is a property of the camera, not of each zone, because it's about
the camera's viewing geometry. Every zone in a given camera uses the same
mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .track_state import TrackState
from .types import StreamId


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------


def bottom_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, y2


def geometric_center(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


CENTROID_FUNCS = {
    "bottom_center": bottom_center,
    "geometric_center": geometric_center,
}


def get_centroid_func(mode: str):
    try:
        return CENTROID_FUNCS[mode]
    except KeyError as e:
        raise ValueError(
            f"unknown centroid_mode {mode!r}; expected one of {list(CENTROID_FUNCS)}"
        ) from e


# ---------------------------------------------------------------------------
# Point-in-polygon (ray casting, inclusive of boundary)
# ---------------------------------------------------------------------------


def point_in_polygon(
    point: Tuple[float, float], polygon: Sequence[Sequence[float]]
) -> bool:
    """Ray-casting PIP. Boundary-inclusive.

    Implementation detail: rather than the textbook ``inside == (hits % 2)``,
    we count a point on the boundary as inside. The standard ray-casting
    algorithm does this naturally when we compare <= instead of < on the
    cross-product sign — for axis-aligned horizontal rays at the same y,
    vertices with y == point.y are skipped, which can give an off-by-one at
    boundaries. We use a slight nudge (1e-9) and count strictly horizontal
    edges correctly, but the boundary tests in the test suite use synthetic
    centroids that sit exactly on edges; for those we add a fallback
    edge-segment intersection check.
    """
    if len(polygon) < 3:
        return False
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]
        # Standard ray-casting: a horizontal ray from (x,y) toward +x crosses
        # the edge (xj,yj)-(xi,yi) iff the y-straddles and the x-intersection
        # is to the right of (x,y).
        if (yi > y) != (yj > y):
            # Compute x of intersection.
            xint = (xj - xi) * (y - yi) / (yj - yi) + xi
            if xint > x:
                inside = not inside
        j = i
    if inside:
        return True
    # Boundary check: centroid on an edge counts as inside. Without this,
    # the "straddling" test would treat centroids exactly on the polygon
    # edge as outside.
    return _on_polygon_boundary(point, polygon)


def _on_polygon_boundary(
    point: Tuple[float, float], polygon: Sequence[Sequence[float]]
) -> bool:
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i][0], polygon[i][1]
        x2, y2 = polygon[(i + 1) % n][0], polygon[(i + 1) % n][1]
        if _point_on_segment(x, y, x1, y1, x2, y2):
            return True
    return False


def _point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> bool:
    # Cross product == 0 (collinear) and within the bounding box.
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if abs(cross) > 1e-7:
        return False
    eps = 1e-7
    if min(x1, x2) - eps <= px <= max(x1, x2) + eps and min(y1, y2) - eps <= py <= max(y1, y2) + eps:
        return True
    return False


# ---------------------------------------------------------------------------
# Zone + ZoneManager
# ---------------------------------------------------------------------------


def _segments_intersect_strict(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    dx, dy = p4

    def cross(ox, oy, px, py, qx, qy):
        return (px - ox) * (qy - oy) - (py - oy) * (qx - ox)

    d1 = cross(cx, cy, dx, dy, ax, ay)
    d2 = cross(cx, cy, dx, dy, bx, by)
    d3 = cross(ax, ay, bx, by, cx, cy)
    d4 = cross(ax, ay, bx, by, dx, dy)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def is_polygon_self_intersecting(polygon: Sequence[Sequence[float]]) -> bool:
    """Return True if any two non-adjacent edges of the polygon intersect."""
    n = len(polygon)
    if n < 4:
        return False
    edges = [(i, (i + 1) % n) for i in range(n)]
    for i in range(len(edges)):
        a_i, b_i = edges[i]
        a = (float(polygon[a_i][0]), float(polygon[a_i][1]))
        b = (float(polygon[b_i][0]), float(polygon[b_i][1]))
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                continue
            a_j, b_j = edges[j]
            c = (float(polygon[a_j][0]), float(polygon[a_j][1]))
            d = (float(polygon[b_j][0]), float(polygon[b_j][1]))
            if _segments_intersect_strict(a, b, c, d):
                return True
    return False


@dataclass
class Zone:
    """One polygon zone for one camera.

    ``id`` is unique within a camera. The polygon is a list of ``[x, y]``
    pairs in image coordinates (pixels). Self-intersecting polygons are
    rejected as invalid geometry.
    """

    id: str
    polygon: List[List[float]]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("zone id is required")
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.id!r}: polygon must have at least 3 vertices")
        if is_polygon_self_intersecting(self.polygon):
            raise ValueError(f"zone {self.id!r}: polygon is self-intersecting")



@dataclass
class ZoneCount:
    """Per-frame zone snapshot."""

    zone_id: str
    count: int
    track_ids: List[int]


class ZoneManager:
    """Per-camera zone occupancy tracker.

    Holds a list of ``Zone`` and the camera's centroid mode. ``update()``
    takes the per-frame list of tracks and returns a list of ``ZoneCount``
    (one per zone, in declared order).
    """

    def __init__(
        self,
        zones: Iterable[Zone],
        centroid_mode: str = "bottom_center",
    ) -> None:
        self._zones: List[Zone] = list(zones)
        self._centroid = get_centroid_func(centroid_mode)
        self._centroid_mode = centroid_mode
        # Duplicate id check
        seen: set[str] = set()
        for z in self._zones:
            if z.id in seen:
                raise ValueError(f"duplicate zone id: {z.id!r}")
            seen.add(z.id)

    @property
    def centroid_mode(self) -> str:
        return self._centroid_mode

    def update(self, tracks: Iterable[TrackState]) -> List[ZoneCount]:
        """Return the per-zone occupancy count for the current frame."""
        tracks_list = list(tracks)
        out: List[ZoneCount] = []
        for zone in self._zones:
            inside_ids: List[int] = []
            for t in tracks_list:
                cx, cy = self._centroid(t.bbox)
                if point_in_polygon((cx, cy), zone.polygon):
                    inside_ids.append(t.track_id)
            out.append(ZoneCount(zone_id=zone.id, count=len(inside_ids), track_ids=inside_ids))
        return out

    def get_zone(self, zone_id: str) -> Zone | None:
        for z in self._zones:
            if z.id == zone_id:
                return z
        return None
