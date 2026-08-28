"""YAML config loading and validation."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

from .tripwire import Tripwire
from .zones import Zone


class ConfigError(ValueError):
    """Raised when the YAML config is missing required fields or invalid."""


@dataclasses.dataclass
class ViolationsConfig:
    """Top-level violations-layer configuration.

    Currently just the QR-match window. ``window_seconds`` is the half-window
    used by ``ViolationService`` to pair a crossing with a scan — a crossing
    matches a scan if the scan is within ``window_seconds`` of the crossing's
    timestamp. Must be strictly greater than 0 (a 0 window would pair every
    crossing with every coincident scan, which is meaningless).
    """

    window_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ConfigError(
                f"violations.window_seconds must be > 0, got {self.window_seconds}"
            )


@dataclasses.dataclass
class ObservabilityConfig:
    """Observability / metrics configuration.

    ``enabled=False`` disables the Prometheus endpoint entirely. The
    in-process counters in ``observability.METRICS`` are always live; this
    only governs whether an HTTP server is opened.
    """

    enabled: bool = True
    metrics_port: int = 0  # 0 means "don't open an HTTP port"

    def __post_init__(self) -> None:
        if self.metrics_port < 0 or self.metrics_port > 65535:
            raise ConfigError(
                f"observability.metrics_port must be 0..65535, got {self.metrics_port}"
            )


@dataclasses.dataclass
class StreamConfig:
    id: str
    source: str
    model_size: str = "yolov8n.pt"
    tracker_type: str = "bytetrack"  # "bytetrack" | "botsort"
    use_appearance: bool = False
    buffer_seconds: int = 5
    fps_fallback: int = 30
    display_conf: float = 0.25
    centroid_mode: str = "bottom_center"  # "bottom_center" | "geometric_center"
    zones: List[Zone] = dataclasses.field(default_factory=list)
    tripwires: List[Tripwire] = dataclasses.field(default_factory=list)
    # Optional ``[width, height]``. When present, tripwire endpoints are
    # bounds-checked against the frame at config-load time. When absent
    # (RTSP where cv2 can't read dimensions reliably), the check is skipped
    # and a warning is logged at build time.
    frame_size: Optional[Tuple[int, int]] = None
    # Stage 6: high-density / crowd alert. When set, the stream's
    # ``DensityRule`` fires if the zone count stays above
    # ``max_density_threshold`` for ``density_dwell_seconds`` (and is then
    # silenced for ``density_cooldown_seconds``).
    max_density_threshold: Optional[int] = None
    density_dwell_seconds: float = 2.0
    density_cooldown_seconds: float = 10.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("stream id is required")
        if not self.source:
            raise ConfigError(f"stream {self.id!r}: source is required")
        if self.tracker_type not in ("bytetrack", "botsort"):
            raise ConfigError(
                f"stream {self.id!r}: tracker_type must be 'bytetrack' or 'botsort', "
                f"got {self.tracker_type!r}"
            )
        if self.use_appearance and self.tracker_type != "botsort":
            raise ConfigError(
                f"stream {self.id!r}: use_appearance=true requires tracker_type='botsort'"
            )
        if self.buffer_seconds <= 0:
            raise ConfigError(f"stream {self.id!r}: buffer_seconds must be > 0")
        if self.fps_fallback <= 0:
            raise ConfigError(f"stream {self.id!r}: fps_fallback must be > 0")
        if not (0.0 <= self.display_conf <= 1.0):
            raise ConfigError(f"stream {self.id!r}: display_conf must be in [0, 1]")
        if self.centroid_mode not in ("bottom_center", "geometric_center"):
            raise ConfigError(
                f"stream {self.id!r}: centroid_mode must be 'bottom_center' or "
                f"'geometric_center', got {self.centroid_mode!r}"
            )
        # Duplicate zone/tripwire ids within a stream.
        zone_seen: set[str] = set()
        for z in self.zones:
            if z.id in zone_seen:
                raise ConfigError(f"stream {self.id!r}: duplicate zone id {z.id!r}")
            zone_seen.add(z.id)
        trip_seen: set[str] = set()
        for t in self.tripwires:
            if t.id in trip_seen:
                raise ConfigError(f"stream {self.id!r}: duplicate tripwire id {t.id!r}")
            trip_seen.add(t.id)
        # Frame-size sanity.
        if self.frame_size is not None:
            w, h = self.frame_size
            if w <= 0 or h <= 0:
                raise ConfigError(
                    f"stream {self.id!r}: frame_size must have positive width and height"
                )
        # Density-rule sanity.
        if self.max_density_threshold is not None and self.max_density_threshold <= 0:
            raise ConfigError(
                f"stream {self.id!r}: max_density_threshold must be > 0"
            )
        if self.density_dwell_seconds < 0:
            raise ConfigError(
                f"stream {self.id!r}: density_dwell_seconds must be >= 0"
            )
        if self.density_cooldown_seconds < 0:
            raise ConfigError(
                f"stream {self.id!r}: density_cooldown_seconds must be >= 0"
            )


@dataclasses.dataclass
class EvidenceConfig:
    """Configuration for the evidence-clip recorder.

    All fields are optional. ``enabled=False`` (the default) keeps the
    recorder off — useful for pure-tracking deployments that don't need
    to retain violation video.
    """

    enabled: bool = False
    base_dir: str = "./evidence_clips"
    pre_seconds: float = 5.0
    post_seconds: float = 5.0
    free_threshold_mb: float = 2048.0
    retention_days: float = 30.0
    fps: float = 30.0
    buffer_memory_ceiling_mb: float = 1024.0  # total raw-frame buffer across all streams
    clip_storage: str = "local"  # "local" | "noop" (other backends pluggable)

    def __post_init__(self) -> None:
        if self.pre_seconds < 0:
            raise ConfigError("evidence.pre_seconds must be >= 0")
        if self.post_seconds < 0:
            raise ConfigError("evidence.post_seconds must be >= 0")
        if self.free_threshold_mb <= 0:
            raise ConfigError("evidence.free_threshold_mb must be > 0")
        if self.retention_days <= 0:
            raise ConfigError("evidence.retention_days must be > 0")
        if self.fps <= 0:
            raise ConfigError("evidence.fps must be > 0")
        if self.buffer_memory_ceiling_mb <= 0:
            raise ConfigError("evidence.buffer_memory_ceiling_mb must be > 0")
        if self.clip_storage not in ("local", "noop"):
            raise ConfigError(
                f"evidence.clip_storage must be 'local' or 'noop', got {self.clip_storage!r}"
            )


@dataclasses.dataclass
class AppConfig:
    streams: List[StreamConfig]
    # Optional global event queue capacity hint; consumers may still pass
    # their own queue.Queue to specific TripwireManagers if they want
    # different routing.
    event_queue_maxsize: Optional[int] = None
    # Evidence-clip configuration. If left at defaults, evidence remains
    # disabled and no recorder is wired into the pipeline.
    evidence: EvidenceConfig = dataclasses.field(default_factory=EvidenceConfig)
    # Stage 5 additions.
    violations: ViolationsConfig = dataclasses.field(default_factory=ViolationsConfig)
    observability: ObservabilityConfig = dataclasses.field(default_factory=ObservabilityConfig)

    def __post_init__(self) -> None:
        if not self.streams:
            raise ConfigError("at least one stream is required")
        seen: set[str] = set()
        for s in self.streams:
            if s.id in seen:
                raise ConfigError(f"duplicate stream id: {s.id!r}")
            seen.add(s.id)


# ---------------------------------------------------------------------------
# Polygon self-intersection check (sweep-line via non-adjacent edge pairs)
# ---------------------------------------------------------------------------


def _segments_intersect_strict(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """True iff the two *closed* segments share any point other than a shared
    endpoint. Collinear overlap returns False (we don't ban polygons whose
    vertices happen to align)."""
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


def _polygon_self_intersects(polygon: List[List[float]]) -> bool:
    """Return True if any two non-adjacent edges of the polygon intersect.

    Adjacent edges share a vertex — they always "intersect" at that
    vertex, but that's normal. We skip those.
    """
    n = len(polygon)
    if n < 4:
        # Triangles can never self-intersect.
        return False
    edges = [(i, (i + 1) % n) for i in range(n)]
    for i in range(len(edges)):
        a_i, b_i = edges[i]
        a = (polygon[a_i][0], polygon[a_i][1])
        b = (polygon[b_i][0], polygon[b_i][1])
        for j in range(i + 2, len(edges)):
            if i == 0 and j == len(edges) - 1:
                # The closing edge of a polygon shares vertices with edge 0.
                continue
            a_j, b_j = edges[j]
            c = (polygon[a_j][0], polygon[a_j][1])
            d = (polygon[b_j][0], polygon[b_j][1])
            if _segments_intersect_strict(a, b, c, d):
                return True
    return False


def _coerce_frame_size(raw: Any, stream_id: str) -> Optional[Tuple[int, int]]:
    if raw is None:
        return None
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise ConfigError(
            f"stream {stream_id!r}: frame_size must be a [width, height] pair"
        )
    try:
        w, h = int(raw[0]), int(raw[1])
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"stream {stream_id!r}: frame_size entries must be integers"
        ) from e
    if w <= 0 or h <= 0:
        raise ConfigError(
            f"stream {stream_id!r}: frame_size must be positive, got [{w}, {h}]"
        )
    return (w, h)


def _coerce_zone(raw: Any, stream_id: str) -> Zone:
    if not isinstance(raw, dict):
        raise ConfigError(f"zone entries must be mappings, got {type(raw).__name__}")
    zid = str(raw.get("id", ""))
    if not zid:
        raise ConfigError(f"stream {stream_id!r}: zone missing id")
    poly = raw.get("polygon")
    if not isinstance(poly, list) or len(poly) < 3:
        raise ConfigError(f"stream {stream_id!r} zone {zid!r}: polygon must be a list of >=3 points")
    normalized: List[List[float]] = []
    for pt in poly:
        if not isinstance(pt, (list, tuple)) or len(pt) != 2:
            raise ConfigError(
                f"stream {stream_id!r} zone {zid!r}: each polygon point must be [x, y]"
            )
        normalized.append([float(pt[0]), float(pt[1])])
    # Stage 5: reject self-intersecting polygons. PIP would still "work"
    # but the result is meaningless and operators frequently misconfigure
    # bow-tie / figure-8 regions. Catch it at config-load time.
    if _polygon_self_intersects(normalized):
        raise ConfigError(
            f"stream {stream_id!r} zone {zid!r}: polygon is self-intersecting"
        )
    return Zone(id=zid, polygon=normalized)


def _coerce_tripwire(
    raw: Any, stream_id: str, frame_size: Optional[Tuple[int, int]]
) -> Tripwire:
    if not isinstance(raw, dict):
        raise ConfigError(f"tripwire entries must be mappings, got {type(raw).__name__}")
    tid = str(raw.get("id", ""))
    if not tid:
        raise ConfigError(f"stream {stream_id!r}: tripwire missing id")
    p1 = raw.get("p1")
    p2 = raw.get("p2")
    if not (isinstance(p1, (list, tuple)) and len(p1) == 2):
        raise ConfigError(f"stream {stream_id!r} tripwire {tid!r}: p1 must be [x, y]")
    if not (isinstance(p2, (list, tuple)) and len(p2) == 2):
        raise ConfigError(f"stream {stream_id!r} tripwire {tid!r}: p2 must be [x, y]")
    p1t = (float(p1[0]), float(p1[1]))
    p2t = (float(p2[0]), float(p2[1]))
    # Stage 5: p1 == p2 is a degenerate segment.
    if p1t == p2t:
        raise ConfigError(
            f"stream {stream_id!r} tripwire {tid!r}: p1 == p2 (degenerate segment)"
        )
    # Stage 5: reject non-finite coordinates.
    for label, pt in (("p1", p1t), ("p2", p2t)):
        if not (math.isfinite(pt[0]) and math.isfinite(pt[1])):
            raise ConfigError(
                f"stream {stream_id!r} tripwire {tid!r}: {label} has non-finite coordinate {pt}"
            )
    # Stage 5: bounds check if a frame_size hint was given.
    if frame_size is not None:
        w, h = frame_size
        for label, (x, y) in (("p1", p1t), ("p2", p2t)):
            if not (0 <= x <= w and 0 <= y <= h):
                raise ConfigError(
                    f"stream {stream_id!r} tripwire {tid!r}: {label} ({x},{y}) "
                    f"is outside frame {w}x{h}"
                )
    recycle_after_frames = int(raw.get("recycle_after_frames", 60))
    if recycle_after_frames < 0:
        raise ConfigError(
            f"stream {stream_id!r} tripwire {tid!r}: recycle_after_frames must be >= 0"
        )
    recycle_distance_px = float(raw.get("recycle_distance_px", 200.0))
    if recycle_distance_px < 0:
        raise ConfigError(
            f"stream {stream_id!r} tripwire {tid!r}: recycle_distance_px must be >= 0"
        )
    return Tripwire(
        id=tid,
        p1=p1t,
        p2=p2t,
        direction_in=str(raw.get("direction_in", "left_to_right")),
        recycle_after_frames=recycle_after_frames,
        recycle_distance_px=recycle_distance_px,
    )


def _coerce_stream(raw: Any) -> StreamConfig:
    if not isinstance(raw, dict):
        raise ConfigError(f"each stream entry must be a mapping, got {type(raw).__name__}")
    sid = str(raw.get("id", ""))
    frame_size = _coerce_frame_size(raw.get("frame_size"), sid)
    zones_raw = raw.get("zones", []) or []
    tripwires_raw = raw.get("tripwires", []) or []
    return StreamConfig(
        id=sid,
        source=str(raw.get("source", "")),
        model_size=str(raw.get("model_size", "yolov8n.pt")),
        tracker_type=str(raw.get("tracker_type", "bytetrack")).lower(),
        use_appearance=bool(raw.get("use_appearance", False)),
        buffer_seconds=int(raw.get("buffer_seconds", 5)),
        fps_fallback=int(raw.get("fps_fallback", 30)),
        display_conf=float(raw.get("display_conf", 0.25)),
        centroid_mode=str(raw.get("centroid_mode", "bottom_center")).lower(),
        zones=[_coerce_zone(z, sid) for z in zones_raw],
        tripwires=[_coerce_tripwire(t, sid, frame_size) for t in tripwires_raw],
        frame_size=frame_size,
        max_density_threshold=(
            int(raw["max_density_threshold"])
            if raw.get("max_density_threshold") is not None
            else None
        ),
        density_dwell_seconds=float(raw.get("density_dwell_seconds", 2.0)),
        density_cooldown_seconds=float(raw.get("density_cooldown_seconds", 10.0)),
    )


def _coerce_violations(raw: Any) -> ViolationsConfig:
    if raw is None:
        return ViolationsConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'violations' must be a mapping")
    window = raw.get("window_seconds", 10.0)
    if window is None:
        window = 10.0
    return ViolationsConfig(window_seconds=float(window))


def _coerce_observability(raw: Any) -> ObservabilityConfig:
    if raw is None:
        return ObservabilityConfig()
    if not isinstance(raw, dict):
        raise ConfigError("'observability' must be a mapping")
    enabled = bool(raw.get("enabled", True))
    port_raw = raw.get("metrics_port", 0)
    if port_raw is None:
        port_raw = 0
    return ObservabilityConfig(enabled=enabled, metrics_port=int(port_raw))


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ConfigError(f"config file is empty: {p}")
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")
    streams_raw = data.get("streams")
    if streams_raw is None:
        raise ConfigError("config must contain a 'streams' list")
    if not isinstance(streams_raw, list):
        raise ConfigError("'streams' must be a list")
    streams = [_coerce_stream(s) for s in streams_raw]
    eqm = data.get("event_queue_maxsize")
    evidence_raw = data.get("evidence") or {}
    if evidence_raw and not isinstance(evidence_raw, dict):
        raise ConfigError("'evidence' must be a mapping")
    evidence = _coerce_evidence(evidence_raw)
    violations = _coerce_violations(data.get("violations"))
    observability = _coerce_observability(data.get("observability"))
    return AppConfig(
        streams=streams,
        event_queue_maxsize=int(eqm) if eqm is not None else None,
        evidence=evidence,
        violations=violations,
        observability=observability,
    )


def _coerce_evidence(raw: dict) -> EvidenceConfig:
    """Coerce an ``evidence:`` mapping into an EvidenceConfig.

    All keys are optional; defaults come from EvidenceConfig.
    """
    def _f(name):
        v = raw.get(name)
        return float(v) if v is not None else None
    def _b(name):
        v = raw.get(name)
        return bool(v) if v is not None else False
    return EvidenceConfig(
        enabled=_b("enabled") if "enabled" in raw else False,
        base_dir=str(raw.get("base_dir", "./evidence_clips")),
        pre_seconds=_f("pre_seconds") if _f("pre_seconds") is not None else 5.0,
        post_seconds=_f("post_seconds") if _f("post_seconds") is not None else 5.0,
        free_threshold_mb=_f("free_threshold_mb") if _f("free_threshold_mb") is not None else 2048.0,
        retention_days=_f("retention_days") if _f("retention_days") is not None else 30.0,
        fps=_f("fps") if _f("fps") is not None else 30.0,
        buffer_memory_ceiling_mb=_f("buffer_memory_ceiling_mb")
            if _f("buffer_memory_ceiling_mb") is not None else 1024.0,
        clip_storage=str(raw.get("clip_storage", "local")),
    )

