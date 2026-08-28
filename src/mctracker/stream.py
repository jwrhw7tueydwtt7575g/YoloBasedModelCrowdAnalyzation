"""Per-stream wiring.

A Stream owns:

* one ``StreamReader`` thread that fills a ``FrameBuffer`` continuously
* one processor thread that pulls from the buffer, runs detection, then
  tracking, then zone occupancy + tripwire crossing, then filters for
  output, then fires the user callback

The crucial invariant is the order of operations in the processor loop:

    detect(frame)  ->  tracker.update(frame, ALL detections)  ->  zone/tripwire update  ->  filter(conf >= display_conf)  ->  callback

The conf filter is applied *after* the tracker, never before. The tracker
is what gets to see every box — including the conf=0.05 ones — so its
confidence-cascade can decide whether to keep or drop them. Zones and
tripwires operate on the tracker's output (every track, regardless of
display conf), so a low-confidence track still contributes to occupancy
counts and can still cross a tripwire.

Stage 5 resilience: each stage call is wrapped in ``try/except`` so a
broken detector / tracker / zone manager / tripwire manager doesn't kill
the processor thread. Per-stage latency is observed into the histograms
in ``observability.METRICS``; per-stage exceptions increment the failure
counter. The ID-switch detector piggybacks on the tracker's output.

Stage 6 high-density: if a ``density_rule`` is wired in, the per-frame
zone counts are also fed into it; on fire, the rule calls
``density_sink(HighDensityViolation)`` which the Pipeline routes to the
high-density repository + evidence recorder.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from .buffer import FrameBuffer, StreamReader, estimate_fps
from .detector import Detector
from .observability import METRICS, time_stage
from .track_state import TrackState
from .tracker import Tracker
from .tripwire import TripwireManager
from .types import Frame, StreamId
from .zones import ZoneCount, ZoneManager, get_centroid_func

log = logging.getLogger(__name__)

# Callback signature: stream_id, displayed tracks, zone counts, new crossings.
ResultsCallback = Callable[
    [StreamId, List[TrackState], List[ZoneCount], list],
    None,
]

# ID-switch detector thresholds.
_ID_SWITCH_SPATIAL_PX = 40.0  # same physical spot, give or take
_ID_SWITCH_TEMPORAL_S = 2.0  # within this many seconds of the previous id


class Stream:
    """One camera. Owns its own buffer, reader, detector, tracker, zones,
    and tripwires.

    The tracker instance is unique to this stream — no cross-stream state
    can leak because no other stream holds a reference to it. The same
    is true for the ZoneManager and TripwireManager.
    """

    def __init__(
        self,
        stream_id: str,
        source: str,
        detector: Detector,
        tracker: Tracker,
        on_results: ResultsCallback,
        buffer_seconds: int = 5,
        fps_fallback: int = 30,
        display_conf: float = 0.25,
        zone_manager: Optional[ZoneManager] = None,
        tripwire_manager: Optional[TripwireManager] = None,
        reader: Optional[StreamReader] = None,
        density_rule=None,  # Optional[DensityRule]
        density_sink: Optional[Callable] = None,  # Callable[[HighDensityViolation], None]
    ) -> None:
        self.id = stream_id
        self.source = source
        self._detector = detector
        self._tracker = tracker
        self._on_results = on_results
        self._display_conf = float(display_conf)
        self._density_rule = density_rule
        self._density_sink = density_sink

        # Buffer sizing: trust cv2 if it can tell us, fall back otherwise.
        fps = estimate_fps(source, fallback=fps_fallback)
        maxlen = max(1, int(fps * buffer_seconds))
        self._buffer = FrameBuffer(maxlen=maxlen)
        self._fps = float(fps)
        self._buffer_seconds = int(buffer_seconds)

        # Reader: caller may pass a fake (tests). Otherwise build a real one.
        self._reader = reader or StreamReader(
            source=source, buffer=self._buffer, stream_id=stream_id
        )

        # Per-camera zones and tripwires.
        self._zone_manager = zone_manager
        self._tripwire_manager = tripwire_manager

        # Active-track registry: per-stream, never shared.
        self._active: dict[int, TrackState] = {}
        self._lock = threading.Lock()

        # ID-switch detector: most-recent centroid per track_id with a wall-
        # clock timestamp. When a new track_id lands at the same spot a
        # *different* track_id just vacated, we count an ID switch.
        self._last_centroid: Dict[int, Tuple[float, float, float]] = {}

        # Processor thread state.
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ start/stop

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._reader.start()
        self._thread = threading.Thread(
            target=self._run_processor, name=f"stream[{self.id}]", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._reader.stop(timeout=timeout)
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def active_track_ids(self) -> set[int]:
        with self._lock:
            return set(self._active.keys())

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer

    @property
    def fps(self) -> float:
        return self._fps

    def get_track(self, track_id: int) -> Optional[TrackState]:
        with self._lock:
            return self._active.get(track_id)

    @property
    def zone_manager(self) -> Optional[ZoneManager]:
        return self._zone_manager

    @property
    def tripwire_manager(self) -> Optional[TripwireManager]:
        return self._tripwire_manager

    # ------------------------------------------------------------------ processor

    def _run_processor(self) -> None:
        # Bounded wait on get_latest: lets the stop event be observed quickly
        # at shutdown, and avoids a tight spin when no frames are arriving.
        centroid_func = (
            get_centroid_func(self._zone_manager.centroid_mode)
            if self._zone_manager is not None
            else None
        )
        while not self._stop.is_set():
            latest = self._buffer.get_latest(timeout=0.25)
            if latest is None:
                continue
            frame, ts = latest
            frame_ts = ts or time.time()

            # ---- detect (every failure logs + metric, never kills thread)
            try:
                with time_stage("detect", self.id):
                    detections = self._detector.detect(frame)
            except Exception:
                log.exception(
                    "detector failed for stream %s",
                    self.id,
                    extra={"stream_id": self.id, "stage": "detect", "event": "stage_failure"},
                )
                METRICS.inc_stage_failure(self.id, "detect")
                METRICS.inc_frame(self.id, dropped=True)
                continue

            # ---- track (every failure logs + metric, never kills thread)
            try:
                with time_stage("track", self.id):
                    tracks = self._tracker.update(frame, detections)
            except Exception:
                log.exception(
                    "tracker failed for stream %s",
                    self.id,
                    extra={"stream_id": self.id, "stage": "track", "event": "stage_failure"},
                )
                METRICS.inc_stage_failure(self.id, "track")
                METRICS.inc_frame(self.id, dropped=True)
                continue

            # ID-switch detector (best-effort, runs even if zones/tripwires crash below).
            self._detect_id_switches(tracks, frame_ts)

            self._update_active(tracks)
            METRICS.inc_frame(self.id, dropped=False)

            # ---- zones
            zone_counts: List[ZoneCount]
            if self._zone_manager is not None and centroid_func is not None:
                try:
                    with time_stage("zone", self.id):
                        zone_counts = self._zone_manager.update(tracks)
                except Exception:
                    log.exception(
                        "zone manager failed for stream %s",
                        self.id,
                        extra={"stream_id": self.id, "stage": "zone", "event": "stage_failure"},
                    )
                    METRICS.inc_stage_failure(self.id, "zone")
                    zone_counts = []
            else:
                zone_counts = []

            # ---- high-density / crowd alert (Stage 6)
            if getattr(self, "_density_rule", None) is not None and zone_counts:
                self._evaluate_density(zone_counts, frame_ts)

            # ---- tripwires
            new_crossings: list
            if self._tripwire_manager is not None and centroid_func is not None:
                try:
                    with time_stage("tripwire", self.id):
                        new_crossings = self._tripwire_manager.update(
                            tracks, centroid_func, timestamp=frame_ts
                        )
                except Exception:
                    log.exception(
                        "tripwire manager failed for stream %s",
                        self.id,
                        extra={"stream_id": self.id, "stage": "tripwire", "event": "stage_failure"},
                    )
                    METRICS.inc_stage_failure(self.id, "tripwire")
                    new_crossings = []
            else:
                new_crossings = []

            displayed = [t for t in tracks if t.confidence >= self._display_conf]
            try:
                self._on_results(self.id, displayed, zone_counts, new_crossings)
            except Exception:  # pragma: no cover - callback author responsibility
                log.exception(
                    "on_results callback raised for stream %s",
                    self.id,
                    extra={"stream_id": self.id, "stage": "callback", "event": "callback_failure"},
                )

            # Yield CPU slice to OS scheduler (critical for single-vCPU cloud platforms like Render)
            time.sleep(0.005)

    def _detect_id_switches(
        self, tracks: List[TrackState], frame_ts: float
    ) -> None:
        """Best-effort ID-switch detection.

        Idea: for each *current* track, check whether a *different* track_id
        occupied the same physical spot within ``_ID_SWITCH_TEMPORAL_S``
        and ``_ID_SWITCH_SPATIAL_PX``. If so, we treat it as the tracker
        having handed the id to a new person — i.e., an ID switch.

        Limitations (documented):
        * Two people walking past each other in a crowd can look like a
          switch if they happen to be near the same centroid. To bound
          false positives we require the previous id to have been seen
          within the last 2 s and the spatial closeness to be tight.
        * The detector is intentionally cheap (no embedding comparison).
          It is a *rate* metric, not a forensic tool.
        """
        if not tracks:
            return
        now = frame_ts
        for t in tracks:
            cx, cy = t.centroid
            for prev_id, (px, py, prev_ts) in list(self._last_centroid.items()):
                if prev_id == t.track_id:
                    continue
                if (now - prev_ts) > _ID_SWITCH_TEMPORAL_S:
                    continue
                dx = cx - px
                dy = cy - py
                if dx * dx + dy * dy > (_ID_SWITCH_SPATIAL_PX ** 2):
                    continue
                METRICS.inc_id_switch(self.id)
                log.warning(
                    "ID switch suspected",
                    extra={
                        "stream_id": self.id,
                        "event": "id_switch",
                        "from_track_id": prev_id,
                        "to_track_id": t.track_id,
                        "dx_px": dx,
                        "dy_px": dy,
                        "dt_s": now - prev_ts,
                    },
                )
                # Clear the previous slot so the same switch isn't double-
                # counted by every track on the new id.
                self._last_centroid.pop(prev_id, None)
                break
        # Update the registry with this frame's positions.
        for t in tracks:
            self._last_centroid[t.track_id] = (t.centroid[0], t.centroid[1], now)
        # Garbage-collect ids we haven't seen for > 5 s.
        stale = [tid for tid, (_, _, ts) in self._last_centroid.items() if (now - ts) > 5.0]
        for tid in stale:
            self._last_centroid.pop(tid, None)

    def _evaluate_density(
        self, zone_counts: List[ZoneCount], frame_ts: float
    ) -> None:
        """Feed the per-frame zone counts to the high-density rule.

        On fire, the rule produces a ``HighDensityViolation``; we invoke
        the configured sink (the Pipeline routes that to a repository and
        the evidence recorder). Wrapped in try/except so a density-rule
        failure never kills the processor.
        """
        # Local import to keep the top-level imports light and avoid a
        # circular dep at module-load time.
        from .violations.rules import (
            ZoneOccupancySnapshot,
            density_snapshots_from_zone_counts,
        )

        try:
            snaps = density_snapshots_from_zone_counts(
                self.id, zone_counts, frame_ts
            )
            for snap in snaps:
                v = self._density_rule.observe(snap)
                if v is not None:
                    METRICS.inc_violation("high_density")
                    log.warning(
                        "high-density alert",
                        extra={
                            "stream_id": self.id,
                            "zone_id": v.zone_id,
                            "event": "high_density",
                            "count": v.density_count,
                            "threshold": v.threshold,
                            "dwell_s": v.dwell_seconds,
                        },
                    )
                    if self._density_sink is not None:
                        try:
                            self._density_sink(v)
                        except Exception:
                            log.exception(
                                "density_sink raised for stream %s",
                                self.id,
                                extra={
                                    "stream_id": self.id,
                                    "event": "density_sink_failure",
                                },
                            )
        except Exception:
            log.exception(
                "density rule failed for stream %s",
                self.id,
                extra={"stream_id": self.id, "stage": "density", "event": "stage_failure"},
            )
            METRICS.inc_stage_failure(self.id, "density")

    def _update_active(self, tracks: List[TrackState]) -> None:
        # Keep a per-stream dict of the most recent state per track id.
        with self._lock:
            for t in tracks:
                prev = self._active.get(t.track_id)
                if prev is not None:
                    # Append current centroid to history, preserving maxlen.
                    t.centroid_history = prev.centroid_history
                    t.centroid_history.append((t.centroid[0], t.centroid[1], t.last_seen_ts))
                    if prev.embedding is not None and t.embedding is None:
                        t.embedding = prev.embedding
                else:
                    t.centroid_history.append((t.centroid[0], t.centroid[1], t.last_seen_ts))
                self._active[t.track_id] = t
