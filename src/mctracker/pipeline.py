"""Multi-stream pipeline orchestrator."""

from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Callable, List, Optional

from .callbacks import default_callback
from .config import AppConfig, EvidenceConfig, StreamConfig, load_config
from .detector import Detector, YOLODetector
from .observability import METRICS, start_prometheus_endpoint
from .stream import Stream
from .tracker import Tracker, make_tracker
from .tripwire import TripwireManager
from .types import StreamId
from .track_state import TrackState
from .zones import ZoneCount, ZoneManager

# Imported lazily to avoid pulling cv2/sqlalchemy when not used.
def _build_evidence_recorder(cfg: EvidenceConfig):
    from .violations import (
        ClipBuilder,
        DiskSpaceGuard,
        EvidenceRecorder,
        LocalDiskClipStorage,
        MemoryBudgetGuard,
        NoopClipStorage,
    )

    if cfg.clip_storage == "noop":
        storage = NoopClipStorage()
    else:
        storage = LocalDiskClipStorage(cfg.base_dir)
    return EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=cfg.fps),
        disk_guard=DiskSpaceGuard(
            base_dir=cfg.base_dir, free_threshold_mb=cfg.free_threshold_mb,
            storage=storage if hasattr(storage, "list_clips_older_than") else None,
        ),
        memory_guard=MemoryBudgetGuard(
            ceiling_bytes=int(cfg.buffer_memory_ceiling_mb * 1024 * 1024)
        ),
        pre_seconds=cfg.pre_seconds,
        post_seconds=cfg.post_seconds,
        buffer_mode_hint="raw",
    )


log = logging.getLogger(__name__)


class Pipeline:
    """Owns the set of Streams and provides lifecycle methods."""

    def __init__(
        self,
        config_path: str | Path,
        on_results: Optional[
            Callable[
                [StreamId, List[TrackState], List[ZoneCount], list],
                None,
            ]
        ] = None,
        event_queue: Optional["queue.Queue"] = None,
        evidence_recorder: Optional["object"] = None,
        metrics_port: int = 0,
        high_density_repo=None,
    ) -> None:
        self._config_path = Path(config_path)
        self._on_results = on_results or default_callback()
        # Event queue is the destination for CrossingEvents from all
        # tripwires in all streams. Stage 5 reads from here.
        self._event_queue = event_queue
        self._evidence_recorder = evidence_recorder  # injectable for tests
        self._metrics_port = int(metrics_port)
        self._high_density_repo = high_density_repo  # Stage 6: optional
        self._streams: List[Stream] = []
        self._config: Optional[AppConfig] = None
        self._shutdown = False

    @property
    def config(self) -> AppConfig:
        if self._config is None:
            raise RuntimeError("Pipeline.build() has not been called")
        return self._config

    @property
    def streams(self) -> List[Stream]:
        return list(self._streams)

    @property
    def evidence_recorder(self) -> Optional["object"]:
        return self._evidence_recorder

    @property
    def metrics_port(self) -> int:
        return self._metrics_port

    def build(self) -> None:
        """Parse the YAML and instantiate one Stream per entry.

        If ``_config`` has already been set on the instance (e.g. by a test
        injecting a config without a YAML file on disk), it is reused.

        Stage 5: per-camera failure isolation. If one stream's build raises
        (bad detector init, bad zone/tripwire config, etc.), it is logged
        and the other streams still come up.
        """
        if self._config is None:
            self._config = load_config(self._config_path)
        if self._config.evidence.enabled and self._evidence_recorder is None:
            self._evidence_recorder = _build_evidence_recorder(self._config.evidence)
        # Per-stream try/except: one bad stream doesn't take down the others.
        built: List[Stream] = []
        for sc in self._config.streams:
            try:
                built.append(self._build_stream(sc))
            except Exception:
                METRICS.inc_build_failure(sc.id)
                log.exception(
                    "failed to build stream; continuing",
                    extra={
                        "stream_id": sc.id,
                        "event": "build_failure",
                    },
                )
        self._streams = built
        # Wire the recorder against each stream. We do this lazily so that
        # the stream's buffer is already created.
        if self._evidence_recorder is not None:
            from .violations import LiveStreamPost
            for s in self._streams:
                post = LiveStreamPost(
                    stream_id=s.id, buffer=s.buffer, fps=s.fps
                )
                self._evidence_recorder.register_stream(post)
        # Stage 6: wire density sinks now that streams exist (and any
        # recorder is registered). Streams with a density_rule get a sink
        # that records the alert and asks the EvidenceRecorder for a clip.
        self._wire_density_sinks()

    def _build_stream(self, sc: StreamConfig) -> Stream:
        detector: Detector = YOLODetector(model_size=sc.model_size)
        tracker: Tracker = make_tracker(
            sc.tracker_type,
            with_reid=(sc.tracker_type == "botsort" and sc.use_appearance),
        )
        zone_manager: Optional[ZoneManager] = None
        if sc.zones:
            zone_manager = ZoneManager(zones=sc.zones, centroid_mode=sc.centroid_mode)
        tripwire_manager: Optional[TripwireManager] = None
        if sc.tripwires:
            tripwire_manager = TripwireManager(
                stream_id=sc.id, tripwires=sc.tripwires, event_queue=self._event_queue
            )
        # Stage 6: density rule + sink. Only constructed if the operator
        # asked for it (max_density_threshold != None) and at least one
        # zone exists. The sink is wired below in ``_wire_density_sinks``.
        density_rule = None
        if sc.max_density_threshold is not None and sc.zones:
            from .violations import DensityRule

            density_rule = DensityRule(
                threshold=sc.max_density_threshold,
                dwell_seconds=sc.density_dwell_seconds,
                cooldown_seconds=sc.density_cooldown_seconds,
            )
        return Stream(
            stream_id=sc.id,
            source=sc.source,
            detector=detector,
            tracker=tracker,
            on_results=self._on_results,
            buffer_seconds=sc.buffer_seconds,
            fps_fallback=sc.fps_fallback,
            display_conf=sc.display_conf,
            zone_manager=zone_manager,
            tripwire_manager=tripwire_manager,
            density_rule=density_rule,
            density_sink=None,  # wired by _wire_density_sinks()
        )

    def run(self) -> None:
        """Start every stream. Blocks until shutdown is called from another thread."""
        if not self._streams:
            self.build()
        if self._metrics_port > 0:
            start_prometheus_endpoint(self._metrics_port)
        for s in self._streams:
            s.start()
        try:
            # Park the main thread; workers do the work.
            stopper = threading.Event()
            stopper.wait()
        except KeyboardInterrupt:
            log.info(
                "KeyboardInterrupt — shutting down",
                extra={"event": "shutdown", "reason": "keyboard_interrupt"},
            )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        for s in self._streams:
            try:
                s.stop()
            except Exception:
                log.exception(
                    "error stopping stream %s",
                    s.id,
                    extra={"stream_id": s.id, "event": "shutdown_failure"},
                )
        if self._evidence_recorder is not None:
            # Run a retention cleanup at shutdown.
            try:
                retention = self._config.evidence.retention_days if self._config else 30.0
                self._evidence_recorder._disk_guard.cleanup_old_clips(retention)
            except Exception:
                log.exception("evidence retention cleanup failed")

    # ------------------------------------------------------------------
    # Stage 5 orchestrator: tie crossings → violations → evidence clips
    # ------------------------------------------------------------------

    def run_violation_consumer(
        self,
        repo,
        scan_store=None,
        window_seconds: Optional[float] = None,
        occupancy_provider=None,
    ) -> threading.Thread:
        """Drain ``self._event_queue`` (crossings) and write violations.

        Wired together:

            crossing → ViolationService.process_crossing → repo.record(v)
            → evidence_recorder.record(v, violation_id=rid)
            → repo.attach_clip(rid, clip.path, clip.url)

        ``window_seconds`` defaults to ``self._config.violations.window_seconds``
        when a config is present. Pass ``0`` to force a literal value.

        Returns the daemon Thread running the loop. Caller is expected
        to keep its process alive (e.g., with the existing ``run()``
        main-thread park).
        """
        from .violations import ViolationService

        if window_seconds is None:
            if self._config is not None and self._config.violations is not None:
                window_seconds = self._config.violations.window_seconds
            else:
                window_seconds = 10.0

        service = ViolationService(
            scan_store=scan_store,
            window_seconds=window_seconds,
            on_violation=self._make_violation_sink(repo),
        )

        def _consumer():
            from .violations.rules import consume_crossing_queue
            consume_crossing_queue(
                service=service,
                crossing_queue=self._event_queue,
                occupancy_provider=occupancy_provider,
                poll_timeout=0.05,
            )

        t = threading.Thread(target=_consumer, name="violation-consumer", daemon=True)
        t.start()
        return t

    def _wire_density_sinks(self) -> None:
        """Attach a density-sink to each stream that has a DensityRule.

        The sink persists the alert to ``self._high_density_repo`` (if
        configured) and asks the ``EvidenceRecorder`` (if configured) for
        a clip. The recorder's existing per-stream ``LiveStreamPost``
        already knows about the stream's ring buffer, so we just pass the
        HighDensityViolation through ``recorder.record``.
        """
        if self._high_density_repo is None and self._evidence_recorder is None:
            return
        for s in self._streams:
            if getattr(s, "_density_rule", None) is None:
                continue
            sink = self._make_density_sink(self._high_density_repo)
            s._density_sink = sink

    def _make_density_sink(self, repo):
        """Build a callable that persists + captures evidence for one alert."""
        recorder = self._evidence_recorder

        def on_density(v):
            vid = repo.record(v) if repo is not None else 0
            if recorder is not None:
                try:
                    clip = recorder.record(v, violation_id=vid)
                except Exception:
                    log.exception(
                        "evidence: record() raised for density alert %d",
                        vid,
                        extra={
                            "violation_id": vid,
                            "event": "evidence_failure",
                        },
                    )
                    clip = None
                if clip is not None and repo is not None:
                    try:
                        repo.attach_clip(vid, clip.path, clip.url)
                    except Exception:
                        log.exception(
                            "repo.attach_clip failed for density alert %d",
                            vid,
                            extra={
                                "violation_id": vid,
                                "event": "attach_clip_failure",
                            },
                        )
            return vid

        return on_density

    def _make_violation_sink(self, repo):
        """Compose persistence + evidence into a single on_violation callback."""
        recorder = self._evidence_recorder

        def on_violation(v):
            vid = repo.record(v)
            METRICS.inc_violation(str(getattr(v, "kind", "unknown")))
            if recorder is not None:
                try:
                    clip = recorder.record(v, violation_id=vid)
                except Exception:
                    log.exception(
                        "evidence: record() raised for violation %d",
                        vid,
                        extra={"violation_id": vid, "event": "evidence_failure"},
                    )
                    clip = None
                if clip is not None:
                    try:
                        repo.attach_clip(vid, clip.path, clip.url)
                    except Exception:
                        log.exception(
                            "repo.attach_clip failed for %d",
                            vid,
                            extra={"violation_id": vid, "event": "attach_clip_failure"},
                        )
            return vid
        return on_violation
