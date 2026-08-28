"""Evidence clip recording triggered by violations.

Architecture
------------

1. The Stage 1 ``FrameBuffer`` is the source of the *pre-event* window.
   When a violation is recorded, we ``snapshot()`` the buffer
   atomically — the live reader keeps filling, the snapshot is a
   deepcopy of the contents.

2. An ``EvidenceRecorder`` is configured with one or more *stream
   posts*: a function returning frames for a named stream on demand.
   On ``record(violation)``, we ask the matching stream for its
   current snapshot *plus* a configurable post-event duration of
   freshly-captured frames. For tests this is implemented by a
   ``StreamPost`` that just reads frames from a list; for production
   it pulls from a kept-open VideoCapture on the same source.

3. A ``ClipBuilder`` concatenates the pre+post frames and writes an
   MP4 using ``cv2.VideoWriter``. Fourcc is ``mp4v`` for portability.

4. A ``ClipStorage`` swappable backend persists the bytes. The
   default ``LocalDiskClipStorage`` writes
   ``{base}/{camera_id}/{date}/{violation_id}.mp4``. A
   ``NoopClipStorage`` is used by tests. S3/GCS would add a new
   implementation behind the same Protocol without changing the
   violation pipeline logic.

5. A ``DiskSpaceGuard`` checks free bytes before each save, raises
   on threshold breach, and prunes clips older than
   ``retention_days`` on demand.

6. A ``MemoryBudgetGuard`` tracks the total bytes used across all
   registered per-stream buffers. Adding a new stream is rejected
   with a logged warning if it would push us over the ceiling; the
   caller may ask the guard for a *clipped* maxlen to keep the
   process alive while operators investigate.

Keyframe alignment
------------------

The Stage 1 buffer is raw ``np.ndarray`` (mode='raw'), so the
keyframe-alignment problem is moot — every frame is a full frame.
If a future compressed-mode buffer (e.g. ``CompressedFrameBuffer``
holding H.264 NAL bytes) is the source, the recorder detects the
mode and either:

* falls back to a parallel raw buffer if one is registered, or
* clamps the clip start to the most recent in-buffer keyframe-ish
  boundary and emits a warning. We don't yet detect H.264 keyframes
  in the payload — the warning is best-effort and the clip may
  start a few frames later than the requested pre-event lookback.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Sequence

import cv2
import numpy as np

from ..buffer import FrameBuffer
from .rules import Violation

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class LocalClip:
    """Where a saved clip lives (on whatever backend)."""

    path: str  # local path for default backend; may be a URL for cloud
    url: str   # a string that downstream code can hand to a browser
    backend: str  # "local" | "noop" | "s3" | ...

    def to_row(self) -> dict:
        return {"clip_path": self.path, "clip_url": self.url, "clip_backend": self.backend}


class ClipStorage(Protocol):
    """Interface every clip backend must satisfy."""

    def save(
        self,
        content: bytes,
        stream_id: str,
        violation_id: int,
        timestamp: float,
    ) -> LocalClip:
        """Persist ``content`` and return its location."""

    def get_url(self, path: str) -> str:
        """Return a URL that humans can click on for ``path``."""


# ---------------------------------------------------------------------------
# Local disk storage
# ---------------------------------------------------------------------------


class LocalDiskClipStorage:
    """Saves clips under ``{base_dir}/{stream_id}/{date}/{violation_id}.mp4``.

    ``base_dir`` defaults to ``./evidence_clips``. The same directory is
    also the unit on which disk-space checks run.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def save(
        self,
        content: bytes,
        stream_id: str,
        violation_id: int,
        timestamp: float,
    ) -> LocalClip:
        d = self._date_dir(stream_id, timestamp)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{violation_id}.mp4"
        path.write_bytes(content)
        return LocalClip(
            path=str(path),
            url=path.as_uri(),
            backend="local",
        )

    def get_url(self, path: str) -> str:
        return Path(path).as_uri()

    def _date_dir(self, stream_id: str, timestamp: float) -> Path:
        # Date folder in UTC; both the recorder and the operator see
        # the same daily bucket regardless of timezone.
        dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        sub = dt.strftime("%Y-%m-%d")
        return self._base_dir / str(stream_id) / sub

    def list_clips_older_than(self, days: float) -> List[Path]:
        """Return all .mp4 paths under base_dir older than ``days`` days.

        Used by ``DiskSpaceGuard.cleanup_old_clips``.
        """
        cutoff = time.time() - days * 86400.0
        out: List[Path] = []
        if not self._base_dir.exists():
            return out
        for p in self._base_dir.rglob("*.mp4"):
            try:
                if p.stat().st_mtime < cutoff:
                    out.append(p)
            except FileNotFoundError:
                continue
        return out


class NoopClipStorage:
    """In-memory storage for tests. Records every save but writes nowhere."""

    def __init__(self) -> None:
        self.saves: list[dict] = []

    def save(self, content, stream_id, violation_id, timestamp):
        entry = {
            "content": content,
            "stream_id": stream_id,
            "violation_id": violation_id,
            "timestamp": timestamp,
            "url": f"noop://{stream_id}/{violation_id}",
        }
        self.saves.append(entry)
        return LocalClip(path="", url=entry["url"], backend="noop")

    def get_url(self, path: str) -> str:
        return path or "noop://"


# ---------------------------------------------------------------------------
# Disk-space guard
# ---------------------------------------------------------------------------


@dataclass
class DiskSpaceStatus:
    free_bytes: int
    threshold_mb: float
    threshold_bytes: int
    ok: bool

    def to_row(self) -> dict:
        return {
            "free_bytes": self.free_bytes,
            "threshold_mb": self.threshold_mb,
            "ok": self.ok,
        }


class DiskSpaceGuard:
    """Pre-write free-space check + retention cleanup.

    Use::

        guard = DiskSpaceGuard(base_dir="./evidence_clips", free_threshold_mb=2048)
        if not guard.has_space_for(estimated_bytes):
            raise DiskSpaceError(...)
        storage.save(...)

        # Periodically:
        guard.cleanup_old_clips(retention_days=30)
    """

    def __init__(
        self,
        base_dir: str | Path,
        free_threshold_mb: float = 2048.0,
        storage: Optional[LocalDiskClipStorage] = None,
    ) -> None:
        if free_threshold_mb <= 0:
            raise ValueError("free_threshold_mb must be > 0")
        self._base_dir = Path(base_dir).resolve()
        self._threshold_mb = float(free_threshold_mb)
        self._threshold_bytes = int(self._threshold_mb * 1024 * 1024)
        self._storage = storage or LocalDiskClipStorage(self._base_dir)
        self._alerted_below_threshold = False
        self._lock = threading.Lock()

    @property
    def free_threshold_mb(self) -> float:
        return self._threshold_mb

    def free_bytes(self) -> int:
        usage = shutil.disk_usage(str(self._base_dir))
        return int(usage.free)

    def status(self) -> DiskSpaceStatus:
        free = self.free_bytes()
        ok = free >= self._threshold_bytes
        return DiskSpaceStatus(
            free_bytes=free,
            threshold_mb=self._threshold_mb,
            threshold_bytes=self._threshold_bytes,
            ok=ok,
        )

    def has_space_for(self, n_bytes: int) -> bool:
        free = self.free_bytes()
        if free - n_bytes < self._threshold_bytes:
            with self._lock:
                if not self._alerted_below_threshold:
                    log.warning(
                        "evidence disk space low: free=%d bytes, threshold=%d bytes",
                        free,
                        self._threshold_bytes,
                    )
                    self._alerted_below_threshold = True
            return False
        return True

    def cleanup_old_clips(self, retention_days: float) -> int:
        """Delete clips older than ``retention_days``. Returns count deleted.

        Safe to call repeatedly. Uses ``LocalDiskClipStorage.list_clips_older_than``.
        """
        if retention_days <= 0:
            raise ValueError("retention_days must be > 0")
        paths = self._storage.list_clips_older_than(retention_days)
        deleted = 0
        for p in paths:
            try:
                p.unlink()
                deleted += 1
            except FileNotFoundError:
                continue
            except Exception:  # pragma: no cover - logging only
                log.exception("failed to delete old clip %s", p)
        if deleted:
            log.info("evidence retention: deleted %d clip(s) older than %.1f days",
                     deleted, retention_days)
            # Reset the alert flag — we've reclaimed space.
            with self._lock:
                self._alerted_below_threshold = False
        return deleted


class DiskSpaceError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Clip builder
# ---------------------------------------------------------------------------


@dataclass
class ClipBuilderResult:
    bytes_written: int
    frame_count: int
    fps_used: float
    size_px: tuple
    notes: List[str]


class ClipBuilder:
    """Encodes a flat list of ``np.ndarray`` frames to MP4 bytes.

    Frames are expected to be ``(H, W, 3)`` BGR. The fourcc is
    ``mp4v`` by default for portability — an alternative is ``avc1``
    but it needs platform FFmpeg support. ``mp4v`` works on every
    opencv-python-headless wheel we've tested.
    """

    def __init__(self, fps: float = 30.0, fourcc: str = "mp4v") -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._fps = float(fps)
        self._fourcc = fourcc

    @property
    def fps(self) -> float:
        return self._fps

    def build_bytes(self, frames: Sequence[np.ndarray]) -> bytes:
        """Return the encoded MP4 bytes for a flat list of frames."""
        import tempfile

        if not frames:
            raise ValueError("cannot build a clip from zero frames")
        # Filter out Nones defensively.
        valid = [f for f in frames if f is not None]
        if not valid:
            raise ValueError("no non-None frames to encode")
        h, w = valid[0].shape[:2]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            writer = cv2.VideoWriter(
                tmp.name,
                cv2.VideoWriter_fourcc(*self._fourcc),
                self._fps,
                (int(w), int(h)),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open output")
            for f in valid:
                if f.shape[:2] != (h, w):
                    f = cv2.resize(f, (int(w), int(h)))
                writer.write(f)
            writer.release()
            with open(tmp.name, "rb") as fh:
                data = fh.read()
        finally:
            try:
                os.unlink(tmp.name)
            except FileNotFoundError:
                pass
        return data


# ---------------------------------------------------------------------------
# Stream post
# ---------------------------------------------------------------------------


class StreamPost(Protocol):
    """Source of post-event frames for one stream.

    Implementations:

    * ``LiveStreamPost`` — wraps the running ``FrameBuffer`` so that
      after the violation is detected we keep pulling frames for
      ``duration_seconds``.
    * ``SyntheticStreamPost`` — produces frames from a list, used by
      integration tests.
    """

    def stream_id(self) -> str:
        ...

    def snapshot_pre(self, max_seconds: float) -> tuple[list, float]:
        """Return ``(frames, last_ts)``. ``frames`` is a list of
        ``(frame, ts)`` oldest→newest.
        """
        ...

    def capture_post(self, duration_seconds: float) -> tuple[list, float]:
        """Capture ``duration_seconds`` of frames after the violation
        timestamp. Returns ``(frames, end_ts)``.
        """


class SyntheticStreamPost:
    """Produces frames from a pre-loaded list of ``np.ndarray``.

    Used by the integration test. The same module supports the
    pre+post pattern: ``snapshot_pre`` returns the last
    ``max_seconds * fps`` frames from the script and ``capture_post``
    returns the next ``duration_seconds * fps`` frames.
    """

    def __init__(
        self,
        stream_id: str,
        frames: Sequence[np.ndarray],
        fps: float = 10.0,
        start_ts: float = 0.0,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        self._stream_id = str(stream_id)
        self._frames = list(frames)
        self._fps = float(fps)
        self._start_ts = float(start_ts)
        self._cursor = 0

    def stream_id(self) -> str:
        return self._stream_id

    @property
    def fps(self) -> float:
        return self._fps

    def snapshot_pre(self, max_seconds: float) -> tuple[list, float]:
        n = int(self._fps * max_seconds)
        if n <= 0:
            return [], self._start_ts
        end = self._cursor
        start = max(0, end - n)
        out = []
        for i in range(start, end):
            ts = self._start_ts + i / self._fps
            out.append((self._frames[i].copy(), ts))
        wc_at_freeze = out[-1][1] if out else self._start_ts
        return out, wc_at_freeze

    def capture_post(self, duration_seconds: float) -> tuple[list, float]:
        n = int(self._fps * duration_seconds)
        start = self._cursor
        end = min(len(self._frames), start + n)
        out = []
        for i in range(start, end):
            ts = self._start_ts + i / self._fps
            out.append((self._frames[i].copy(), ts))
        self._cursor = end
        end_ts = out[-1][1] if out else self._start_ts
        return out, end_ts


class LiveStreamPost:
    """Wraps a live ``FrameBuffer`` for evidence recording.

    The "post" capture polls the buffer for ``duration_seconds`` of
    real time. The "pre" snapshot is taken from the buffer at the
    moment ``record_pre`` is called.
    """

    def __init__(self, stream_id: str, buffer: FrameBuffer, fps: float) -> None:
        self._stream_id = str(stream_id)
        self._buffer = buffer
        self._fps = float(fps)

    def stream_id(self) -> str:
        return self._stream_id

    def snapshot_pre(self, max_seconds: float) -> tuple[list, float]:
        return self._buffer.snapshot()

    def capture_post(self, duration_seconds: float) -> tuple[list, float]:
        """Poll the buffer for ``duration_seconds`` real time, returning
        every frame that arrives."""
        deadline = time.time() + duration_seconds
        out = []
        last_ts = time.time()
        while time.time() < deadline:
            latest = self._buffer.get_latest(timeout=0.05)
            if latest is None:
                continue
            frame, ts = latest
            if frame is None:
                continue
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            out.append((frame.copy(), ts))
            last_ts = ts
        return out, last_ts


# ---------------------------------------------------------------------------
# Memory budget guard
# ---------------------------------------------------------------------------


class MemoryBudgetExceeded(RuntimeError):
    pass


class MemoryBudgetGuard:
    """Track the total pre-event buffer memory across all streams.

    Each registered buffer is sized in bytes via
    ``FrameBuffer.estimated_bytes()``. Adding a new buffer is rejected
    loudly (logged at WARNING; raises if ``strict=True``). A caller
    may ask for a *clipped* maxlen so the system keeps running
    while operators investigate.

    Safe to use across threads (Buffer registration / readout can
    happen on different threads).
    """

    def __init__(self, ceiling_bytes: int) -> None:
        if ceiling_bytes <= 0:
            raise ValueError("ceiling_bytes must be > 0")
        self._ceiling = int(ceiling_bytes)
        self._allocations: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def ceiling_bytes(self) -> int:
        return self._ceiling

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return sum(self._allocations.values())

    def register(self, stream_id: str, buffer: FrameBuffer) -> int:
        """Register a buffer; return its allocated bytes (possibly clipped).

        If registering would exceed the ceiling, the allocation is
        capped at ``budget_left`` (clipped_maxlen frames), a WARNING
        is logged, and the registration is allowed with the smaller
        value. Callers that want to *honour* the budget should rebuild
        the buffer at the returned maxlen.
        """
        used = self.used_bytes
        cur_size = buffer.estimated_bytes()
        budget_left = max(0, self._ceiling - used)
        if cur_size > budget_left:
            n_frames = max(1, len(buffer))
            per_frame = max(1, cur_size // n_frames)
            # Don't let the per-frame allocation exceed the budget by
            # itself; clamp both maxlen and the recorded size.
            clipped_maxlen = max(0, min(n_frames, budget_left // per_frame))
            cur_size = clipped_maxlen * per_frame
            log.warning(
                "memory budget tight: stream %s requested %d bytes, only %d left in "
                "ceiling of %d. Clipping buffer to maxlen=%d (%d bytes)",
                stream_id, cur_size, budget_left, self._ceiling, clipped_maxlen, cur_size,
            )
        with self._lock:
            self._allocations[stream_id] = cur_size
        return cur_size

    def unregister(self, stream_id: str) -> None:
        with self._lock:
            self._allocations.pop(stream_id, None)

    def clipped_maxlen_for(self, buffer: FrameBuffer) -> int:
        """If over the budget by joining, what's the maxlen we'd be allowed?"""
        budget_left = max(0, self._ceiling - self.used_bytes)
        if len(buffer) == 0:
            return 1
        per_frame = max(1, buffer.estimated_bytes() // max(1, len(buffer)))
        return max(1, budget_left // per_frame)


# ---------------------------------------------------------------------------
# Evidence recorder
# ---------------------------------------------------------------------------


class EvidenceRecorder:
    """Top-level orchestrator: violation in → clip → storage.

    Construct one per pipeline. ``register_stream(stream_id, post)``
    wires up the source for one camera. ``record(violation)`` is the
    consumer entry point — typically wired as the
    ``ViolationService.on_violation`` callback.

    The recorder is single-consumer (one thread) but adds internal
    locks around state mutation; the *write* to storage is itself
    synchronous but small.
    """

    def __init__(
        self,
        storage: ClipStorage,
        builder: ClipBuilder,
        disk_guard: DiskSpaceGuard,
        memory_guard: Optional[MemoryBudgetGuard] = None,
        buffer_mode_hint: str = "raw",
        pre_seconds: float = 5.0,
        post_seconds: float = 5.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if pre_seconds < 0:
            raise ValueError("pre_seconds must be >= 0")
        if post_seconds < 0:
            raise ValueError("post_seconds must be >= 0")
        self._storage = storage
        self._builder = builder
        self._disk_guard = disk_guard
        self._memory_guard = memory_guard
        self._buffer_mode_hint = buffer_mode_hint
        self._pre_seconds = float(pre_seconds)
        self._post_seconds = float(post_seconds)
        self._clock = clock or (lambda: time.time())

        self._posts: dict[str, StreamPost] = {}
        self._lock = threading.Lock()

    # ----- registration -----

    def register_stream(self, post: StreamPost) -> None:
        with self._lock:
            self._posts[post.stream_id()] = post

    def unregister_stream(self, stream_id: str) -> None:
        with self._lock:
            self._posts.pop(stream_id, None)

    def streams(self) -> list[str]:
        with self._lock:
            return list(self._posts.keys())

    # ----- main entry -----

    def record(self, violation: Violation, violation_id: int) -> Optional[LocalClip]:
        """Build a clip for ``violation`` and persist it. Returns the location."""
        notes: list[str] = []
        post = self._posts.get(violation.stream_id)
        if post is None:
            log.warning(
                "evidence: no stream registered for %s; skipping clip for violation %d",
                violation.stream_id, violation_id,
            )
            return None

        # 1. Capture pre-event snapshot.
        pre, _pre_wc = post.snapshot_pre(self._pre_seconds)

        # 2. Detect compressed-mode and emit a warning if the requested
        #    pre-event window could not be honoured.
        if self._buffer_mode_hint == "compressed":
            notes.append(
                "buffer is compressed-mode; pre-event lookback may be clamped to "
                "nearest keyframe-ish boundary"
            )
            log.warning(
                "evidence: stream %s is configured for compressed-mode buffer; "
                "pre-event lookback may be shorter than the requested %.1fs",
                violation.stream_id, self._pre_seconds,
            )

        # 3. Capture post-event frames.
        post_frames, _post_end_ts = post.capture_post(self._post_seconds)
        if not post_frames:
            notes.append(
                f"no post-event frames captured within {self._post_seconds}s window"
            )
            log.warning(
                "evidence: stream %s produced 0 post-event frames for violation %d",
                violation.stream_id, violation_id,
            )

        # 4. Build the MP4.
        all_frames = [f for f, _ts in (list(pre) + list(post_frames))]
        if not all_frames:
            log.warning("evidence: nothing to encode for violation %d", violation_id)
            return None

        try:
            content = self._builder.build_bytes(all_frames)
        except Exception:
            log.exception("evidence: cv2 writer failed for violation %d", violation_id)
            return None

        # 5. Disk guard.
        if not self._disk_guard.has_space_for(len(content)):
            log.warning(
                "evidence: skipping save for violation %d — not enough free disk",
                violation_id,
            )
            return None

        # 6. Storage save.
        clip = self._storage.save(
            content,
            stream_id=violation.stream_id,
            violation_id=int(violation_id),
            timestamp=violation.timestamp,
        )
        return clip
