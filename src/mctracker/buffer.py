"""Thread-safe fixed-size ring buffer and a continuous capture thread.

The contract that matters:

* ``FrameBuffer.put`` is non-blocking. When the deque is full it atomically
  drops the oldest frame — the reader thread never waits on the processor.
* ``FrameBuffer.get_latest`` returns the most recent frame, or ``None`` if
  the buffer is empty.
* ``FrameBuffer.snapshot`` returns a frame-by-frame copy of the buffer
  contents. Used by the evidence-clip recorder to "freeze" the
  pre-event window without blocking the live reader.
* ``StreamReader`` runs on its own thread, reads from a ``cv2.VideoCapture``,
  and feeds the buffer continuously. It never blocks on the processor.

This is the decoupling guarantee that lets a slow detector/tracker never stall
the camera ingest.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .observability import METRICS
from .types import Frame

log = logging.getLogger(__name__)


def _now() -> float:
    return time.time()


class FrameBuffer:
    """A fixed-size ring buffer of (frame, timestamp) pairs.

    Thread-safe for one writer and any number of readers. The writer is the
    capture thread; readers are the per-stream processor thread.

    Uses ``threading.Condition`` for blocking ``get_latest`` so that the
    processor wakes exactly when a new frame arrives, instead of spinning
    on a sticky Event.

    Holds **raw** frames (``np.ndarray``) by default. For evidence-clip
    purposes this is the simple mode: ``snapshot()`` returns frames that
    can be passed directly to ``cv2.VideoWriter``. A subclass
    ``CompressedFrameBuffer`` flags itself as ``mode='compressed'``;
    downstream code (evidence clip builder) detects this and either
    falls back to raw frames (if available) or warns that clips can
    only start at the next keyframe.
    """

    def __init__(self, maxlen: int, mode: str = "raw") -> None:
        if maxlen <= 0:
            raise ValueError(f"maxlen must be > 0, got {maxlen}")
        if mode not in ("raw", "compressed"):
            raise ValueError(f"mode must be 'raw' or 'compressed', got {mode!r}")
        self._dq: collections.deque = collections.deque(maxlen=maxlen)
        self._cond = threading.Condition()
        self._maxlen = int(maxlen)
        self._mode = mode
        # Timestamp of the last frame returned by get_latest. Used so that
        # subsequent calls block until a strictly-newer frame is available.
        # -1.0 means "nothing returned yet".
        self._last_returned_ts: float = -1.0

    # ----- raw write path -----

    def put(self, frame: Frame, ts: float) -> None:
        with self._cond:
            self._dq.append((frame, ts))
            self._cond.notify_all()

    # ----- reader path -----

    def get_latest(self, timeout: Optional[float] = 0.0) -> Optional[tuple[Frame, float]]:
        """Return the most recent (frame, ts), or ``None`` on timeout / empty.

        Successive calls block until a *new* frame (strictly larger timestamp)
        has been put, so a fast processor cannot re-process the same frame
        in a tight loop. ``timeout=None`` blocks forever; ``timeout=0.0`` is
        non-blocking; a positive value is a bounded wait.
        """
        with self._cond:
            if not self._dq:
                if timeout == 0.0:
                    return None
                if timeout is None:
                    while not self._dq:
                        self._cond.wait()
                else:
                    if not self._cond.wait_for(lambda: len(self._dq) > 0, timeout=timeout):
                        return None
            frame, ts = self._dq[-1]
            if ts <= self._last_returned_ts:
                if timeout == 0.0:
                    return None
                if timeout is None:
                    while ts <= self._last_returned_ts:
                        self._cond.wait()
                        frame, ts = self._dq[-1]
                else:
                    deadline = _now() + timeout
                    while ts <= self._last_returned_ts:
                        remaining = deadline - _now()
                        if remaining <= 0:
                            return None
                        self._cond.wait(timeout=remaining)
                        frame, ts = self._dq[-1]
            self._last_returned_ts = ts
            return frame, ts

    # ----- introspection -----

    def __len__(self) -> int:
        with self._cond:
            return len(self._dq)

    @property
    def maxlen(self) -> int:
        return self._maxlen

    @property
    def mode(self) -> str:
        """Whether frames are held as raw arrays or compressed bytes.

        Evidence clip building is straightforward for ``raw`` — frames
        are passed directly to ``cv2.VideoWriter``. For ``compressed``,
        the clip builder either needs to fall back to raw (which means
        re-decoding on the fly) or clamp the start to the next keyframe.
        """
        return self._mode

    def snapshot(self) -> tuple[list, float]:
        """Return an independent copy of the current contents.

        Returned shape: ``(frames, last_returned_ts)`` where
        ``frames`` is a list of ``(frame, ts)`` ordered oldest→newest.
        Subsequent ``put()`` calls do NOT corrupt the snapshot — raw
        frames are ``.copy()``-ed so they don't share memory with the
        live deque.

        For compressed-mode buffers, frames are bytes-like and are
        returned as-is.
        """
        with self._cond:
            pairs = list(self._dq)
            wc_at_freeze = self._last_returned_ts
        if self._mode == "raw":
            copied: list[tuple] = []
            for frame, ts in pairs:
                copied.append((frame.copy() if frame is not None else None, ts))
            return copied, wc_at_freeze
        return pairs, wc_at_freeze

    def estimated_bytes(self) -> int:
        """Estimate the in-memory size of the current buffer.

        For ``raw`` mode, this is roughly ``len(self) * (H*W*3)``,
        computed from the first stored frame if available. For
        ``compressed``, we estimate from the average byte length of
        stored payloads.
        """
        with self._cond:
            n = len(self._dq)
            if n == 0:
                return 0
            sample = self._dq[0][0]
        if sample is None:
            return 0
        if self._mode == "raw":
            if hasattr(sample, "nbytes"):
                return n * sample.nbytes
            return n * (480 * 640 * 3)
        if hasattr(sample, "__len__"):
            return len(sample) * n
        return 64 * 1024 * n


class CompressedFrameBuffer(FrameBuffer):
    """A buffer that stores compressed payloads (H.264 NAL bytes, JPEGs, …).

    Frames are bytes-like objects instead of ``np.ndarray``. The
    evidence-clip builder detects this via ``mode == 'compressed'`` and
    either falls back to a raw-frame buffer (if maintained alongside)
    or issues a warning that the clip can only start at the next
    keyframe (i.e. the requested pre-event lookback may be shorter
    than requested).
    """

    def __init__(self, maxlen: int) -> None:
        super().__init__(maxlen=maxlen, mode="compressed")

    def put(self, frame: bytes, ts: float) -> None:
        with self._cond:
            self._dq.append((frame, ts))
            self._cond.notify_all()


def estimate_fps(source: str, fallback: int = 30) -> int:
    """Best-effort FPS read for sizing the ring buffer.

    Falls back to ``fallback`` if cv2 can't tell (RTSP streams in
    particular often return 0). Logs a warning so the operator knows
    the buffer is sized from a guess.
    """
    cap = cv2.VideoCapture(source)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or math.isnan(fps) or fps < 1:
        log.warning(
            "could not read FPS for source %r; falling back to %d", source, fallback
        )
        return int(fallback)
    return int(round(fps))


class StreamReader:
    """Continuous capture thread.

    Spawns one daemon thread that calls ``cap.read()`` in a loop and
    pushes every frame into the buffer. When the source stalls (no new
    frame within ``stall_timeout`` seconds) it logs a warning but keeps
    trying — RTSP streams are known to hiccup.

    Stage 5: for *live* sources (anything that isn't a video file or a
    webcam integer index), a read failure on a healthy stream triggers a
    reconnect-with-backoff: close the capture, sleep with exponential
    backoff capped at ``max_reconnect_backoff_s`` (default 30s), reopen,
    increment ``METRICS.stream_reconnects_total``. Video files and webcams
    keep the original "exit on EOF" behaviour — the operator is
    expected to restart the pipeline if a file ends.
    """

    # Backoff schedule in seconds. Stays short for the first few attempts
    # so a temporary hiccup recovers within seconds, then plateaus.
    _BACKOFF_S: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)

    def __init__(
        self,
        source: str,
        buffer: FrameBuffer,
        stall_timeout: float = 2.0,
        stream_id: str = "",
        max_reconnect_backoff_s: float = 30.0,
    ) -> None:
        self._source = source
        self._buffer = buffer
        self._stall_timeout = float(stall_timeout)
        self._stream_id = stream_id
        self._max_reconnect_backoff_s = float(max_reconnect_backoff_s)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_frame_ts: float = 0.0
        self._lock = threading.Lock()

    @property
    def stream_id(self) -> str:
        return self._stream_id

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"reader[{self._stream_id or self._source}]", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _backoff_for(self, attempt: int) -> float:
        # Capped exponential backoff. attempt=0 -> 1s, attempt=1 -> 2s, ...
        idx = min(attempt, len(self._BACKOFF_S) - 1)
        return min(self._BACKOFF_S[idx], self._max_reconnect_backoff_s)

    def _is_file_like(self) -> bool:
        return self._source.isdigit() or _looks_like_video(self._source)

    def _run(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            log.error(
                "could not open source %r",
                self._source,
                extra={"stream_id": self._stream_id, "event": "open_failed"},
            )
            return
        # For live sources, attempt counter starts at -1 so the first
        # reconnect after a failure uses attempt 0 -> 1s.
        reconnect_attempt = -1
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                now = time.time()
                if not ok or frame is None:
                    if self._is_file_like():
                        # File/webcam: end of stream is terminal.
                        return
                    # Live source: reconnect-with-backoff.
                    cap.release()
                    reconnect_attempt += 1
                    backoff = self._backoff_for(reconnect_attempt)
                    METRICS.inc_reconnect(self._stream_id or "unknown")
                    log.warning(
                        "stream stalled; reconnecting",
                        extra={
                            "stream_id": self._stream_id,
                            "event": "reconnect",
                            "attempt": reconnect_attempt,
                            "backoff_s": backoff,
                        },
                    )
                    # Interruptible sleep so shutdown() doesn't have to
                    # wait the full backoff window.
                    if self._stop.wait(timeout=backoff):
                        return
                    cap = cv2.VideoCapture(self._source)
                    if not cap.isOpened():
                        # Couldn't reopen; loop and try again with the
                        # next backoff tier.
                        continue
                    # Successfully reopened — reset the counter so a
                    # *future* failure starts fresh from 1s.
                    reconnect_attempt = -1
                    continue
                self._buffer.put(frame, now)
                with self._lock:
                    self._last_frame_ts = now
        finally:
            cap.release()


def _looks_like_video(source: str) -> bool:
    s = source.lower()
    return s.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"))


def make_blank_frame(h: int = 480, w: int = 640) -> Frame:
    """Helper for tests and the synthetic occlusion scenario."""
    return np.zeros((h, w, 3), dtype=np.uint8)
