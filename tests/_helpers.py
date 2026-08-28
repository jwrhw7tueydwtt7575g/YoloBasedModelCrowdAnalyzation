"""Test helpers shared across the test files.

Lives outside ``conftest.py`` so the test modules can do a plain
``from _helpers import ...`` — pytest's conftest auto-import magic only
works for fixtures, not for ``from conftest import ...``.
"""

from __future__ import annotations

import threading
import time
from typing import Iterable, List, Optional

import numpy as np

from mctracker.buffer import FrameBuffer
from mctracker.detector import Detector
from mctracker.stream import Stream
from mctracker.tracker import Tracker
from mctracker.types import Detection, Frame


# ---------------------------------------------------------------------------
# Frame / detection helpers
# ---------------------------------------------------------------------------


def make_frame(h: int = 480, w: int = 640, color: tuple = (0, 0, 0)) -> Frame:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def make_detection(
    x1: float = 100.0,
    y1: float = 100.0,
    x2: float = 200.0,
    y2: float = 200.0,
    conf: float = 0.9,
    cls: int = 0,
    det_id: Optional[int] = None,
) -> Detection:
    return Detection(
        xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        conf=conf,
        cls=cls,
        det_id=det_id,
    )


# ---------------------------------------------------------------------------
# ScriptedDetector — drives frames from a per-call script
# ---------------------------------------------------------------------------


class ScriptedDetector(Detector):
    def __init__(self) -> None:
        self._script: List[Optional[List[Detection]]] = []
        self.calls: List[Frame] = []

    def set_script(self, script: Iterable[Iterable[Detection] | None]) -> None:
        self._script = [list(d) if d is not None else None for d in script]

    def detect(self, frame: Frame) -> List[Detection]:
        self.calls.append(frame)
        if not self._script:
            return []
        out = self._script.pop(0)
        return list(out) if out is not None else []


# ---------------------------------------------------------------------------
# FakeReader — synchronous, in-test replacement for StreamReader
# ---------------------------------------------------------------------------


class FakeReader:
    def __init__(self, source: str = "fake://test", buffer: Optional[FrameBuffer] = None) -> None:
        self._source = source
        self._buffer = buffer if buffer is not None else FrameBuffer(maxlen=16)
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self, timeout: float = 2.0) -> None:
        self.stopped = True

    def feed(self, frame: Frame) -> None:
        self._buffer.put(frame, time.time())

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer


def build_stream_for_test(
    stream_id: str,
    detector: Detector,
    tracker: Tracker,
    on_results,
    buffer_maxlen: int = 8,
    display_conf: float = 0.25,
) -> tuple[Stream, FakeReader]:
    """Construct a Stream with a FakeReader, bypassing cv2 entirely."""
    fake = FakeReader(source=f"fake://{stream_id}")
    s = Stream.__new__(Stream)
    s.id = stream_id
    s.source = f"fake://{stream_id}"
    s._detector = detector
    s._tracker = tracker
    s._on_results = on_results
    s._display_conf = float(display_conf)
    s._buffer = fake.buffer
    s._reader = fake
    s._zone_manager = None
    s._tripwire_manager = None
    s._active = {}
    s._lock = threading.Lock()
    s._last_centroid = {}
    s._stop = threading.Event()
    s._thread = None
    s._fps = 30.0
    return s, fake


def wait_for(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
