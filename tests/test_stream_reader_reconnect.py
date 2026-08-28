"""Stage 5 RTSP reconnect-with-backoff tests.

We don't drive a real RTSP stream in tests — that would be flaky and slow.
Instead we verify the backoff logic against a fake capture object that
counts open/close/read calls. The capture contract matches what
``StreamReader._run`` actually expects from ``cv2.VideoCapture``.
"""

from __future__ import annotations

import threading
import time

import pytest

from mctracker.buffer import FrameBuffer, StreamReader
from mctracker.observability import METRICS, reset_metrics


class _FakeCapture:
    """Fake cv2.VideoCapture that fails N times before producing frames.

    ``open=False`` mimics "could not open" at construction time (the reader
    will exit immediately). ``fail_count`` mimics "open succeeded but reads
    keep returning (False, None)".
    """

    def __init__(self, frames: list, fail_count: int = 0, open: bool = True):
        self._frames = list(frames)
        self._fail_count = fail_count
        self._open = open
        self.read_count = 0
        self.release_count = 0

    def isOpened(self) -> bool:
        return self._open

    def read(self):
        self.read_count += 1
        if self._fail_count > 0:
            self._fail_count -= 1
            return (False, None)
        if not self._frames:
            return (False, None)
        f = self._frames.pop(0)
        return (True, f)

    def release(self) -> None:
        self.release_count += 1


@pytest.fixture(autouse=True)
def _isolated_metrics():
    """Each test starts with a fresh METRICS to avoid cross-pollution."""
    reset_metrics()
    yield
    reset_metrics()


def test_stream_reader_reconnect_after_read_failure(monkeypatch, capsys):
    """A fake capture that fails N times should trigger N reconnect attempts
    on the metrics counter, then start producing frames."""
    reset_metrics()
    frames = [b"\x00\x00\x00\x00"] * 4
    fake = _FakeCapture(frames=frames, fail_count=3)
    buf = FrameBuffer(maxlen=8)

    # Patch cv2.VideoCapture to always return our fake.
    import mctracker.buffer as buf_mod

    monkeypatch.setattr(
        buf_mod.cv2, "VideoCapture", lambda *_a, **_kw: fake,
    )

    reader = StreamReader(
        source="rtsp://fake/stream",
        buffer=buf,
        stream_id="rtsp_test_cam",
        max_reconnect_backoff_s=0.05,  # short for speed
    )
    reader.start()
    # Wait for frames to land. Generous timeout to absorb the backoff sleeps.
    deadline = time.time() + 2.0
    while time.time() < deadline and len(buf) < 4:
        time.sleep(0.05)
    reader.stop(timeout=5.0)

    # After 3 read failures followed by 4 successful reads, the metrics
    # counter for this stream_id should be at least 3.
    counters = METRICS.stream_reconnects_total.labelled()
    assert counters.get(("rtsp_test_cam",), 0) >= 3
    # The reader is still alive enough that frames made it to the buffer.
    assert len(buf) == 4


def test_stream_reader_file_source_exits_on_eof(monkeypatch):
    """Video files must NOT trigger reconnect-with-backoff; they exit on EOF."""
    # Run the test in a fresh METRICS so prior tests' reader threads
    # (which may still be tearing down) don't pollute the counter.
    reset_metrics()
    frames = [b"\x00\x00\x00\x00"] * 2
    fake = _FakeCapture(frames=frames, fail_count=0)
    buf = FrameBuffer(maxlen=8)
    import mctracker.buffer as buf_mod
    monkeypatch.setattr(
        buf_mod.cv2, "VideoCapture", lambda *_a, **_kw: fake,
    )
    reader = StreamReader(
        source="/videos/sample.mp4",  # file-like source
        buffer=buf,
        stream_id="file_test_cam",
    )
    reader.start()
    # Generous join window: the file-source path doesn't sleep.
    reader.stop(timeout=5.0)
    # Snapshot only this stream's counter; previous tests use other stream_ids.
    snap = METRICS.snapshot()
    counters = METRICS.stream_reconnects_total.labelled()
    # The file-source test contributes 0 reconnects under its own stream id.
    assert counters.get(("file_test_cam",), 0) == 0


def test_stream_reader_backoff_grows_then_caps(monkeypatch):
    """Backoff schedule: 1, 2, 4, 8, 15, 30 (capped at max_reconnect_backoff_s)."""
    reader = StreamReader(
        source="rtsp://x", buffer=FrameBuffer(maxlen=4), max_reconnect_backoff_s=30.0,
    )
    schedule = [reader._backoff_for(i) for i in range(10)]
    assert schedule[0] == 1.0
    assert schedule[1] == 2.0
    assert schedule[2] == 4.0
    assert schedule[3] == 8.0
    assert schedule[4] == 15.0
    assert schedule[5] == 30.0
    # Capped.
    assert schedule[9] == 30.0
    # Smaller cap honoured.
    cap_reader = StreamReader(
        source="rtsp://x", buffer=FrameBuffer(maxlen=4), max_reconnect_backoff_s=0.1,
    )
    assert cap_reader._backoff_for(0) == 0.1


def test_stream_reader_open_failure_exits_cleanly(monkeypatch):
    """An open failure on the very first read must NOT loop forever."""
    fake = _FakeCapture(frames=[], fail_count=0, open=False)
    import mctracker.buffer as buf_mod
    monkeypatch.setattr(
        buf_mod.cv2, "VideoCapture", lambda *_a, **_kw: fake,
    )
    reader = StreamReader(source="rtsp://dead", buffer=FrameBuffer(maxlen=4))
    reader.start()
    reader.stop(timeout=2.0)