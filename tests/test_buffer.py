"""Tests for the ring buffer and reader behavior."""

from __future__ import annotations

import time

import pytest

from mctracker.buffer import FrameBuffer


def test_buffer_init_rejects_zero_maxlen():
    with pytest.raises(ValueError):
        FrameBuffer(maxlen=0)


def test_buffer_put_get_latest_returns_most_recent():
    buf = FrameBuffer(maxlen=3)
    f1 = bytes([1])  # anything hashable; we don't care about contents here
    f2 = bytes([2])
    f3 = bytes([3])
    buf.put(f1, 1.0)
    buf.put(f2, 2.0)
    buf.put(f3, 3.0)
    frame, ts = buf.get_latest(timeout=0.1)
    assert ts == 3.0
    assert frame is f3


def test_buffer_drops_oldest_when_full():
    buf = FrameBuffer(maxlen=2)
    buf.put("a", 1.0)
    buf.put("b", 2.0)
    buf.put("c", 3.0)
    assert len(buf) == 2
    # The first frame should have been evicted; the latest is "c"
    frame, ts = buf.get_latest(timeout=0.1)
    assert frame == "c" and ts == 3.0


def test_buffer_get_latest_returns_none_on_empty():
    buf = FrameBuffer(maxlen=2)
    assert buf.get_latest(timeout=0.0) is None


def test_buffer_get_latest_blocks_until_put():
    buf = FrameBuffer(maxlen=2)

    def _late_put():
        time.sleep(0.05)
        buf.put("late", 99.0)

    import threading
    t = threading.Thread(target=_late_put)
    t.start()
    frame, ts = buf.get_latest(timeout=1.0)
    t.join()
    assert frame == "late" and ts == 99.0


def test_buffer_is_thread_safe_under_concurrent_writers():
    import threading
    buf = FrameBuffer(maxlen=1000)

    def writer(prefix: str, n: int):
        for i in range(n):
            buf.put(f"{prefix}-{i}", float(i))

    threads = [threading.Thread(target=writer, args=(f"w{i}", 100)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf) == 500  # 5 writers * 100 writes, but maxlen=1000 so all fit
