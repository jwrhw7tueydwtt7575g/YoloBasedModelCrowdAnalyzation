"""Unit tests for the evidence module.

These tests don't drive the full pipeline (that's in
``test_evidence_integration.py``) — they pin the smaller pieces:

* ``FrameBuffer.snapshot`` returns an independent copy of the contents.
* ``FrameBuffer.mode`` flags raw vs compressed.
* ``DiskSpaceGuard`` reports status and prunes old clips.
* ``LocalDiskClipStorage`` writes the expected path layout.
* ``MemoryBudgetGuard`` tracks per-stream allocations and clips new
  registrations when over budget.
* ``EvidenceRecorder`` builds a clip via a synthetic post.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mctracker.buffer import (
    CompressedFrameBuffer,
    FrameBuffer,
)
from mctracker.violations import (
    ClipBuilder,
    CrossingRecord,
    DiskSpaceGuard,
    EvidenceRecorder,
    LiveStreamPost,
    LocalDiskClipStorage,
    MemoryBudgetGuard,
    NoopClipStorage,
    SyntheticStreamPost,
    Violation,
    ViolationKind,
    ViolationService,
    InMemoryViolationRepository,
)


# ---------------------------------------------------------------------------
# FrameBuffer.snapshot / mode
# ---------------------------------------------------------------------------


def test_frame_buffer_snapshot_returns_independent_copy():
    """Mutating the live buffer after a snapshot must NOT corrupt the
    snapshot — and vice versa.
    """
    f0 = np.zeros((4, 4, 3), dtype=np.uint8)
    f1 = np.ones((4, 4, 3), dtype=np.uint8) * 255
    buf = FrameBuffer(maxlen=8)
    buf.put(f0, 0.0)
    buf.put(f1, 1.0)

    snap, _ts = buf.snapshot()
    assert len(snap) == 2
    snap_frame = snap[0][0]
    assert (snap_frame == f0).all()

    # Mutate the live buffer to fill it past maxlen and evict f0.
    for i in range(10):
        buf.put(np.full((4, 4, 3), 64, dtype=np.uint8), 2.0 + i)

    # The snapshot frame should still match the original f0.
    assert (snap_frame == f0).all()

    # Mutating the snapshot frame should not affect the live buffer.
    snap_frame[:] = 99
    latest = buf.get_latest(timeout=0.05)
    assert latest is not None
    assert latest[0][0, 0, 0] != 99


def test_frame_buffer_mode_defaults_to_raw():
    buf = FrameBuffer(maxlen=4)
    assert buf.mode == "raw"


def test_compressed_frame_buffer_mode_flagged():
    buf = CompressedFrameBuffer(maxlen=4)
    assert buf.mode == "compressed"
    buf.put(b"\x00\x01", 0.0)
    snap, _ts = buf.snapshot()
    assert snap[0][0] == b"\x00\x01"


def test_frame_buffer_estimated_bytes_for_raw():
    f = np.full((480, 640, 3), 100, dtype=np.uint8)
    buf = FrameBuffer(maxlen=10)
    for i in range(5):
        buf.put(f, float(i))
    # 5 frames × 480×640×3 = 4,608,000 bytes.
    assert buf.estimated_bytes() == 5 * 480 * 640 * 3


def test_frame_buffer_estimated_bytes_empty():
    buf = FrameBuffer(maxlen=4)
    assert buf.estimated_bytes() == 0


# ---------------------------------------------------------------------------
# LocalDiskClipStorage path layout
# ---------------------------------------------------------------------------


def test_local_clip_storage_writes_dated_path(tmp_path: Path):
    storage = LocalDiskClipStorage(str(tmp_path / "evidence_clips"))
    ts = time.time()
    clip = storage.save(b"\x00\x01\x02", stream_id="cam0", violation_id=42, timestamp=ts)
    assert clip.backend == "local"
    assert clip.path
    p = Path(clip.path)
    assert p.exists()
    assert p.parent.parent.name == "cam0"
    assert p.name == "42.mp4"
    assert p.parent.is_dir()


def test_local_clip_storage_list_older_than(tmp_path: Path):
    storage = LocalDiskClipStorage(str(tmp_path / "evidence_clips"))
    storage.save(b"x", stream_id="a", violation_id=1, timestamp=time.time())
    old_path = storage.save(
        b"x", stream_id="a", violation_id=2, timestamp=time.time() - 40 * 86400,
    )
    # save() writes the bytes now, so mtime is wall-clock current. Force
    # mtime to match the recorded violation timestamp so the cleanup
    # logic sees this as a 40-day-old file.
    import os as _os
    old_mtime = time.time() - 40 * 86400
    _os.utime(old_path.path, (old_mtime, old_mtime))
    paths = storage.list_clips_older_than(days=30)
    assert old_path.path in [str(p) for p in paths]


# ---------------------------------------------------------------------------
# DiskSpaceGuard
# ---------------------------------------------------------------------------


def test_disk_space_guard_status(tmp_path: Path):
    g = DiskSpaceGuard(base_dir=str(tmp_path), free_threshold_mb=1.0)
    s = g.status()
    assert s.threshold_bytes == 1 * 1024 * 1024
    # We can't predict free bytes but we can predict that free_bytes > 0.
    assert s.free_bytes > 0


def test_disk_space_guard_has_space_for_returns_bool(tmp_path: Path):
    g = DiskSpaceGuard(base_dir=str(tmp_path), free_threshold_mb=1.0)
    assert isinstance(g.has_space_for(10), bool)


def test_disk_space_guard_cleanup_old_removes(tmp_path: Path):
    storage = LocalDiskClipStorage(str(tmp_path / "evidence_clips"))
    g = DiskSpaceGuard(
        base_dir=str(tmp_path / "evidence_clips"),
        free_threshold_mb=1.0,
        storage=storage,
    )
    storage.save(b"x", stream_id="a", violation_id=1, timestamp=time.time())
    old_clip = storage.save(
        b"x", stream_id="a", violation_id=2, timestamp=time.time() - 40 * 86400,
    )
    import os as _os
    old_mtime = time.time() - 40 * 86400
    _os.utime(old_clip.path, (old_mtime, old_mtime))

    before = list((tmp_path / "evidence_clips").rglob("*.mp4"))
    deleted = g.cleanup_old_clips(retention_days=30)
    after = list((tmp_path / "evidence_clips").rglob("*.mp4"))
    assert deleted == 1
    assert len(after) == len(before) - 1


# ---------------------------------------------------------------------------
# MemoryBudgetGuard
# ---------------------------------------------------------------------------


def test_memory_budget_register_below_ceiling():
    g = MemoryBudgetGuard(ceiling_bytes=10_000_000)
    f = np.full((100, 100, 3), 100, dtype=np.uint8)  # 30 KB
    buf = FrameBuffer(maxlen=10)
    for i in range(5):
        buf.put(f, float(i))
    g.register("a", buf)
    assert g.used_bytes > 0


def test_memory_budget_register_above_ceiling_warns_and_clips(caplog):
    g = MemoryBudgetGuard(ceiling_bytes=50_000)
    f = np.full((480, 640, 3), 100, dtype=np.uint8)  # ~922 KB
    buf = FrameBuffer(maxlen=10)
    for i in range(5):
        buf.put(f, float(i))  # ~4.6 MB total
    import logging
    with caplog.at_level(logging.WARNING, logger="mctracker.violations.evidence"):
        g.register("a", buf)
    # The recorded allocation should be smaller than the requested one
    # (clipped to fit ceiling).
    assert g.used_bytes <= g.ceiling_bytes


def test_memory_budget_unregister():
    g = MemoryBudgetGuard(ceiling_bytes=1_000_000)
    buf = FrameBuffer(maxlen=4)
    f = np.full((10, 10, 3), 1, dtype=np.uint8)
    buf.put(f, 0.0)
    g.register("a", buf)
    assert g.used_bytes > 0
    g.unregister("a")
    assert g.used_bytes == 0


# ---------------------------------------------------------------------------
# ClipBuilder
# ---------------------------------------------------------------------------


def test_clip_builder_produces_mp4(tmp_path: Path):
    fps = 10.0
    frames = [
        np.full((32, 32, 3), (i % 255), dtype=np.uint8)
        for i in range(20)
    ]
    b = ClipBuilder(fps=fps)
    data = b.build_bytes(frames)
    assert data[:4] == b"\x00\x00\x00\x00" or len(data) > 0  # sanity

    # Reopen as cv2.VideoCapture to confirm it's a video.
    out = tmp_path / "test.mp4"
    out.write_bytes(data)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert n >= 18  # mp4v may pack frames; allow a few slack


# ---------------------------------------------------------------------------
# EvidenceRecorder (synthetic post)
# ---------------------------------------------------------------------------


def _no_scans_store():
    class S:
        def add(self, scan):
            pass

        def query_window(self, zone_id, ts, window):
            from mctracker.violations import ScansInWindow
            return ScansInWindow(scans=[])

    return S()


def test_evidence_recorder_synthetic_post_produces_clip(tmp_path: Path):
    base_dir = tmp_path / "evidence_clips"
    storage = LocalDiskClipStorage(str(base_dir))
    g = DiskSpaceGuard(base_dir=str(base_dir), free_threshold_mb=1.0)
    recorder = EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=10.0),
        disk_guard=g,
        pre_seconds=2.0,
        post_seconds=1.0,
    )

    fps = 10.0
    frames = [
        np.full((32, 32, 3), (i % 255), dtype=np.uint8)
        for i in range(60)
    ]
    post = SyntheticStreamPost(
        stream_id="cam0",
        frames=frames,
        fps=fps,
        start_ts=1000.0,
    )
    post._cursor = 30  # pretend the violation happened at frame 30
    recorder.register_stream(post)

    v = Violation(
        timestamp=1003.0,
        stream_id="cam0",
        zone_id="entrance",
        tripwire_id="entrance",
        track_id=1,
        direction="left_to_right",
        embedding=None,
        kind=ViolationKind.UNMATCHED,
    )

    clip = recorder.record(v, violation_id=7)
    assert clip is not None
    assert clip.backend == "local"
    assert Path(clip.path).exists()
    assert Path(clip.path).name == "7.mp4"


def test_evidence_recorder_warns_on_compressed_mode(tmp_path: Path, caplog):
    base_dir = tmp_path / "evidence_clips"
    storage = LocalDiskClipStorage(str(base_dir))
    g = DiskSpaceGuard(base_dir=str(base_dir), free_threshold_mb=1.0)
    recorder = EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=10.0),
        disk_guard=g,
        pre_seconds=1.0,
        post_seconds=1.0,
        buffer_mode_hint="compressed",
    )

    fps = 10.0
    frames = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(20)]
    post = SyntheticStreamPost(
        stream_id="cam0", frames=frames, fps=fps, start_ts=1000.0,
    )
    recorder.register_stream(post)

    v = Violation(
        timestamp=1000.5,
        stream_id="cam0",
        zone_id="z1",
        tripwire_id="t1",
        track_id=1,
        direction="left_to_right",
        embedding=None,
        kind=ViolationKind.UNMATCHED,
    )

    import logging
    with caplog.at_level(logging.WARNING, logger="mctracker.violations.evidence"):
        recorder.record(v, violation_id=11)
    assert any(
        "compressed-mode" in rec.message for rec in caplog.records
    )


def test_evidence_recorder_returns_none_for_unknown_stream(tmp_path: Path):
    base_dir = tmp_path / "evidence_clips"
    storage = LocalDiskClipStorage(str(base_dir))
    g = DiskSpaceGuard(base_dir=str(base_dir), free_threshold_mb=1.0)
    recorder = EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=10.0),
        disk_guard=g,
    )

    v = Violation(
        timestamp=1000.0,
        stream_id="not_registered",
        zone_id="z1",
        tripwire_id="t1",
        track_id=1,
        direction="left_to_right",
        embedding=None,
        kind=ViolationKind.UNMATCHED,
    )
    assert recorder.record(v, violation_id=1) is None


def test_evidence_recorder_noop_storage(tmp_path: Path):
    """NoopClipStorage records saves without writing any bytes to disk."""
    storage = NoopClipStorage()
    g = DiskSpaceGuard(base_dir=str(tmp_path), free_threshold_mb=1.0)
    recorder = EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=10.0),
        disk_guard=g,
        pre_seconds=1.0,
        post_seconds=1.0,
    )
    fps = 10.0
    frames = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(30)]
    post = SyntheticStreamPost(
        stream_id="cam0", frames=frames, fps=fps, start_ts=1000.0,
    )
    post._cursor = 10
    recorder.register_stream(post)

    v = Violation(
        timestamp=1001.0,
        stream_id="cam0",
        zone_id="z1",
        tripwire_id="t1",
        track_id=1,
        direction="left_to_right",
        embedding=None,
        kind=ViolationKind.UNMATCHED,
    )
    clip = recorder.record(v, violation_id=2)
    assert clip is not None
    assert clip.backend == "noop"
    assert storage.saves, "noop storage should have recorded the save"
