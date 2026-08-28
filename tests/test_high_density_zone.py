"""Tests for Stage 6 high-density / crowd alerting.

Covers:

* ``DensityRule`` semantics:
  - fires when count > threshold for > dwell_seconds
  - does NOT fire when count stays at or below the threshold
  - respects per-zone cooldown after a fire
  - resets dwell after the count drops
* ``HighDensityViolation`` schema + ``InMemoryHighDensityRepository``
* Evidence clip recording for high-density alerts (via the existing
  ``EvidenceRecorder``)
* Stream → DensityRule → sink → repo wiring (end-to-end smoke test
  driven by ``_run_processor`` directly)
"""

from __future__ import annotations

import threading
import time
from typing import List

import numpy as np
import pytest

from mctracker.violations import (
    DensityRule,
    EvidenceRecorder,
    HighDensityViolation,
    InMemoryHighDensityRepository,
    NoopClipStorage,
    SyntheticStreamPost,
    Violation,
    ViolationKind,
    ZoneOccupancySnapshot,
    density_snapshots_from_zone_counts,
)
from mctracker.violations.evidence import (
    ClipBuilder,
    DiskSpaceGuard,
)
from mctracker.zones import ZoneCount


# ---------------------------------------------------------------------------
# DensityRule — core semantics
# ---------------------------------------------------------------------------


def _snap(stream_id: str, zone_id: str, count: int, ts: float) -> ZoneOccupancySnapshot:
    return ZoneOccupancySnapshot(
        timestamp=ts, zone_id=zone_id, count=count, stream_id=stream_id
    )


def test_density_rule_fires_when_count_stays_above_threshold():
    """Count=6, threshold=5, dwell=1.0 → fires after 1s."""
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=5,
        dwell_seconds=1.0,
        cooldown_seconds=10.0,
        on_violation=fired.append,
    )
    # First snapshot: starts the dwell timer.
    assert rule.observe(_snap("cam0", "z1", 6, ts=100.0)) is None
    # 0.5s later — still no fire (dwell not reached).
    assert rule.observe(_snap("cam0", "z1", 6, ts=100.5)) is None
    # 1.2s after start — fire!
    v = rule.observe(_snap("cam0", "z1", 7, ts=101.2))
    assert v is not None
    assert v.zone_id == "z1"
    assert v.stream_id == "cam0"
    assert v.density_count == 7  # peak observed during the run
    assert v.threshold == 5
    assert v.dwell_seconds >= 1.0
    assert len(fired) == 1
    assert fired[0] is v


def test_density_rule_does_not_fire_when_count_at_or_below_threshold():
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=5,
        dwell_seconds=2.0,
        cooldown_seconds=10.0,
        on_violation=fired.append,
    )
    # Count == threshold must NOT fire (the rule is strict "above").
    for i in range(20):
        ts = 100.0 + i * 0.1
        assert rule.observe(_snap("cam0", "z1", 5, ts=ts)) is None
    # Count below threshold — also should not fire.
    for i in range(20):
        ts = 100.0 + i * 0.1
        assert rule.observe(_snap("cam0", "z1", 3, ts=ts)) is None
    assert fired == []


def test_density_rule_resets_dwell_when_count_drops():
    """If count drops below threshold then rises again, the second run
    must re-accumulate from zero rather than fire instantly."""
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=5,
        dwell_seconds=1.0,
        cooldown_seconds=10.0,
        on_violation=fired.append,
    )
    # Run 1: go above for 0.5s, then drop to 3. No fire expected.
    rule.observe(_snap("cam0", "z1", 6, ts=100.0))
    rule.observe(_snap("cam0", "z1", 6, ts=100.5))
    rule.observe(_snap("cam0", "z1", 3, ts=100.6))
    # Run 2: jump back to 6 — must NOT fire instantly because dwell
    # was reset.
    rule.observe(_snap("cam0", "z1", 6, ts=101.0))
    rule.observe(_snap("cam0", "z1", 6, ts=101.5))
    # Now we are 1.0s into Run 2's dwell (101.0 → 102.0) → fire.
    v = rule.observe(_snap("cam0", "z1", 6, ts=102.0))
    assert v is not None
    assert fired == [v]


def test_density_rule_cooldown_silences_subsequent_fires():
    """Once it fires for a zone, the rule must stay silent for at
    least ``cooldown_seconds`` even if the count stays above the
    threshold the whole time."""
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=5,
        dwell_seconds=1.0,
        cooldown_seconds=10.0,
        on_violation=fired.append,
    )
    # First fire at t=101.0.
    rule.observe(_snap("cam0", "z1", 6, ts=100.0))
    rule.observe(_snap("cam0", "z1", 6, ts=100.5))
    v1 = rule.observe(_snap("cam0", "z1", 6, ts=101.0))
    assert v1 is not None
    assert len(fired) == 1
    # Stay above threshold — but cooldown (10s) not yet elapsed.
    rule.observe(_snap("cam0", "z1", 7, ts=105.0))
    rule.observe(_snap("cam0", "z1", 8, ts=109.0))
    rule.observe(_snap("cam0", "z1", 9, ts=110.0))
    rule.observe(_snap("cam0", "z1", 9, ts=112.0))
    # Still no second fire because cooldown has not passed AND the dwell
    # was reset when the first alert fired.
    assert len(fired) == 1
    # Drop the count to 3 → below threshold → exit cooldown countdown?
    # Per design: cooldown only elapses once the count has *dropped*
    # below threshold. So after the drop the next above-threshold run
    # still has to wait the cooldown after the FIRST fire.
    rule.observe(_snap("cam0", "z1", 3, ts=120.0))
    # After the drop, a fresh above-threshold run must restart dwell.
    rule.observe(_snap("cam0", "z1", 7, ts=121.0))
    # t=122.5 is 21.5s after the FIRST fire (at t=101.0): cooldown ok
    # but dwell is only 1.5s which is >= 1.0s → fires.
    v2 = rule.observe(_snap("cam0", "z1", 7, ts=122.5))
    assert v2 is not None
    assert v2 is not v1
    assert len(fired) == 2


def test_density_rule_per_zone_isolation():
    """State for (cam0, z1) must not bleed into (cam0, z2)."""
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=3, dwell_seconds=0.5, cooldown_seconds=2.0,
        on_violation=fired.append,
    )
    # z2: above for 1s → fires.
    rule.observe(_snap("cam0", "z2", 5, ts=100.0))
    v = rule.observe(_snap("cam0", "z2", 5, ts=101.0))
    assert v is not None
    # z1: never goes above → must NOT fire.
    rule.observe(_snap("cam0", "z1", 2, ts=100.5))
    assert fired == [v]
    # z2 has just fired — z1 still has its own untouched state.
    rule.observe(_snap("cam0", "z1", 4, ts=102.0))
    # 0.3s dwell < 0.5s required → must NOT fire yet.
    rule.observe(_snap("cam0", "z1", 4, ts=102.3))
    assert len(fired) == 1
    # 0.6s dwell >= 0.5s required → fires.
    v_z1 = rule.observe(_snap("cam0", "z1", 4, ts=102.6))
    assert v_z1 is not None
    assert len(fired) == 2


def test_density_rule_zero_dwell_fires_immediately():
    """``dwell_seconds=0`` means fire on the very first above-threshold
    observation."""
    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=2, dwell_seconds=0.0, cooldown_seconds=10.0,
        on_violation=fired.append,
    )
    v = rule.observe(_snap("cam0", "z1", 3, ts=100.0))
    assert v is not None
    assert v.dwell_seconds == 0.0
    assert fired == [v]


def test_density_rule_invalid_args_raise():
    with pytest.raises(ValueError):
        DensityRule(threshold=0)
    with pytest.raises(ValueError):
        DensityRule(threshold=-1)
    with pytest.raises(ValueError):
        DensityRule(threshold=5, dwell_seconds=-0.1)
    with pytest.raises(ValueError):
        DensityRule(threshold=5, cooldown_seconds=-0.1)


def test_density_rule_reset_zone_clears_state():
    rule = DensityRule(threshold=5, dwell_seconds=1.0, cooldown_seconds=10.0)
    # accumulate dwell for z1.
    rule.observe(_snap("cam0", "z1", 6, ts=100.0))
    rule.observe(_snap("cam0", "z1", 6, ts=100.5))
    # Reset z1 only.
    rule.reset_zone("cam0", "z1")
    # Fresh above-threshold run with same synthetic timestamps — must
    # NOT fire instantly (dwell was reset).
    assert rule.observe(_snap("cam0", "z1", 6, ts=100.6)) is None
    assert rule.observe(_snap("cam0", "z1", 6, ts=101.1)) is None
    rule.reset()


# ---------------------------------------------------------------------------
# density_snapshots_from_zone_counts
# ---------------------------------------------------------------------------


def test_density_snapshots_from_zone_counts_converts_and_filters():
    zcs = [
        ZoneCount(zone_id="z1", count=4, track_ids=[1, 2, 3, 4]),
        ZoneCount(zone_id="z2", count=7, track_ids=list(range(7))),
    ]
    snaps = density_snapshots_from_zone_counts("cam0", zcs, timestamp=1234.5)
    assert len(snaps) == 2
    s1, s2 = sorted(snaps, key=lambda s: s.zone_id)
    assert s1.zone_id == "z1"
    assert s1.count == 4
    assert s1.stream_id == "cam0"
    assert s1.timestamp == 1234.5
    assert s2.zone_id == "z2"
    assert s2.count == 7
    assert s2.stream_id == "cam0"


def test_density_snapshots_from_zone_counts_skips_non_zonecount_entries():
    """Hardening: bad entries in the list must not break the conversion."""
    zcs = [
        ZoneCount(zone_id="z1", count=4, track_ids=[]),
        "garbage",  # type: ignore[list-item]
        None,        # type: ignore[list-item]
    ]
    snaps = density_snapshots_from_zone_counts("cam0", zcs, timestamp=1.0)
    assert len(snaps) == 1
    assert snaps[0].zone_id == "z1"


# ---------------------------------------------------------------------------
# HighDensityViolation + repository
# ---------------------------------------------------------------------------


def test_high_density_violation_to_row_has_required_keys():
    v = HighDensityViolation(
        timestamp=200.0,
        stream_id="cam0",
        zone_id="lobby",
        density_count=12,
        threshold=10,
        dwell_seconds=3.5,
        notes="peak 12",
    )
    row = v.to_row()
    assert row["timestamp"] == 200.0
    assert row["stream_id"] == "cam0"
    assert row["zone_id"] == "lobby"
    assert row["density_count"] == 12
    assert row["threshold"] == 10
    assert row["dwell_seconds"] == 3.5
    assert row["notes"] == "peak 12"


def test_in_memory_high_density_repository_round_trip():
    repo = InMemoryHighDensityRepository()
    v1 = HighDensityViolation(100.0, "cam0", "z1", 8, 5, 2.5)
    v2 = HighDensityViolation(110.0, "cam0", "z2", 12, 10, 4.0)
    rid1 = repo.record(v1)
    rid2 = repo.record(v2)
    assert (rid1, rid2) == (1, 2)

    rows = repo.list_recent()
    assert len(rows) == 2
    # Newest first.
    assert rows[0]["zone_id"] == "z2"
    assert rows[1]["zone_id"] == "z1"

    rows_z1 = repo.list_recent(zone_id="z1")
    assert len(rows_z1) == 1 and rows_z1[0]["zone_id"] == "z1"

    # attach_clip attaches to the right row.
    repo.attach_clip(rid2, "/tmp/clip.mp4", "/api/clips/2.mp4")
    rows = repo.list_recent(zone_id="z2")
    assert rows[0]["clip_path"] == "/tmp/clip.mp4"
    assert rows[0]["clip_url"] == "/api/clips/2.mp4"


# ---------------------------------------------------------------------------
# DensityRule fires → evidence recorder stores a clip
# ---------------------------------------------------------------------------


def _build_recorder(tmp_path) -> EvidenceRecorder:
    storage = NoopClipStorage()
    recorder = EvidenceRecorder(
        storage=storage,
        builder=ClipBuilder(fps=10.0),
        disk_guard=DiskSpaceGuard(base_dir=str(tmp_path), free_threshold_mb=1.0),
        pre_seconds=1.0,
        post_seconds=1.0,
        buffer_mode_hint="raw",
    )
    return recorder


def test_high_density_alert_triggers_evidence_clip(tmp_path):
    """When a high-density alert fires, the recorder must store a clip
    keyed by the alert id and the repo must reflect the clip path."""
    repo = InMemoryHighDensityRepository()
    recorder = _build_recorder(tmp_path)

    fps = 10.0
    frames = [
        np.full((16, 16, 3), (i % 255), dtype=np.uint8)
        for i in range(60)
    ]
    post = SyntheticStreamPost(
        stream_id="cam0", frames=frames, fps=fps, start_ts=1000.0,
    )
    post._cursor = 30  # event "happens" near frame 30
    recorder.register_stream(post)

    rule = DensityRule(
        threshold=5,
        dwell_seconds=1.0,
        cooldown_seconds=10.0,
    )
    # Feed the rule with snapshots until it fires.
    fired = None
    for i, ts in enumerate([100.0, 100.5, 101.2]):
        v = rule.observe(_snap("cam0", "z1", 7, ts=ts))
        if v is not None:
            fired = v
            break
    assert fired is not None, "rule should have fired"

    # Now run the same flow the Pipeline does.
    vid = repo.record(fired)
    clip = recorder.record(fired, violation_id=vid)
    assert clip is not None
    repo.attach_clip(vid, clip.path, clip.url)

    rows = repo.list_recent()
    assert len(rows) == 1
    assert rows[0]["id"] == vid
    assert rows[0]["density_count"] == 7
    assert rows[0]["threshold"] == 5
    assert rows[0]["clip_path"] == clip.path
    assert rows[0]["clip_url"] == clip.url


def test_high_density_alerts_path_does_not_affect_violation_kind():
    """The high-density alert is distinct from a tripwire violation: it
    has no ``kind`` and no tripwire_id, while a regular Violation does."""
    hdv = HighDensityViolation(
        timestamp=101.2, stream_id="cam0", zone_id="z1",
        density_count=7, threshold=5, dwell_seconds=1.2,
    )
    assert not hasattr(hdv, "kind")
    assert not hasattr(hdv, "tripwire_id")
    # Sanity: a Violation still has those, to make sure the two types
    # remain clearly distinct downstream.
    v = Violation(
        timestamp=101.2, stream_id="cam0", zone_id="z1",
        tripwire_id="t1", track_id=1, direction="left_to_right",
        embedding=None, kind=ViolationKind.UNMATCHED,
    )
    assert v.kind is ViolationKind.UNMATCHED


# ---------------------------------------------------------------------------
# End-to-end: Pipeline._make_density_sink records + captures
# ---------------------------------------------------------------------------


def test_pipeline_density_sink_persists_and_captures(tmp_path):
    """Build a sink (mirroring Pipeline._make_density_sink) and exercise
    it against a violation + recorder + repo — to cover the cross-package
    path that Pipeline uses."""
    from mctracker.pipeline import Pipeline

    repo = InMemoryHighDensityRepository()
    recorder = _build_recorder(tmp_path)
    fps = 10.0
    frames = [
        np.full((16, 16, 3), (i % 255), dtype=np.uint8)
        for i in range(60)
    ]
    post = SyntheticStreamPost(
        stream_id="cam0", frames=frames, fps=fps, start_ts=1000.0,
    )
    recorder.register_stream(post)

    # Construct just enough of a Pipeline to use ``_make_density_sink``.
    p = Pipeline.__new__(Pipeline)
    p._evidence_recorder = recorder
    sink = Pipeline._make_density_sink(p, repo)

    alert = HighDensityViolation(
        timestamp=101.0, stream_id="cam0", zone_id="lobby",
        density_count=8, threshold=5, dwell_seconds=2.5,
    )
    vid = sink(alert)
    assert vid == 1
    rows = repo.list_recent()
    assert len(rows) == 1
    # Clip attached.
    assert rows[0]["clip_path"] is not None
    assert rows[0]["clip_url"] is not None


def test_pipeline_density_sink_handles_no_recorder():
    """If no recorder is configured, the sink must still persist the
    alert (just without a clip)."""
    from mctracker.pipeline import Pipeline

    repo = InMemoryHighDensityRepository()
    p = Pipeline.__new__(Pipeline)
    p._evidence_recorder = None
    sink = Pipeline._make_density_sink(p, repo)

    alert = HighDensityViolation(
        timestamp=101.0, stream_id="cam0", zone_id="lobby",
        density_count=8, threshold=5, dwell_seconds=2.5,
    )
    vid = sink(alert)
    assert vid == 1
    rows = repo.list_recent()
    assert len(rows) == 1
    assert rows[0]["clip_path"] is None


def test_pipeline_density_sink_recorder_failure_does_not_break_repo():
    """If the recorder raises, the repo row must still be present (just
    without the clip fields)."""
    from mctracker.pipeline import Pipeline

    repo = InMemoryHighDensityRepository()

    class BoomRecorder:
        def record(self, *_a, **_kw):
            raise RuntimeError("disk full")

    p = Pipeline.__new__(Pipeline)
    p._evidence_recorder = BoomRecorder()
    sink = Pipeline._make_density_sink(p, repo)

    alert = HighDensityViolation(
        timestamp=101.0, stream_id="cam0", zone_id="lobby",
        density_count=8, threshold=5, dwell_seconds=2.5,
    )
    vid = sink(alert)
    assert vid == 1
    rows = repo.list_recent()
    assert rows[0]["clip_path"] is None


# ---------------------------------------------------------------------------
# Stream._evaluate_density — direct call (no processor thread required)
# ---------------------------------------------------------------------------


def test_stream_evaluate_density_fires_via_real_stream(tmp_path):
    """Wires a real ``Stream`` with a scripted detector/tracker that
    produce enough tracks to push a zone over its threshold, then
    directly invokes ``_evaluate_density`` and confirms a sink was
    called."""
    from mctracker.stream import Stream
    from mctracker.zones import Zone, ZoneManager

    # Build a zone covering the test frame coordinates.
    zone = Zone(id="lobby", polygon=[[0, 0], [400, 0], [400, 400], [0, 400]])
    zm = ZoneManager(zones=[zone], centroid_mode="geometric_center")

    repo = InMemoryHighDensityRepository()

    fired: List[HighDensityViolation] = []
    rule = DensityRule(
        threshold=3, dwell_seconds=0.5, cooldown_seconds=10.0,
    )

    def sink(v):
        fired.append(v)
        repo.record(v)

    s = Stream.__new__(Stream)
    s.id = "cam0"
    s.source = "fake://cam0"
    s._detector = None
    s._tracker = None
    s._on_results = lambda *a, **k: None
    s._display_conf = 0.25
    s._density_rule = rule
    s._density_sink = sink
    s._buffer = None  # type: ignore[assignment]
    s._reader = None  # type: ignore[assignment]
    s._zone_manager = zm
    s._tripwire_manager = None
    s._active = {}
    s._lock = threading.Lock()
    s._last_centroid = {}
    s._stop = threading.Event()
    s._thread = None
    s._fps = 30.0
    s._buffer_seconds = 5

    # 4 zone counts, all with count=5 → above threshold.
    zone_counts = [
        ZoneCount(zone_id="lobby", count=5, track_ids=[1, 2, 3, 4, 5]),
    ]
    s._evaluate_density(zone_counts, frame_ts=100.0)
    assert fired == []  # dwell not yet reached

    s._evaluate_density(zone_counts, frame_ts=100.6)
    # Now dwell ≥ 0.5s → fires.
    assert len(fired) == 1
    assert fired[0].stream_id == "cam0"
    assert fired[0].zone_id == "lobby"
    assert fired[0].density_count == 5
    assert fired[0].threshold == 3

    rows = repo.list_recent()
    assert len(rows) == 1
    assert rows[0]["zone_id"] == "lobby"
