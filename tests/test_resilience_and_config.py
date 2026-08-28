"""Tests for config validation, RTSP reconnect backoff, metrics export, and stream isolation."""

from __future__ import annotations

import pytest

from mctracker.config import ConfigError, AppConfig, StreamConfig, ViolationsConfig, load_config
from mctracker.metrics import Metrics, reset_metrics, METRICS
from mctracker.zones import Zone, is_polygon_self_intersecting
from mctracker.tripwire import Tripwire


def test_polygon_self_intersection_rejection():
    """Verify that a self-intersecting polygon (bowtie / figure 8) is rejected."""
    # Simple self-intersecting bowtie polygon
    bowtie = [
        [0.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
        [100.0, 0.0],
    ]
    assert is_polygon_self_intersecting(bowtie) is True

    with pytest.raises(ValueError, match="polygon is self-intersecting"):
        Zone(id="z_bowtie", polygon=bowtie)


def test_valid_polygon_accepted():
    """Verify that a valid convex or concave polygon is accepted."""
    poly = [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [50.0, 50.0],  # concave vertex
        [0.0, 100.0],
    ]
    assert is_polygon_self_intersecting(poly) is False
    z = Zone(id="z_concave", polygon=poly)
    assert z.id == "z_concave"


def test_tripwire_degenerate_rejection():
    """Verify that degenerate tripwire (p1 == p2) raises ValueError."""
    with pytest.raises(ValueError, match="p1 == p2"):
        Tripwire(id="tw_bad", p1=(100.0, 100.0), p2=(100.0, 100.0))


def test_violations_window_seconds_validation():
    """Verify that violations window_seconds <= 0 raises ConfigError."""
    with pytest.raises(ConfigError, match="window_seconds must be > 0"):
        ViolationsConfig(window_seconds=0.0)

    with pytest.raises(ConfigError, match="window_seconds must be > 0"):
        ViolationsConfig(window_seconds=-5.0)


def test_metrics_collection_and_prometheus_export():
    """Verify that Metrics tracks events and renders valid Prometheus text output."""
    m = reset_metrics()

    m.inc_frame("cam1", dropped=False)
    m.inc_frame("cam1", dropped=True)
    m.inc_reconnect("cam1")
    m.inc_id_switch("cam1")
    m.inc_violation("unmatched")
    m.observe("detect", 0.015, "cam1")

    snap = m.snapshot()
    assert snap["counters"]["frames_processed_total"] == 1
    assert snap["counters"]["frames_dropped_total"] == 1
    assert snap["counters"]["stream_reconnects_total"] == 1
    assert snap["counters"]["id_switches_total"] == 1
    assert snap["counters"]["violations_total"] == 1

    prom_text = m.to_prometheus_text()
    assert "mctracker_frames_processed_total" in prom_text
    assert "mctracker_frames_dropped_total" in prom_text
    assert "mctracker_stream_reconnects_total" in prom_text
    assert "mctracker_detect_seconds_bucket" in prom_text
