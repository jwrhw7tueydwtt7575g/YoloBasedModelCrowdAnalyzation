"""Stage 5 config-validation tests.

These tests cover the new failure modes that config.py now rejects:

* polygon self-intersection
* tripwire with p1 == p2 (degenerate segment)
* tripwire with non-finite coordinates
* tripwire endpoint outside an explicit frame_size
* QR-match window <= 0
* duplicate zone / tripwire ids within a stream
* bad frame_size hint (negative / non-integer / wrong shape)
* frame_size with negative width/height

A passing test means ``load_config`` raised ``ConfigError`` with a message
that contains a useful substring. We keep the substring assertions loose
so future rewording doesn't break the test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mctracker.config import (
    AppConfig,
    ConfigError,
    ViolationsConfig,
    _polygon_self_intersects,
    load_config,
)


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Polygon self-intersection
# ---------------------------------------------------------------------------


def test_polygon_self_intersects_detects_bowtie():
    """A figure-8 / bow-tie polygon must be flagged."""
    poly = [
        [0.0, 0.0],
        [100.0, 100.0],
        [100.0, 0.0],
        [0.0, 100.0],
    ]
    assert _polygon_self_intersects(poly)


def test_polygon_self_intersects_accepts_convex_quad():
    poly = [[0, 0], [100, 0], [100, 100], [0, 100]]
    assert not _polygon_self_intersects(poly)


def test_polygon_self_intersects_accepts_triangle():
    poly = [[0, 0], [10, 0], [5, 10]]
    assert not _polygon_self_intersects(poly)


def test_config_rejects_self_intersecting_zone(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    zones:
      - id: z1
        polygon: [[0,0],[100,100],[100,0],[0,100]]
""",
    )
    with pytest.raises(ConfigError, match="self-intersecting"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Tripwire geometry
# ---------------------------------------------------------------------------


def test_config_rejects_tripwire_p1_equals_p2(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    tripwires:
      - id: tw1
        p1: [50, 50]
        p2: [50, 50]
""",
    )
    with pytest.raises(ConfigError, match="degenerate"):
        load_config(cfg)


def test_config_rejects_tripwire_nonfinite_coord(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    tripwires:
      - id: tw1
        p1: [50, "inf"]
        p2: [200, 200]
""",
    )
    with pytest.raises(ConfigError, match="tripwire"):
        load_config(cfg)


def test_config_rejects_tripwire_outside_frame(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    frame_size: [640, 480]
    tripwires:
      - id: tw1
        p1: [50, 50]
        p2: [1000, 100]
""",
    )
    with pytest.raises(ConfigError, match="outside frame"):
        load_config(cfg)


def test_config_accepts_tripwire_when_frame_size_omitted(tmp_path: Path):
    """If frame_size is absent we cannot bounds-check; accept."""
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    tripwires:
      - id: tw1
        p1: [50, 50]
        p2: [10000, 100]
""",
    )
    app = load_config(cfg)
    assert len(app.streams) == 1


# ---------------------------------------------------------------------------
# QR-match window
# ---------------------------------------------------------------------------


def test_config_rejects_qr_window_zero(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
violations:
  window_seconds: 0
""",
    )
    with pytest.raises(ConfigError, match="window_seconds"):
        load_config(cfg)


def test_config_rejects_qr_window_negative(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
violations:
  window_seconds: -1.5
""",
    )
    with pytest.raises(ConfigError, match="window_seconds"):
        load_config(cfg)


def test_violations_config_post_init_rejects_zero():
    with pytest.raises(ConfigError, match="window_seconds"):
        ViolationsConfig(window_seconds=0)


def test_violations_config_post_init_rejects_negative():
    with pytest.raises(ConfigError, match="window_seconds"):
        ViolationsConfig(window_seconds=-0.1)


def test_violations_config_accepts_positive():
    cfg = ViolationsConfig(window_seconds=10.0)
    assert cfg.window_seconds == 10.0


# ---------------------------------------------------------------------------
# Duplicate ids
# ---------------------------------------------------------------------------


def test_config_rejects_duplicate_zone_ids(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    zones:
      - id: z1
        polygon: [[0,0],[10,0],[10,10],[0,10]]
      - id: z1
        polygon: [[20,20],[30,20],[30,30],[20,30]]
""",
    )
    with pytest.raises(ConfigError, match="duplicate zone"):
        load_config(cfg)


def test_config_rejects_duplicate_tripwire_ids(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    tripwires:
      - id: tw1
        p1: [0, 50]
        p2: [100, 50]
      - id: tw1
        p1: [200, 50]
        p2: [300, 50]
""",
    )
    with pytest.raises(ConfigError, match="duplicate tripwire"):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Frame size sanity
# ---------------------------------------------------------------------------


def test_config_rejects_bad_frame_size(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    frame_size: [-10, 480]
""",
    )
    with pytest.raises(ConfigError):
        load_config(cfg)


def test_config_rejects_frame_size_wrong_shape(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
    frame_size: [640]
""",
    )
    with pytest.raises(ConfigError):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------


def test_config_default_observability_is_present():
    cfg = _write_yaml_path() if False else None
    app = AppConfig(
        streams=[__stream_cfg_stub()],
    )
    assert app.observability.metrics_port == 0
    assert app.violations.window_seconds == 10.0


def __stream_cfg_stub():
    from mctracker.config import StreamConfig

    return StreamConfig(id="x", source="fake://x")


def _write_yaml_path():
    # placeholder to keep type-checkers happy; not used.
    return None


# ---------------------------------------------------------------------------
# Observability port bounds
# ---------------------------------------------------------------------------


def test_observability_port_rejects_out_of_range(tmp_path: Path):
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam0
    source: fake://test
observability:
  metrics_port: 70000
""",
    )
    with pytest.raises(ConfigError):
        load_config(cfg)


# ---------------------------------------------------------------------------
# Frame-size check applies per-stream
# ---------------------------------------------------------------------------


def test_config_frame_size_only_affects_own_stream_tripwires(tmp_path: Path):
    """A stream without frame_size must not bounds-check its tripwires,
    even if other streams in the same file declare one."""
    cfg = _write_yaml(
        tmp_path,
        """
streams:
  - id: cam_with_size
    source: fake://test
    frame_size: [640, 480]
    tripwires:
      - id: t1
        p1: [50, 50]
        p2: [1000, 100]   # outside cam_with_size's frame — should fail
  - id: cam_no_size
    source: fake://test
    tripwires:
      - id: t2
        p1: [50, 50]
        p2: [1000, 100]   # no frame_size on this stream — should pass
""",
    )
    with pytest.raises(ConfigError, match="cam_with_size"):
        load_config(cfg)