"""Config loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mctracker.config import ConfigError, StreamConfig, load_config


def write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_minimal_config(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: cam1
    source: rtsp://example/stream
  - id: cam2
    source: /tmp/v.mp4
""",
    )
    cfg = load_config(p)
    assert len(cfg.streams) == 2
    assert cfg.streams[0].id == "cam1"
    assert cfg.streams[0].tracker_type == "bytetrack"
    assert cfg.streams[0].buffer_seconds == 5
    assert cfg.streams[0].display_conf == 0.25


def test_load_full_config(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: lobby
    source: rtsp://10.0.0.1/stream1
    model_size: yolov8s.pt
    tracker_type: botsort
    use_appearance: true
    buffer_seconds: 10
    fps_fallback: 25
    display_conf: 0.4
""",
    )
    cfg = load_config(p)
    s = cfg.streams[0]
    assert s.model_size == "yolov8s.pt"
    assert s.tracker_type == "botsort"
    assert s.use_appearance is True
    assert s.buffer_seconds == 10
    assert s.fps_fallback == 25
    assert s.display_conf == 0.4


def test_use_appearance_requires_botsort(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: cam1
    source: foo
    tracker_type: bytetrack
    use_appearance: true
""",
    )
    with pytest.raises(ConfigError, match="use_appearance"):
        load_config(p)


def test_unknown_tracker_type_rejected(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: cam1
    source: foo
    tracker_type: deepsort
""",
    )
    with pytest.raises(ConfigError, match="tracker_type"):
        load_config(p)


def test_duplicate_stream_ids_rejected(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: dup
    source: a
  - id: dup
    source: b
""",
    )
    with pytest.raises(ConfigError, match="duplicate stream id"):
        load_config(p)


def test_empty_config_rejected(tmp_path):
    p = write_yaml(tmp_path, "streams: []\n")
    with pytest.raises(ConfigError, match="at least one stream"):
        load_config(p)


def test_missing_source_rejected(tmp_path):
    p = write_yaml(
        tmp_path,
        """
streams:
  - id: cam1
""",
    )
    with pytest.raises(ConfigError, match="source is required"):
        load_config(p)
