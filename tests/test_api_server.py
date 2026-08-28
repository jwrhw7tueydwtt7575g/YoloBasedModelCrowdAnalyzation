"""Unit tests for FastAPI REST API endpoints and stream routes."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from mctracker.api.server import app

client = TestClient(app)

SAMPLE_VIDEO_PATH = "/home/vivek/Downloads/Testing Videos/kling_20260828_VIDEO_A_realisti_3962_0.mp4"


def test_dashboard_ui_returns_200():
    response = client.get("/")
    assert response.status_code == 200
    assert "Multi-Camera Tracking & Analytics Dashboard" in response.text


def test_list_streams_empty_by_default():
    response = client.get("/api/v1/streams")
    assert response.status_code == 200
    data = response.json()
    assert "streams" in data
    assert isinstance(data["streams"], list)


def test_process_video_endpoint_rejects_empty():
    response = client.post("/api/v1/process-video")
    assert response.status_code == 400


def test_process_video_with_file():
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip(f"Sample video file not found at {SAMPLE_VIDEO_PATH}")

    with open(SAMPLE_VIDEO_PATH, "rb") as vf:
        response = client.post(
            "/api/v1/process-video",
            files={"file": ("test.mp4", vf, "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "stream_id" in data
    stream_id = data["stream_id"]

    # Test report endpoint
    report_resp = client.get(f"/api/v1/streams/{stream_id}/report")
    assert report_resp.status_code == 200
    report_data = report_resp.json()
    assert report_data["stream_id"] == stream_id
    assert "metrics" in report_data
    assert "unique_persons_tracked" in report_data["metrics"]
