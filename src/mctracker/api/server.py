"""FastAPI REST Web Service & MJPEG Streamer for mctracker."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

from mctracker.annotator import FrameAnnotator
from mctracker.config import StreamConfig, AppConfig
from mctracker.detector import Detector
from mctracker.metrics import METRICS, reset_metrics
from mctracker.stream import Stream
from mctracker.tracker import Tracker, make_tracker
from mctracker.tripwire import Tripwire, TripwireManager, CrossingEvent
from mctracker.types import Detection, Frame
from mctracker.zones import Zone, ZoneManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("mctracker.api")

app = FastAPI(
    title="Multi-Camera Video Tracking & Analytics API",
    description="Production API for real-time video object tracking, tripwire crossing analytics, and live MJPEG streaming.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registry storing active stream sessions
ACTIVE_STREAMS: Dict[str, Dict] = {}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "mctracker_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class OpenCVContourDetector(Detector):
    """Fallback background subtraction motion detector when ultralytics is not available."""
    def __init__(self):
        self._fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def detect(self, frame: Frame) -> List[Detection]:
        fgmask = self._fgbg.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.002
        for c in contours:
            area = cv2.contourArea(c)
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(c)
                dets.append(
                    Detection(
                        xyxy=np.array([float(x), float(y), float(x + bw), float(y + bh)], dtype=np.float32),
                        conf=0.85,
                        cls=0,
                        det_id=None,
                    )
                )
        return dets


def _create_detector(model_size: str = "n") -> Detector:
    try:
        from mctracker.detector import YOLODetector
        det = YOLODetector(model_size=model_size)
        log.info("Loaded YOLODetector with Ultralytics model weights.")
        return det
    except Exception as e:
        log.warning(f"YOLODetector unavailable ({e}); using OpenCV Contour Motion Detector.")
        return OpenCVContourDetector()


def _create_tracker(tracker_type: str = "botsort", fps: int = 24) -> Tracker:
    try:
        from mctracker.tracker import make_tracker
        return make_tracker(tracker_type, frame_rate=fps, with_reid=False)
    except Exception as e:
        log.warning(f"make_tracker unavailable ({e}); using FakeTracker fallback.")
        from mctracker.tracker import FakeTracker
        return FakeTracker(inner=None)


class ProcessVideoRequest(BaseModel):
    source_url: Optional[str] = None
    tripwire_y_percent: float = 0.5
    zone_margin_percent: float = 0.1


@app.post("/api/v1/process-video")
async def process_video(
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    source_url: Optional[str] = Form(None),
):
    """Upload a video file or submit an RTSP/HTTP stream URL for live tracking."""
    if not file and not source_url:
        raise HTTPException(status_code=400, detail="Must provide either a video file upload or a source_url.")

    stream_id = f"stream_{int(time.time() * 1000)}"

    if file:
        file_path = UPLOAD_DIR / f"{stream_id}_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        video_source = str(file_path)
    else:
        video_source = source_url

    # Inspect video resolution
    cap = cv2.VideoCapture(video_source)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()

    # Construct Tripwire and Zone
    tripwire_y = height * 0.5
    tw = Tripwire(
        id="tw_main",
        p1=(50.0, float(tripwire_y)),
        p2=(float(width - 50), float(tripwire_y)),
        direction_in="left_to_right",
    )
    zn = Zone(
        id="zone_center",
        polygon=[
            [width * 0.1, height * 0.1],
            [width * 0.9, height * 0.1],
            [width * 0.9, height * 0.9],
            [width * 0.1, height * 0.9],
        ],
    )

    annotator = FrameAnnotator()
    detector = _create_detector("n")
    tracker = _create_tracker("botsort", fps=int(fps))
    zone_mgr = ZoneManager(zones=[zn])
    tripwire_mgr = TripwireManager(stream_id=stream_id, tripwires=[tw])

    session = {
        "stream_id": stream_id,
        "source": video_source,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": 0,
        "start_time": time.time(),
        "status": "processing",
        "annotated_frame": None,
        "tracks_seen": set(),
        "crossings": [],
        "zones": [zn],
        "tripwires": [tw],
        "annotator": annotator,
        "annotated_video_path": str(UPLOAD_DIR / f"{stream_id}_out.mp4"),
    }

    # Video Writer for downloading processed MP4
    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(session["annotated_video_path"], fourcc, fps, (width, height))
        if not writer.isOpened():
            writer = None
    except Exception as e:
        log.warning(f"VideoWriter unavailable ({e}); disabling MP4 recording.")
    session["writer"] = writer

    def on_results(sid: str, tracks, zones, crossings):
        session["processed_frames"] += 1
        for t in tracks:
            session["tracks_seen"].add(t.track_id)
        if crossings:
            session["crossings"].extend(crossings)

        # Stage 6: high-density / crowd alerts. The API server doesn't load
        # the full Pipeline config, so we apply a heuristic threshold here
        # purely for UI demo / API exposure. The real Pipeline-level
        # DensityRule (with dwell + cooldown) is what fires the evidence
        # recorder in production. We only emit here for *visual* feedback.
        try:
            zlist = list(zones.values()) if isinstance(zones, dict) else list(zones)
        except Exception:
            zlist = []
        for zc in zlist:
            threshold = session.get("high_density_threshold", 8)
            if zc.count > threshold:
                last_fired = session.setdefault("_hd_last_fired", {}).get(zc.zone_id, 0.0)
                now = time.time()
                # simple 10s cooldown so the dashboard doesn't redraw an alert every frame
                if now - last_fired >= 10.0:
                    session["_hd_last_fired"][zc.zone_id] = now
                    session.setdefault("high_density_alerts", []).append({
                        "timestamp": round(now, 2),
                        "stream_id": sid,
                        "zone_id": zc.zone_id,
                        "density_count": zc.count,
                        "threshold": threshold,
                        "clip_path": None,
                        "clip_url": None,
                    })

        # Retrieve current frame buffer for rendering
        if stream._buffer and len(stream._buffer) > 0:
            latest = stream._buffer.peek_latest()
            if latest is not None:
                raw_frame, ts = latest
                ann_frame = annotator.annotate(
                    raw_frame,
                    tracks=tracks,
                    zones=session.get("zones", []),
                    tripwires=session.get("tripwires", []),
                    recent_crossings=crossings,
                )
                session["annotated_frame"] = ann_frame
                if writer is not None and writer.isOpened():
                    try:
                        writer.write(ann_frame)
                    except Exception:
                        pass

    stream = Stream(
        stream_id=stream_id,
        source=video_source,
        detector=detector,
        tracker=tracker,
        on_results=on_results,
        buffer_seconds=5,
        fps_fallback=int(fps),
        display_conf=0.20,
        zone_manager=zone_mgr,
        tripwire_manager=tripwire_mgr,
    )
    session["stream"] = stream
    ACTIVE_STREAMS[stream_id] = session

    stream.start()
    return {
        "status": "success",
        "stream_id": stream_id,
        "video_info": {"resolution": f"{width}x{height}", "fps": fps, "total_frames": total_frames},
        "links": {
            "feed": f"/api/v1/streams/{stream_id}/feed",
            "report": f"/api/v1/streams/{stream_id}/report",
            "download": f"/api/v1/streams/{stream_id}/download",
        },
    }


@app.get("/api/v1/streams")
async def list_streams():
    """List all registered processing sessions."""
    out = []
    for sid, s in ACTIVE_STREAMS.items():
        out.append({
            "stream_id": sid,
            "source": s["source"],
            "processed_frames": s["processed_frames"],
            "status": s["status"],
            "unique_persons_tracked": len(s["tracks_seen"]),
            "crossings_count": len(s["crossings"]),
        })
    return {"streams": out}


@app.get("/api/v1/streams/{stream_id}/feed")
async def stream_mjpeg_feed(stream_id: str):
    """Returns a live annotated MJPEG video stream."""
    if stream_id not in ACTIVE_STREAMS:
        raise HTTPException(status_code=404, detail="Stream session not found.")

    session = ACTIVE_STREAMS[stream_id]

    async def generate_mjpeg():
        while True:
            frame = session.get("annotated_frame")
            if frame is not None:
                ret, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )
            await asyncio.sleep(0.04)

    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/v1/streams/{stream_id}/report")
async def get_stream_report(stream_id: str):
    """Generate and return a structured JSON analytics report."""
    if stream_id not in ACTIVE_STREAMS:
        raise HTTPException(status_code=404, detail="Stream session not found.")

    s = ACTIVE_STREAMS[stream_id]
    elapsed = time.time() - s["start_time"]
    fps_throughput = s["processed_frames"] / float(elapsed + 1e-6)

    crossings_data = []
    for c in s["crossings"]:
        crossings_data.append({
            "timestamp": round(c.timestamp, 2),
            "track_id": c.track_id,
            "tripwire_id": c.tripwire_id,
            "direction": c.direction,
            "centroid": [round(c.centroid[0], 1), round(c.centroid[1], 1)],
        })

    snap = METRICS.snapshot()
    stage_latencies = {}
    if "histograms" in snap and "stage_seconds" in snap["histograms"]:
        for (st_id, stage), stats in snap["histograms"]["stage_seconds"].items():
            if st_id == stream_id:
                stage_latencies[stage] = {
                    "count": stats["count"],
                    "avg_ms": round((stats["sum"] / (stats["count"] + 1e-6)) * 1000.0, 2),
                }

    return {
        "report_id": f"rep_{stream_id}",
        "stream_id": stream_id,
        "source": s["source"],
        "metrics": {
            "total_frames_processed": s["processed_frames"],
            "total_video_frames": s["total_frames"],
            "processing_duration_seconds": round(elapsed, 2),
            "pipeline_fps": round(fps_throughput, 2),
            "unique_persons_tracked": len(s["tracks_seen"]),
            "person_track_ids": sorted(list(s["tracks_seen"])),
            "total_tripwire_crossings": len(s["crossings"]),
            "total_high_density_alerts": len(s.get("high_density_alerts", [])),
        },
        "tripwire_crossings": crossings_data,
        "high_density_alerts": list(s.get("high_density_alerts", [])),
        "stage_latencies": stage_latencies,
    }


@app.get("/api/v1/streams/{stream_id}/download")
async def download_annotated_video(stream_id: str):
    """Download recorded annotated MP4 video."""
    if stream_id not in ACTIVE_STREAMS:
        raise HTTPException(status_code=404, detail="Stream session not found.")

    path = ACTIVE_STREAMS[stream_id]["annotated_video_path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Annotated video file not generated yet.")

    return FileResponse(path, media_type="video/mp4", filename=f"{stream_id}_annotated.mp4")


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Interactive Web Dashboard for uploading video, viewing live MJPEG detection feed, and fetching reports."""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Multi-Camera Tracking & Analytics Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0f172a;
                --card-bg: #1e293b;
                --accent: #38bdf8;
                --text: #f8fafc;
                --text-muted: #94a3b8;
                --border: #334155;
            }
            body {
                font-family: 'Inter', sans-serif;
                background-color: var(--bg);
                color: var(--text);
                margin: 0;
                padding: 24px;
            }
            .header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 24px;
                border-bottom: 1px solid var(--border);
                padding-bottom: 16px;
            }
            .header h1 { margin: 0; font-size: 24px; color: var(--accent); }
            .grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
            }
            .card {
                background-color: var(--card-bg);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 20px;
            }
            .card h3 { margin-top: 0; font-size: 18px; color: var(--text); border-bottom: 1px solid var(--border); padding-bottom: 8px;}
            input[type="file"], input[type="text"] {
                width: 100%;
                padding: 10px;
                background: #0f172a;
                border: 1px solid var(--border);
                color: white;
                border-radius: 6px;
                margin-bottom: 12px;
                box-sizing: border-box;
            }
            button {
                background-color: var(--accent);
                color: #0f172a;
                font-weight: 600;
                padding: 10px 18px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                width: 100%;
            }
            button:hover { opacity: 0.9; }
            .video-container {
                background: #000;
                border-radius: 8px;
                overflow: hidden;
                min-height: 360px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .video-container img {
                max-width: 100%;
                height: auto;
                display: block;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 12px;
                margin-top: 16px;
            }
            .stat-box {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                text-align: center;
            }
            .stat-value { font-size: 22px; font-weight: 700; color: var(--accent); }
            .stat-value.warn { color: #fbbf24; }
            .stat-label { font-size: 12px; color: var(--text-muted); }
            .alert-list {
                margin-top: 12px;
                max-height: 220px;
                overflow-y: auto;
            }
            .alert-item {
                background: #0f172a;
                border-left: 4px solid #fbbf24;
                border-radius: 6px;
                padding: 10px 12px;
                margin-bottom: 8px;
                font-size: 13px;
            }
            .alert-item .alert-zone {
                color: #fbbf24;
                font-weight: 600;
            }
            .alert-item .alert-count {
                color: var(--accent);
                font-weight: 700;
            }
            .alert-item a {
                color: #38bdf8;
                text-decoration: none;
                margin-left: 8px;
            }
            .alert-item .alert-meta {
                color: var(--text-muted);
                font-size: 11px;
                margin-top: 4px;
            }
            pre {
                background: #0f172a;
                padding: 12px;
                border-radius: 8px;
                max-height: 240px;
                overflow-y: auto;
                font-size: 12px;
                color: #38bdf8;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>📹 Multi-Camera Tracking & Analytics Dashboard</h1>
            <span>FastAPI Live Inference Engine</span>
        </div>

        <div class="grid">
            <div class="card">
                <h3>1. Upload Video or Submit Stream</h3>
                <form id="uploadForm">
                    <label>Select Local Video File (.mp4):</label>
                    <input type="file" id="videoFile" accept="video/mp4">
                    <div style="text-align: center; margin: 8px; color: var(--text-muted);">OR</div>
                    <label>Enter RTSP / HTTP Video Stream URL:</label>
                    <input type="text" id="streamUrl" placeholder="rtsp://192.168.1.100:554/live">
                    <button type="submit">🚀 Start Stream Processing</button>
                </form>

                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-value" id="fpsVal">0.0</div>
                        <div class="stat-label">Pipeline FPS</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="tracksVal">0</div>
                        <div class="stat-label">Unique Persons</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="crossingsVal">0</div>
                        <div class="stat-label">Tripwire Crossings</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value warn" id="densityVal">0</div>
                        <div class="stat-label">⚠ High-Density Alerts</div>
                    </div>
                </div>

                <h3 style="margin-top: 24px;">2. Analytics Report</h3>
                <button onclick="fetchReport()" style="background: #10b981; color: white;">📊 Fetch JSON Analytics Report</button>
                <pre id="reportBox">// Click button to load analytics report</pre>

                <h3 style="margin-top: 24px;">⚠ Recent High-Density Alerts</h3>
                <div class="alert-list" id="alertList">
                    <p style="color: var(--text-muted); font-size: 13px; margin: 0;">No alerts yet.</p>
                </div>
            </div>

            <div class="card">
                <h3>Live Visual Stream (Annotated Detections & Tripwires)</h3>
                <div class="video-container" id="videoWrapper">
                    <p style="color: var(--text-muted);">No active stream rendering</p>
                </div>
            </div>
        </div>

        <script>
            let activeStreamId = null;

            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const fileInput = document.getElementById('videoFile');
                const urlInput = document.getElementById('streamUrl');

                const formData = new FormData();
                if (fileInput.files.length > 0) {
                    formData.append('file', fileInput.files[0]);
                } else if (urlInput.value) {
                    formData.append('source_url', urlInput.value);
                } else {
                    alert('Please select a file or enter a stream URL');
                    return;
                }

                try {
                    const res = await fetch('/api/v1/process-video', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (data.status === 'success') {
                        activeStreamId = data.stream_id;
                        document.getElementById('videoWrapper').innerHTML = `<img src="/api/v1/streams/${activeStreamId}/feed" alt="Live Feed">`;
                        startPollingStats();
                    } else {
                        alert('Error starting stream');
                    }
                } catch (err) {
                    alert('Failed to connect to API server');
                }
            });

            async function fetchReport() {
                if (!activeStreamId) {
                    alert('No active stream session. Please process a video first.');
                    return;
                }
                const res = await fetch(`/api/v1/streams/${activeStreamId}/report`);
                const data = await res.json();
                document.getElementById('reportBox').innerText = JSON.stringify(data, null, 2);
            }

            function startPollingStats() {
                setInterval(async () => {
                    if (!activeStreamId) return;
                    const res = await fetch(`/api/v1/streams/${activeStreamId}/report`);
                    const data = await res.json();
                    if (data.metrics) {
                        document.getElementById('fpsVal').innerText = data.metrics.pipeline_fps;
                        document.getElementById('tracksVal').innerText = data.metrics.unique_persons_tracked;
                        document.getElementById('crossingsVal').innerText = data.metrics.total_tripwire_crossings;
                        document.getElementById('densityVal').innerText = data.metrics.total_high_density_alerts || 0;
                    }
                    if (data.high_density_alerts) {
                        renderAlertList(data.high_density_alerts);
                    }
                }, 1500);
            }

            function renderAlertList(alerts) {
                const list = document.getElementById('alertList');
                if (!alerts || alerts.length === 0) {
                    list.innerHTML = '<p style="color: var(--text-muted); font-size: 13px; margin: 0;">No alerts yet.</p>';
                    return;
                }
                // newest first
                const sorted = [...alerts].sort((a, b) => b.timestamp - a.timestamp);
                list.innerHTML = sorted.slice(0, 20).map(a => {
                    const when = new Date(a.timestamp * 1000).toLocaleTimeString();
                    const clipLink = a.clip_url
                        ? `<a href="${a.clip_url}" target="_blank">▶ evidence clip</a>`
                        : '';
                    return `<div class="alert-item">
                        <span class="alert-zone">⚠ zone: ${a.zone_id}</span>
                        <span class="alert-count">count=${a.density_count}</span>
                        <span style="color: var(--text-muted);">threshold=${a.threshold}</span>
                        ${clipLink}
                        <div class="alert-meta">stream: ${a.stream_id} • ${when}</div>
                    </div>`;
                }).join('');
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
