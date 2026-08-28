<div align="center">

# 📹 Multi-Camera Person Tracking & Analytics Pipeline (`mctracker`)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Railway](https://img.shields.io/badge/Railway-Live%20Deploy-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)](https://mctracker-api-production.up.railway.app/)
[![Tests](https://img.shields.io/badge/Tests-152%20Passed-2EA44F?style=for-the-badge&logo=github-actions&logoColor=white)](tests/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<p align="center">
  <b>A real-time, multi-stream computer vision pipeline for person detection, multi-object tracking, spatial tripwire counting, high-density crowd alerting, and automated MP4 evidence recording.</b>
</p>

[Live Production Dashboard](https://mctracker-api-production.up.railway.app/) • [Features](#-key-features) • [Architecture](#-architecture) • [API Reference](#-api-reference) • [Docker & Deployment](#-docker--railway-deployment) • [Testing](#-testing--benchmarking)

</div>

---

## 🌟 Key Features

- 🎥 **Multi-Stream Async Pipeline:** Dedicated thread-isolated ingestion, ring buffers (`deque` / zero-copy memory budget), and background frame processors.
- 🎯 **No Pre-Filter Invariant:** All detections—including low-confidence candidates (`conf >= 0.05`)—are passed directly to the tracker (`ByteTrack` or `BoT-SORT`) to maintain trajectory continuity. Confidence filtering is applied *only* at the output/display stage.
- 📐 **Polygon Zone Occupancy & Tripwires:** Real-time point-in-polygon counting (`bottom_center` or `geometric_center` centroids) and vector cross-product tripwire crossing detection with track ID recycling guards.
- 🚨 **Stage 6 High-Density Crowd Alerting:** Real-time zone density monitoring. Stateful `_ZoneDensityState` tracking with configurable occupancy threshold, dwell duration (sustained crowding), and cooldown period to prevent duplicate alert spam.
- 📹 **Automated Pre/Post Event Evidence Recording:** Captures rolling in-memory snapshots from the frame buffer prior to a violation and pairs it with live post-event capture to generate standalone MP4 evidence clips (`LocalDiskClipStorage` / S3 ready).
- 🏷️ **Access Control & Tailgating Detection:** Correlates camera tripwire crossings against an external scan/badge stream (`ScanStore` with $O(\log N)$ `SortedDict` lookups) to detect tailgating and unauthorized access.
- ⚡ **Production-Ready FastAPI Server & Dashboard:** Out-of-the-box REST API for streaming MJPEG feeds, asynchronous video file processing, real-time analytics reporting, and a responsive dark-mode HTML5 dashboard.

---

## 🏗️ Architecture

```
                                  ┌────────────────────────────────────────────────────────┐
                                  │                Multi-Camera Pipeline                   │
                                  └────────────────────────────────────────────────────────┘
                                                              │
         ┌────────────────────────────────────────────────────┴────────────────────────────────────────────────────┐
         ▼                                                                                                         ▼
┌──────────────────┐                                                                                     ┌──────────────────┐
│  RTSP / Video 0  │                                                                                     │  RTSP / Video 1  │
└──────────────────┘                                                                                     └──────────────────┘
         │                                                                                                         │
         ▼                                                                                                         ▼
┌──────────────────┐                                                                                     ┌──────────────────┐
│  StreamReader    │ (Thread)                                                                            │  StreamReader    │ (Thread)
└──────────────────┘                                                                                     └──────────────────┘
         │                                                                                                         │
         ▼                                                                                                         ▼
┌──────────────────┐                                                                                     ┌──────────────────┐
│   FrameBuffer    │ (Rolling Ring Buffer, ~5s)                                                           │   FrameBuffer    │ (Rolling Ring Buffer, ~5s)
└──────────────────┘                                                                                     └──────────────────┘
         │                                                                                                         │
         ▼                                                                                                         ▼
┌──────────────────┐                                                                                     ┌──────────────────┐
│  Stream Processor│ (YOLOv8 + ByteTrack / BoT-SORT)                                                     │  Stream Processor│ (YOLOv8 + ByteTrack / BoT-SORT)
└──────────────────┘                                                                                     └──────────────────┘
         │                                                                                                         │
         └────────────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │            Zone Occupancy & Tripwire Engine            │
                                  └────────────────────────────────────────────────────────┘
                                                              │
                                    ┌─────────────────────────┴─────────────────────────┐
                                    ▼                                                   ▼
                       ┌─────────────────────────┐                         ┌─────────────────────────┐
                       │  High-Density Rule      │                         │  Violation Service      │
                       │  (Threshold, Dwell,     │                         │  (Crossing vs ScanStore │
                       │   Cooldown State)       │                         │   Tailgating Rule)      │
                       └─────────────────────────┘                         └─────────────────────────┘
                                    │                                                   │
                                    └─────────────────────────┬─────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │       EvidenceRecorder & DB Storage (Postgres/Memory)   │
                                  │       - Snapshots Pre-Event Buffer                     │
                                  │       - Appends Post-Event Capture -> MP4              │
                                  └────────────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                  ┌────────────────────────────────────────────────────────┐
                                  │       FastAPI Production Server & Dashboard            │
                                  └────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Local Environment Setup

```bash
# Clone repository
git clone https://github.com/vivekchaudhari17/mctracker.git
cd "opencv yolo model"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core runtime dependencies
pip install -e ".[dev]"
```

For GPU acceleration and heavy tracking dependencies (YOLOv8 + BoxMOT):

```bash
pip install -e ".[dev,heavy]"
```

---

### 2. Running Local Pipeline

To run the pipeline with a YAML configuration file:

```bash
python main.py --config config.example.yaml
```

To run a lightweight webcam smoke test:

```bash
python main.py --config config.smoke.yaml
```

---

### 3. Launching FastAPI API & Live Dashboard

```bash
uvicorn mctracker.api.server:app --host 0.0.0.0 --port 8000
```

Access the dashboard at `http://localhost:8000/`.

---

## 🔌 API Reference

### Core Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Interactive Live Monitoring Dashboard (HTML5) |
| `/api/v1/process-video` | `POST` | Asynchronously process uploaded video file (`multipart/form-data`) |
| `/api/v1/streams` | `GET` | List active camera streams & status |
| `/api/v1/streams/{id}/feed` | `GET` | Live MJPEG Video Stream Feed (`multipart/x-mixed-replace`) |
| `/api/v1/streams/{id}/report` | `GET` | Detailed JSON Analytics Report (Occupancy, Crossings, Alerts, Latencies) |
| `/api/v1/streams/{id}/download` | `GET` | Download processed MP4 annotated video file |
| `/api/v1/scans` | `POST` | Badge / QR Code scan webhook ingestion (`ScanStore`) |
| `/metrics` | `GET` | Prometheus exporter metrics |

---

### Example API Request (Async Video Upload)

```bash
curl -X POST "https://mctracker-api-production.up.railway.app/api/v1/process-video" \
  -F "file=@/path/to/sample_video.mp4"
```

**Response:**

```json
{
  "status": "success",
  "stream_id": "stream_1787924041387",
  "video_info": {
    "resolution": "640x480",
    "fps": 30.0,
    "total_frames": 90
  },
  "links": {
    "feed": "/api/v1/streams/stream_1787924041387/feed",
    "report": "/api/v1/streams/stream_1787924041387/report",
    "download": "/api/v1/streams/stream_1787924041387/download"
  }
}
```

---

## 🐳 Docker & Railway Deployment

### Build and Tag Docker Image

```bash
docker build -t vivekchaudhari17/mctracker-api:latest -t vivekchaudhari17/mctracker-api:v1.1 .
```

### Push to Registry

```bash
docker push vivekchaudhari17/mctracker-api:latest
docker push vivekchaudhari17/mctracker-api:v1.1
```

### Docker Compose Local Deployment

```bash
docker compose up -d
```

### Railway Production Environment
- **Live URL:** `https://mctracker-api-production.up.railway.app`
- **Image:** `docker.io/vivekchaudhari17/mctracker-api:latest`
- **Port:** Binds automatically to Railway's dynamic `$PORT` environment variable.

---

## 🧪 Testing & Benchmarking

The codebase is protected by a comprehensive unit & integration test suite covering stream resilience, zone occupancy, tripwire logic, memory guards, evidence recording, and high-density alerting.

```bash
# Run complete test suite
.venv/bin/pytest

# Run high-density zone tests specifically
.venv/bin/pytest tests/test_high_density_zone.py
```

### Test Suite Results

```text
============================= test session starts ==============================
collected 152 items

tests/test_api_server.py ....                                            [  2%]
tests/test_buffer.py ......                                              [  6%]
tests/test_config.py .......                                             [ 11%]
tests/test_config_validation.py ....................                     [ 24%]
tests/test_evidence_integration.py .                                     [ 25%]
tests/test_evidence_unit.py ..................                           [ 36%]
tests/test_high_density_zone.py ..................                       [ 48%]
tests/test_low_conf_reaches_tracker.py ...                               [ 50%]
tests/test_pipeline_smoke.py ..                                          [ 51%]
tests/test_resilience_and_config.py .....                                [ 55%]
tests/test_scan_store.py .........                                       [ 61%]
tests/test_stream_reader_reconnect.py ....                               [ 63%]
tests/test_stream_resilience.py ....                                     [ 66%]
tests/test_tracker_isolation.py ....                                     [ 69%]
tests/test_tracker_occlusion.py .X.                                      [ 71%]
tests/test_tripwire.py ...............                                   [ 80%]
tests/test_violation_service.py ...............                          [ 90%]
tests/test_violations_repository.py .....                                [ 94%]
tests/test_zones.py .........                                            [100%]

================== 151 passed, 1 xpassed, 1 warning in 5.31s ===================
```

---

## 📜 Configuration (`config.yaml`)

```yaml
streams:
  - id: camera_lobby
    source: "rtsp://admin:pass@192.168.1.100:554/live"
    model_size: yolov8n.pt
    tracker_type: botsort
    use_appearance: false
    centroid_mode: bottom_center
    display_conf: 0.25
    zones:
      - id: zone_main_lobby
        polygon: [[100, 100], [500, 100], [500, 500], [100, 500]]
        density_alert:
          enabled: true
          threshold: 5
          dwell_seconds: 2.0
          cooldown_seconds: 15.0
    tripwires:
      - id: tripwire_entrance
        p1: [100.0, 300.0]
        p2: [500.0, 300.0]
        direction_in: left_to_right

evidence:
  enabled: true
  base_dir: ./evidence_clips
  pre_seconds: 5.0
  post_seconds: 5.0
  free_threshold_mb: 2048.0
  retention_days: 30.0
  clip_storage: local
```

---

## 📝 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
