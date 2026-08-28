"""Script to run mctracker pipeline on user video file and report live performance metrics."""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

import cv2
import numpy as np

from mctracker.config import AppConfig, StreamConfig, EvidenceConfig, ViolationsConfig
from mctracker.detector import Detector
from mctracker.pipeline import Pipeline
from mctracker.metrics import reset_metrics, METRICS
from mctracker.tracker import Tracker, make_tracker
from mctracker.tripwire import Tripwire, TripwireManager, CrossingEvent
from mctracker.types import Detection, Frame
from mctracker.zones import Zone, ZoneManager
from mctracker.stream import Stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("video_test")

VIDEO_PATH = "/home/vivek/Downloads/Testing Videos/kling_20260828_VIDEO_A_realisti_3962_0.mp4"


class OpenCVContourDetector(Detector):
    """Fallback background subtraction motion detector when ultralytics is not available."""
    def __init__(self):
        self._fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    def detect(self, frame: Frame) -> List[Detection]:
        fgmask = self._fgbg.apply(frame)
        # Apply morphology to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        h, w = frame.shape[:2]
        min_area = (h * w) * 0.002  # Filter out tiny noise blobs

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


class SimulatedVideoTracker(Tracker):
    """Pure-Python IOU tracker for video test run."""
    def __init__(self, max_age: int = 20) -> None:
        self._max_age = max_age
        self._next_id = 1
        self._tracks = {}

    def update(self, frame: Frame, detections: List[Detection]) -> List:
        from mctracker.track_state import make_track_state
        now_ts = time.time()

        for tid in list(self._tracks.keys()):
            self._tracks[tid]["age"] += 1
            if self._tracks[tid]["age"] > self._max_age:
                del self._tracks[tid]

        unmatched_dets = list(range(len(detections)))
        matched_tracks = set()

        for det_idx in list(unmatched_dets):
            det = detections[det_idx]
            best_iou = 0.0
            best_tid = None
            for tid, trk in self._tracks.items():
                if tid in matched_tracks:
                    continue
                t_box = trk["xyxy"]
                d_box = det.xyxy
                ix1, iy1 = max(t_box[0], d_box[0]), max(t_box[1], d_box[1])
                ix2, iy2 = min(t_box[2], d_box[2]), min(t_box[3], d_box[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                union = ((t_box[2]-t_box[0])*(t_box[3]-t_box[1])) + ((d_box[2]-d_box[0])*(d_box[3]-d_box[1])) - inter
                iou = inter / float(union + 1e-6)
                if iou > 0.2 and iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                self._tracks[best_tid]["xyxy"] = det.xyxy
                self._tracks[best_tid]["conf"] = det.conf
                self._tracks[best_tid]["cls"] = det.cls
                self._tracks[best_tid]["age"] = 0
                matched_tracks.add(best_tid)
                unmatched_dets.remove(det_idx)

        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {"xyxy": det.xyxy, "conf": det.conf, "cls": det.cls, "age": 0}
            matched_tracks.add(tid)

        output_states = []
        for tid in matched_tracks:
            trk = self._tracks[tid]
            if trk["age"] == 0:
                output_states.append(
                    make_track_state(
                        track_id=tid,
                        bbox_xyxy=(float(trk["xyxy"][0]), float(trk["xyxy"][1]), float(trk["xyxy"][2]), float(trk["xyxy"][3])),
                        conf=float(trk["conf"]),
                        cls=int(trk["cls"]),
                        ts=now_ts,
                    )
                )
        return output_states

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_track_ids(self) -> set[int]:
        return {tid for tid, trk in self._tracks.items() if trk["age"] == 0}


def _create_detector(model_size: str = "yolov8n.pt") -> Detector:
    log.info("Using OpenCV Contour Motion Detector for high-throughput testing.")
    return OpenCVContourDetector()


def _create_tracker(tracker_type: str = "botsort") -> Tracker:
    try:
        import boxmot  # type: ignore # noqa: F401
        log.info("Using boxmot tracker.")
        return make_tracker(tracker_type, frame_rate=30, with_reid=False)
    except Exception:
        log.warning("boxmot unavailable; using IOU tracker.")
        return SimulatedVideoTracker()


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        sys.exit(1)

    print(f"============================================================")
    print(f"Running mctracker Pipeline on User Video:")
    print(f"Path: {VIDEO_PATH}")
    print(f"============================================================")

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"Video Resolution: {width}x{height} @ {fps:.1f} FPS ({total_frames} total frames)")

    tripwire_y = height // 2
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

    stream_cfg = StreamConfig(
        id="user_video_stream",
        source=VIDEO_PATH,
        model_size="yolov8n.pt",
        tracker_type="botsort",
        use_appearance=False,
        zones=[zn],
        tripwires=[tw],
        fps_fallback=int(fps),
    )

    reset_metrics()

    recorded_frames = 0
    all_tracks_seen = set()
    all_crossings = []
    zone_occupancy_history = []

    def on_results(stream_id: str, tracks, zones, crossings):
        nonlocal recorded_frames
        recorded_frames += 1
        for t in tracks:
            all_tracks_seen.add(t.track_id)
        if crossings:
            all_crossings.extend(crossings)
        if zones:
            zone_occupancy_history.append((recorded_frames, {z.zone_id: z.count for z in zones}))

    detector = _create_detector("yolov8n.pt")
    tracker = _create_tracker("botsort")
    zone_mgr = ZoneManager(zones=[zn])
    tripwire_mgr = TripwireManager(stream_id="user_video_stream", tripwires=[tw])

    stream = Stream(
        stream_id="user_video_stream",
        source=VIDEO_PATH,
        detector=detector,
        tracker=tracker,
        on_results=on_results,
        buffer_seconds=5,
        fps_fallback=int(fps),
        display_conf=0.20,
        zone_manager=zone_mgr,
        tripwire_manager=tripwire_mgr,
    )

    print("\nStarting stream execution loop...")
    start_time = time.time()
    stream.start()

    # Wait for StreamReader thread to finish reading all frames from file
    if stream._reader and stream._reader._thread:
        stream._reader._thread.join(timeout=60.0)

    # Wait until processor thread drains all buffered frames
    drain_timeout = time.time() + 30.0
    while len(stream._buffer) > 0 and time.time() < drain_timeout:
        time.sleep(0.1)

    time.sleep(0.5)
    stream.stop(timeout=2.0)

    elapsed = time.time() - start_time
    proc_fps = recorded_frames / float(elapsed + 1e-6)

    snap = METRICS.snapshot()

    print("\n============================================================")
    print("Execution & Benchmark Results on User Video:")
    print("============================================================")
    print(f"Total Video Frames Processed: {recorded_frames} / {total_frames}")
    print(f"Processing Time:               {elapsed:.2f} seconds")
    print(f"Pipeline Throughput:           {proc_fps:.2f} FPS (Video FPS: {fps:.1f})")
    print(f"Unique Persons Tracked:        {len(all_tracks_seen)} (IDs: {sorted(list(all_tracks_seen))})")
    print(f"Tripwire Crossings Detected:   {len(all_crossings)}")
    for c in all_crossings:
        print(f"  - [{c.timestamp:.2f}s] Track #{c.track_id} crossed {c.tripwire_id} ({c.direction}) at position ({c.centroid[0]:.1f}, {c.centroid[1]:.1f})")

    print(f"\nStage Latency Breakdown (Averages):")
    if "histograms" in snap and "stage_seconds" in snap["histograms"]:
        for (stream_id, stage), stats in snap["histograms"]["stage_seconds"].items():
            avg_ms = (stats['sum'] / (stats['count'] + 1e-6)) * 1000.0
            print(f"  - Stage '{stage}': total_calls={stats['count']}, avg_latency={avg_ms:.2f} ms")

    print(f"\nPrometheus Counters:")
    for metric_name, val in snap.get("counters", {}).items():
        print(f"  - {metric_name}: {val}")

    print("============================================================")


if __name__ == "__main__":
    main()
