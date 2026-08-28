"""Visual Frame Annotator module for drawing detections, tracks, tripwires, and zones onto BGR OpenCV frames."""

from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mctracker.track_state import TrackState
from mctracker.tripwire import Tripwire, CrossingEvent
from mctracker.types import Frame
from mctracker.zones import Zone


def _get_track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a distinct, bright BGR color deterministically for a track ID."""
    golden_ratio = 0.618033988749895
    hue = (track_id * golden_ratio) % 1.0
    # High saturation and brightness for vivid display on dark/light video backgrounds
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


class FrameAnnotator:
    """Renders visual bounding boxes, track IDs, zone polygons, and tripwire indicators onto frames."""

    def __init__(self, line_thickness: int = 2, font_scale: float = 0.5) -> None:
        self.line_thickness = line_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate(
        self,
        frame: Frame,
        tracks: List[TrackState],
        zones: Optional[List[Zone]] = None,
        tripwires: Optional[List[Tripwire]] = None,
        recent_crossings: Optional[List[CrossingEvent]] = None,
    ) -> Frame:
        """Draw annotations onto a copy of the BGR input frame."""
        out = frame.copy()
        h, w = out.shape[:2]

        # 1. Draw Zones (Semi-transparent polygon overlays)
        if zones:
            overlay = out.copy()
            for zone in zones:
                polygon = getattr(zone, "polygon", None)
                if not polygon:
                    continue
                pts = np.array(polygon, dtype=np.int32)
                is_occupied = getattr(zone, "count", 0) > 0
                # Green for vacant zone, Red for occupied
                fill_color = (0, 0, 200) if is_occupied else (0, 180, 0)
                border_color = (0, 0, 255) if is_occupied else (0, 255, 0)

                cv2.fillPoly(overlay, [pts], fill_color)
                cv2.polylines(out, [pts], isClosed=True, color=border_color, thickness=2)

                # Zone label at centroid of polygon
                M = cv2.moments(pts)
                if M["m00"] != 0:
                    cx = int(M["10"] / M["m00"])
                    cy = int(M["01"] / M["m00"])
                    label = f"Zone: {zone.id} (Occupancy: {getattr(zone, 'count', 0)})"
                    (tw_w, tw_h), _ = cv2.getTextSize(label, self.font, 0.5, 1)
                    cv2.rectangle(out, (cx - 5, cy - tw_h - 5), (cx + tw_w + 5, cy + 5), (0, 0, 0), -1)
                    cv2.putText(out, label, (cx, cy), self.font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Apply alpha blend for zone fill
            alpha = 0.25
            cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

        # 2. Draw Tripwires
        if tripwires:
            active_tripwire_ids = {c.tripwire_id for c in (recent_crossings or [])}
            for tw in tripwires:
                p1 = (int(tw.p1[0]), int(tw.p1[1]))
                p2 = (int(tw.p2[0]), int(tw.p2[1]))
                
                is_active = tw.id in active_tripwire_ids
                line_color = (0, 0, 255) if is_active else (255, 255, 0)  # Bright Red if crossing, Cyan otherwise
                thickness = 4 if is_active else 2

                cv2.line(out, p1, p2, line_color, thickness)
                cv2.circle(out, p1, 5, line_color, -1)
                cv2.circle(out, p2, 5, line_color, -1)

                # Tripwire ID label
                mid_x = (p1[0] + p2[0]) // 2
                mid_y = (p1[1] + p2[1]) // 2
                status_str = " [CROSSING!]" if is_active else ""
                tw_label = f"Tripwire: {tw.id}{status_str}"
                cv2.putText(out, tw_label, (mid_x - 40, mid_y - 10), self.font, 0.45, line_color, 1, cv2.LINE_AA)

        # 3. Draw Tracked Objects (Bounding boxes & ID tags)
        for trk in tracks:
            x1, y1, x2, y2 = map(int, trk.bbox)
            color = _get_track_color(trk.track_id)

            # Draw bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, self.line_thickness)

            # Draw centroid (bottom-center tracking point)
            cx, cy = map(int, trk.centroid)
            cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

            # Build label text
            conf_str = f"{trk.confidence*100:.0f}%" if trk.confidence else ""
            label_text = f"ID #{trk.track_id} {conf_str}".strip()

            # Draw filled background pill for label
            (txt_w, txt_h), baseline = cv2.getTextSize(label_text, self.font, self.font_scale, 1)
            lbl_y1 = max(0, y1 - txt_h - 6)
            cv2.rectangle(out, (x1, lbl_y1), (x1 + txt_w + 6, y1), color, -1)
            cv2.putText(
                out,
                label_text,
                (x1 + 3, y1 - 4),
                self.font,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return out
