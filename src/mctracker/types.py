"""Core types shared across the package."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class Detection(NamedTuple):
    """A single detection as produced by a Detector.

    Attributes:
        xyxy: Bounding box as ``[x1, y1, x2, y2]`` in pixel coords, float32, shape (4,).
        conf: Confidence in ``[0, 1]``. **Never filtered before being passed to the
            tracker** — the tracker's confidence-cascade decides what to drop.
        cls: Class id. We use ``0`` for ``person``.
        det_id: Optional index into the per-frame detection list. Forwarded to boxmot
            so it can report which input detection produced each track.
    """

    xyxy: np.ndarray
    conf: float
    cls: int
    det_id: int | None = None


# A frame is a BGR uint8 ndarray (H, W, 3) — what cv2 returns.
Frame = np.ndarray
StreamId = str
