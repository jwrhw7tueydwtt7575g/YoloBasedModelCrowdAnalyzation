"""Person detection.

The Detector is an ABC so tests can inject a scripted fake. The production
implementation wraps ultralytics' YOLOv8 with ``classes=[0]`` (person).

**Critical contract: no confidence filter is applied here.** Every box
ultralytics returns is passed downstream to the tracker, even ones at conf=0.05.
Filtering happens at the display/output stage only (see ``Stream``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np

from .types import Detection, Frame


class Detector(ABC):
    """Abstract detector. ``detect(frame)`` returns a list of Detection."""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Detection]:
        ...


class YOLODetector(Detector):
    """YOLOv8 person detector, ultralytics-backed.

    Imports ultralytics lazily so tests that only need ``DummyDetector`` don't
    pay the ~3s first-import cost (and don't need a torch install).
    """

    def __init__(
        self,
        model_size: str = "yolov8n.pt",
        device: str | None = None,
        imgsz: int = 640,
        person_class_id: int = 0,
    ) -> None:
        # Lazy import: ultralytics first import is slow.
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(model_size)
        self._device = device
        self._imgsz = int(imgsz)
        self._person_class_id = int(person_class_id)

    def detect(self, frame: Frame) -> List[Detection]:
        # verbose=False keeps the terminal clean. classes=[self._person_class_id]
        # limits inference to person — saves a little time and matches the spec.
        results = self._model.predict(
            frame,
            classes=[self._person_class_id],
            verbose=False,
            device=self._device,
            imgsz=self._imgsz,
        )
        if not results:
            return []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        out: List[Detection] = []
        xyxy = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls
        for i in range(len(xyxy)):
            box = xyxy[i].detach().cpu().numpy().astype(np.float32)
            c = float(conf[i].detach().cpu().item())
            k = int(cls[i].detach().cpu().item())
            out.append(Detection(xyxy=box, conf=c, cls=k, det_id=i))
        return out


class DummyDetector(Detector):
    """Scripted fake. Tests build one with a per-frame list of detections."""

    def __init__(self, scripted: Iterable[Iterable[Detection] | None] | None = None) -> None:
        self._script: list[list[Detection] | None] = []
        if scripted is not None:
            self.set_script(scripted)
        self._i = 0
        self.calls: list[Frame] = []  # for assertions in tests

    def set_script(self, scripted: Iterable[Iterable[Detection] | None]) -> None:
        self._script = [list(d) if d is not None else [] for d in scripted]
        self._i = 0

    def detect(self, frame: Frame) -> List[Detection]:
        self.calls.append(frame)
        if self._i >= len(self._script):
            return []
        out = self._script[self._i]
        self._i += 1
        return list(out)
