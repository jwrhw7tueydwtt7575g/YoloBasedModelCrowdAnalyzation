"""Appearance embeddings for BoT-SORT.

We use OSNet, the same family boxmot uses by default. Loading is lazy so that
importing this module does not pull torch.

The embedder is given a frame and an xyxy bbox; it returns a 1D float32
feature vector. boxmot's BoT-SORT calls its own embedder internally for
association; our embedder is used to populate ``TrackState.embedding`` for
downstream consumers (re-identification across cameras, etc.).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import numpy as np

from .types import Frame

log = logging.getLogger(__name__)


class OSNetEmbedder:
    """OSNet-based appearance embedder.

    The first call lazily downloads (via torch.hub) the default OSNet weights
    if they aren't already cached. If torch is not installed, every call
    returns ``None``-friendly behaviour via :meth:`__call__` raising a clear
    error — the embedder is opt-in, so this should not affect ByteTrack users.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._weights = weights
        self._device = device
        self._model = None
        self._transform = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
            from torchvision import transforms  # type: ignore
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "torch + torchvision are required for OSNetEmbedder. "
                "Install with: pip install torch torchvision"
            ) from e

        try:
            self._model = torch.hub.load(
                "mikel-brostrom/osnet_iaa_pytorch_testing:main",
                "osnet_iaa_x1_0_pytorch",
                pretrained=True,
                trust_repo=True,
            )
        except Exception as e:
            log.warning("failed to load OSNet from torch.hub: %s", e)
            raise
        if self._device is not None:
            self._model.to(self._device)
        self._model.eval()
        self._transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, frame: Frame, bbox_xyxy: Tuple[float, float, float, float]) -> np.ndarray:
        """Return a 1D float32 feature vector for the person inside ``bbox_xyxy``."""
        self._load()
        import torch  # type: ignore

        x1, y1, x2, y2 = (int(round(v)) for v in bbox_xyxy)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((512,), dtype=np.float32)
        crop_bgr = frame[y1:y2, x1:x2]
        crop_rgb = crop_bgr[:, :, ::-1]  # BGR -> RGB
        tensor = self._transform(crop_rgb).unsqueeze(0)
        if self._device is not None:
            tensor = tensor.to(self._device)
        with torch.no_grad():
            feat = self._model(tensor)
        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)
        # L2-normalize so cosine == dot.
        n = np.linalg.norm(feat)
        if n > 0:
            feat = feat / n
        return feat
