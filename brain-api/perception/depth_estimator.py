"""Depth Anything 3 inference wrapper.

The official PyTorch implementation is expected to follow the Depth-Anything-3
repo layout. This module provides a thin adapter so the rest of the system only
depends on ``predict(image) -> depth_map``.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Load and run Depth Anything 3 for monocular depth estimation."""

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        input_size: int = 518,
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self._model: Any | None = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the DA3 checkpoint.

        This is a placeholder that attempts the most common pattern:
        ``DepthAnything3Pipeline`` from the official repo. Adjust once the
        exact hub/model layout is known.
        """
        try:
            # Try the official Depth-Anything-3 import path first.
            # pylint: disable=import-outside-toplevel
            from depth_anything_3 import DepthAnything3Pipeline

            self._model = DepthAnything3Pipeline.from_pretrained(self.model_path)
            self._model.to(self.device)
            logger.info("Loaded Depth Anything 3 from %s on %s", self.model_path, self.device)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not import DepthAnything3Pipeline: %s", exc)

        try:
            # Fallback: transformers AutoModelForDepthEstimation if the hub
            # provides a compatible config.
            # pylint: disable=import-outside-toplevel
            from transformers import AutoProcessor, AutoModelForDepthEstimation

            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = AutoModelForDepthEstimation.from_pretrained(self.model_path)
            self._model.to(self.device)
            logger.info("Loaded depth model via transformers from %s", self.model_path)
            return
        except Exception as exc:  # pragma: no cover
            logger.warning("Transformers fallback failed: %s", exc)

        raise RuntimeError(
            f"Unable to load Depth Anything 3 from {self.model_path}. "
            "Please install the official depth-anything-3 package or verify the checkpoint path."
        )

    def _decode_image(self, image_data: str | bytes | np.ndarray) -> np.ndarray:
        """Accept base64 string, raw bytes, or numpy array and return RGB array."""
        if isinstance(image_data, np.ndarray):
            return image_data

        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)

        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            return np.array(image)

        raise TypeError(f"Unsupported image type: {type(image_data)}")

    def predict(
        self,
        image: str | bytes | np.ndarray,
        return_numpy: bool = True,
    ) -> np.ndarray:
        """Estimate depth for a single RGB image.

        Args:
            image: Base64 JPEG string, raw bytes, or HxWx3 RGB numpy array.
            return_numpy: If True return a HxW float32 depth map (meters),
                otherwise return the raw model output.

        Returns:
            Depth map in meters.
        """
        rgb = self._decode_image(image)
        h, w = rgb.shape[:2]

        if self._model is None:
            raise RuntimeError("Model not loaded")

        # Official DA3 pipeline path
        if hasattr(self._model, "infer"):
            depth = self._model.infer(rgb, input_size=self.input_size)
        elif hasattr(self._model, "__call__") and hasattr(self, "_processor"):
            pil = Image.fromarray(rgb)
            inputs = self._processor(images=pil, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                prediction = outputs.predicted_depth
            depth = prediction.squeeze().cpu().numpy()
        else:
            raise RuntimeError("Model has no supported inference method")

        if depth.shape != (h, w):
            depth = Image.fromarray(depth).resize((w, h), Image.BILINEAR)
            depth = np.asarray(depth, dtype=np.float32)

        if return_numpy:
            return np.asarray(depth, dtype=np.float32)
        return depth

    def to_colormap(self, depth: np.ndarray) -> np.ndarray:
        """Convert depth map to uint8 RGB colormap for visualization."""
        import cv2

        depth_norm = depth - depth.min()
        if depth_norm.max() > 0:
            depth_norm = depth_norm / depth_norm.max()
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)[:, :, ::-1]
