"""Standalone FastAPI service for Depth Anything 3 3D perception."""

from __future__ import annotations

import base64
import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from perception import CameraIntrinsics, DepthEstimator, LocalMap3D, LocalNavigator, RGBDFrame
from perception.pose_tracker import PoseTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("da3_service")

# Global state managed via lifespan.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(__file__).parent.parent / "models" / "depth-anything-3"
    if not model_path.exists():
        logger.warning("Model path %s does not exist. Depth estimation will fail.", model_path)

    try:
        _state["depth"] = DepthEstimator(model_path=model_path)
        logger.info("Depth estimator loaded")
    except Exception as exc:
        logger.error("Failed to load depth estimator: %s", exc)
        _state["depth"] = None

    _state["pose_tracker"] = PoseTracker()
    _state["map"] = LocalMap3D()
    _state["navigator"] = LocalNavigator()
    yield
    _state.clear()


app = FastAPI(title="OpenBot Depth Anything 3 Perception Service", lifespan=lifespan)


class ImageRequest(BaseModel):
    image_base64: str
    camera_fov_deg: float = Field(default=60.0, ge=10.0, le=170.0)


class PoseRequest(BaseModel):
    current_image_base64: str
    previous_image_base64: str
    current_depth_base64: str | None = None
    previous_depth_base64: str | None = None
    camera_fov_deg: float = Field(default=60.0, ge=10.0, le=170.0)


class MapUpdateRequest(BaseModel):
    image_base64: str
    depth_base64: str | None = None
    camera_fov_deg: float = Field(default=60.0, ge=10.0, le=170.0)
    timestamp: float | None = None


class NavigateRequest(BaseModel):
    target_angle_deg: float = Field(default=0.0, ge=-180.0, le=180.0)


class DepthResponse(BaseModel):
    depth_map_base64: str
    min_m: float
    max_m: float


class PoseResponse(BaseModel):
    pose: dict[str, float]


class ObstaclesResponse(BaseModel):
    obstacles: list[dict[str, Any]]
    navigation_hint: str
    map_info: dict[str, Any]


def _decode_image_b64(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")


def _decode_depth_b64(b64: str) -> np.ndarray:
    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        arr = np.asarray(img, dtype=np.float32)
        # If stored as uint16 millimeters, convert to meters.
        if arr.dtype == np.uint16:
            arr = arr / 1000.0
        return arr
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid depth map: {exc}")


def _encode_depth(depth: np.ndarray) -> str:
    """Encode depth map as uint16 mm PNG and return base64."""
    depth_mm = (depth * 1000.0).astype(np.uint16)
    img = Image.fromarray(depth_mm)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "depth_loaded": _state.get("depth") is not None}


@app.post("/depth", response_model=DepthResponse)
def depth_estimate(req: ImageRequest) -> DepthResponse:
    estimator = _state.get("depth")
    if estimator is None:
        raise HTTPException(status_code=503, detail="Depth model not loaded")

    rgb = _decode_image_b64(req.image_base64)
    depth = estimator.predict(rgb)

    return DepthResponse(
        depth_map_base64=_encode_depth(depth),
        min_m=float(depth.min()),
        max_m=float(depth.max()),
    )


@app.post("/pose", response_model=PoseResponse)
def pose_estimate(req: PoseRequest) -> PoseResponse:
    tracker = _state.get("pose_tracker")
    if tracker is None:
        raise HTTPException(status_code=503, detail="Pose tracker not initialized")

    rgb_curr = _decode_image_b64(req.current_image_base64)
    rgb_prev = _decode_image_b64(req.previous_image_base64)

    depth_curr = (
        _decode_depth_b64(req.current_depth_base64)
        if req.current_depth_base64
        else np.ones(rgb_curr.shape[:2], dtype=np.float32)
    )
    depth_prev = (
        _decode_depth_b64(req.previous_depth_base64)
        if req.previous_depth_base64
        else np.ones(rgb_prev.shape[:2], dtype=np.float32)
    )

    K = CameraIntrinsics.from_image_size(rgb_curr.shape[1], rgb_curr.shape[0], req.camera_fov_deg)
    tracker.reset()
    tracker.estimate(rgb_prev, depth_prev, K)
    pose = tracker.estimate(rgb_curr, depth_curr, K)

    return PoseResponse(pose=pose.to_dict())


@app.post("/map/update")
def map_update(req: MapUpdateRequest) -> dict[str, Any]:
    estimator = _state.get("depth")
    tracker = _state.get("pose_tracker")
    local_map = _state.get("map")
    if estimator is None or tracker is None or local_map is None:
        raise HTTPException(status_code=503, detail="Perception pipeline not initialized")

    rgb = _decode_image_b64(req.image_base64)
    depth = (
        _decode_depth_b64(req.depth_base64)
        if req.depth_base64
        else estimator.predict(rgb)
    )

    K = CameraIntrinsics.from_image_size(rgb.shape[1], rgb.shape[0], req.camera_fov_deg)
    pose = tracker.estimate(rgb, depth, K)

    frame = RGBDFrame(
        rgb=rgb,
        depth=depth,
        pose=pose,
        timestamp=req.timestamp or time.time(),
    )
    local_map.update(frame, K)

    return {"pose": pose.to_dict(), "map": local_map.to_dict()}


@app.post("/map/obstacles", response_model=ObstaclesResponse)
def map_obstacles(req: NavigateRequest | None = None) -> ObstaclesResponse:
    local_map = _state.get("map")
    navigator = _state.get("navigator")
    if local_map is None or navigator is None:
        raise HTTPException(status_code=503, detail="Perception pipeline not initialized")

    target_angle = req.target_angle_deg if req else 0.0
    obstacles = navigator.detect_obstacles(local_map.get_obstacle_points())
    hint, _ = navigator.compute_hint(local_map.get_obstacle_points(), target_angle)

    return ObstaclesResponse(
        obstacles=[o.to_dict() for o in obstacles],
        navigation_hint=hint,
        map_info=local_map.to_dict(),
    )


@app.post("/navigate", response_model=ObstaclesResponse)
def navigate(req: NavigateRequest) -> ObstaclesResponse:
    return map_obstacles(req)


@app.post("/map/save")
def map_save(path: str = "local_map.ply") -> dict[str, Any]:
    local_map = _state.get("map")
    if local_map is None:
        raise HTTPException(status_code=503, detail="Map not initialized")
    local_map.save(path)
    return {"saved_to": path}


@app.post("/reset")
def reset() -> dict[str, Any]:
    if "pose_tracker" in _state:
        _state["pose_tracker"].reset()
    if "map" in _state:
        _state["map"].reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("perception.da3_service:app", host="0.0.0.0", port=8002, reload=False)
