"""Incremental local 3D mapping from RGB-D frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import open3d as o3d

from perception.pose_tracker import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)


@dataclass
class RGBDFrame:
    """A single RGB-D keyframe."""

    rgb: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW float32 meters
    pose: Pose
    timestamp: float


class LocalMap3D:
    """Maintain a sliding-window local point cloud map."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        max_keyframes: int = 60,
        voxel_size: float = 0.05,
        ground_threshold: float = 0.05,
        max_range: float = 5.0,
    ) -> None:
        self.intrinsics = intrinsics
        self.max_keyframes = max_keyframes
        self.voxel_size = voxel_size
        self.ground_threshold = ground_threshold
        self.max_range = max_range
        self._keyframes: list[RGBDFrame] = []
        self._global_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()

    def reset(self) -> None:
        self._keyframes.clear()
        self._global_cloud.clear()

    def update(self, frame: RGBDFrame, intrinsics: CameraIntrinsics | None = None) -> None:
        """Add a new RGB-D frame and rebuild the local map."""
        K = intrinsics or self.intrinsics
        if K is None:
            K = CameraIntrinsics.from_image_size(frame.rgb.shape[1], frame.rgb.shape[0])

        self._keyframes.append(frame)
        if len(self._keyframes) > self.max_keyframes:
            self._keyframes.pop(0)

        self._rebuild_map(K)

    def _rebuild_map(self, K: CameraIntrinsics) -> None:
        merged = o3d.geometry.PointCloud()
        for frame in self._keyframes:
            cloud = self._frame_to_cloud(frame, K)
            if cloud is not None and len(cloud.points) > 0:
                merged += cloud

        if len(merged.points) > 0:
            merged = merged.voxel_down_sample(self.voxel_size)
        self._global_cloud = merged
        logger.debug("Local map size: %d points", len(self._global_cloud.points))

    def _frame_to_cloud(
        self, frame: RGBDFrame, K: CameraIntrinsics
    ) -> o3d.geometry.PointCloud | None:
        rgb = frame.rgb
        depth = frame.depth
        h, w = depth.shape

        # Create meshgrid of pixel coordinates.
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)

        # Filter by max range and valid depth.
        valid = (z > 0.1) & (z < self.max_range) & np.isfinite(z)
        if valid.sum() == 0:
            return None

        fx, fy = K.fx, K.fy
        cx, cy = K.cx, K.cy

        x = (u[valid].astype(np.float32) - cx) * z[valid] / fx
        y = (v[valid].astype(np.float32) - cy) * z[valid] / fy
        z_valid = z[valid]

        points_camera = np.stack([x, y, z_valid], axis=-1)

        # Transform to world coordinates.
        T = self._pose_to_matrix(frame.pose)
        R = T[:3, :3]
        t = T[:3, 3]
        points_world = points_camera @ R.T + t

        colors = rgb[valid] / 255.0

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points_world)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        return cloud

    @staticmethod
    def _pose_to_matrix(pose: Pose) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [pose.x, pose.y, pose.z]
        return T

    def get_obstacle_points(
        self, robot_height: float = 0.15
    ) -> np.ndarray:
        """Return obstacle points above ground plane and below robot camera height.

        Uses a simple height-based ground removal: points with z < ground_threshold
        are considered ground; points between ground_threshold and robot_height are
        obstacles.
        """
        if len(self._global_cloud.points) == 0:
            return np.empty((0, 3))

        points = np.asarray(self._global_cloud.points)
        obstacles = points[
            (points[:, 2] > self.ground_threshold)
            & (points[:, 2] < robot_height)
        ]
        return obstacles

    def get_map(self) -> o3d.geometry.PointCloud:
        return self._global_cloud

    def save(self, path: str) -> None:
        """Save current map to .ply or .pcd."""
        o3d.io.write_point_cloud(path, self._global_cloud)
        logger.info("Saved local map to %s", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_points": len(self._global_cloud.points),
            "num_keyframes": len(self._keyframes),
            "voxel_size": self.voxel_size,
            "max_range": self.max_range,
        }
