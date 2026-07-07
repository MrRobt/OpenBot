"""Incremental local 3D mapping from RGB-D frames (numpy-only)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.pose_tracker import CameraIntrinsics, Pose

logger = logging.getLogger(__name__)


try:
    # Optional: use open3d for visualization/saving if available.
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False


@dataclass
class RGBDFrame:
    """A single RGB-D keyframe."""

    rgb: np.ndarray  # HxWx3 uint8
    depth: np.ndarray  # HxW float32 meters
    pose: Pose
    timestamp: float


class LocalMap3D:
    """Maintain a sliding-window local point cloud map using numpy only."""

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
        self._points: np.ndarray = np.empty((0, 3))
        self._colors: np.ndarray = np.empty((0, 3))

    def reset(self) -> None:
        self._keyframes.clear()
        self._points = np.empty((0, 3))
        self._colors = np.empty((0, 3))

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
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []
        for frame in self._keyframes:
            pts, cols = self._frame_to_cloud(frame, K)
            if pts is not None and len(pts) > 0:
                all_points.append(pts)
                all_colors.append(cols)

        if all_points:
            merged_points = np.concatenate(all_points, axis=0)
            merged_colors = np.concatenate(all_colors, axis=0)
            merged_points, idx = self._voxel_down_sample(merged_points)
            merged_colors = merged_colors[idx]
            self._points = merged_points
            self._colors = merged_colors
        else:
            self._points = np.empty((0, 3))
            self._colors = np.empty((0, 3))

        logger.debug("Local map size: %d points", len(self._points))

    def _voxel_down_sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return downsampled points and the indices of kept unique voxels."""
        if len(points) == 0:
            return points, np.array([], dtype=int)
        voxels = np.floor(points / self.voxel_size).astype(np.int32)
        uniq, idx = np.unique(voxels, axis=0, return_index=True)
        return points[idx], idx

    def _frame_to_cloud(
        self, frame: RGBDFrame, K: CameraIntrinsics
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        rgb = frame.rgb
        depth = frame.depth
        h, w = depth.shape

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.astype(np.float32)

        valid = (z > 0.1) & (z < self.max_range) & np.isfinite(z)
        if valid.sum() == 0:
            return None, None

        fx, fy = K.fx, K.fy
        cx, cy = K.cx, K.cy

        x = (u[valid].astype(np.float32) - cx) * z[valid] / fx
        y = (v[valid].astype(np.float32) - cy) * z[valid] / fy
        z_valid = z[valid]

        points_camera = np.stack([x, y, z_valid], axis=-1)

        T = self._pose_to_matrix(frame.pose)
        R = T[:3, :3]
        t = T[:3, 3]
        points_world = points_camera @ R.T + t
        colors = rgb[valid] / 255.0

        return points_world, colors

    @staticmethod
    def _pose_to_matrix(pose: Pose) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [pose.x, pose.y, pose.z]
        return T

    def get_obstacle_points(
        self, robot_height: float = 0.15
    ) -> np.ndarray:
        """Return obstacle points above ground plane and below robot camera height."""
        if len(self._points) == 0:
            return np.empty((0, 3))

        obstacles = self._points[
            (self._points[:, 2] > self.ground_threshold)
            & (self._points[:, 2] < robot_height)
        ]
        return obstacles

    def get_map(self) -> tuple[np.ndarray, np.ndarray]:
        return self._points, self._colors

    def save(self, path: str) -> None:
        """Save current map to .ply (uses open3d if available, else simple ASCII)."""
        if HAS_OPEN3D:
            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(self._points)
            cloud.colors = o3d.utility.Vector3dVector(self._colors)
            o3d.io.write_point_cloud(path, cloud)
        else:
            self._save_ascii(path)
        logger.info("Saved local map to %s", path)

    def _save_ascii(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self._points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for pt, col in zip(self._points, self._colors):
                r, g, b = (col * 255).astype(int)
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {r} {g} {b}\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_points": len(self._points),
            "num_keyframes": len(self._keyframes),
            "voxel_size": self.voxel_size,
            "max_range": self.max_range,
        }
