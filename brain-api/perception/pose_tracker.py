"""Frame-to-frame camera pose tracker using RGB-D frames."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Pose:
    """SE(3) pose represented by translation + quaternion."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "qw": self.qw,
            "qx": self.qx,
            "qy": self.qy,
            "qz": self.qz,
        }


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @classmethod
    def from_image_size(
        cls, width: int, height: int, hfov_deg: float = 60.0
    ) -> "CameraIntrinsics":
        """Estimate intrinsics from image size and horizontal FOV."""
        fx = width / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy)


class PoseTracker:
    """Estimate relative camera motion between consecutive RGB-D frames."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        max_features: int = 500,
        min_matches: int = 20,
    ) -> None:
        self.intrinsics = intrinsics
        self.max_features = max_features
        self.min_matches = min_matches
        self._orb = cv2.ORB_create(nfeatures=max_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._prev: dict[str, Any] | None = None
        self._absolute_pose = Pose()

    def reset(self) -> None:
        self._prev = None
        self._absolute_pose = Pose()

    def estimate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics | None = None,
    ) -> Pose:
        """Estimate pose of current frame relative to the first frame.

        Args:
            rgb: HxWx3 RGB image.
            depth: HxW depth map in meters.
            intrinsics: Optional camera intrinsics (uses stored if None).

        Returns:
            Absolute pose as Pose (translation in meters + quaternion).
        """
        K = (intrinsics or self.intrinsics)
        if K is None:
            K = CameraIntrinsics.from_image_size(rgb.shape[1], rgb.shape[0])
        K_mat = K.as_matrix()

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kpts, desc = self._orb.detectAndCompute(gray, None)
        if desc is None:
            logger.warning("No features detected, returning previous pose")
            return self._absolute_pose

        if self._prev is None:
            self._prev = {"kpts": kpts, "desc": desc, "depth": depth, "K": K_mat}
            return self._absolute_pose

        matches = self._bf.match(self._prev["desc"], desc)
        matches = sorted(matches, key=lambda m: m.distance)
        matches = matches[: min(len(matches), 100)]

        if len(matches) < self.min_matches:
            logger.warning(
                "Too few matches (%d < %d), returning previous pose", len(matches), self.min_matches
            )
            self._prev = {"kpts": kpts, "desc": desc, "depth": depth, "K": K_mat}
            return self._absolute_pose

        prev_pts = np.float32([self._prev["kpts"][m.queryIdx].pt for m in matches])
        curr_pts = np.float32([kpts[m.trainIdx].pt for m in matches])

        # Use depth at previous frame to get 3D points.
        prev_depth = self._prev["depth"]
        prev_3d = self._unproject(prev_pts, prev_depth, K_mat)

        # Filter valid depths.
        valid = (prev_3d[:, 2] > 0.1) & np.isfinite(prev_3d[:, 2])
        if valid.sum() < self.min_matches:
            logger.warning("Too few valid depths for PnP")
            self._prev = {"kpts": kpts, "desc": desc, "depth": depth, "K": K_mat}
            return self._absolute_pose

        object_points = prev_3d[valid]
        image_points = curr_pts[valid]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            K_mat,
            None,
            reprojectionError=8.0,
            iterationsCount=200,
        )

        if not success or inliers is None or len(inliers) < 8:
            logger.warning("PnP failed, returning previous pose")
            self._prev = {"kpts": kpts, "desc": desc, "depth": depth, "K": K_mat}
            return self._absolute_pose

        R, _ = cv2.Rodrigues(rvec)
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3] = tvec.squeeze()

        T_abs = self._pose_to_matrix(self._absolute_pose) @ T_rel
        self._absolute_pose = self._matrix_to_pose(T_abs)

        self._prev = {"kpts": kpts, "desc": desc, "depth": depth, "K": K_mat}
        return self._absolute_pose

    @staticmethod
    def _unproject(pts_2d: np.ndarray, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        u = np.clip(pts_2d[:, 0].astype(int), 0, depth.shape[1] - 1)
        v = np.clip(pts_2d[:, 1].astype(int), 0, depth.shape[0] - 1)
        z = depth[v, u]
        x = (pts_2d[:, 0] - cx) * z / fx
        y = (pts_2d[:, 1] - cy) * z / fy
        return np.stack([x, y, z], axis=1)

    @staticmethod
    def _pose_to_matrix(pose: Pose) -> np.ndarray:
        T = np.eye(4)
        T[:3, 3] = [pose.x, pose.y, pose.z]
        q = np.array([pose.qw, pose.qx, pose.qy, pose.qz])
        q = q / (np.linalg.norm(q) + 1e-8)
        T[:3, :3] = PoseTracker._quat_to_rot(q)
        return T

    @staticmethod
    def _matrix_to_pose(T: np.ndarray) -> Pose:
        R = T[:3, :3]
        t = T[:3, 3]
        q = PoseTracker._rot_to_quat(R)
        return Pose(x=float(t[0]), y=float(t[1]), z=float(t[2]), qw=q[0], qx=q[1], qy=q[2], qz=q[3])

    @staticmethod
    def _quat_to_rot(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return np.array([w, x, y, z])
