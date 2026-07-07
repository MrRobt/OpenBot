"""Simple local navigation using 3D obstacle map."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


@dataclass
class Obstacle:
    distance_m: float
    angle_deg: float
    width_m: float
    label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_m": self.distance_m,
            "angle_deg": self.angle_deg,
            "width_m": self.width_m,
            "label": self.label,
        }


class LocalNavigator:
    """Compute navigation hints from obstacle point cloud."""

    HINT_CLEAR = "CLEAR_AHEAD"
    HINT_LEFT = "FREE_LEFT"
    HINT_RIGHT = "FREE_RIGHT"
    HINT_BLOCKED = "BLOCKED_FRONT"

    def __init__(
        self,
        min_distance_m: float = 0.3,
        max_distance_m: float = 2.0,
        fov_deg: float = 90.0,
        angular_resolution_deg: float = 10.0,
        safety_margin_m: float = 0.15,
    ) -> None:
        self.min_distance_m = min_distance_m
        self.max_distance_m = max_distance_m
        self.fov_deg = fov_deg
        self.angular_resolution_deg = angular_resolution_deg
        self.safety_margin_m = safety_margin_m

    def detect_obstacles(
        self, obstacle_points: np.ndarray
    ) -> list[Obstacle]:
        """Cluster obstacle points and summarize each cluster."""
        if len(obstacle_points) == 0:
            return []

        # Use x,y only for clustering.
        xy = obstacle_points[:, :2]
        clustering = DBSCAN(eps=0.15, min_samples=5).fit(xy)
        labels = clustering.labels_

        obstacles: list[Obstacle] = []
        for label in set(labels):
            if label == -1:
                continue
            cluster = obstacle_points[labels == label]
            center = cluster.mean(axis=0)
            distance_m = float(np.linalg.norm(center[:2]))
            angle_deg = float(np.degrees(np.arctan2(center[1], center[0])))

            # Width as max spread perpendicular to ray.
            angles = np.degrees(np.arctan2(cluster[:, 1], cluster[:, 0]))
            width_m = float(np.linalg.norm(cluster[:, :2].max(axis=0) - cluster[:, :2].min(axis=0)))

            obstacles.append(
                Obstacle(
                    distance_m=distance_m,
                    angle_deg=angle_deg,
                    width_m=width_m,
                    label="obstacle",
                )
            )

        obstacles.sort(key=lambda o: o.distance_m)
        return obstacles

    def compute_hint(
        self,
        obstacle_points: np.ndarray,
        target_angle_deg: float = 0.0,
    ) -> tuple[str, list[Obstacle]]:
        """Return a navigation hint and the list of nearby obstacles.

        The hint is chosen based on obstacle density in front of the robot.
        If a target angle is provided, the hint favors the target direction when
        it is clear.
        """
        obstacles = self.detect_obstacles(obstacle_points)

        # Filter to obstacles within the safety corridor.
        nearby = [
            o
            for o in obstacles
            if o.distance_m < self.max_distance_m
            and abs(o.angle_deg) < self.fov_deg / 2
        ]

        if not nearby:
            return self.HINT_CLEAR, obstacles

        # Build angular cost map.
        angles = np.arange(
            -self.fov_deg / 2,
            self.fov_deg / 2 + self.angular_resolution_deg,
            self.angular_resolution_deg,
        )
        costs = np.zeros_like(angles, dtype=float)

        for obs in nearby:
            # Cost decreases with distance and increases with width.
            cost = (1.0 / max(obs.distance_m, self.min_distance_m)) * obs.width_m
            angle_diff = np.abs(angles - obs.angle_deg)
            costs += cost * np.exp(-(angle_diff ** 2) / (2 * 15.0 ** 2))

        # Add bias toward target direction if it is reasonably clear.
        target_idx = np.argmin(np.abs(angles - target_angle_deg))
        if costs[target_idx] < 1.0 / self.max_distance_m:
            if target_angle_deg < -10:
                return self.HINT_LEFT, obstacles
            if target_angle_deg > 10:
                return self.HINT_RIGHT, obstacles
            return self.HINT_CLEAR, obstacles

        # Choose the direction with lowest cost.
        left_angles = angles < 0
        right_angles = angles > 0
        left_cost = costs[left_angles].min() if left_angles.any() else float("inf")
        right_cost = costs[right_angles].min() if right_angles.any() else float("inf")

        if left_cost < right_cost:
            return self.HINT_LEFT, obstacles
        if right_cost < left_cost:
            return self.HINT_RIGHT, obstacles
        return self.HINT_BLOCKED, obstacles
