# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""One calibration sample: robot pose, target pose and quality metrics."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .geometry import matrix_to_pose, pose_to_matrix


@dataclass
class Sample:
    """A single acquisition.

    Poses use ROS conventions: translation in metres, quaternion (x, y, z, w).
    ``robot_*`` is the flange pose in the base frame (base -> flange).
    ``target_*`` is the target pose in the camera optical frame (camera -> target).
    """

    index: int = -1
    timestamp: float = 0.0
    robot_translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    robot_rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    target_translation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    target_rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    joint_names: List[str] = field(default_factory=list)
    joint_values: List[float] = field(default_factory=list)
    image_path: str = ""
    decorated_image_path: str = ""
    # PnP metrics
    reprojection_error_mean: float = 0.0
    reprojection_error_max: float = 0.0
    n_markers: int = 0
    n_corners: int = 0
    target_distance: float = 0.0
    target_tilt: float = 0.0          # deg, angle between optical axis and target normal
    pnp_center_z: Optional[float] = None
    # Depth metrics (None when no depth is available)
    depth_plane_distance: Optional[float] = None
    depth_plane_normal: Optional[List[float]] = None
    depth_inliers: int = 0
    depth_center_z: Optional[float] = None
    # Flags
    excluded: bool = False
    forced: bool = False              # taken although the pose was not stable

    # -- transforms --------------------------------------------------------
    @property
    def T_base_flange(self) -> np.ndarray:
        return pose_to_matrix(self.robot_translation, self.robot_rotation)

    @T_base_flange.setter
    def T_base_flange(self, T: np.ndarray) -> None:
        self.robot_translation, self.robot_rotation = matrix_to_pose(T)

    @property
    def T_camera_target(self) -> np.ndarray:
        return pose_to_matrix(self.target_translation, self.target_rotation)

    @T_camera_target.setter
    def T_camera_target(self, T: np.ndarray) -> None:
        self.target_translation, self.target_rotation = matrix_to_pose(T)

    @property
    def depth_dz(self) -> Optional[float]:
        """PnP minus depth-plane Z at the target centre (m), if available."""
        if self.pnp_center_z is None or self.depth_center_z is None:
            return None
        return float(self.pnp_center_z - self.depth_center_z)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return _plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Sample":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _plain(value: Any) -> Any:
    """Convert numpy scalars/arrays to plain Python for YAML."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
