# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Candidate camera poses around the target (random and Fibonacci hemisphere).

Target frame (OpenCV board convention): X right, Y down, Z into the board.
A camera looking at the board frontally has the same axes as the target and
sits on the -Z side. All camera poses keep the target centre on the optical
axis.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from .config import CalibConfig
from .geometry import invert, pose7_to_matrix
from .evaluation import estimate_target_pose
from .sample import Sample

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def _rot(axis: str, angle_rad: float) -> np.ndarray:
    return Rotation.from_euler(axis, angle_rad).as_matrix()


def orbit_camera_pose(center: Sequence[float], distance: float, roll_deg: float = 0.0,
                      pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    """Camera pose in the target frame (T_target_camera).

    Starting from the frontal view at ``distance`` from ``center``, the camera
    orbits around the centre by ``yaw`` (about Y) and ``pitch`` (about X), then
    rotates by ``roll`` about its optical axis.
    """
    R_orbit = _rot("y", np.radians(yaw_deg)) @ _rot("x", np.radians(pitch_deg))
    T = np.eye(4)
    T[:3, :3] = R_orbit @ _rot("z", np.radians(roll_deg))
    T[:3, 3] = np.asarray(center, dtype=float) - distance * R_orbit[:, 2]
    return T


def look_at(position: Sequence[float], center: Sequence[float], roll_deg: float = 0.0) -> np.ndarray:
    """Camera pose in target frame at ``position`` looking at ``center``.

    The image Y axis is kept as close as possible to the target Y axis.
    """
    p = np.asarray(position, dtype=float)
    z = np.asarray(center, dtype=float) - p
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross([1.0, 0.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z]) @ _rot("z", np.radians(roll_deg))
    T[:3, 3] = p
    return T


def random_camera_pose(rng: np.random.Generator, center: Sequence[float],
                       config: CalibConfig) -> np.ndarray:
    """Random T_target_camera with distance and roll/pitch/yaw within the config bounds."""
    u = rng.uniform
    return orbit_camera_pose(
        center,
        u(config.random_z_min, max(config.random_z_min, config.random_z_max)),
        u(config.random_roll_min, max(config.random_roll_min, config.random_roll_max)),
        u(config.random_pitch_min, max(config.random_pitch_min, config.random_pitch_max)),
        u(config.random_yaw_min, max(config.random_yaw_min, config.random_yaw_max)))


def fibonacci_hemisphere(n: int, distance: float, max_polar_deg: float,
                         center: Sequence[float], roll_deg: float = 0.0,
                         alternate_roll_deg: float = 0.0) -> List[np.ndarray]:
    """``n`` camera poses (T_target_camera) on a spherical cap, Fibonacci spiral.

    Args:
        max_polar_deg: aperture of the cap measured from the frontal view.
        alternate_roll_deg: roll added with alternating sign, to vary the
            rotation about the optical axis (useful for hand-eye conditioning).
    """
    cos_max = np.cos(np.radians(max_polar_deg))
    c = np.asarray(center, dtype=float)
    poses = []
    for i in range(n):
        cos_p = 1.0 - (1.0 - cos_max) * (i + 0.5) / n
        sin_p = np.sqrt(max(0.0, 1.0 - cos_p ** 2))
        az = i * GOLDEN_ANGLE
        direction = np.array([sin_p * np.cos(az), sin_p * np.sin(az), -cos_p])
        roll = roll_deg + (alternate_roll_deg if i % 2 else -alternate_roll_deg)
        poses.append(look_at(c + distance * direction, c, roll))
    return poses


def flange_goal(T_base_target: np.ndarray, T_target_camera: np.ndarray,
                T_flange_camera: np.ndarray) -> np.ndarray:
    """Flange pose in the base frame that puts the camera at ``T_target_camera``."""
    return T_base_target @ T_target_camera @ invert(T_flange_camera)


def flange_camera_guess(config: CalibConfig, result_T: np.ndarray | None = None) -> np.ndarray:
    """Current best flange -> optical transform: the calibration if any, else the config guess."""
    if result_T is not None:
        return np.asarray(result_T)
    return pose7_to_matrix(config.initial_flange_to_camera)


def target_in_base_guess(samples: Sequence[Sample], T_flange_camera: np.ndarray) -> np.ndarray:
    """Target pose in base from the acquired samples and a flange -> camera transform."""
    return estimate_target_pose(samples, T_flange_camera)
