# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Candidate camera poses around the target (random and Fibonacci hemisphere).

Poses are generated as the camera pose relative to the target
(T_target_camera); ``flange_goal`` turns them into a flange goal for either
calibration type.

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
from .solvers import EYE_IN_HAND, EYE_TO_HAND, check_mode

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


def flange_goal(T_fixed_target: np.ndarray, T_target_camera: np.ndarray,
                X: np.ndarray, mode: str = EYE_IN_HAND) -> np.ndarray:
    """Flange pose in the base frame that puts the camera at ``T_target_camera``.

    Eye-in-hand: ``T_fixed_target`` is the target in base and ``X`` is
    T_flange_camera; the camera moves around the fixed target.
    Eye-to-hand: ``T_fixed_target`` is the target in the flange frame and
    ``X`` is T_base_camera; the target moves in front of the fixed camera so
    that their relative pose is the same.
    """
    if check_mode(mode) == EYE_IN_HAND:
        return T_fixed_target @ T_target_camera @ invert(X)
    return X @ invert(T_target_camera) @ invert(T_fixed_target)


def hand_eye_guess(config: CalibConfig, result_T: np.ndarray | None = None) -> np.ndarray:
    """Current best hand-eye transform: the calibration if any, else the config guess.

    The guess is ``initial_flange_to_camera`` for eye-in-hand and
    ``initial_base_to_camera`` for eye-to-hand.
    """
    if result_T is not None:
        return np.asarray(result_T)
    if config.calibration_type == EYE_TO_HAND:
        return pose7_to_matrix(config.initial_base_to_camera)
    return pose7_to_matrix(config.initial_flange_to_camera)


def fixed_target_guess(samples: Sequence[Sample], X: np.ndarray,
                       mode: str = EYE_IN_HAND) -> np.ndarray:
    """Target pose in its fixed frame (base or flange) from the samples and X."""
    return estimate_target_pose(samples, X, mode)


# Backward-compatible names (eye-in-hand).
flange_camera_guess = hand_eye_guess


def target_in_base_guess(samples: Sequence[Sample], T_flange_camera: np.ndarray) -> np.ndarray:
    return fixed_target_guess(samples, T_flange_camera, EYE_IN_HAND)
