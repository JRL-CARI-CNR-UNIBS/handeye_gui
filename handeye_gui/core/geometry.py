# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Small geometry helpers: homogeneous transforms, averaging, plane fitting.

Quaternions are always in ROS order (x, y, z, w).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """4x4 matrix from a translation and an (x, y, z, w) quaternion."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quaternion, dtype=float)).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=float)
    return T


def matrix_to_pose(T: np.ndarray) -> Tuple[list, list]:
    """(translation, quaternion xyzw) lists from a 4x4 matrix."""
    q = Rotation.from_matrix(T[:3, :3]).as_quat()
    if q[3] < 0:  # canonical sign, easier to read in files
        q = -q
    return [float(v) for v in T[:3, 3]], [float(v) for v in q]


def pose7_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """4x4 matrix from [x, y, z, qx, qy, qz, qw]."""
    return pose_to_matrix(pose[:3], pose[3:7])


def matrix_to_pose7(T: np.ndarray) -> list:
    t, q = matrix_to_pose(T)
    return t + q


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform."""
    Ti = np.eye(4)
    R = T[:3, :3]
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ T[:3, 3]
    return Ti


def rotation_angle(R: np.ndarray) -> float:
    """Rotation angle (rad) of a 3x3 rotation matrix."""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def average_quaternions(quaternions: Iterable[Sequence[float]]) -> np.ndarray:
    """Arithmetic mean of quaternions after sign alignment, renormalised.

    Accurate when the quaternions are close to each other, which is the case
    for a stability window or a calibration distribution.
    """
    q = np.asarray(list(quaternions), dtype=float)
    if q.ndim != 2 or len(q) == 0:
        raise ValueError("need at least one quaternion")
    signs = np.sign(q @ q[0])
    signs[signs == 0] = 1.0
    mean = (q * signs[:, None]).mean(axis=0)
    return mean / np.linalg.norm(mean)


def average_poses(transforms: Iterable[np.ndarray]) -> np.ndarray:
    """Mean pose: mean translation and mean quaternion."""
    Ts = list(transforms)
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    q = average_quaternions(Rotation.from_matrix(T[:3, :3]).as_quat() for T in Ts)
    return pose_to_matrix(t, q)


def pose_dispersion(transforms: Sequence[np.ndarray]) -> Tuple[float, float]:
    """RMS position deviation (m) and RMS angular deviation (rad) around the mean pose."""
    if len(transforms) < 2:
        return 0.0, 0.0
    mean = average_poses(transforms)
    pos = np.array([T[:3, 3] - mean[:3, 3] for T in transforms])
    ang = np.array([rotation_angle(mean[:3, :3].T @ T[:3, :3]) for T in transforms])
    return float(np.sqrt((pos ** 2).sum(axis=1).mean())), float(np.sqrt((ang ** 2).mean()))


def rotvec_delta(R_ref: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotation vector of R_ref^T R (small-angle parameterisation around R_ref)."""
    return Rotation.from_matrix(R_ref.T @ R).as_rotvec()


# --------------------------------------------------------------------------
# Plane fitting
# --------------------------------------------------------------------------
@dataclass
class PlaneFit:
    """Plane n . p + d = 0, with n pointing away from the camera (n_z > 0)."""

    normal: np.ndarray
    offset: float
    inliers: int
    mask: np.ndarray

    @property
    def distance(self) -> float:
        """Distance of the plane from the camera origin (m)."""
        return float(abs(self.offset))

    def z_along_ray(self, direction: Sequence[float]) -> Optional[float]:
        """Z of the intersection between the plane and a ray from the origin."""
        d = np.asarray(direction, dtype=float)
        den = float(self.normal @ d)
        if abs(den) < 1e-9:
            return None
        return float((-self.offset / den) * d[2])


def _plane_from_points(points: np.ndarray) -> Tuple[np.ndarray, float]:
    centroid = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    return n, float(-n @ centroid)


def fit_plane_ransac(points: np.ndarray, threshold: float, iterations: int = 200,
                     rng: Optional[np.random.Generator] = None,
                     min_inliers: int = 10) -> Optional[PlaneFit]:
    """Robust plane fit (vectorised RANSAC + least-squares refinement).

    Args:
        points: (N, 3) array in the camera frame.
        threshold: inlier distance (m).
        iterations: number of random triplets.
    Returns:
        PlaneFit or None if there are not enough points.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < max(3, min_inliers):
        return None
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(pts), size=(iterations, 3))
    p1, p2, p3 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    normals = np.cross(p2 - p1, p3 - p1)
    norms = np.linalg.norm(normals, axis=1)
    valid = norms > 1e-12
    if not valid.any():
        return None
    normals = normals[valid] / norms[valid, None]
    offsets = -(normals * p1[valid]).sum(axis=1)
    dist = np.abs(pts @ normals.T + offsets)          # (N, iterations)
    counts = (dist < threshold).sum(axis=0)
    best = int(np.argmax(counts))
    mask = dist[:, best] < threshold
    if mask.sum() < min_inliers:
        return None
    # Two refinement rounds on the inliers.
    for _ in range(2):
        n, d = _plane_from_points(pts[mask])
        mask = np.abs(pts @ n + d) < threshold
        if mask.sum() < min_inliers:
            return None
    n, d = _plane_from_points(pts[mask])
    return PlaneFit(normal=n, offset=d, inliers=int(mask.sum()), mask=mask)
