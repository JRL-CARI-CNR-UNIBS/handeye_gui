# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""ArUco grid detection, PnP, depth-plane check and image decoration.

Pure OpenCV/numpy: used by the ROS CameraManager and testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from .config import CalibConfig
from .geometry import PlaneFit, fit_plane_ransac


@dataclass
class Detection:
    corners: list                   # list of (1, 4, 2) float32 arrays
    ids: Optional[np.ndarray]       # (N, 1) int32 or None

    @property
    def n_markers(self) -> int:
        return 0 if self.ids is None else len(self.ids)


@dataclass
class PnpResult:
    rvec: np.ndarray
    tvec: np.ndarray
    T: np.ndarray                   # target pose in the camera frame (camera -> target)
    reprojection_error_mean: float  # px
    reprojection_error_max: float   # px
    n_markers: int
    n_corners: int
    distance: float                 # m, camera origin to target centre
    tilt: float                     # deg, optical axis vs target normal
    center: np.ndarray              # target centre in the camera frame
    image_points: np.ndarray        # (N, 2) detected corners used by PnP


@dataclass
class DepthResult:
    plane: PlaneFit
    center_z: Optional[float]       # depth-plane Z along the ray through the PnP centre
    n_points: int

    @property
    def normal(self) -> list:
        return [float(v) for v in self.plane.normal]


class BoardModel:
    """Grid of ArUco markers.

    Marker (column c, row r) has id ``first_marker_id + r * markers_x + c``.
    Its top-left corner is at (c * (w + sx), r * (h + sy), 0) in the target
    frame: same layout as ``cv2.aruco.GridBoard`` but with independent marker
    width/height and separations.
    """

    def __init__(self, config: CalibConfig):
        self.config = config
        w, h, sx, sy = config.marker_dimensions()
        nx, ny = config.markers_x, config.markers_y
        if not hasattr(cv2.aruco, config.aruco_dictionary):
            raise ValueError(f"unknown ArUco dictionary {config.aruco_dictionary}")
        self.dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.aruco_dictionary))
        n_ids = self.dictionary.bytesList.shape[0]
        last_id = config.first_marker_id + nx * ny - 1
        if last_id >= n_ids:
            raise ValueError(f"{config.aruco_dictionary} has ids 0..{n_ids - 1}, but the grid "
                             f"needs ids {config.first_marker_id}..{last_id}")
        obj_points, ids = [], []
        for r in range(ny):
            for c in range(nx):
                x0, y0 = c * (w + sx), r * (h + sy)
                obj_points.append(np.array([[x0, y0, 0], [x0 + w, y0, 0],
                                            [x0 + w, y0 + h, 0], [x0, y0 + h, 0]], np.float32))
                ids.append(config.first_marker_id + r * nx + c)
        self.ids = np.array(ids, dtype=np.int32)
        self.board = cv2.aruco.Board(obj_points, self.dictionary, self.ids)
        self.size = (nx * w + (nx - 1) * sx, ny * h + (ny - 1) * sy)
        self.center = np.array([self.size[0] / 2.0, self.size[1] / 2.0, 0.0])
        params = cv2.aruco.DetectorParameters()
        params.detectInvertedMarker = bool(config.detect_inverted)
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    # ------------------------------------------------------------------
    def detect(self, gray: np.ndarray) -> Detection:
        """Detect the markers that belong to this board."""
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return Detection([], None)
        keep = [i for i, mid in enumerate(ids.ravel()) if mid in set(self.ids.tolist())]
        if not keep:
            return Detection([], None)
        return Detection([corners[i] for i in keep], ids[keep].reshape(-1, 1))

    def estimate_pose(self, detection: Detection, K: np.ndarray, D: np.ndarray) -> Optional[PnpResult]:
        """PnP on all detected corners (IPPE initialisation + LM refinement)."""
        if detection.n_markers == 0:
            return None
        obj, img = self.board.matchImagePoints(detection.corners, detection.ids)
        if obj is None or len(obj) < 4:
            return None
        obj = obj.reshape(-1, 3).astype(np.float64)
        img = img.reshape(-1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, D, rvec, tvec)
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = np.linalg.norm(proj.reshape(-1, 2) - img, axis=1)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.ravel()
        center = R @ self.center + tvec.ravel()
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        return PnpResult(rvec, tvec, T, float(err.mean()), float(err.max()),
                         detection.n_markers, len(obj), float(np.linalg.norm(center)), tilt,
                         center, img)

    # ------------------------------------------------------------------
    def draw(self, image_bgr: np.ndarray, detection: Detection, pnp: Optional[PnpResult],
             K: Optional[np.ndarray], D: Optional[np.ndarray],
             depth: Optional[DepthResult] = None, extra_lines: Iterable[str] = ()) -> np.ndarray:
        """Copy of the image with markers, axes and metrics drawn on it."""
        out = image_bgr.copy()
        if detection.n_markers:
            cv2.aruco.drawDetectedMarkers(out, detection.corners, detection.ids)
        lines = []
        if pnp is not None and K is not None:
            cv2.drawFrameAxes(out, K, D, pnp.rvec, pnp.tvec, 0.5 * min(self.size), 2)
            lines += [f"markers {pnp.n_markers}  corners {pnp.n_corners}",
                      f"reproj mean {pnp.reprojection_error_mean:.2f} px  "
                      f"max {pnp.reprojection_error_max:.2f} px",
                      f"distance {pnp.distance:.3f} m  tilt {pnp.tilt:.1f} deg"]
            if depth is not None and depth.center_z is not None:
                dz = pnp.center[2] - depth.center_z
                lines.append(f"depth plane {depth.plane.distance:.3f} m  dZ {1e3 * dz:+.1f} mm  "
                             f"inliers {depth.plane.inliers}/{depth.n_points}")
        elif detection.n_markers:
            lines.append(f"markers {detection.n_markers} (no pose)")
        else:
            lines.append("no target")
        lines += list(extra_lines)
        scale = max(0.5, out.shape[1] / 1280.0)
        y = int(30 * scale)
        for text in lines:
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 0, 0),
                        int(4 * scale), cv2.LINE_AA)
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, (0, 255, 255),
                        max(1, int(2 * scale)), cv2.LINE_AA)
            y += int(28 * scale)
        return out


def depth_plane(depth_m: np.ndarray, K: np.ndarray, polygon_px: np.ndarray, shrink: float,
                threshold: float, pnp_center: Optional[Sequence[float]] = None,
                max_points: int = 4000, rng: Optional[np.random.Generator] = None
                ) -> Optional[DepthResult]:
    """Fit a plane to the depth pixels inside the (shrunk) target polygon.

    The depth image must be aligned to the colour image, so the colour
    intrinsics ``K`` apply to it.
    """
    pts = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2)
    c = hull.mean(axis=0)
    hull = c + shrink * (hull - c)
    mask = np.zeros(depth_m.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(hull).astype(np.int32)], 1)
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0)
    vs, us = np.nonzero(valid)
    if len(us) < 10:
        return None
    rng = rng or np.random.default_rng(0)
    if len(us) > max_points:
        sel = rng.choice(len(us), max_points, replace=False)
        us, vs = us[sel], vs[sel]
    z = depth_m[vs, us].astype(np.float64)
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    plane = fit_plane_ransac(np.column_stack([x, y, z]), threshold, rng=rng)
    if plane is None:
        return None
    center_z = plane.z_along_ray(pnp_center) if pnp_center is not None else None
    return DepthResult(plane, center_z, len(us))


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """16UC1 in millimetres or 32FC1 in metres -> float32 metres."""
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) * 1e-3
    return depth.astype(np.float32)
