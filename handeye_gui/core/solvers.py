# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around ``cv2.calibrateHandEye``, eye-in-hand and eye-to-hand.

Both modes use the same solver. Only the robot pose fed to it changes:

* eye-in-hand (camera on the flange): ``A_i = T_base_flange``, the result is
  ``T_flange_optical`` and the target is fixed in the base frame;
* eye-to-hand (camera fixed, target on the flange): ``A_i = T_flange_base``,
  the result is ``T_base_optical`` and the target is fixed in the flange frame.

In both cases ``A_i @ X @ T_camera_target_i`` is constant: the target pose in
its fixed frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Sequence

import cv2
import numpy as np

from .geometry import invert, matrix_to_pose
from .sample import Sample


class Method(Enum):
    """The five hand-eye solvers available in OpenCV."""

    TSAI = cv2.CALIB_HAND_EYE_TSAI
    PARK = cv2.CALIB_HAND_EYE_PARK
    HORAUD = cv2.CALIB_HAND_EYE_HORAUD
    ANDREFF = cv2.CALIB_HAND_EYE_ANDREFF
    DANIILIDIS = cv2.CALIB_HAND_EYE_DANIILIDIS

    @classmethod
    def parse(cls, value: "Method | str") -> "Method":
        if isinstance(value, Method):
            return value
        return cls[str(value).strip().upper()]


MIN_SAMPLES = 3

EYE_IN_HAND = "eye_in_hand"
EYE_TO_HAND = "eye_to_hand"
MODES = (EYE_IN_HAND, EYE_TO_HAND)


def check_mode(mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"unknown calibration type {mode!r}, expected one of {MODES}")
    return mode


def result_parent(mode: str) -> str:
    """Name of the frame the hand-eye result is expressed in ("flange" or "base")."""
    return "flange" if check_mode(mode) == EYE_IN_HAND else "base"


def target_parent(mode: str) -> str:
    """Name of the frame the target is fixed to ("base" or "flange")."""
    return "base" if check_mode(mode) == EYE_IN_HAND else "flange"


def robot_pose(sample: Sample, mode: str = EYE_IN_HAND) -> np.ndarray:
    """Robot transform that enters the chain: T_base_flange, or its inverse for eye-to-hand."""
    T = sample.T_base_flange
    return T if check_mode(mode) == EYE_IN_HAND else invert(T)


@dataclass
class HandEyeResult:
    """Calibration result. ``T`` is the camera (optical) pose in the flange frame
    (eye-in-hand) or in the base frame (eye-to-hand)."""

    T: np.ndarray
    method: Method
    n_samples: int
    mode: str = EYE_IN_HAND

    @property
    def translation(self) -> list:
        return matrix_to_pose(self.T)[0]

    @property
    def quaternion(self) -> list:
        return matrix_to_pose(self.T)[1]

    def to_dict(self) -> Dict[str, Any]:
        return {"method": self.method.name, "mode": self.mode, "n_samples": self.n_samples,
                "translation": self.translation, "quaternion_xyzw": self.quaternion,
                "matrix": self.T.tolist()}


def solve_matrices(T_base_flange: Sequence[np.ndarray], T_camera_target: Sequence[np.ndarray],
                   method: Method | str = Method.TSAI) -> np.ndarray:
    """Solve AX = XB from paired transforms; returns T_flange_camera (4x4).

    For eye-to-hand pass the inverted robot poses (T_flange_base): the result
    is then T_base_camera.
    """
    method = Method.parse(method)
    if len(T_base_flange) != len(T_camera_target):
        raise ValueError("pose lists have different lengths")
    if len(T_base_flange) < MIN_SAMPLES:
        raise ValueError(f"at least {MIN_SAMPLES} samples are needed, got {len(T_base_flange)}")
    R_g2b = [T[:3, :3] for T in T_base_flange]
    t_g2b = [T[:3, 3].reshape(3, 1) for T in T_base_flange]
    R_t2c = [T[:3, :3] for T in T_camera_target]
    t_t2c = [T[:3, 3].reshape(3, 1) for T in T_camera_target]
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method.value)
    X = np.eye(4)
    X[:3, :3] = R
    X[:3, 3] = np.asarray(t).ravel()
    if not np.all(np.isfinite(X)):
        raise RuntimeError(f"{method.name} returned a non-finite solution")
    return X


def solve(samples: Sequence[Sample], method: Method | str = Method.TSAI,
          mode: str = EYE_IN_HAND) -> HandEyeResult:
    """Calibrate from samples (the caller decides which samples to pass)."""
    method = Method.parse(method)
    X = solve_matrices([robot_pose(s, mode) for s in samples],
                       [s.T_camera_target for s in samples], method)
    return HandEyeResult(T=X, method=method, n_samples=len(samples), mode=mode)
