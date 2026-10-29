# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Calibration configuration (target, frames, topics, thresholds)."""

from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

ARUCO_DICTIONARIES = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11",
]


def _f(default: Any, group: str, label: str, help: str = "", **extra: Any):
    """Dataclass field with GUI metadata (group, label, tooltip, ranges, choices)."""
    meta = {"group": group, "label": label, "help": help}
    meta.update(extra)
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata=meta)
    return field(default=default, metadata=meta)


@dataclass
class CalibConfig:
    """All user-tunable settings. Lengths are in metres, angles in degrees."""

    # --- Target -------------------------------------------------------------
    aruco_dictionary: str = _f("DICT_4X4_50", "Target", "ArUco dictionary",
                               choices=ARUCO_DICTIONARIES)
    markers_x: int = _f(3, "Target", "Markers along X", min=1, max=50)
    markers_y: int = _f(4, "Target", "Markers along Y", min=1, max=50)
    marker_size: float = _f(0.04, "Target", "Marker side [m]",
                            "Used when width/height are 0.", min=0.0, max=2.0, decimals=5)
    marker_width: float = _f(0.0, "Target", "Marker width [m]",
                             "0 = use marker side.", min=0.0, max=2.0, decimals=5)
    marker_height: float = _f(0.0, "Target", "Marker height [m]",
                              "0 = use marker side.", min=0.0, max=2.0, decimals=5)
    marker_separation: float = _f(0.01, "Target", "Separation X [m]", min=0.0, max=1.0, decimals=5)
    marker_separation_y: float = _f(0.0, "Target", "Separation Y [m]",
                                    "0 = same as X.", min=0.0, max=1.0, decimals=5)
    first_marker_id: int = _f(0, "Target", "First marker id", min=0, max=10000)
    detect_inverted: bool = _f(False, "Target", "Detect inverted markers",
                               "Enable for white-on-black markers (detectInvertedMarker).")

    # --- Frames -------------------------------------------------------------
    base_frame: str = _f("base_link", "Frames", "Base frame")
    flange_frame: str = _f("tool0", "Frames", "Flange frame")
    camera_frame: str = _f("camera_link", "Frames", "Camera frame",
                           "Frame to report the result in. The solver works in the image "
                           "(optical) frame; the result is converted with TF if they differ.")

    # --- Topics -------------------------------------------------------------
    image_topic: str = _f("/camera/camera/color/image_raw", "Topics", "Image topic")
    camera_info_topic: str = _f("/camera/camera/color/camera_info", "Topics", "Camera info topic")
    depth_topic: str = _f("/camera/camera/aligned_depth_to_color/image_raw", "Topics",
                          "Depth topic (optional)",
                          "Depth aligned to the colour image. Empty = disabled.")
    joint_states_topic: str = _f("/joint_states", "Topics", "Joint states topic")

    # --- Stability ----------------------------------------------------------
    stability_window: float = _f(0.5, "Stability", "Window [s]", min=0.05, max=10.0, decimals=2)
    stability_pos_threshold: float = _f(0.001, "Stability", "Position RMS threshold [m]",
                                        min=0.0, max=1.0, decimals=5)
    stability_rot_threshold: float = _f(0.2, "Stability", "Rotation RMS threshold [deg]",
                                        min=0.0, max=45.0, decimals=3)

    # --- Motion -------------------------------------------------------------
    planning_group: str = _f("manipulator", "Motion", "Planning group")
    move_group_action: str = _f("/move_action", "Motion", "MoveGroup action")
    execute_action: str = _f("/execute_trajectory", "Motion", "ExecuteTrajectory action")
    velocity_scaling: float = _f(0.1, "Motion", "Velocity scaling", min=0.01, max=1.0, decimals=2)
    acceleration_scaling: float = _f(0.1, "Motion", "Acceleration scaling",
                                     min=0.01, max=1.0, decimals=2)
    planning_time: float = _f(5.0, "Motion", "Planning time [s]", min=0.1, max=60.0, decimals=1)
    joint_names: List[str] = _f([], "Motion", "Joint names",
                                "Comma separated. Empty = all joints in joint_states.")
    initial_flange_to_camera: List[float] = _f(
        [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 1.0], "Motion", "Initial flange->camera guess",
        "x y z qx qy qz qw of the optical frame in the flange frame. "
        "Used to generate poses before a calibration exists.")

    # --- Pose generation ----------------------------------------------------
    random_z_min: float = _f(0.30, "Pose generation", "Random distance min [m]",
                             min=0.01, max=5.0, decimals=3)
    random_z_max: float = _f(0.50, "Pose generation", "Random distance max [m]",
                             min=0.01, max=5.0, decimals=3)
    random_roll_min: float = _f(-30.0, "Pose generation", "Random roll min [deg]",
                                "Rotation about the optical axis.", min=-180.0, max=180.0)
    random_roll_max: float = _f(30.0, "Pose generation", "Random roll max [deg]",
                                min=-180.0, max=180.0)
    random_pitch_min: float = _f(-25.0, "Pose generation", "Random pitch min [deg]",
                                 "Orbit about the camera X axis, around the target centre.",
                                 min=-89.0, max=89.0)
    random_pitch_max: float = _f(25.0, "Pose generation", "Random pitch max [deg]",
                                 min=-89.0, max=89.0)
    random_yaw_min: float = _f(-25.0, "Pose generation", "Random yaw min [deg]",
                               "Orbit about the camera Y axis, around the target centre.",
                               min=-89.0, max=89.0)
    random_yaw_max: float = _f(25.0, "Pose generation", "Random yaw max [deg]",
                               min=-89.0, max=89.0)
    hemisphere_n: int = _f(20, "Pose generation", "Hemisphere poses", min=1, max=500)
    hemisphere_distance: float = _f(0.40, "Pose generation", "Hemisphere radius [m]",
                                    min=0.01, max=5.0, decimals=3)
    hemisphere_max_polar: float = _f(35.0, "Pose generation", "Hemisphere max polar [deg]",
                                     min=0.0, max=85.0)

    # --- Evaluation ---------------------------------------------------------
    outlier_sigma: float = _f(3.0, "Evaluation", "Outlier threshold [std]",
                              min=0.5, max=10.0, decimals=2)
    max_reprojection_error: float = _f(1.0, "Evaluation", "Max reprojection error [px]",
                                       "Samples above this are marked in the table.",
                                       min=0.0, max=100.0, decimals=2)
    depth_ransac_threshold: float = _f(0.004, "Evaluation", "Depth RANSAC threshold [m]",
                                       min=0.0001, max=0.1, decimals=4)
    depth_roi_shrink: float = _f(0.85, "Evaluation", "Depth ROI shrink factor",
                                 "Target polygon is scaled by this factor around its centroid.",
                                 min=0.1, max=1.0, decimals=2)

    # ------------------------------------------------------------------------
    def marker_dimensions(self) -> tuple:
        """Return (width, height, separation_x, separation_y) in metres."""
        w = self.marker_width if self.marker_width > 0 else self.marker_size
        h = self.marker_height if self.marker_height > 0 else self.marker_size
        sy = self.marker_separation_y if self.marker_separation_y > 0 else self.marker_separation
        return w, h, self.marker_separation, sy

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any] | None) -> "CalibConfig":
        """Build a config, ignoring unknown keys (with a warning)."""
        data = dict(data or {})
        known = {f.name: f for f in dataclasses.fields(cls)}
        kwargs = {}
        for key, value in data.items():
            if key not in known:
                warnings.warn(f"Unknown config key ignored: {key}")
                continue
            kwargs[key] = _coerce(known[key], value)
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "CalibConfig":
        with open(path) as stream:
            return cls.from_dict(yaml.safe_load(stream))


def _coerce(fld: dataclasses.Field, value: Any) -> Any:
    """Convert a YAML value to the type of the given field."""
    default = fld.default if fld.default is not dataclasses.MISSING else fld.default_factory()
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        items = list(value or [])
        if default and isinstance(default[0], float):
            return [float(v) for v in items]
        return [str(v) for v in items] if not default else items
    return str(value) if value is not None else ""
