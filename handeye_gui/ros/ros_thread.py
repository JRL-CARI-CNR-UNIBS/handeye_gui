# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""rclpy in a dedicated QThread, bridged to the GUI with Qt signals.

* ROS callbacks run in an executor spun by this thread. They never touch
  widgets: they emit signals, which Qt queues to the GUI thread.
* Image processing runs in its own worker thread (see CameraManager), so all
  executor callbacks are short. A SingleThreadedExecutor is used on purpose:
  measured with 1280x720 RGB-D at 15 Hz, rclpy's MultiThreadedExecutor held
  the GIL so much that frames took ~0.6 s instead of ~35 ms and the TF timer
  dropped from 30 Hz to ~4 Hz.
* GUI requests are plain method calls (slots) executed in the GUI thread.
  They only read thread-safe state or send asynchronous action goals.
* Every ROS callback and every slot is wrapped in try/except: an exception
  becomes an ``error`` signal instead of killing the application.
"""

from __future__ import annotations

import functools
import time
import traceback
from typing import Callable, List, Optional, Sequence

import numpy as np
import rclpy
from PySide6.QtCore import QThread, Signal, Slot
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time

from ..core.config import CalibConfig
from ..core.geometry import matrix_to_pose7
from ..core.sample import Sample
from .camera_manager import CameraManager
from .motion_manager import Candidate, MotionManager
from .tf_manager import TfManager

MAX_FRAME_AGE = 1.0   # s, a sample needs a camera frame newer than this


class RosThread(QThread):
    """Owns the ROS node and the three managers."""

    image_ready = Signal(object, object)        # RGB np.ndarray, info dict
    stability_changed = Signal(bool, object)    # stable, info dict
    motion_done = Signal(str, bool, str)        # "plan"/"execute", success, message
    sample_ready = Signal(object, object, object, object)  # Sample, BGR image, BGR decorated, meta
    status = Signal(str)
    error = Signal(str)
    started_ok = Signal()

    def __init__(self, config: CalibConfig, args: Optional[List[str]] = None, parent=None):
        super().__init__(parent)
        self._config = config
        self._args = args
        self._executor: Optional[SingleThreadedExecutor] = None
        self.node = None
        self.camera: Optional[CameraManager] = None
        self.tf: Optional[TfManager] = None
        self.motion: Optional[MotionManager] = None

    # -- helpers -----------------------------------------------------------
    def guard(self, fn: Callable) -> Callable:
        """Wrap a callback so that exceptions are reported, not raised."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - deliberately broad
                name = getattr(fn, "__qualname__", repr(fn))
                self.error.emit(f"{name}: {exc}")
                if self.node is not None:
                    self.node.get_logger().error(f"{name}: {exc}\n{traceback.format_exc()}")
        return wrapper

    def _require_ready(self) -> None:
        if self.node is None:
            raise RuntimeError("ROS is not running yet")

    # -- thread ------------------------------------------------------------
    def run(self) -> None:
        owns_context = False
        executor = None
        try:
            if not rclpy.ok():
                rclpy.init(args=self._args)
                owns_context = True
            self.node = rclpy.create_node("handeye_gui")
            self.tf = TfManager(self.node, self._config, self.stability_changed.emit, self.guard)
            self.camera = CameraManager(self.node, self._config, self.image_ready.emit,
                                        self.tf.push_target, self.guard)
            self.motion = MotionManager(self.node, self._config, self.motion_done.emit, self.guard)
            executor = SingleThreadedExecutor()
            executor.add_node(self.node)
            self._executor = executor
            self.status.emit("ROS node 'handeye_gui' started")
            self.started_ok.emit()
            executor.spin()               # returns when stop() shuts the executor down
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"ROS thread stopped: {exc}\n{traceback.format_exc()}")
        finally:
            if self.camera is not None:
                self.camera.close()
            if executor is not None:
                if self.node is not None:
                    executor.remove_node(self.node)
                executor.shutdown(timeout_sec=1.0)
                self._executor = None
            if self.node is not None:
                self.node.destroy_node()
                self.node = None
            if owns_context and rclpy.ok():
                rclpy.shutdown()

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=0.0)
        self.wait(3000)

    # -- slots (called from the GUI thread) --------------------------------
    def _slot(fn):  # noqa: N805 - decorator used inside the class body
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            try:
                self._require_ready()
                return fn(self, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self.error.emit(f"{fn.__name__}: {exc}")
        return wrapper

    @Slot(object)
    def apply_config(self, config: CalibConfig) -> None:
        self._config = config          # used by run() if ROS is not started yet
        if self.node is not None:
            self._reconfigure(config)

    @_slot
    def _reconfigure(self, config: CalibConfig) -> None:
        self.tf.configure(config)
        self.camera.configure(config)
        self.motion.configure(config)
        self.status.emit("configuration applied")

    @Slot(bool)
    @_slot
    def request_sample(self, forced: bool = False) -> None:
        """Build a Sample from the latest frame, TF and joints; emits ``sample_ready``."""
        frame = self.camera.latest()
        if frame is None:
            raise RuntimeError(f"no image received on {self._config.image_topic}")
        if time.monotonic() - frame.received > MAX_FRAME_AGE:
            raise RuntimeError("the last camera frame is too old")
        if frame.pnp is None:
            raise RuntimeError("target not detected in the last frame")
        T_bf = self.tf.flange_pose(frame.stamp)
        if T_bf is None:
            raise RuntimeError(f"no TF {self._config.base_frame} -> {self._config.flange_frame}")
        pnp, depth = frame.pnp, frame.depth
        names, values = self.motion.current_joints()
        s = Sample(timestamp=frame.stamp.nanoseconds * 1e-9, joint_names=names, joint_values=values,
                   reprojection_error_mean=pnp.reprojection_error_mean,
                   reprojection_error_max=pnp.reprojection_error_max,
                   n_markers=pnp.n_markers, n_corners=pnp.n_corners,
                   target_distance=pnp.distance, target_tilt=pnp.tilt,
                   pnp_center_z=float(pnp.center[2]), forced=bool(forced))
        s.T_base_flange = T_bf
        s.T_camera_target = pnp.T
        if depth is not None:
            s.depth_plane_distance = depth.plane.distance
            s.depth_plane_normal = depth.normal
            s.depth_inliers = depth.plane.inliers
            s.depth_center_z = depth.center_z
        meta = {"image_frame": frame.frame_id}
        T_oc = self.optical_to_camera_frame(frame.frame_id)
        if T_oc is not None:
            meta["T_optical_camera"] = matrix_to_pose7(T_oc)
        self.sample_ready.emit(s, frame.image, frame.decorated, meta)

    def optical_to_camera_frame(self, optical_frame: str) -> Optional[np.ndarray]:
        """Pose of ``camera_frame`` in the image frame, identity if they coincide."""
        if not optical_frame or optical_frame == self._config.camera_frame:
            return np.eye(4)
        return self.tf.lookup(optical_frame, self._config.camera_frame, Time(), 0.1)

    @Slot(float, float)
    @_slot
    def plan_current(self, vel: float, acc: float) -> None:
        """Plan to the current candidate (the first one if none was planned yet)."""
        self.motion.plan_candidate(max(self.motion.current, 0), vel, acc)

    @Slot(object, float, float)
    @_slot
    def plan_pose(self, T_base_flange: np.ndarray, vel: float, acc: float) -> None:
        self.motion.plan_pose(T_base_flange, vel, acc)

    @Slot()
    @_slot
    def execute(self) -> None:
        self.motion.execute()

    @Slot(float, float)
    @_slot
    def plan_and_execute_next(self, vel: float, acc: float) -> None:
        self.motion.plan_and_execute_next(vel, acc)

    @Slot()
    @_slot
    def stop_motion(self) -> None:
        self.motion.stop()

    @Slot(object)
    @_slot
    def set_candidates(self, candidates: Sequence[Candidate]) -> None:
        self.motion.set_candidates(candidates)
        self.status.emit(f"{len(candidates)} candidate poses loaded")

    def candidate_progress(self) -> tuple:
        if self.motion is None:
            return -1, 0
        return self.motion.current, len(self.motion.candidates)

    def current_joints(self) -> tuple:
        return self.motion.current_joints() if self.motion is not None else ([], [])

    def stability(self) -> tuple:
        return self.tf.stability() if self.tf is not None else (False, {})

    del _slot
