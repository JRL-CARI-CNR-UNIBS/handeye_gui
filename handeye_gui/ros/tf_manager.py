# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""TF lookups and pose stability over a sliding window."""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

import numpy as np
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from ..core.config import CalibConfig
from ..core.geometry import pose_dispersion, pose_to_matrix

TICK_RATE = 30.0       # Hz, flange sampling rate
EMIT_PERIOD = 0.2      # s, stability is reported at most this often (and on changes)


def transform_to_matrix(tf) -> np.ndarray:
    t, r = tf.transform.translation, tf.transform.rotation
    return pose_to_matrix([t.x, t.y, t.z], [r.x, r.y, r.z, r.w])


class TfManager:
    """Reads base -> flange from TF and judges whether robot and target are still.

    The pose is ``stable`` when, over the last ``stability_window`` seconds,
    both the flange pose and the target pose in camera have RMS position and
    rotation deviations below the configured thresholds, and the target was
    visible in every frame.
    """

    def __init__(self, node: Node, config: CalibConfig,
                 on_stability: Callable[[bool, dict], None],
                 guard: Callable[[Callable], Callable]):
        self._node = node
        self._on_stability = on_stability
        self.config = config
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, node, spin_thread=False)
        self._lock = threading.Lock()
        self._flange: deque = deque()   # (t, T)
        self._target: deque = deque()   # (t, T or None)
        self._stable: Optional[bool] = None
        self._info: dict = {}
        self._last_emit = 0.0
        self._timer = node.create_timer(1.0 / TICK_RATE, guard(self._tick),
                                        callback_group=MutuallyExclusiveCallbackGroup())

    def configure(self, config: CalibConfig) -> None:
        with self._lock:
            self.config = config
            self._flange.clear()
            self._target.clear()

    # ------------------------------------------------------------------
    def lookup(self, target_frame: str, source_frame: str, stamp: Optional[Time] = None,
               timeout: float = 0.0) -> Optional[np.ndarray]:
        """Pose of ``source_frame`` in ``target_frame`` (4x4), or None."""
        try:
            tf = self.buffer.lookup_transform(target_frame, source_frame, stamp or Time(),
                                              timeout=Duration(seconds=timeout))
        except TransformException:
            return None
        return transform_to_matrix(tf)

    def flange_pose(self, stamp: Optional[Time] = None, timeout: float = 0.1) -> Optional[np.ndarray]:
        """base -> flange at ``stamp`` (falls back to the latest transform)."""
        cfg = self.config
        T = None
        if stamp is not None:
            T = self.lookup(cfg.base_frame, cfg.flange_frame, stamp, timeout)
        return T if T is not None else self.lookup(cfg.base_frame, cfg.flange_frame)

    def push_target(self, T: Optional[np.ndarray]) -> None:
        """Called by the camera manager for every processed frame."""
        now = self._now()
        with self._lock:
            self._target.append((now, T))
            self._trim(self._target, now)

    def stability(self) -> tuple:
        with self._lock:
            return bool(self._stable), dict(self._info)

    # ------------------------------------------------------------------
    def _now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9

    def _trim(self, window: deque, now: float) -> None:
        while window and now - window[0][0] > self.config.stability_window:
            window.popleft()

    def _tick(self) -> None:
        now = self._now()
        T = self.flange_pose(timeout=0.0)
        with self._lock:
            if T is not None:
                self._flange.append((now, T))
            self._trim(self._flange, now)
            self._trim(self._target, now)
            stable, info = self._evaluate(now)
            changed = stable != self._stable
            self._stable, self._info = stable, info
        if changed or now - self._last_emit >= EMIT_PERIOD or now < self._last_emit:
            self._last_emit = now
            self._on_stability(stable, info)

    def _evaluate(self, now: float) -> tuple:
        cfg = self.config
        info = {"reason": ""}
        min_span = 0.6 * cfg.stability_window
        flange = [T for _, T in self._flange]
        if len(flange) < 3 or now - self._flange[0][0] < min_span:
            info["reason"] = f"no TF {cfg.base_frame} -> {cfg.flange_frame}" if not flange \
                else "collecting"
            return False, info
        targets = [T for _, T in self._target]
        if not targets or any(T is None for T in targets):
            info["reason"] = "target not visible"
        fp, fr = pose_dispersion(flange)
        info.update(flange_pos=fp, flange_rot=np.degrees(fr))
        visible = [T for T in targets if T is not None]
        if len(visible) >= 2:
            tp, tr = pose_dispersion(visible)
            info.update(target_pos=tp, target_rot=np.degrees(tr))
        pos = max(fp, info.get("target_pos", 0.0))
        rot = max(np.degrees(fr), info.get("target_rot", 0.0))
        pos_ok = pos <= cfg.stability_pos_threshold
        rot_ok = rot <= cfg.stability_rot_threshold
        if not info["reason"] and not (pos_ok and rot_ok):
            info["reason"] = "moving"
        return not info["reason"], info
