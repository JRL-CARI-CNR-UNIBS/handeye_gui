# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Camera subscriptions, ArUco detection, PnP and depth-plane check.

The ROS callbacks only store the newest messages. A dedicated worker thread
does the heavy processing (detection, PnP, RANSAC, drawing). Long callbacks
inside a rclpy executor make it spin hot and starve the other callbacks
(TF, joints, actions), so they are kept out of it.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image

from ..core.config import CalibConfig
from ..core.target import BoardModel, DepthResult, PnpResult, depth_plane, depth_to_meters

MAX_DEPTH_AGE = 0.1   # s, max stamp difference between colour and matched depth
DEPTH_HISTORY = 15    # depth messages kept for stamp matching

# Only the newest image matters: older ones are dropped, never queued.
IMAGE_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.BEST_EFFORT)


@dataclass
class Frame:
    """Latest processed camera frame."""

    stamp: Time
    frame_id: str
    image: np.ndarray            # BGR
    decorated: np.ndarray        # BGR
    pnp: Optional[PnpResult]
    depth: Optional[DepthResult]
    received: float              # time.monotonic() when processed


class CameraManager:
    """Keeps the latest processed frame.

    ``on_image`` is invoked from the worker thread at most ``display_rate``
    times per second with an RGB image for the GUI; ``on_target`` for every
    processed frame with the target pose (or None).
    """

    def __init__(self, node: Node, config: CalibConfig,
                 on_image: Callable[[np.ndarray, dict], None],
                 on_target: Callable[[Optional[np.ndarray]], None],
                 guard: Callable[[Callable], Callable], display_rate: float = 15.0):
        self._node = node
        self._on_image = on_image
        self._on_target = on_target
        self._guard = guard
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._group = ReentrantCallbackGroup()
        self._subs = []
        self._K: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._depths: deque = deque(maxlen=DEPTH_HISTORY)   # (stamp_ns, Image msg)
        self._latest: Optional[Frame] = None
        self._last_display = 0.0
        self._period = 1.0 / display_rate
        self._rng = np.random.default_rng(0)
        self.config = config
        self.board = BoardModel(config)
        # Worker thread state
        self._pending: Optional[Image] = None
        self._wakeup = threading.Condition()
        self._running = True
        self._worker = threading.Thread(target=self._work, name="handeye_camera", daemon=True)
        self._worker.start()
        self.configure(config)

    # ------------------------------------------------------------------
    def configure(self, config: CalibConfig) -> None:
        """(Re)create the board model and the subscriptions."""
        board = BoardModel(config)       # raises on a bad dictionary, before touching state
        with self._lock:
            for sub in self._subs:
                self._node.destroy_subscription(sub)
            self._subs = []
            self.config = config
            self.board = board
            self._K = self._D = None
            self._depths.clear()
            self._latest = None
            sub = self._node.create_subscription
            self._subs.append(sub(CameraInfo, config.camera_info_topic, self._guard(self._on_info),
                                  qos_profile_sensor_data, callback_group=self._group))
            self._subs.append(sub(Image, config.image_topic, self._guard(self._on_color), IMAGE_QOS,
                                  callback_group=self._group))
            if config.depth_topic.strip():
                self._subs.append(sub(Image, config.depth_topic, self._guard(self._on_depth),
                                      qos_profile_sensor_data, callback_group=self._group))

    def close(self) -> None:
        """Stop the worker thread."""
        with self._wakeup:
            self._running = False
            self._wakeup.notify()
        self._worker.join(timeout=2.0)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._latest

    @property
    def intrinsics(self):
        return self._K, self._D

    # -- ROS callbacks (cheap) ---------------------------------------------
    def _on_info(self, msg: CameraInfo) -> None:
        K = np.array(msg.k, dtype=float).reshape(3, 3)
        if K[0, 0] > 0:
            self._K, self._D = K, np.array(msg.d, dtype=float)

    def _on_depth(self, msg: Image) -> None:
        self._depths.append((Time.from_msg(msg.header.stamp).nanoseconds, msg))

    def _on_color(self, msg: Image) -> None:
        with self._wakeup:
            self._pending = msg
            self._wakeup.notify()

    # -- worker ------------------------------------------------------------
    def _work(self) -> None:
        process = self._guard(self._process)
        while True:
            with self._wakeup:
                while self._running and self._pending is None:
                    self._wakeup.wait()
                if not self._running:
                    return
                msg, self._pending = self._pending, None
            process(msg)

    def _match_depth(self, stamp_ns: int) -> Optional[np.ndarray]:
        """Depth image (metres) with the closest stamp, if within MAX_DEPTH_AGE."""
        candidates = list(self._depths)
        if not candidates:
            return None
        t, msg = min(candidates, key=lambda item: abs(item[0] - stamp_ns))
        if abs(t - stamp_ns) * 1e-9 > MAX_DEPTH_AGE:
            return None
        return depth_to_meters(self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough"))

    def _process(self, msg: Image) -> None:
        bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        stamp = Time.from_msg(msg.header.stamp)
        K, D = self._K, self._D
        board, config = self.board, self.config
        detection = board.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        pnp = board.estimate_pose(detection, K, D) if K is not None else None
        depth_res = None
        depth = self._match_depth(stamp.nanoseconds) if pnp is not None else None
        if depth is not None and depth.shape[:2] == bgr.shape[:2]:
            depth_res = depth_plane(depth, K, pnp.image_points, config.depth_roi_shrink,
                                    config.depth_ransac_threshold, pnp.center, rng=self._rng)
        extra = [] if K is not None else [f"waiting for {config.camera_info_topic}"]
        decorated = board.draw(bgr, detection, pnp, K, D, depth_res, extra)
        frame = Frame(stamp, msg.header.frame_id, bgr, decorated, pnp, depth_res, time.monotonic())
        with self._lock:
            self._latest = frame
        self._on_target(pnp.T if pnp is not None else None)

        now = time.monotonic()
        if now - self._last_display >= self._period:
            self._last_display = now
            info = {"markers": detection.n_markers, "has_pose": pnp is not None,
                    "has_depth": depth_res is not None, "frame_id": msg.header.frame_id}
            self._on_image(cv2.cvtColor(decorated, cv2.COLOR_BGR2RGB), info)
