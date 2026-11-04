# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Robot description (URDF) and semantic description (SRDF) from their latched topics.

robot_state_publisher publishes ``/robot_description`` and move_group
publishes ``/robot_description_semantic``, both transient-local, so a late
subscriber still receives them.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from ..core.robot_model import RobotModel

LATCHED_QOS = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)


class RobotDescription:
    def __init__(self, node: Node, topic: str, guard: Callable[[Callable], Callable],
                 on_error: Callable[[str], None] = lambda msg: None):
        self._node = node
        self._guard = guard
        self._on_error = on_error
        self._lock = threading.Lock()
        self._group = ReentrantCallbackGroup()
        self._subs = []
        self._urdf: Optional[str] = None
        self._srdf: Optional[str] = None
        self._model: Optional[RobotModel] = None
        self.topic = ""
        self.configure(topic)

    def configure(self, topic: str) -> None:
        topic = topic.strip() or "/robot_description"
        if topic == self.topic:
            return
        with self._lock:
            for sub in self._subs:
                self._node.destroy_subscription(sub)
            self.topic = topic
            self._urdf = self._srdf = None
            self._model = None
            self._subs = [
                self._node.create_subscription(String, topic, self._guard(self._on_urdf),
                                               LATCHED_QOS, callback_group=self._group),
                self._node.create_subscription(String, topic + "_semantic",
                                               self._guard(self._on_srdf), LATCHED_QOS,
                                               callback_group=self._group)]

    def _on_urdf(self, msg: String) -> None:
        with self._lock:
            self._urdf = msg.data
            self._model = None

    def _on_srdf(self, msg: String) -> None:
        with self._lock:
            self._srdf = msg.data
            self._model = None

    def model(self) -> Optional[RobotModel]:
        """Parsed model, or None if neither description has been received."""
        with self._lock:
            if self._model is None and (self._urdf or self._srdf):
                try:
                    self._model = RobotModel.from_xml(self._urdf, self._srdf)
                except Exception as exc:  # noqa: BLE001 - malformed XML must not kill the GUI
                    self._on_error(f"cannot parse the robot description: {exc}")
                    self._urdf = self._srdf = None
                    return None
            return self._model
