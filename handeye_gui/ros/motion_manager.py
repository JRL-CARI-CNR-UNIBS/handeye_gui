# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""MoveIt access through the move_group actions (no Python bindings needed).

Planning uses ``MoveGroup`` with ``plan_only = True``; execution sends the
stored trajectory to ``ExecuteTrajectory``. Both are asynchronous: results
come back through the ``on_done(kind, ok, message)`` callback.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
from geometry_msgs.msg import Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import (Constraints, JointConstraint, MotionPlanRequest, MoveItErrorCodes,
                             OrientationConstraint, PositionConstraint)
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive

from ..core.config import CalibConfig
from ..core.geometry import matrix_to_pose

JOINT_TOLERANCE = 1e-3       # rad
POSITION_TOLERANCE = 1e-3    # m, radius of the goal sphere
ORIENTATION_TOLERANCE = 5e-3  # rad per axis

_ERROR_NAMES = {v: k for k, v in vars(MoveItErrorCodes).items()
                if k.isupper() and isinstance(v, int)}


def error_name(code: int) -> str:
    return _ERROR_NAMES.get(code, str(code))


@dataclass
class Candidate:
    """A goal in the candidate list: joint values or a flange pose in base."""

    joint_names: List[str] = field(default_factory=list)
    joint_values: List[float] = field(default_factory=list)
    pose: Optional[np.ndarray] = None     # T_base_flange
    label: str = ""

    @property
    def is_pose(self) -> bool:
        return self.pose is not None


class MotionManager:
    """Plans and executes motions, and walks through a list of candidate goals."""

    def __init__(self, node: Node, config: CalibConfig,
                 on_done: Callable[[str, bool, str], None],
                 guard: Callable[[Callable], Callable]):
        self._node = node
        self._on_done = on_done
        self._guard = guard
        self._lock = threading.Lock()
        self._group = ReentrantCallbackGroup()
        self.config = config
        self._move_client: Optional[ActionClient] = None
        self._exec_client: Optional[ActionClient] = None
        self._joint_sub = None
        self._joints: dict = {}       # name -> latest position
        self._trajectory = None
        self._goal_handle = None
        self.candidates: List[Candidate] = []
        self.current = -1           # index of the candidate last planned
        self.configure(config)

    def configure(self, config: CalibConfig) -> None:
        with self._lock:
            old = (self.config.move_group_action, self.config.execute_action,
                   self.config.joint_states_topic)
            new = (config.move_group_action, config.execute_action, config.joint_states_topic)
            self.config = config
            if self._move_client is not None and old == new:
                return
            for client in (self._move_client, self._exec_client):
                if client is not None:
                    client.destroy()
            if self._joint_sub is not None:
                self._node.destroy_subscription(self._joint_sub)
            self._move_client = ActionClient(self._node, MoveGroup, config.move_group_action,
                                             callback_group=self._group)
            self._exec_client = ActionClient(self._node, ExecuteTrajectory, config.execute_action,
                                             callback_group=self._group)
            self._joint_sub = self._node.create_subscription(
                JointState, config.joint_states_topic, self._guard(self._on_joints), 10,
                callback_group=self._group)

    # -- joint states ------------------------------------------------------
    def _on_joints(self, msg: JointState) -> None:
        if not msg.name:
            return
        with self._lock:
            # Merge, so that several publishers (arm, gripper) add up.
            merged = dict(self._joints)
            merged.update(zip(msg.name, msg.position))
            self._joints = merged

    def current_joints(self) -> tuple:
        """(names, values) of the configured joints (all joints if none configured)."""
        values = self._joints
        names = self.config.joint_names or list(values)
        names = [n for n in names if n in values]
        return names, [float(values[n]) for n in names]

    # -- candidates --------------------------------------------------------
    def set_candidates(self, candidates: Sequence[Candidate]) -> None:
        with self._lock:
            self.candidates = list(candidates)
            self.current = -1

    def plan_candidate(self, i: int, vel: float, acc: float, auto_execute: bool = False) -> None:
        if not self.candidates:
            raise IndexError("no candidate poses: load joints or generate hemisphere poses")
        if not 0 <= i < len(self.candidates):
            raise IndexError(f"candidate {i + 1} out of range (1..{len(self.candidates)})")
        self.current = i
        c = self.candidates[i]
        if c.is_pose:
            self.plan_pose(c.pose, vel, acc, auto_execute)
        else:
            self.plan(dict(zip(c.joint_names, c.joint_values)), vel, acc, auto_execute)

    def plan_and_execute_next(self, vel: float, acc: float) -> None:
        nxt = self.current + 1
        if nxt >= len(self.candidates):
            raise IndexError("no more candidate poses")
        self.plan_candidate(nxt, vel, acc, auto_execute=True)

    # -- planning ----------------------------------------------------------
    def _request(self, vel: float, acc: float) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.group_name = self.config.planning_group
        req.num_planning_attempts = 5
        req.allowed_planning_time = float(self.config.planning_time)
        req.max_velocity_scaling_factor = float(np.clip(vel, 0.01, 1.0))
        req.max_acceleration_scaling_factor = float(np.clip(acc, 0.01, 1.0))
        req.start_state.is_diff = True
        return req

    def plan(self, target_joints: dict, vel: float, acc: float, auto_execute: bool = False) -> None:
        """Plan to a joint configuration {name: value}."""
        req = self._request(vel, acc)
        c = Constraints()
        for name, value in target_joints.items():
            c.joint_constraints.append(JointConstraint(
                joint_name=name, position=float(value), tolerance_above=JOINT_TOLERANCE,
                tolerance_below=JOINT_TOLERANCE, weight=1.0))
        req.goal_constraints = [c]
        self._send_plan(req, auto_execute)

    def plan_pose(self, T_base_flange: np.ndarray, vel: float, acc: float,
                  auto_execute: bool = False) -> None:
        """Plan to a flange pose expressed in the base frame."""
        cfg = self.config
        t, q = matrix_to_pose(T_base_flange)
        req = self._request(vel, acc)
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = cfg.base_frame
        pc.link_name = cfg.flange_frame
        pc.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.SPHERE,
                                                          dimensions=[POSITION_TOLERANCE])]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = t
        pose.orientation.w = 1.0
        pc.constraint_region.primitive_poses = [pose]
        pc.weight = 1.0
        oc = OrientationConstraint()
        oc.header.frame_id = cfg.base_frame
        oc.link_name = cfg.flange_frame
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = q
        oc.absolute_x_axis_tolerance = oc.absolute_y_axis_tolerance = ORIENTATION_TOLERANCE
        oc.absolute_z_axis_tolerance = ORIENTATION_TOLERANCE
        oc.weight = 1.0
        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        req.goal_constraints = [c]
        self._send_plan(req, auto_execute)

    def _send_plan(self, req: MotionPlanRequest, auto_execute: bool) -> None:
        if not self._move_client.server_is_ready():
            raise RuntimeError(f"action server {self.config.move_group_action} not available")
        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = False
        goal.planning_options.planning_scene_diff.is_diff = True
        goal.planning_options.planning_scene_diff.robot_state.is_diff = True
        self._trajectory = None
        future = self._move_client.send_goal_async(goal)
        future.add_done_callback(self._guard(lambda f: self._on_plan_accepted(f, auto_execute)))

    def _on_plan_accepted(self, future, auto_execute: bool) -> None:
        handle = future.result()
        if not handle.accepted:
            self._on_done("plan", False, "planning goal rejected")
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(
            self._guard(lambda f: self._on_plan_result(f, auto_execute)))

    def _on_plan_result(self, future, auto_execute: bool) -> None:
        self._goal_handle = None
        result = future.result().result
        code = result.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self._on_done("plan", False, f"planning failed: {error_name(code)}")
            return
        self._trajectory = result.planned_trajectory
        n = len(self._trajectory.joint_trajectory.points)
        label = f" (candidate {self.current + 1}/{len(self.candidates)})" if self.current >= 0 else ""
        self._on_done("plan", True, f"plan ready, {n} points{label}")
        if auto_execute:
            self.execute()

    # -- execution ---------------------------------------------------------
    @property
    def has_plan(self) -> bool:
        return self._trajectory is not None

    def execute(self) -> None:
        if self._trajectory is None:
            raise RuntimeError("no plan to execute")
        if not self._exec_client.server_is_ready():
            raise RuntimeError(f"action server {self.config.execute_action} not available")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = self._trajectory
        self._trajectory = None        # a plan is valid only from its start state
        future = self._exec_client.send_goal_async(goal)
        future.add_done_callback(self._guard(self._on_exec_accepted))

    def _on_exec_accepted(self, future) -> None:
        handle = future.result()
        if not handle.accepted:
            self._on_done("execute", False, "execution goal rejected")
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._guard(self._on_exec_result))

    def _on_exec_result(self, future) -> None:
        self._goal_handle = None
        code = future.result().result.error_code.val
        ok = code == MoveItErrorCodes.SUCCESS
        self._on_done("execute", ok, "motion done" if ok else f"execution failed: {error_name(code)}")

    def stop(self) -> None:
        """Cancel the running planning or execution goal."""
        handle = self._goal_handle
        if handle is not None:
            handle.cancel_goal_async()
