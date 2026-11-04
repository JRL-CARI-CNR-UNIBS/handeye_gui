# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Check configured ROS names (topics, frames, actions, groups) against the ROS graph.

``GraphSnapshot`` is filled by the ROS thread; everything here is plain
Python so that the GUI and the tests can use it without ROS.

Field kinds come from the ``suggest`` metadata of CalibConfig fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

IMAGE = "sensor_msgs/msg/Image"
CAMERA_INFO = "sensor_msgs/msg/CameraInfo"
JOINT_STATE = "sensor_msgs/msg/JointState"

OK, WARN, ERROR, UNKNOWN = "ok", "warn", "error", "unknown"

TOPIC_KINDS = {"image_topic": IMAGE, "camera_info_topic": CAMERA_INFO, "depth_topic": IMAGE,
               "joint_states_topic": JOINT_STATE}


@dataclass
class GraphSnapshot:
    topics: Dict[str, List[str]] = field(default_factory=dict)      # name -> types
    publishers: Dict[str, int] = field(default_factory=dict)        # name -> count
    actions: Dict[str, List[str]] = field(default_factory=dict)     # servers only
    services: Dict[str, List[str]] = field(default_factory=dict)    # servers only
    frames: Optional[List[str]] = None       # None: no TF received yet
    groups: Optional[List[str]] = None       # None: SRDF not received

    def topics_of(self, msg_type: str, published: bool = True) -> List[str]:
        """Topics of a type; with ``published`` only those that have a publisher."""
        return sorted(n for n, types in self.topics.items()
                      if msg_type in types and (not published or self.publishers.get(n, 1) > 0))


def normalize(name: str) -> str:
    """Topic/action/service names are resolved in the root namespace."""
    name = name.strip()
    return name if not name or name.startswith("/") else "/" + name


def split_kind(kind: str) -> Tuple[str, str]:
    """``"action:pkg/action/T"`` -> ("action", "pkg/action/T")."""
    base, _, arg = kind.partition(":")
    return base, arg


def suggestions(kind: str, snap: Optional[GraphSnapshot]) -> List[str]:
    """Existing names that fit a field of the given kind."""
    if snap is None:
        return []
    base, arg = split_kind(kind)
    if base in TOPIC_KINDS:
        names = snap.topics_of(TOPIC_KINDS[base])
        if base == "depth_topic":
            names.sort(key=lambda n: (0 if "depth" in n else 1, n))
        elif base == "image_topic":
            names.sort(key=lambda n: (1 if "depth" in n else 0, n))
        return names
    if base == "topic":
        return snap.topics_of(arg) if arg else sorted(snap.topics)
    if base == "frame":
        return sorted(snap.frames or [])
    if base == "planning_group":
        return sorted(snap.groups or [])
    if base == "action":
        return sorted(n for n, t in snap.actions.items() if not arg or arg in t)
    if base == "service":
        return sorted(n for n, t in snap.services.items() if not arg or arg in t)
    return []


def check(kind: str, value: str, snap: Optional[GraphSnapshot]) -> Tuple[str, str]:
    """(status, message) of a configured name."""
    if snap is None:
        return UNKNOWN, "not checked (ROS not running)"
    base, arg = split_kind(kind)
    value = value.strip()
    if base == "depth_topic" and not value:
        return OK, "depth check disabled"
    if not value:
        return ERROR, "empty"
    if base in TOPIC_KINDS or base == "topic":
        return _check_topic(normalize(value), TOPIC_KINDS.get(base, arg), snap,
                            optional=base == "topic")
    if base == "frame":
        if not snap.frames:
            return UNKNOWN, "no TF received yet"
        return (OK, "frame found") if value in snap.frames else (ERROR, "frame not in TF")
    if base == "planning_group":
        if snap.groups is None:
            return UNKNOWN, "SRDF not received: groups cannot be checked"
        if value in snap.groups:
            return OK, "planning group found"
        return ERROR, "no such planning group in the SRDF"
    if base in ("action", "service"):
        names = snap.actions if base == "action" else snap.services
        types = names.get(normalize(value))
        if types is None:
            return ERROR, f"{base} server not found"
        if arg and arg not in types:
            return ERROR, f"{base} has type {', '.join(types)}, expected {arg}"
        return OK, f"{base} found"
    return UNKNOWN, ""


def _check_topic(name: str, msg_type: str, snap: GraphSnapshot,
                 optional: bool = False) -> Tuple[str, str]:
    """A topic is valid when someone publishes it (the GUI itself subscribes to it)."""
    missing = WARN if optional else ERROR
    types = snap.topics.get(name)
    if types is None or snap.publishers.get(name, 1) == 0:
        return missing, "no publisher on this topic"
    if msg_type and msg_type not in types:
        return ERROR, f"topic has type {', '.join(types)}, expected {msg_type}"
    return OK, "topic published"


def camera_info_for(image_topic: str, snap: Optional[GraphSnapshot]) -> Optional[str]:
    """camera_info topic next to an image topic (``.../image_raw`` -> ``.../camera_info``)."""
    if snap is None:
        return None
    name = normalize(image_topic)
    infos = set(snap.topics_of(CAMERA_INFO))
    parts = name.rstrip("/").split("/")
    # Walk up the namespace: /a/b/image_raw -> /a/b/camera_info, /a/camera_info, ...
    for k in range(len(parts) - 1, 0, -1):
        cand = "/".join(parts[:k] + ["camera_info"])
        if cand in infos:
            return cand
    return None


MOTION_FIELDS = ("planning_group", "move_group_action", "execute_action")
