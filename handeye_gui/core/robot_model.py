# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Minimal URDF/SRDF reader: joint limits and planning-group joints.

Only the standard library XML parser is used (no urdfdom or MoveIt bindings).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional

ACTUATED = ("revolute", "continuous", "prismatic")


@dataclass
class JointInfo:
    name: str
    type: str
    parent: str
    child: str
    lower: Optional[float] = None
    upper: Optional[float] = None
    mimic: bool = False

    @property
    def actuated(self) -> bool:
        return self.type in ACTUATED and not self.mimic

    @property
    def angular(self) -> bool:
        return self.type in ("revolute", "continuous")


def parse_urdf(xml: str) -> Dict[str, JointInfo]:
    """Joints of a URDF, by name."""
    root = ET.fromstring(xml)
    joints = {}
    for j in root.iter("joint"):
        name, jtype = j.get("name"), j.get("type", "fixed")
        parent, child = j.find("parent"), j.find("child")
        if not name or parent is None or child is None:
            continue
        info = JointInfo(name, jtype, parent.get("link", ""), child.get("link", ""),
                         mimic=j.find("mimic") is not None)
        lim = j.find("limit")
        if jtype != "continuous" and lim is not None:
            if lim.get("lower") is not None:
                info.lower = float(lim.get("lower"))
            if lim.get("upper") is not None:
                info.upper = float(lim.get("upper"))
        joints[name] = info
    return joints


def parse_srdf_groups(xml: str, joints: Dict[str, JointInfo]) -> Dict[str, List[str]]:
    """Actuated joints of every planning group of an SRDF.

    Supports ``<joint>``, ``<link>`` (its parent joint), ``<chain>`` and
    nested ``<group>`` elements. Without the URDF joints only ``<joint>``
    entries can be resolved.
    """
    root = ET.fromstring(xml)
    elements = {g.get("name"): g for g in root.findall("group") if g.get("name")}
    by_child = {j.child: j for j in joints.values()}

    def chain(base: str, tip: str) -> List[str]:
        out, link = [], tip
        while link != base and link in by_child:
            j = by_child[link]
            out.append(j.name)
            link = j.parent
        return list(reversed(out)) if link == base else []

    def resolve(name: str, seen: set) -> List[str]:
        if name in seen or name not in elements:
            return []
        seen = seen | {name}
        out: List[str] = []
        for e in elements[name]:
            if e.tag == "joint":
                out.append(e.get("name", ""))
            elif e.tag == "link" and e.get("name") in by_child:
                out.append(by_child[e.get("name")].name)
            elif e.tag == "chain":
                out += chain(e.get("base_link", ""), e.get("tip_link", ""))
            elif e.tag == "group":
                out += resolve(e.get("name", ""), seen)
        return out

    groups = {}
    for name in elements:
        names = []
        for j in resolve(name, set()):
            info = joints.get(j)
            if j and j not in names and (info is None or info.actuated):
                names.append(j)
        groups[name] = names
    return groups


@dataclass
class RobotModel:
    joints: Dict[str, JointInfo]
    groups: Optional[Dict[str, List[str]]] = None     # None until the SRDF is known

    @classmethod
    def from_xml(cls, urdf: Optional[str], srdf: Optional[str] = None) -> "RobotModel":
        joints = parse_urdf(urdf) if urdf else {}
        groups = parse_srdf_groups(srdf, joints) if srdf else None
        return cls(joints, groups)

    def limits(self, name: str) -> tuple:
        """(lower, upper) in rad or m; infinite for continuous or unknown joints."""
        j = self.joints.get(name)
        lo = j.lower if j is not None and j.lower is not None else -math.inf
        hi = j.upper if j is not None and j.upper is not None else math.inf
        return lo, hi

    def is_angular(self, name: str) -> bool:
        j = self.joints.get(name)
        return j is None or j.angular        # unknown joints are assumed revolute
