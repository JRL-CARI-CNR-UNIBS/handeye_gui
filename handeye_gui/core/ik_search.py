# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Choose an IK solution close to the current joint configuration.

MoveIt's IK returns *a* solution, which can be far from the current
configuration in joint space (elbow flip, wrist turned by a full revolution).
``IkSearch`` drives repeated IK calls:

1. the first seed is the current configuration;
2. a solution is accepted when every joint moves at most ``max_step``;
3. otherwise the next seed is drawn around the current configuration, in a
   neighbourhood that grows with every attempt, until ``attempts`` calls.

Before the check, a revolute joint whose range exceeds one turn (e.g. +/-360
deg on a UR) is moved by whole turns to the equivalent angle nearest to the
current one, if that angle is within the joint limits.

The class is pure Python: the ROS service calls are done by MotionManager.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * math.pi
MIN_AMPLITUDE = 0.05      # rad, seed neighbourhood of the second attempt


def nearest_turn(q: float, ref: float, lower: float = -math.inf, upper: float = math.inf) -> float:
    """``q + 2*pi*k`` closest to ``ref`` within [lower, upper] (q itself if none fits)."""
    k = round((ref - q) / TWO_PI)
    best = q
    for kk in (k - 1, k, k + 1):
        cand = q + kk * TWO_PI
        if lower - 1e-9 <= cand <= upper + 1e-9 and abs(cand - ref) < abs(best - ref):
            best = cand
    return best


class IkSearch:
    """State of the search for one pose goal.

    Args:
        current: current joint positions (all joints are kept in the seeds).
        joints: joints that are checked against ``max_step`` (the planning group).
        max_step: maximum allowed motion per joint, rad (m for prismatic joints).
        attempts: maximum number of IK calls.
        limits: ``name -> (lower, upper)``; missing joints are unbounded.
        angular: ``name -> bool``, whether whole-turn equivalents apply.
    """

    def __init__(self, current: Dict[str, float], joints: Sequence[str], max_step: float,
                 attempts: int, limits: Optional[Dict[str, Tuple[float, float]]] = None,
                 angular: Optional[Callable[[str], bool]] = None,
                 rng: Optional[np.random.Generator] = None):
        missing = [j for j in joints if j not in current]
        if missing:
            raise ValueError(f"no joint state for {', '.join(missing)}")
        self.current = dict(current)
        self.joints = list(joints)
        self.max_step = float(max_step)
        self.attempts = max(1, int(attempts))
        self.limits = limits or {}
        self.angular = angular or (lambda name: True)
        self.rng = rng or np.random.default_rng()
        self.tried = 0
        self.best: Optional[Tuple[float, str]] = None    # (largest step, joint) of the best rejected

    def _limits(self, name: str) -> Tuple[float, float]:
        return self.limits.get(name, (-math.inf, math.inf))

    def amplitude(self, attempt: int) -> float:
        """Half-width of the seed neighbourhood of ``attempt`` (1-based after the first)."""
        if self.attempts <= 2:
            return max(MIN_AMPLITUDE, self.max_step)
        frac = (attempt - 1) / (self.attempts - 2)
        return MIN_AMPLITUDE + frac * max(0.0, self.max_step - MIN_AMPLITUDE)

    def next_seed(self) -> Optional[Dict[str, float]]:
        """Seed of the next IK call, or None when the attempts are used up."""
        if self.tried >= self.attempts:
            return None
        attempt = self.tried
        self.tried += 1
        if attempt == 0:
            return {j: self.current[j] for j in self.joints}
        amp = self.amplitude(attempt)
        seed = {}
        for j in self.joints:
            lo, hi = self._limits(j)
            seed[j] = float(np.clip(self.current[j] + self.rng.uniform(-amp, amp), lo, hi))
        return seed

    def adjust(self, solution: Dict[str, float]) -> Dict[str, float]:
        """Group joints of ``solution``, moved by whole turns towards the current values."""
        out = {}
        for j in self.joints:
            q = float(solution.get(j, self.current[j]))
            if self.angular(j):
                q = nearest_turn(q, self.current[j], *self._limits(j))
            out[j] = q
        return out

    def step(self, q: Dict[str, float]) -> Tuple[float, str]:
        """(largest absolute joint motion, joint name) from the current configuration."""
        worst = max(self.joints, key=lambda j: abs(q[j] - self.current[j]))
        return abs(q[worst] - self.current[worst]), worst

    def evaluate(self, solution: Dict[str, float]) -> Optional[Dict[str, float]]:
        """The adjusted solution if it respects ``max_step``, else None (and keep the best)."""
        q = self.adjust(solution)
        step = self.step(q)
        if step[0] <= self.max_step:
            return q
        if self.best is None or step[0] < self.best[0]:
            self.best = step
        return None

    def failure_message(self, angular_unit: Callable[[float], str] = None) -> str:
        fmt = angular_unit or (lambda v: f"{math.degrees(v):.1f} deg")
        msg = (f"no IK solution within {fmt(self.max_step)} of the current configuration "
               f"after {self.tried} attempts")
        if self.best is not None:
            msg += f" (closest: {fmt(self.best[0])} on {self.best[1]})"
        return msg
