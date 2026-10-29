# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""How well the samples cover distances, viewing directions and rotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .geometry import invert
from .sample import Sample

# Viewing direction bins: a central cap plus rings split in azimuth sectors.
POLAR_RINGS_DEG = (10.0, 25.0, 40.0, 60.0)
AZIMUTH_SECTORS = 8
ROLL_BINS = 8           # rotation about the optical axis, 45 deg each
DISTANCE_BINS = 5
TILT_EDGES_DEG = np.arange(0.0, 70.0, 10.0)


@dataclass
class CoverageReport:
    n_samples: int
    distance_hist: np.ndarray
    distance_edges: np.ndarray
    tilt_hist: np.ndarray
    tilt_edges: np.ndarray
    view_occupied: int
    view_total: int
    roll_occupied: int
    distance_occupied: int
    index: float            # 0 (poor) .. 1 (every bin covered)

    def to_dict(self) -> Dict[str, Any]:
        return {"n_samples": self.n_samples, "index": float(self.index),
                "distance_hist": self.distance_hist.tolist(),
                "distance_edges": self.distance_edges.tolist(),
                "tilt_hist": self.tilt_hist.tolist(), "tilt_edges": self.tilt_edges.tolist(),
                "view_bins": [self.view_occupied, self.view_total],
                "roll_bins": [self.roll_occupied, ROLL_BINS],
                "distance_bins": [self.distance_occupied, DISTANCE_BINS]}


def _view_bin(direction: np.ndarray) -> int:
    """Bin of the direction target -> camera, expressed in the target frame.

    The camera sees the target from its -Z side (OpenCV board convention), so
    a frontal view has direction (0, 0, -1).
    """
    polar = np.degrees(np.arccos(np.clip(-direction[2], -1.0, 1.0)))
    if polar < POLAR_RINGS_DEG[0]:
        return 0
    ring = int(np.searchsorted(POLAR_RINGS_DEG, polar, side="right")) - 1
    ring = min(ring, len(POLAR_RINGS_DEG) - 1)
    az = (np.degrees(np.arctan2(direction[1], direction[0])) + 360.0) % 360.0
    return 1 + ring * AZIMUTH_SECTORS + int(az // (360.0 / AZIMUTH_SECTORS))


def coverage_report(samples: Sequence[Sample],
                    distance_range: Tuple[float, float] = (0.2, 1.0)) -> CoverageReport:
    """Histograms of distance and tilt plus a synthetic coverage index.

    The index is the mean of three occupancy fractions: viewing directions,
    rotations about the optical axis and distance bins over ``distance_range``.
    """
    view_total = 1 + len(POLAR_RINGS_DEG) * AZIMUTH_SECTORS
    d_edges = np.linspace(distance_range[0], distance_range[1], DISTANCE_BINS + 1)
    if not samples:
        return CoverageReport(0, np.zeros(DISTANCE_BINS, int), d_edges,
                              np.zeros(len(TILT_EDGES_DEG) - 1, int), TILT_EDGES_DEG,
                              0, view_total, 0, 0, 0.0)
    views, rolls = set(), set()
    distances, tilts = [], []
    for s in samples:
        T_ct = s.T_camera_target
        p = invert(T_ct)[:3, 3]                       # camera position in target frame
        n = np.linalg.norm(p)
        if n > 1e-9:
            views.add(_view_bin(p / n))
        roll = (np.degrees(np.arctan2(T_ct[1, 0], T_ct[0, 0])) + 360.0) % 360.0
        rolls.add(int(roll // (360.0 / ROLL_BINS)))
        distances.append(s.target_distance or float(np.linalg.norm(T_ct[:3, 3])))
        tilts.append(s.target_tilt)
    d_hist, _ = np.histogram(np.clip(distances, d_edges[0], d_edges[-1]), bins=d_edges)
    t_hist, _ = np.histogram(np.clip(tilts, TILT_EDGES_DEG[0], TILT_EDGES_DEG[-1]),
                             bins=TILT_EDGES_DEG)
    d_occ = int((d_hist > 0).sum())
    index = (len(views) / view_total + len(rolls) / ROLL_BINS + d_occ / DISTANCE_BINS) / 3.0
    return CoverageReport(len(samples), d_hist, d_edges, t_hist, TILT_EDGES_DEG,
                          len(views), view_total, len(rolls), d_occ, float(index))
