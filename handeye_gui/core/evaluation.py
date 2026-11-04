# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Quality assessment of a hand-eye calibration.

Eye-in-hand chain: for every sample i the target pose in the base frame is

    T_base_target_i = T_base_flange_i @ X @ T_camera_target_i

and must be the same for all samples, because the target does not move.
The spread of T_base_target_i around their mean is the chain-closure error.

Eye-to-hand uses the same machinery with the target fixed to the flange:

    T_flange_target_i = T_flange_base_i @ X @ T_camera_target_i

(see ``solvers.robot_pose``). "Target pose" below means the target in its
fixed frame: base (eye-in-hand) or flange (eye-to-hand).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from .coverage import CoverageReport, coverage_report
from .geometry import average_poses, average_quaternions, matrix_to_pose, rotation_angle, rotvec_delta
from .sample import Sample
from .solvers import (EYE_IN_HAND, HandEyeResult, Method, check_mode, result_parent, robot_pose,
                      solve, solve_matrices, target_parent)

PARAM_NAMES = ("x", "y", "z", "rx", "ry", "rz")


# --------------------------------------------------------------------------
# Per-sample errors
# --------------------------------------------------------------------------
@dataclass
class SampleError:
    index: int
    position: np.ndarray        # target position in its fixed frame (m)
    dx: float
    dy: float
    dz: float
    angle: float                # deg

    @property
    def translation(self) -> float:
        return float(np.sqrt(self.dx ** 2 + self.dy ** 2 + self.dz ** 2))


def target_pose(sample: Sample, X: np.ndarray, mode: str = EYE_IN_HAND) -> np.ndarray:
    """Target pose in its fixed frame (base for eye-in-hand, flange for eye-to-hand)."""
    return robot_pose(sample, mode) @ X @ sample.T_camera_target


def target_in_base(sample: Sample, X: np.ndarray) -> np.ndarray:
    """Eye-in-hand target pose in the base frame (kept for backward compatibility)."""
    return target_pose(sample, X, EYE_IN_HAND)


def estimate_target_pose(samples: Sequence[Sample], X: np.ndarray,
                         mode: str = EYE_IN_HAND) -> np.ndarray:
    """Mean target pose in its fixed frame given the hand-eye transform X."""
    if not samples:
        raise ValueError("no samples")
    return average_poses(target_pose(s, X, mode) for s in samples)


def per_sample_errors(samples: Sequence[Sample], result: HandEyeResult | np.ndarray,
                      reference: Optional[np.ndarray] = None,
                      mode: Optional[str] = None) -> List[SampleError]:
    """Chain-closure error (XYZ in the target's fixed frame and angle) of every sample.

    Args:
        reference: target pose used as ground truth. Defaults to the mean over
            ``samples``; pass the identification estimate when evaluating
            validation samples.
        mode: calibration type; defaults to ``result.mode`` (or eye-in-hand
            when ``result`` is a plain matrix).
    """
    X, mode = _unpack(result, mode)
    if reference is None:
        reference = estimate_target_pose(samples, X, mode)
    errors = []
    for s in samples:
        T = target_pose(s, X, mode)
        d = T[:3, 3] - reference[:3, 3]
        ang = np.degrees(rotation_angle(reference[:3, :3].T @ T[:3, :3]))
        errors.append(SampleError(s.index, T[:3, 3].copy(), float(d[0]), float(d[1]),
                                  float(d[2]), float(ang)))
    return errors


def _unpack(result: HandEyeResult | np.ndarray, mode: Optional[str]) -> Tuple[np.ndarray, str]:
    if isinstance(result, HandEyeResult):
        return result.T, check_mode(mode or result.mode)
    return np.asarray(result), check_mode(mode or EYE_IN_HAND)


def flag_outliers(errors: Sequence[SampleError], n_sigma: float) -> List[int]:
    """Indices whose translational or angular error exceeds mean + n_sigma * std."""
    if len(errors) < 3:
        return []
    t = np.array([e.translation for e in errors])
    a = np.array([e.angle for e in errors])
    t_lim = t.mean() + n_sigma * t.std()
    a_lim = a.mean() + n_sigma * a.std()
    return [e.index for e, ti, ai in zip(errors, t, a) if ti > t_lim or ai > a_lim]


@dataclass
class ErrorStats:
    n: int
    translation_mean: float      # m
    translation_std: float
    translation_max: float
    angle_mean: float            # deg
    angle_std: float
    angle_max: float

    @classmethod
    def from_errors(cls, errors: Sequence[SampleError]) -> Optional["ErrorStats"]:
        if not errors:
            return None
        t = np.array([e.translation for e in errors])
        a = np.array([e.angle for e in errors])
        return cls(len(errors), float(t.mean()), float(t.std()), float(t.max()),
                   float(a.mean()), float(a.std()), float(a.max()))

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


# --------------------------------------------------------------------------
# Identification / validation
# --------------------------------------------------------------------------
def split(samples: Sequence[Sample], ratio: float, seed: int = 0
          ) -> Tuple[List[Sample], List[Sample]]:
    """Reproducible random split into identification and validation sets.

    ``ratio`` is the identification fraction. At least 3 identification
    samples are kept; ratio >= 1 gives an empty validation set.
    """
    n = len(samples)
    n_id = min(n, max(3, int(round(ratio * n))))
    order = np.random.default_rng(seed).permutation(n)
    ident = sorted(order[:n_id])
    valid = sorted(order[n_id:])
    return [samples[i] for i in ident], [samples[i] for i in valid]


def validation_error(ident: Sequence[Sample], valid: Sequence[Sample],
                     result: HandEyeResult | np.ndarray,
                     mode: Optional[str] = None) -> Optional[ErrorStats]:
    """Errors of the validation samples against the target pose estimated on ``ident``."""
    if not valid:
        return None
    X, mode = _unpack(result, mode)
    ref = estimate_target_pose(ident, X, mode)
    return ErrorStats.from_errors(per_sample_errors(valid, X, ref, mode))


@dataclass
class Distribution:
    """Distribution of calibrated parameters over resampled problems.

    ``params`` rows are (x, y, z, rx, ry, rz): translation (m) and rotation
    vector (rad) relative to ``mean_T``.
    """

    kind: str                       # "bootstrap" or "repeated_split"
    params: np.ndarray              # (N, 6)
    mean_T: np.ndarray
    validation_errors: np.ndarray   # (N,) mean translational validation error, m (nan if none)
    failures: int = 0

    @property
    def std(self) -> np.ndarray:
        return self.params.std(axis=0, ddof=1) if len(self.params) > 1 else np.zeros(6)

    @property
    def covariance(self) -> np.ndarray:
        return np.cov(self.params.T) if len(self.params) > 1 else np.zeros((6, 6))

    def to_dict(self) -> Dict[str, Any]:
        t, q = matrix_to_pose(self.mean_T)
        ve = self.validation_errors[np.isfinite(self.validation_errors)]
        return {"kind": self.kind, "n": int(len(self.params)), "failures": self.failures,
                "mean_translation": t, "mean_quaternion_xyzw": q,
                "param_names": list(PARAM_NAMES), "std": self.std.tolist(),
                "covariance": self.covariance.tolist(),
                "validation_error_mean": float(ve.mean()) if len(ve) else None,
                "validation_error_std": float(ve.std()) if len(ve) else None}


def _distribution(kind: str, solutions: List[np.ndarray], val_errors: List[float],
                  failures: int) -> Distribution:
    if not solutions:
        raise RuntimeError(f"{kind}: every resampled calibration failed")
    t = np.array([X[:3, 3] for X in solutions])
    q = average_quaternions(_quat(X) for X in solutions)
    mean = np.eye(4)
    mean[:3, :3] = Rotation.from_quat(q).as_matrix()
    mean[:3, 3] = t.mean(axis=0)
    rot = np.array([rotvec_delta(mean[:3, :3], X[:3, :3]) for X in solutions])
    return Distribution(kind, np.hstack([t, rot]), mean, np.array(val_errors), failures)


def _quat(X: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(X[:3, :3]).as_quat()


def _mean_translation_error(ident: Sequence[Sample], valid: Sequence[Sample],
                            X: np.ndarray, mode: str) -> float:
    stats = validation_error(ident, valid, X, mode)
    return stats.translation_mean if stats else float("nan")


def repeated_split(samples: Sequence[Sample], method: Method | str, ratio: float,
                   n: int, seed: int = 0, mode: str = EYE_IN_HAND) -> Distribution:
    """Calibrate on ``n`` random identification sets, validate on the rest."""
    solutions, errors, failures = [], [], 0
    for k in range(n):
        ident, valid = split(samples, ratio, seed + k)
        try:
            X = solve(ident, method, mode).T
        except (ValueError, RuntimeError):
            failures += 1
            continue
        solutions.append(X)
        errors.append(_mean_translation_error(ident, valid, X, mode))
    return _distribution("repeated_split", solutions, errors, failures)


def bootstrap(samples: Sequence[Sample], method: Method | str, n: int,
              seed: int = 0, mode: str = EYE_IN_HAND) -> Distribution:
    """Bootstrap the calibration.

    Each round draws len(samples) indices with replacement and keeps the
    distinct ones (duplicated poses add zero-motion pairs that some solvers
    handle badly). Out-of-bag samples give the validation error.
    """
    rng = np.random.default_rng(seed)
    N = len(samples)
    solutions, errors, failures = [], [], 0
    for _ in range(n):
        idx = np.unique(rng.integers(0, N, N))
        if len(idx) < 3:
            failures += 1
            continue
        ident = [samples[i] for i in idx]
        oob = [samples[i] for i in sorted(set(range(N)) - set(idx.tolist()))]
        try:
            X = solve(ident, method, mode).T
        except (ValueError, RuntimeError):
            failures += 1
            continue
        solutions.append(X)
        errors.append(_mean_translation_error(ident, oob, X, mode))
    return _distribution("bootstrap", solutions, errors, failures)


def depth_vs_pnp(samples: Sequence[Sample]) -> List[Tuple[int, float]]:
    """(index, z_pnp - z_depth) in metres for samples with a depth plane."""
    return [(s.index, s.depth_dz) for s in samples if s.depth_dz is not None]


# --------------------------------------------------------------------------
# Full evaluation used by the GUI and the CLI
# --------------------------------------------------------------------------
@dataclass
class Report:
    result: HandEyeResult
    reference: np.ndarray                    # target pose in its fixed frame (identification)
    ident_indices: List[int]
    valid_indices: List[int]
    excluded_indices: List[int]
    errors: List[SampleError]                # every sample, excluded too
    outliers: List[int]
    ident_stats: Optional[ErrorStats]
    valid_stats: Optional[ErrorStats]
    depth: List[Tuple[int, float]]
    coverage: CoverageReport
    distribution: Optional[Distribution] = None
    T_camera_frame: Optional[np.ndarray] = None   # result expressed in camera_frame
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> str:
        return self.result.mode

    @property
    def T_flange_camera_frame(self) -> Optional[np.ndarray]:
        """Backward-compatible name of ``T_camera_frame`` (eye-in-hand)."""
        return self.T_camera_frame

    def error_of(self, index: int) -> Optional[SampleError]:
        for e in self.errors:
            if e.index == index:
                return e
        return None

    def to_dict(self) -> Dict[str, Any]:
        parent = result_parent(self.mode)
        out: Dict[str, Any] = {"settings": self.settings, "calibration_type": self.mode,
                               f"{parent}_to_optical": self.result.to_dict()}
        if self.T_camera_frame is not None:
            t, q = matrix_to_pose(self.T_camera_frame)
            out[f"{parent}_to_camera_frame"] = {"translation": t, "quaternion_xyzw": q,
                                                "matrix": self.T_camera_frame.tolist()}
        t, q = matrix_to_pose(self.reference)
        out[f"target_in_{target_parent(self.mode)}"] = {"translation": t, "quaternion_xyzw": q}
        out["identification"] = self.ident_indices
        out["validation"] = self.valid_indices
        out["excluded"] = self.excluded_indices
        out["outliers"] = self.outliers
        out["identification_error"] = self.ident_stats.to_dict() if self.ident_stats else None
        out["validation_error"] = self.valid_stats.to_dict() if self.valid_stats else None
        out["per_sample"] = [{"index": e.index, "dx": e.dx, "dy": e.dy, "dz": e.dz,
                              "translation": e.translation, "angle_deg": e.angle}
                             for e in self.errors]
        out["depth_vs_pnp"] = [{"index": i, "dz": dz} for i, dz in self.depth]
        out["coverage"] = self.coverage.to_dict()
        out["distribution"] = self.distribution.to_dict() if self.distribution else None
        return out


def evaluate(samples: Sequence[Sample], method: Method | str = Method.TSAI,
             ident_ratio: float = 0.7, seed: int = 0, bootstrap_n: int = 0,
             n_sigma: float = 3.0, T_optical_camera: Optional[np.ndarray] = None,
             distance_range: Tuple[float, float] = (0.2, 1.0),
             mode: str = EYE_IN_HAND) -> Report:
    """Calibrate on the identification set and assess every sample.

    Args:
        samples: all samples; excluded ones are reported but not used.
        T_optical_camera: pose of ``camera_frame`` in the optical frame, used to
            express the result in ``camera_frame``.
        mode: ``"eye_in_hand"`` or ``"eye_to_hand"``.
    """
    method = Method.parse(method)
    check_mode(mode)
    used = [s for s in samples if not s.excluded]
    ident, valid = split(used, ident_ratio, seed)
    result = solve(ident, method, mode)
    reference = estimate_target_pose(ident, result.T, mode)
    errors = per_sample_errors(samples, result, reference)
    used_idx = {s.index for s in used}
    outliers = flag_outliers([e for e in errors if e.index in used_idx], n_sigma)
    ident_idx = {s.index for s in ident}
    valid_idx = {s.index for s in valid}
    report = Report(
        result=result, reference=reference,
        ident_indices=sorted(ident_idx), valid_indices=sorted(valid_idx),
        excluded_indices=sorted(s.index for s in samples if s.excluded),
        errors=errors, outliers=outliers,
        ident_stats=ErrorStats.from_errors([e for e in errors if e.index in ident_idx]),
        valid_stats=ErrorStats.from_errors([e for e in errors if e.index in valid_idx]),
        depth=depth_vs_pnp(samples), coverage=coverage_report(used, distance_range),
        settings={"method": method.name, "calibration_type": mode,
                  "ident_ratio": float(ident_ratio), "seed": int(seed),
                  "bootstrap_n": int(bootstrap_n), "n_sigma": float(n_sigma)})
    if bootstrap_n > 0:
        report.distribution = bootstrap(used, method, bootstrap_n, seed, mode)
    if T_optical_camera is not None:
        report.T_camera_frame = result.T @ T_optical_camera
    return report


def format_report(report: Report) -> str:
    """Human readable summary (used by the GUI and the CLI)."""
    lines = []
    r = report.result
    t, q = r.translation, r.quaternion
    parent = result_parent(report.mode)
    lines.append(f"Method: {r.method.name}   type: {report.mode.replace('_', '-')}   "
                 f"identification samples: {r.n_samples}")
    lines.append(f"{parent} -> optical frame:")
    lines.append(f"  xyz [m]   = {t[0]: .6f} {t[1]: .6f} {t[2]: .6f}")
    lines.append(f"  q xyzw    = {q[0]: .6f} {q[1]: .6f} {q[2]: .6f} {q[3]: .6f}")
    for row in r.T:
        lines.append("  [" + " ".join(f"{v: .6f}" for v in row) + "]")
    if report.T_camera_frame is not None:
        tc, qc = matrix_to_pose(report.T_camera_frame)
        lines.append(f"{parent} -> camera frame:")
        lines.append(f"  xyz [m]   = {tc[0]: .6f} {tc[1]: .6f} {tc[2]: .6f}")
        lines.append(f"  q xyzw    = {qc[0]: .6f} {qc[1]: .6f} {qc[2]: .6f} {qc[3]: .6f}")

    def stats(name: str, s: Optional[ErrorStats]) -> None:
        if s is None:
            lines.append(f"{name}: n/a")
            return
        lines.append(f"{name} (n={s.n}): translation {1e3 * s.translation_mean:.2f} "
                     f"+/- {1e3 * s.translation_std:.2f} mm (max {1e3 * s.translation_max:.2f}), "
                     f"angle {s.angle_mean:.3f} +/- {s.angle_std:.3f} deg (max {s.angle_max:.3f})")
    lines.append(f"Chain-closure errors: target pose in the {target_parent(report.mode)} frame")
    stats("Identification error", report.ident_stats)
    stats("Validation error", report.valid_stats)
    lines.append(f"Outliers: {report.outliers or 'none'}   excluded: {report.excluded_indices or 'none'}")
    d = report.distribution
    if d is not None:
        lines.append(f"{d.kind} (n={len(d.params)}, failures={d.failures}):")
        lines.append("  std  x,y,z [mm] = " + " ".join(f"{1e3 * v:.3f}" for v in d.std[:3]))
        lines.append("  std  rx,ry,rz [deg] = " + " ".join(f"{np.degrees(v):.4f}" for v in d.std[3:]))
        ve = d.validation_errors[np.isfinite(d.validation_errors)]
        if len(ve):
            lines.append(f"  validation error = {1e3 * ve.mean():.2f} +/- {1e3 * ve.std():.2f} mm")
        lines.append("  covariance (m, rad):")
        for row in d.covariance:
            lines.append("    [" + " ".join(f"{v: .2e}" for v in row) + "]")
    if report.depth:
        dz = np.array([v for _, v in report.depth])
        lines.append(f"Depth - PnP: mean {1e3 * dz.mean():.2f} mm, std {1e3 * dz.std():.2f} mm "
                     f"over {len(dz)} samples")
    c = report.coverage
    lines.append(f"Coverage index: {c.index:.2f} (views {c.view_occupied}/{c.view_total}, "
                 f"rolls {c.roll_occupied}/8, distances {c.distance_occupied}/5)")
    return "\n".join(lines)


__all__ = ["SampleError", "ErrorStats", "Distribution", "Report", "per_sample_errors",
           "target_pose", "estimate_target_pose", "flag_outliers", "split", "validation_error",
           "repeated_split", "bootstrap", "depth_vs_pnp", "evaluate", "format_report",
           "solve_matrices"]
