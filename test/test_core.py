# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Core tests on synthetic eye-in-hand data with known ground truth."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from handeye_gui.core import CalibConfig, Method, Sample, Session, evaluate, solve
from handeye_gui.core import evaluation, geometry, poses
from handeye_gui.core.coverage import coverage_report
from handeye_gui.core.geometry import invert, pose7_to_matrix, rotation_angle

X_TRUE = pose7_to_matrix([0.03, -0.045, 0.06, *Rotation.from_euler("xyz", [2, -3, 91], degrees=True).as_quat()])
T_BASE_TARGET = pose7_to_matrix([0.6, 0.1, 0.0, *Rotation.from_euler("xyz", [180, 0, 10], degrees=True).as_quat()])
CENTER = np.array([0.07, 0.1, 0.0])


def make_samples(n=20, noise_t=0.0, noise_r=0.0, seed=1):
    rng = np.random.default_rng(seed)
    cfg = CalibConfig()
    out = []
    for i in range(n):
        T_tc = poses.random_camera_pose(rng, CENTER, cfg)          # camera in target
        T_bf = poses.flange_goal(T_BASE_TARGET, T_tc, X_TRUE)
        T_ct = invert(T_tc)
        if noise_t or noise_r:
            N = np.eye(4)
            N[:3, :3] = Rotation.from_rotvec(rng.normal(0, noise_r, 3)).as_matrix()
            N[:3, 3] = rng.normal(0, noise_t, 3)
            T_ct = T_ct @ N
        s = Sample(joint_names=["j1", "j2"], joint_values=[0.1 * i, -0.1 * i])
        s.T_base_flange = T_bf
        s.T_camera_target = T_ct
        s.target_distance = float(np.linalg.norm(T_ct[:3, 3]))
        out.append(s)
    for i, s in enumerate(out):
        s.index = i
    return out


@pytest.mark.parametrize("method", list(Method))
def test_solvers_recover_ground_truth(method):
    res = solve(make_samples(), method)
    assert np.allclose(res.T[:3, 3], X_TRUE[:3, 3], atol=1e-4)
    assert rotation_angle(res.T[:3, :3].T @ X_TRUE[:3, :3]) < 1e-3


def test_solve_needs_three_samples():
    with pytest.raises(ValueError):
        solve(make_samples(2))


def test_errors_zero_without_noise():
    samples = make_samples()
    errs = evaluation.per_sample_errors(samples, X_TRUE)
    assert max(e.translation for e in errs) < 1e-9
    ref = evaluation.estimate_target_pose(samples, X_TRUE)
    assert np.allclose(ref, T_BASE_TARGET, atol=1e-9)


def test_split_reproducible_and_disjoint():
    s = make_samples(10)
    a1, b1 = evaluation.split(s, 0.7, seed=3)
    a2, b2 = evaluation.split(s, 0.7, seed=3)
    assert [x.index for x in a1] == [x.index for x in a2]
    assert len(a1) == 7 and len(b1) == 3
    assert not {x.index for x in a1} & {x.index for x in b1}
    ident, valid = evaluation.split(s, 1.0)
    assert len(ident) == 10 and not valid


def test_outlier_flagged():
    samples = make_samples(25, noise_t=0.0005, noise_r=0.0005)
    bad = samples[7]
    T = bad.T_camera_target
    T[:3, 3] += [0.03, 0.0, 0.0]
    bad.T_camera_target = T
    rep = evaluate(samples, Method.PARK, ident_ratio=1.0, n_sigma=3.0)
    assert 7 in rep.outliers
    samples[7].excluded = True
    rep2 = evaluate(samples, Method.PARK, ident_ratio=1.0, n_sigma=3.0)
    assert 7 in rep2.excluded_indices and 7 not in rep2.ident_indices
    assert rep2.ident_stats.translation_mean < rep.ident_stats.translation_mean


def test_bootstrap_and_repeated_split():
    samples = make_samples(20, noise_t=0.001, noise_r=0.001)
    d = evaluation.bootstrap(samples, Method.TSAI, 30, seed=0)
    assert d.params.shape[1] == 6 and len(d.params) + d.failures == 30
    assert np.all(d.std[:3] < 0.01)
    assert d.covariance.shape == (6, 6)
    assert np.allclose(d.mean_T[:3, 3], X_TRUE[:3, 3], atol=0.01)
    r = evaluation.repeated_split(samples, Method.TSAI, 0.7, 10, seed=0)
    assert np.all(np.isfinite(r.validation_errors))
    rep = evaluate(samples, Method.TSAI, 0.7, bootstrap_n=10)
    assert rep.distribution is not None
    assert "validation_error" in rep.to_dict()
    assert "flange -> optical" in evaluation.format_report(rep)


def test_session_roundtrip(tmp_path):
    session = Session(CalibConfig(markers_x=5))
    img = np.zeros((20, 30, 3), np.uint8)
    for s in make_samples(5):
        session.add_sample(s, img, img)
    session.set_excluded(2, True)
    session.remove_sample(4)
    session.metadata["T_optical_camera"] = [0, 0, 0, 0, 0, 0, 1]
    session.results = evaluate(session.samples).to_dict()
    session.save(tmp_path)
    loaded = Session.load(tmp_path)
    assert loaded.config.markers_x == 5
    assert [s.index for s in loaded.samples] == [0, 1, 2, 3]
    assert loaded.get(2).excluded
    assert (tmp_path / loaded.get(0).image_path).exists()
    assert np.allclose(loaded.get(1).T_base_flange, session.get(1).T_base_flange)
    assert loaded.results["flange_to_optical"]["method"] == "TSAI"
    new = loaded.add_sample(Sample())
    assert new.index == 5  # indices keep increasing after a load


def test_joint_states_roundtrip(tmp_path):
    session = Session()
    for s in make_samples(4):
        session.add_sample(s)
    path = tmp_path / "joints.yaml"
    assert session.export_joint_states(path) == 4
    names, values = Session.import_joint_states(path)
    assert names == ["j1", "j2"] and len(values) == 4
    assert values[3] == pytest.approx([0.3, -0.3])


def test_config_roundtrip(tmp_path):
    cfg = CalibConfig(joint_names=["a", "b"], detect_inverted=True, marker_width=0.05)
    cfg.save(tmp_path / "c.yaml")
    back = CalibConfig.load(tmp_path / "c.yaml")
    assert back == cfg
    assert back.marker_dimensions()[:2] == (0.05, 0.04)


def test_plane_ransac():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-0.1, 0.1, (2000, 2))
    n = np.array([0.1, -0.2, 1.0]); n /= np.linalg.norm(n)
    z = (0.5 - n[0] * xy[:, 0] - n[1] * xy[:, 1]) / n[2]
    pts = np.column_stack([xy, z]) + rng.normal(0, 0.0005, (2000, 3))
    pts[:200, 2] += rng.uniform(0.05, 0.2, 200)        # outliers
    fit = geometry.fit_plane_ransac(pts, 0.003, rng=rng)
    assert fit is not None and fit.inliers > 1700
    assert abs(fit.distance - 0.5) < 2e-3
    assert np.degrees(np.arccos(fit.normal @ n)) < 1.0


def test_poses_keep_target_on_axis():
    rng = np.random.default_rng(0)
    cfg = CalibConfig()
    for T in [poses.random_camera_pose(rng, CENTER, cfg) for _ in range(20)] + \
            poses.fibonacci_hemisphere(15, 0.4, 35.0, CENTER, alternate_roll_deg=20):
        c_cam = invert(T) @ np.append(CENTER, 1.0)
        assert np.allclose(c_cam[:2], 0.0, atol=1e-9) and c_cam[2] > 0
    hemi = poses.fibonacci_hemisphere(15, 0.4, 35.0, CENTER)
    tilts = [np.degrees(np.arccos(T[2, 2])) for T in hemi]
    assert max(tilts) <= 35.0 + 1e-6


def test_coverage_index_grows():
    few = coverage_report(make_samples(2))
    many = coverage_report(make_samples(40))
    assert 0.0 < few.index < many.index <= 1.0
    assert coverage_report([]).index == 0.0


def test_cli(tmp_path, capsys):
    from handeye_gui import cli
    session = Session()
    for s in make_samples(12, noise_t=0.0005, noise_r=0.0005):
        session.add_sample(s)
    session.save(tmp_path)
    assert cli.main([str(tmp_path), "--method", "all", "--bootstrap", "5", "--write"]) == 0
    out = capsys.readouterr().out
    assert "DANIILIDIS" in out and "Summary" in out
    assert Session.load(tmp_path).results is not None
