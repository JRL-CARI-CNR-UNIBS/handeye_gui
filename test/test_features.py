# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Eye-to-hand, autosave, IK seed search, robot model, ROS name checks, target print."""

import math
import os
import time

import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from handeye_gui.core import CalibConfig, Method, Sample, Session, evaluate, solve
from handeye_gui.core import autosave, evaluation, poses, ros_names, target_print
from handeye_gui.core.export import write_calibration
from handeye_gui.core.geometry import invert, pose7_to_matrix, rotation_angle
from handeye_gui.core.ik_search import IkSearch, nearest_turn
from handeye_gui.core.robot_model import RobotModel
from handeye_gui.core.solvers import EYE_IN_HAND, EYE_TO_HAND
from handeye_gui.core.target import BoardModel

# Eye-to-hand ground truth: camera fixed in the cell, target on the flange.
T_BASE_CAMERA = pose7_to_matrix([0.7, -0.1, 0.9, *Rotation.from_euler("xyz", [178, 3, 85],
                                                                      degrees=True).as_quat()])
T_FLANGE_TARGET = pose7_to_matrix([0.02, -0.06, 0.03, *Rotation.from_euler("xyz", [5, 175, 20],
                                                                          degrees=True).as_quat()])
CENTER = np.array([0.07, 0.1, 0.0])


def eye_to_hand_samples(n=20, noise=0.0, seed=2):
    rng = np.random.default_rng(seed)
    cfg = CalibConfig()
    out = []
    for i in range(n):
        T_tc = poses.random_camera_pose(rng, CENTER, cfg)
        T_bf = poses.flange_goal(T_FLANGE_TARGET, T_tc, T_BASE_CAMERA, EYE_TO_HAND)
        T_ct = invert(T_tc)
        if noise:
            N = np.eye(4)
            N[:3, :3] = Rotation.from_rotvec(rng.normal(0, noise, 3)).as_matrix()
            N[:3, 3] = rng.normal(0, noise, 3)
            T_ct = T_ct @ N
        s = Sample(index=i)
        s.T_base_flange = T_bf
        s.T_camera_target = T_ct
        s.target_distance = float(np.linalg.norm(T_ct[:3, 3]))
        out.append(s)
    return out


# -- eye-to-hand ---------------------------------------------------------------
def test_eye_to_hand_goal_is_consistent():
    s = eye_to_hand_samples(1)[0]
    # target in base, through the robot and through the camera, must coincide
    assert np.allclose(s.T_base_flange @ T_FLANGE_TARGET, T_BASE_CAMERA @ s.T_camera_target,
                       atol=1e-9)


@pytest.mark.parametrize("method", list(Method))
def test_eye_to_hand_recovers_ground_truth(method):
    res = solve(eye_to_hand_samples(), method, EYE_TO_HAND)
    assert res.mode == EYE_TO_HAND
    assert np.allclose(res.T[:3, 3], T_BASE_CAMERA[:3, 3], atol=1e-4)
    assert rotation_angle(res.T[:3, :3].T @ T_BASE_CAMERA[:3, :3]) < 1e-3


def test_eye_to_hand_evaluation():
    samples = eye_to_hand_samples(20, noise=0.0005)
    rep = evaluate(samples, Method.PARK, 0.7, bootstrap_n=10, mode=EYE_TO_HAND)
    assert rep.mode == EYE_TO_HAND
    assert np.allclose(rep.reference[:3, 3], T_FLANGE_TARGET[:3, 3], atol=0.005)
    d = rep.to_dict()
    assert "base_to_optical" in d and "target_in_flange" in d
    text = evaluation.format_report(rep)
    assert "base -> optical" in text and "flange frame" in text
    # solving eye-to-hand data as eye-in-hand gives large chain-closure errors
    wrong = evaluate(samples, Method.PARK, 1.0, mode=EYE_IN_HAND)
    assert wrong.ident_stats.translation_mean > 10 * rep.ident_stats.translation_mean


def test_hand_eye_guess_follows_mode():
    cfg = CalibConfig(calibration_type="eye_to_hand")
    assert np.allclose(poses.hand_eye_guess(cfg), pose7_to_matrix(cfg.initial_base_to_camera))
    cfg = CalibConfig()
    assert np.allclose(poses.hand_eye_guess(cfg), pose7_to_matrix(cfg.initial_flange_to_camera))


def test_export_mode(tmp_path):
    write_calibration(tmp_path / "c.yaml", "base_link", "camera_link", [0] * 6 + [1], EYE_TO_HAND)
    data = yaml.safe_load((tmp_path / "c.yaml").read_text())
    assert data["calibration_type"] == EYE_TO_HAND and data["parent_frame"] == "base_link"
    write_calibration(tmp_path / "c.py", "base_link", "camera_link", [0] * 6 + [1], EYE_TO_HAND)
    compile((tmp_path / "c.py").read_text(), "c.py", "exec")


# -- autosave ------------------------------------------------------------------
def _session_with_samples(n=3):
    session = Session()
    img = np.zeros((8, 8, 3), np.uint8)
    for s in eye_to_hand_samples(n):
        session.add_sample(s, img, img)
    return session


def test_unsaved_session_lives_under_temp_root(temp_root):
    session = Session()
    assert session.is_temporary and session.folder.parent == temp_root


def test_autosave_and_restore_temporary(temp_root):
    session = _session_with_samples()
    saver = autosave.Autosaver()
    folder = saver.save(session)
    assert folder == session.folder
    assert not autosave.find_orphans()          # owned by this (running) process
    marker = folder / autosave.MARKER_FILE
    data = yaml.safe_load(marker.read_text())
    data["pid"] = 999999999                     # simulate a crashed process
    marker.write_text(yaml.safe_dump(data))
    orphans = autosave.find_orphans()
    assert len(orphans) == 1 and orphans[0].n_samples == 3
    restored = autosave.restore(orphans[0])
    assert restored.is_temporary and len(restored.samples) == 3
    assert (restored.folder / restored.samples[0].image_path).exists()
    assert not autosave.find_orphans()          # claimed again


def test_autosave_mirror_of_saved_session(tmp_path, temp_root):
    session = _session_with_samples()
    user = tmp_path / "user"
    session.save(user)
    saver = autosave.Autosaver()
    session.remove_sample(0)
    mirror = saver.save(session)
    assert mirror != user and mirror.parent == temp_root
    assert len(Session.load(mirror).samples) == 2
    assert len(Session.load(user).samples) == 3            # user folder untouched
    info = yaml.safe_load((mirror / autosave.MARKER_FILE).read_text())
    assert info["source_folder"] == str(user)
    saver.discard(session)
    assert not mirror.exists() and user.exists()


def test_discard_temporary_and_empty_cleanup(temp_root):
    session = _session_with_samples()
    saver = autosave.Autosaver()
    saver.save(session)
    saver.discard(session)
    assert not session.folder.exists()
    empty = Session().folder
    old = time.time() - 2 * autosave.EMPTY_MIN_AGE
    os.utime(empty, (old, old))
    assert not autosave.find_orphans() and not empty.exists()


def test_atomic_write_leaves_no_temp_files(tmp_path):
    session = _session_with_samples()
    session.write_files()
    assert not [p for p in session.folder.iterdir() if p.name.endswith(".tmp")]


# -- IK search -----------------------------------------------------------------
def test_nearest_turn():
    assert nearest_turn(math.radians(350), math.radians(-5)) == pytest.approx(math.radians(-10))
    # out of limits: keep the IK value
    assert nearest_turn(3.0, -3.0, -math.pi, math.pi) == 3.0
    assert nearest_turn(3.0 - 2 * math.pi, 3.0, -2 * math.pi, 2 * math.pi) == pytest.approx(3.0)


def test_ik_search_accepts_and_rejects():
    cur = {"a": 0.0, "b": 1.0, "gripper": 0.02}
    s = IkSearch(cur, ["a", "b"], math.radians(30), attempts=5,
                 limits={"a": (-2 * math.pi, 2 * math.pi)}, rng=np.random.default_rng(0))
    first = s.next_seed()
    assert first == {"a": 0.0, "b": 1.0}
    # 'a' is a full turn away: the equivalent angle is used
    q = s.evaluate({"a": 2 * math.pi + 0.1, "b": 1.2, "gripper": 0.02})
    assert q == pytest.approx({"a": 0.1, "b": 1.2})
    assert s.evaluate({"a": 1.5, "b": 1.0}) is None
    assert s.best[1] == "a"
    seeds = [s.next_seed() for _ in range(4)]
    assert all(abs(x["b"] - 1.0) <= math.radians(30) + 1e-9 for x in seeds)
    assert s.next_seed() is None
    assert "after 5 attempts" in s.failure_message()


def test_ik_search_amplitude_grows():
    s = IkSearch({"a": 0.0}, ["a"], 1.0, attempts=10)
    amps = [s.amplitude(k) for k in range(1, 10)]
    assert amps == sorted(amps) and amps[-1] == pytest.approx(1.0)


def test_ik_search_missing_joint():
    with pytest.raises(ValueError):
        IkSearch({"a": 0.0}, ["a", "b"], 1.0, 3)


# -- robot model -----------------------------------------------------------------
URDF = """<robot name="r">
  <link name="base"/><link name="l1"/><link name="l2"/><link name="tool"/><link name="finger"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="l1"/>
    <limit lower="-6.28" upper="6.28" effort="1" velocity="1"/></joint>
  <joint name="j2" type="continuous"><parent link="l1"/><child link="l2"/></joint>
  <joint name="j3" type="fixed"><parent link="l2"/><child link="tool"/></joint>
  <joint name="g" type="prismatic"><parent link="tool"/><child link="finger"/>
    <limit lower="0" upper="0.04" effort="1" velocity="1"/></joint>
</robot>"""
SRDF = """<robot name="r">
  <group name="arm"><chain base_link="base" tip_link="tool"/></group>
  <group name="hand"><joint name="g"/></group>
  <group name="all"><group name="arm"/><group name="hand"/></group>
</robot>"""


def test_robot_model():
    m = RobotModel.from_xml(URDF, SRDF)
    assert m.groups == {"arm": ["j1", "j2"], "hand": ["g"], "all": ["j1", "j2", "g"]}
    assert m.limits("j1") == (-6.28, 6.28)
    assert m.limits("j2") == (-math.inf, math.inf)
    assert not m.is_angular("g") and m.is_angular("j2")
    assert RobotModel.from_xml(URDF).groups is None


# -- ROS names -------------------------------------------------------------------
def _snapshot():
    return ros_names.GraphSnapshot(
        topics={"/cam/color/image_raw": [ros_names.IMAGE],
                "/cam/color/camera_info": [ros_names.CAMERA_INFO],
                "/cam/depth/image_rect_raw": [ros_names.IMAGE],
                "/joint_states": [ros_names.JOINT_STATE],
                "/silent": [ros_names.IMAGE]},
        publishers={"/cam/color/image_raw": 1, "/silent": 0},
        actions={"/move_action": ["moveit_msgs/action/MoveGroup"]},
        services={"/compute_ik": ["moveit_msgs/srv/GetPositionIK"]},
        frames=["base_link", "tool0"], groups=["manipulator"])


def test_name_checks():
    snap = _snapshot()
    c = ros_names.check
    assert c("image_topic", "/cam/color/image_raw", snap)[0] == ros_names.OK
    assert c("image_topic", "cam/color/image_raw", snap)[0] == ros_names.OK   # relative name
    assert c("image_topic", "/nope", snap)[0] == ros_names.ERROR
    assert c("image_topic", "/joint_states", snap)[0] == ros_names.ERROR      # wrong type
    assert c("image_topic", "/silent", snap)[0] == ros_names.ERROR           # no publisher
    assert "/silent" not in ros_names.suggestions("image_topic", snap)
    assert c("topic:std_msgs/msg/String", "/robot_description", snap)[0] == ros_names.WARN
    assert c("depth_topic", "", snap)[0] == ros_names.OK
    assert c("frame", "tool0", snap)[0] == ros_names.OK
    assert c("frame", "camera_link", snap)[0] == ros_names.ERROR
    assert c("planning_group", "arm", snap)[0] == ros_names.ERROR
    assert c("action:moveit_msgs/action/MoveGroup", "/move_action", snap)[0] == ros_names.OK
    assert c("action:moveit_msgs/action/ExecuteTrajectory", "/execute_trajectory",
             snap)[0] == ros_names.ERROR
    assert c("service:moveit_msgs/srv/GetPositionIK", "/compute_ik", snap)[0] == ros_names.OK
    assert c("frame", "x", None)[0] == ros_names.UNKNOWN
    snap.groups = None
    assert c("planning_group", "arm", snap)[0] == ros_names.UNKNOWN


def test_suggestions_and_camera_info():
    snap = _snapshot()
    assert ros_names.suggestions("depth_topic", snap)[0] == "/cam/depth/image_rect_raw"
    assert ros_names.suggestions("image_topic", snap)[0] == "/cam/color/image_raw"
    assert ros_names.suggestions("frame", snap) == ["base_link", "tool0"]
    assert ros_names.camera_info_for("/cam/color/image_raw", snap) == "/cam/color/camera_info"
    assert ros_names.camera_info_for("/other/image", snap) is None


def test_config_fields_have_valid_suggest_kinds():
    for f in CalibConfig.__dataclass_fields__.values():
        kind = f.metadata.get("suggest")
        if kind:
            base, _ = ros_names.split_kind(kind)
            assert base in set(ros_names.TOPIC_KINDS) | {"frame", "planning_group", "action",
                                                         "service", "topic"}


# -- target print ------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [CalibConfig(),
                                 CalibConfig(aruco_dictionary="DICT_5X5_100", markers_x=4,
                                             markers_y=2, first_marker_id=20, marker_size=0.03,
                                             marker_separation=0.006)])
def test_rendered_target_is_detected(cfg):
    img = target_print.render_image(cfg, 6.0)
    det = BoardModel(cfg).detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    assert sorted(det.ids.ravel().tolist()) == sorted(BoardModel(cfg).ids.tolist())


def test_target_export_real_size(tmp_path):
    cfg = CalibConfig()
    page = target_print.save_target(cfg, tmp_path / "t.pdf")
    assert page == "A4 portrait" and (tmp_path / "t.pdf").stat().st_size > 0
    fig = target_print.target_figure(cfg)
    w, h = fig.get_size_inches() * target_print.INCH
    assert (w, h) == pytest.approx((0.210, 0.297))
    with pytest.raises(ValueError):
        BoardModel(CalibConfig(markers_x=10, markers_y=10))     # 100 ids, DICT_4X4_50
    big = CalibConfig(aruco_dictionary="DICT_4X4_250", markers_x=10, markers_y=10,
                      marker_size=0.05)
    assert target_print.page_size((0.6, 0.6))[0] == "custom"
    assert target_print.save_target(big, tmp_path / "big.svg") == "custom"
