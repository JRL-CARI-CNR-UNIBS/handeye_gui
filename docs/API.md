# handeye_gui Python API

The `handeye_gui.core` package has no dependency on Qt or ROS and can be used
in scripts and notebooks. Units are metres and radians unless a name says
`_deg`; quaternions are `(x, y, z, w)`.

## core.config

### `CalibConfig`

Dataclass with every setting. Each field carries GUI metadata (`group`, `label`,
`help`, ranges, and `suggest` for fields that name a ROS entity); the
Configuration tab is generated from it.

| Group | Fields |
| --- | --- |
| Calibration | `calibration_type` (`eye_in_hand` or `eye_to_hand`) |
| Target | `aruco_dictionary`, `markers_x`, `markers_y`, `marker_size`, `marker_width`, `marker_height`, `marker_separation`, `marker_separation_y`, `first_marker_id`, `detect_inverted` |
| Frames | `base_frame`, `flange_frame`, `camera_frame` |
| Topics | `image_topic`, `camera_info_topic`, `depth_topic` (empty = off), `joint_states_topic` |
| Stability | `stability_window` [s], `stability_pos_threshold` [m], `stability_rot_threshold` [deg] |
| Motion | `planning_group`, `move_group_action`, `execute_action`, `velocity_scaling`, `acceleration_scaling`, `planning_time`, `joint_names`, `max_joint_step` [deg, 0 = no IK check], `ik_attempts`, `ik_service`, `robot_description_topic`, `initial_flange_to_camera` (eye-in-hand), `initial_base_to_camera` (eye-to-hand) |
| Pose generation | `random_z_min/max`, `random_roll/pitch/yaw_min/max` [deg], `hemisphere_n`, `hemisphere_distance`, `hemisphere_max_polar` [deg] |
| Evaluation | `outlier_sigma`, `max_reprojection_error` [px], `depth_ransac_threshold` [m], `depth_roi_shrink` |

Methods: `marker_dimensions() -> (w, h, sx, sy)`, `to_dict()`,
`from_dict(d)` (unknown keys ignored with a warning), `save(path)`, `load(path)`.

## core.sample

### `Sample`

One acquisition: `index`, `timestamp`, `robot_translation/rotation`
(base -> flange), `target_translation/rotation` (camera -> target),
`joint_names/values`, `image_path`, `decorated_image_path`, PnP metrics
(`reprojection_error_mean/max`, `n_markers`, `n_corners`, `target_distance`,
`target_tilt` [deg], `pnp_center_z`), depth metrics (`depth_plane_distance`,
`depth_plane_normal`, `depth_inliers`, `depth_center_z`), flags `excluded`
and `forced`.

Properties: `T_base_flange`, `T_camera_target` (4x4, read/write), `depth_dz`
(PnP minus depth Z at the target centre, or `None`).

## core.session

### `Session(config=None, folder=None, temporary=None)`

Configuration + samples. Without `folder` a new working folder under
`TEMP_ROOT` (`/tmp/handeye_calibration_gui_temporary_save`) is used until the
first `save`. YAML files are written atomically (`write_yaml_atomic`).

| Method | Description |
| --- | --- |
| `add_sample(sample, image=None, decorated=None)` | assigns `sample.index`, writes the BGR images |
| `remove_sample(index)` | removes by sample index |
| `set_excluded(index, value)` | exclusion flag |
| `get(index)`, `active_samples()` | access |
| `save(folder)` / `Session.load(folder, temporary=False)` | `config.yaml`, `samples.yaml`, `results.yaml`, `images/`; `save` moves the session to `folder` |
| `write_files(folder=None)` | writes everything to `folder` (default: own folder) without moving the session |
| `export_joint_states(path)` / `import_joint_states(path)` | moveit_calibration format (`joint_names`, `joint_values`) |

Attributes: `results` (dict written to `results.yaml`), `metadata` (e.g.
`image_frame`, `T_optical_camera` as `[x y z qx qy qz qw]`).

## core.autosave

- `Autosaver`: `save(session)` (own folder for unsaved sessions, a mirror under
  `TEMP_ROOT` for saved ones), `claim(session)`, `discard(session)`.
- `find_orphans()`: autosaves of processes that are no longer running, newest
  first (`AutosaveInfo`: `folder`, `saved_at`, `n_samples`, `source_folder`).
- `restore(info) -> Session` (unsaved session), `discard(info)`.

## core.solvers

- `Method`: enum `TSAI`, `PARK`, `HORAUD`, `ANDREFF`, `DANIILIDIS`;
  `Method.parse("park")`.
- `EYE_IN_HAND`, `EYE_TO_HAND`; `robot_pose(sample, mode)` (T_base_flange or
  its inverse), `result_parent(mode)` ("flange"/"base"), `target_parent(mode)`.
- `solve(samples, method, mode=EYE_IN_HAND) -> HandEyeResult`: needs >= 3 samples.
  The caller chooses the samples (excluded ones are not filtered here).
- `solve_matrices(A_list, T_camera_target_list, method) -> X`.
- `HandEyeResult`: `T` (flange -> optical, or base -> optical for eye-to-hand),
  `method`, `n_samples`, `mode`, `translation`, `quaternion`, `to_dict()`.

## core.evaluation

| Function | Returns |
| --- | --- |
| `target_pose(sample, X, mode)` | target in its fixed frame (base, or flange for eye-to-hand) |
| `per_sample_errors(samples, result, reference=None, mode=None)` | list of `SampleError(index, position, dx, dy, dz, angle[deg])`; `translation` property |
| `estimate_target_pose(samples, X, mode=EYE_IN_HAND)` | mean target pose in its fixed frame |
| `split(samples, ratio, seed)` | `(ident, valid)`, reproducible, >= 3 identification samples |
| `validation_error(ident, valid, result, mode=None)` | `ErrorStats` on the validation set |
| `repeated_split(samples, method, ratio, n, seed, mode)` | `Distribution` |
| `bootstrap(samples, method, n, seed, mode)` | `Distribution` (out-of-bag validation error) |
| `flag_outliers(errors, n_sigma)` | sample indices above mean + n_sigma·std (translation or angle) |
| `depth_vs_pnp(samples)` | `[(index, dz)]` |
| `evaluate(samples, method, ident_ratio, seed, bootstrap_n, n_sigma, T_optical_camera, distance_range, mode)` | `Report` |
| `format_report(report)` | text summary |

`Distribution`: `params` (N x 6: x, y, z, rx, ry, rz; rotation vector around
`mean_T`), `mean_T`, `std`, `covariance`, `validation_errors`, `failures`.

`Report`: `result`, `reference`, `ident_indices`, `valid_indices`,
`excluded_indices`, `errors`, `outliers`, `ident_stats`, `valid_stats`,
`depth`, `coverage`, `distribution`, `T_camera_frame` (result expressed in
`camera_frame`; old name `T_flange_camera_frame` still works), `settings`, `mode`;
`error_of(index)`, `to_dict()` (keys `flange_to_optical`/`target_in_base` for
eye-in-hand, `base_to_optical`/`target_in_flange` for eye-to-hand).

## core.coverage

`coverage_report(samples, distance_range=(0.2, 1.0)) -> CoverageReport` with
distance and tilt histograms and `index` in [0, 1]: the mean of the occupied
fractions of 33 viewing-direction bins, 8 roll bins and 5 distance bins.

## core.target

- `BoardModel(config)`: `detect(gray) -> Detection`,
  `estimate_pose(detection, K, D) -> PnpResult | None`,
  `draw(image, detection, pnp, K, D, depth=None, extra_lines=())`; attributes
  `center`, `size`, `ids`. Raises `ValueError` if the grid needs ids beyond the
  dictionary size.
- `depth_plane(depth_m, K, polygon_px, shrink, threshold, pnp_center)
  -> DepthResult | None`: depth must be aligned to the colour image.
- `depth_to_meters(depth)`: 16UC1 (mm) or 32FC1 (m) to float metres.

## core.target_print

- `target_figure(config, show_ids=False, show_axes=False, caption=True, page=True)`:
  matplotlib `Figure` with the markers as vector rectangles at real size.
- `save_target(config, path)`: PDF/SVG/PNG on an A4/A3 page when the board fits;
  returns the page name.
- `render_image(config, px_per_mm)`: RGB raster; `describe(config)`.

## core.poses

- `orbit_camera_pose(center, distance, roll, pitch, yaw)`: `T_target_camera`.
- `random_camera_pose(rng, center, config)`: within the config bounds.
- `fibonacci_hemisphere(n, distance, max_polar_deg, center, roll_deg=0, alternate_roll_deg=0)`.
- `flange_goal(T_fixed_target, T_target_camera, X, mode=EYE_IN_HAND) -> T_base_flange`:
  eye-in-hand moves the camera around the fixed target, eye-to-hand moves the
  target in front of the fixed camera with the same relative pose.
- `hand_eye_guess(config, result_T=None)`, `fixed_target_guess(samples, X, mode)`
  (old names `flange_camera_guess`, `target_in_base_guess` still work).

## core.ik_search

`IkSearch(current, joints, max_step, attempts, limits, angular)`: `next_seed()`
(current configuration first, then random seeds in a growing neighbourhood),
`evaluate(solution)` (moves revolute joints by whole turns towards the current
value when the limits allow it, accepts if every joint moves <= `max_step`),
`failure_message()`. `nearest_turn(q, ref, lower, upper)`.

## core.robot_model

`RobotModel.from_xml(urdf, srdf)`: `joints` (`JointInfo` with type and limits),
`groups` (planning group -> actuated joints; `<joint>`, `<link>`, `<chain>` and
nested `<group>` supported), `limits(name)`, `is_angular(name)`.

## core.ros_names

`GraphSnapshot` (topics with publisher counts, action and service servers, TF
frames, planning groups), `suggestions(kind, snapshot)`,
`check(kind, value, snapshot) -> (status, message)` with status `ok`, `warn`,
`error` or `unknown`, `camera_info_for(image_topic, snapshot)`.

## core.geometry

`pose_to_matrix`, `matrix_to_pose`, `pose7_to_matrix`, `matrix_to_pose7`,
`invert`, `rotation_angle`, `average_quaternions`, `average_poses`,
`pose_dispersion`, `rotvec_delta`, `fit_plane_ransac -> PlaneFit`.

## core.export

`write_calibration(path, parent, child, pose7, mode="eye_in_hand")`: a
`static_transform_publisher` launch file when `path` ends in `.py`, YAML otherwise.

## ROS layer (`handeye_gui.ros`)

### `RosThread(config, args)` — `QThread`

Signals: `image_ready(rgb, info)`, `stability_changed(stable, info)`,
`motion_done(kind, ok, message)`, `sample_ready(sample, image, decorated, meta)`,
`status(text)`, `error(text)`.

Slots: `apply_config(config)`, `request_sample(forced)`, `plan_current(vel, acc)`,
`plan_pose(T_base_flange, vel, acc)`, `execute()`, `plan_and_execute_next(vel, acc)`,
`stop_motion()`, `set_candidates(list[Candidate])`, `stop()`.
`graph_snapshot() -> GraphSnapshot` (called from the GUI thread).

### Managers

- `CameraManager`: subscriptions, worker thread for detection/PnP/depth,
  `latest() -> Frame`.
- `TfManager`: `lookup(target, source, stamp)`, `flange_pose(stamp)`,
  stability over `stability_window`.
- `MotionManager`: `plan(joints, vel, acc)`, `plan_pose(T, vel, acc)` (through
  `/compute_ik` and `IkSearch` when `max_joint_step > 0`), `execute()`,
  `plan_and_execute_next(vel, acc)`, `stop()` (also cancels an IK search),
  candidate list (`Candidate` with joints or a flange pose), `robot`
  (`RobotDescription`: URDF/SRDF from their latched topics).

## Example

```python
from handeye_gui.core import Session, evaluate, format_report

session = Session.load("/home/me/calib/s1")
session.set_excluded(7, True)
report = evaluate(session.samples, "PARK", ident_ratio=0.7, bootstrap_n=200)
print(format_report(report))
print(report.result.T)
```
