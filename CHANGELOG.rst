^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package handeye_gui
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.2.0 (2026-11-04)
------------------
* Eye-to-hand calibration (camera fixed in the cell, target on the flange),
  selected with the new ``calibration_type`` setting. Same OpenCV solvers with
  inverted robot poses; the result is base -> optical (and base ->
  ``camera_frame``). Pose generation, scatter plots, report, export and CLI
  follow the selected type. New ``initial_base_to_camera`` guess.
* Pose goals are solved with MoveIt IK (``/compute_ik``, any IK plugin) and
  rejected when a joint would move more than ``max_joint_step`` (default 90 deg)
  from the current configuration. Further seeds are drawn around the current
  configuration in a growing neighbourhood, up to ``ik_attempts``. Joints with a
  range beyond one turn are moved by whole turns to the nearest equivalent angle.
  Joint limits and planning-group joints are read from the URDF/SRDF topics.
  ``max_joint_step = 0`` keeps the previous behaviour.
* *View target* window in the Configuration tab: the configured grid with marker
  ids and target frame, and export at real size (PDF/SVG/PNG, on A4/A3 when it fits).
* Autosave of the session after every change in
  ``/tmp/handeye_calibration_gui_temporary_save`` (atomic writes); saved sessions
  are mirrored there, so the user folder changes only on an explicit save. After
  a crash the next start offers to restore the autosave.
* Drop-downs with the topics, TF frames, planning groups, actions and services
  found in the ROS graph. Choosing the image topic fills the camera info topic.
  Missing names are shown in red; Plan and Execute are disabled while the MoveIt
  names are wrong.
* Configuration changes not applied yet are flagged (``*`` in the tab title,
  highlighted Apply button, Revert button) with a warning when leaving the tab:
  apply, discard or stay.
* A grid that needs marker ids beyond the size of the ArUco dictionary now gives
  a clear error instead of an OpenCV exception.
* Session YAML files are written atomically (temporary file + rename).
* Patch for matplotlib < 3.6 with PySide6 >= 6.4 (Ubuntu 22.04 / Humble).
* API: ``solve``, ``evaluate``, ``estimate_target_pose``, ``per_sample_errors``,
  ``bootstrap``, ``repeated_split``, ``flange_goal`` and ``write_calibration`` take
  a calibration type (default eye-in-hand). ``Report.T_flange_camera_frame`` is
  now ``Report.T_camera_frame``; the old names still work. In eye-to-hand,
  ``results.yaml`` uses the keys ``base_to_optical`` and ``target_in_flange``.
* New dependencies: ``std_msgs``, ``builtin_interfaces``.

0.1.0 (2026-10-30)
------------------
* First release: eye-in-hand calibration GUI (PySide6, no RViz) with ArUco grid
  detection, PnP, depth-plane cross-check, stability indicator, MoveIt motion
  (joint-state files, Fibonacci hemisphere, random poses), the five OpenCV
  solvers, identification/validation split, bootstrap, outlier detection,
  coverage index, session folders and offline CLI.
