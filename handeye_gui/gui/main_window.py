# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Main window: owns the Session and the RosThread, wires the tabs together.

Importing this module first runs ``handeye_gui.gui.__init__``, which prepares
the Qt environment (see there).
"""

from __future__ import annotations

import argparse
import copy
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QMainWindow, QMessageBox,
                               QPlainTextEdit, QTabWidget)

from ..core import autosave
from ..core.config import CalibConfig
from ..core.coverage import coverage_report
from ..core.evaluation import Report, evaluate
from ..core.export import write_calibration
from ..core.geometry import matrix_to_pose7, pose7_to_matrix
from ..core.poses import (fibonacci_hemisphere, fixed_target_guess, flange_goal, hand_eye_guess,
                          random_camera_pose)
from ..core.ros_names import ERROR, MOTION_FIELDS
from ..core.session import Session, read_joint_states
from ..core.solvers import result_parent
from ..core.target import BoardModel
from .tab_acquisition import AcquisitionTab
from .tab_config import ConfigTab
from .tab_results import ResultsTab

GRAPH_REFRESH_MS = 3000     # ROS names are re-read this often
CONFIG_TAB = 0


class ComputeWorker(QThread):
    """Runs ``evaluate`` outside the GUI thread (bootstrap can take a while)."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, samples, kwargs, parent=None):
        super().__init__(parent)
        self._samples = samples
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            self.done.emit(evaluate(self._samples, **self._kwargs))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, session: Session, use_ros: bool = True, ros_args=None,
                 restored: bool = False):
        super().__init__()
        self.session = session
        self.autosaver = autosave.Autosaver()
        self.report: Optional[Report] = None
        self.ros = None
        self._worker: Optional[ComputeWorker] = None
        self._pending_compute = False
        self._dirty = False
        self._rng = np.random.default_rng()
        cfg = session.config

        self.config_tab = ConfigTab(cfg)
        self.acq_tab = AcquisitionTab(cfg.velocity_scaling, cfg.acceleration_scaling)
        self.results_tab = ResultsTab()
        tabs = QTabWidget()
        tabs.addTab(self.config_tab, "Configuration")
        tabs.addTab(self.acq_tab, "Acquisition")
        tabs.addTab(self.results_tab, "Results")
        tabs.setCurrentIndex(1)
        self._current_tab = 1
        self._switching_tab = False
        tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs = tabs
        self.setCentralWidget(tabs)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        dock = QDockWidget("Log", self)
        dock.setObjectName("log")
        dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [120], Qt.Vertical)

        self._build_menu()
        self._connect_tabs()
        self._refresh_samples()
        if session.results:
            self.log_message(f"Loaded session {session.folder} (previous results in results.yaml)")

        if use_ros:
            from ..ros.ros_thread import RosThread  # ROS is optional for offline review
            self.ros = RosThread(cfg, ros_args, self)
            self.ros.image_ready.connect(lambda rgb, info: self.acq_tab.image.set_image(rgb))
            self.ros.stability_changed.connect(self.acq_tab.set_stability)
            self.ros.motion_done.connect(self._on_motion_done)
            self.ros.sample_ready.connect(self._on_sample_ready)
            self.ros.status.connect(self.log_message)
            self.ros.error.connect(self.log_error)
            self.ros.started_ok.connect(lambda: QTimer.singleShot(1500, self.refresh_graph))
            self.ros.start()
            self._graph_timer = QTimer(self)
            self._graph_timer.timeout.connect(self.refresh_graph)
            self._graph_timer.start(GRAPH_REFRESH_MS)
        else:
            self.acq_tab.set_stability(False, {"reason": "offline mode (no ROS)"})
            self.acq_tab.set_motion_enabled(False, "offline mode")
            self.log_message("Offline mode: acquisition and motion are disabled.")
        if restored:
            self._dirty = True
            self.log_message(f"Autosaved session restored from {session.folder}: "
                             f"{len(session.samples)} samples. Save it to keep it.")
        self.autosaver.claim(self.session)
        self._update_title()
        self._fit_to_screen()

    def _fit_to_screen(self) -> None:
        """Start at most 1500x950, never larger than the available screen area."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 950)
            return
        avail = screen.availableGeometry()
        self.resize(min(1500, int(0.95 * avail.width())), min(950, int(0.9 * avail.height())))

    def toggle_fullscreen(self) -> None:
        self.showNormal() if self.isFullScreen() else self.showFullScreen()

    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, slot, key in (("&New session", self.new_session, QKeySequence.New),
                                ("&Open session...", self.open_session, QKeySequence.Open),
                                ("&Save session", self.save_session, QKeySequence.Save),
                                ("Save session &as...", self.save_session_as, QKeySequence.SaveAs),
                                ("&Quit", self.close, QKeySequence.Quit)):
            action = QAction(text, self)
            action.setShortcut(key)
            action.triggered.connect(slot)
            menu.addAction(action)
        view = self.menuBar().addMenu("&View")
        full = QAction("&Full screen", self)
        full.setShortcut(QKeySequence.FullScreen)          # F11 on Linux
        full.triggered.connect(self.toggle_fullscreen)
        view.addAction(full)
        maximize = QAction("&Maximize", self)
        maximize.triggered.connect(lambda: self.showNormal() if self.isMaximized() else self.showMaximized())
        view.addAction(maximize)

    def _connect_tabs(self) -> None:
        a = self.acq_tab
        a.take_sample_requested.connect(self.take_sample)
        a.remove_sample_requested.connect(self.remove_sample)
        a.plan_requested.connect(lambda: self._ros_call("plan_current", a.velocity(), a.acceleration()))
        a.execute_requested.connect(lambda: self._ros_call("execute"))
        a.next_requested.connect(lambda: self._ros_call("plan_and_execute_next",
                                                        a.velocity(), a.acceleration()))
        a.random_requested.connect(self.random_pose)
        a.hemisphere_requested.connect(self.hemisphere_poses)
        a.stop_requested.connect(lambda: self._ros_call("stop_motion"))
        a.save_session_requested.connect(self.save_session_as)
        a.load_session_requested.connect(self.open_session)
        a.export_joints_requested.connect(self.export_joints)
        a.import_joints_requested.connect(self.import_joints)
        a.compute_requested.connect(self.compute)
        self.config_tab.config_applied.connect(self.apply_config)
        self.config_tab.pending_changed.connect(self._on_config_pending)
        self.config_tab.refresh_requested.connect(self.refresh_graph)
        self.results_tab.exclusion_changed.connect(self.set_excluded)
        self.results_tab.export_requested.connect(self.export_calibration)
        self.results_tab.table.sample_selected.connect(self._show_sample_image)

    # -- logging -----------------------------------------------------------
    def log_message(self, text: str) -> None:
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {text}")
        self.statusBar().showMessage(text, 5000)

    def log_error(self, text: str) -> None:
        self.log.appendPlainText(f"[{time.strftime('%H:%M:%S')}] ERROR {text}")
        self.statusBar().showMessage(f"Error: {text.splitlines()[0]}", 8000)

    def _ros_call(self, name: str, *args) -> bool:
        if self.ros is None:
            self.log_error("ROS is disabled (offline mode)")
            return False
        getattr(self.ros, name)(*args)
        self._update_candidates()
        return True

    # -- config ------------------------------------------------------------
    def apply_config(self, cfg: CalibConfig) -> None:
        old = self.session.config
        self.session.config = cfg
        if self.ros is not None:
            self.ros.apply_config(cfg)
        if old.calibration_type != cfg.calibration_type and self.report is not None:
            # The result of the other calibration type means something else.
            self.report = None
            self.session.results = None
            self.results_tab.show_report(self.session.samples, None, cfg.max_reprojection_error)
            self.results_tab.set_status("calibration type changed: press Compute")
        self._mark_dirty()
        self.log_message(f"Configuration updated ({cfg.calibration_type.replace('_', '-')})")
        self._refresh_samples()
        self._update_motion_enabled()

    def _on_config_pending(self, pending: bool) -> None:
        self.tabs.setTabText(CONFIG_TAB, "Configuration *" if pending else "Configuration")

    def _on_tab_changed(self, index: int) -> None:
        """Leaving the Configuration tab with changes not applied: ask what to do."""
        previous, self._current_tab = self._current_tab, index
        if self._switching_tab or previous != CONFIG_TAB or index == CONFIG_TAB:
            return
        if not self.config_tab.has_pending_changes():
            return
        box = QMessageBox(QMessageBox.Warning, "Configuration not applied",
                          "Some configuration changes have not been applied yet, so they are "
                          "not used.", parent=self)
        apply_btn = box.addButton("Apply", QMessageBox.AcceptRole)
        discard_btn = box.addButton("Discard changes", QMessageBox.DestructiveRole)
        box.addButton("Stay in Configuration", QMessageBox.RejectRole)
        box.setDefaultButton(apply_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is apply_btn and self.config_tab.apply():
            return
        if clicked is discard_btn:
            self.config_tab.discard_changes()
            return
        self._switching_tab = True
        self.tabs.setCurrentIndex(CONFIG_TAB)
        self._current_tab = CONFIG_TAB
        self._switching_tab = False

    # -- ROS names -----------------------------------------------------------
    def refresh_graph(self) -> None:
        if self.ros is None:
            return
        self.config_tab.set_graph(self.ros.graph_snapshot())
        self._update_motion_enabled()

    def _update_motion_enabled(self) -> None:
        """Plan/Execute need valid MoveIt names in the applied configuration."""
        if self.ros is None:
            return
        cfg = self.session.config
        fields = list(MOTION_FIELDS) + (["ik_service"] if cfg.max_joint_step > 0 else [])
        statuses = self.config_tab.check_config(cfg)
        bad = [f"{CalibConfig.__dataclass_fields__[f].metadata['label']} {getattr(cfg, f)!r}: "
               f"{statuses[f][1]}" for f in fields if statuses[f][0] == ERROR]
        self.acq_tab.set_motion_enabled(not bad, "; ".join(bad))

    # -- samples -----------------------------------------------------------
    def take_sample(self, forced: bool) -> None:
        self._ros_call("request_sample", forced)

    def _on_sample_ready(self, sample, image, decorated, meta) -> None:
        self.session.add_sample(sample, image, decorated)
        self.session.metadata.update(meta)
        self._mark_dirty()
        self.log_message(f"Sample {sample.index} taken"
                         + (" (forced, pose not stable)" if sample.forced else "")
                         + f": reproj {sample.reprojection_error_mean:.2f} px, "
                         f"dist {sample.target_distance:.3f} m, tilt {sample.target_tilt:.1f} deg")
        self._refresh_samples()

    def remove_sample(self) -> None:
        if not self.session.samples:
            return
        idx = self.results_tab.table.selected_index()
        if idx is None:
            idx = self.session.samples[-1].index
        self.session.remove_sample(idx)
        self._mark_dirty()
        self.log_message(f"Sample {idx} removed")
        self.report = None
        self._refresh_samples()

    def set_excluded(self, index: int, value: bool) -> None:
        self.session.set_excluded(index, value)
        self._mark_dirty()
        self._refresh_samples(table=False)
        if len(self.session.active_samples()) >= 3:
            self.compute()
        else:
            self._refresh_samples()

    def _refresh_samples(self, table: bool = True) -> None:
        s = self.session
        self.acq_tab.set_sample_count(len(s.samples), len(s.active_samples()))
        cfg = s.config
        self.acq_tab.set_coverage(coverage_report(s.active_samples(),
                                                  (cfg.random_z_min, cfg.random_z_max)))
        if table:
            self.results_tab.show_samples(s.samples, cfg.max_reprojection_error)
        self._update_title()

    def _show_sample_image(self, index: int) -> None:
        path = self.session.image_file(self.session.get(index), decorated=True)
        if path is None or not path.exists():
            self.results_tab.preview.setText("no image")
            return
        import cv2
        img = cv2.imread(str(path))
        if img is not None:
            self.results_tab.preview.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # -- motion ------------------------------------------------------------
    def _on_motion_done(self, kind: str, ok: bool, message: str) -> None:
        self.acq_tab.set_motion_status(f"{kind}: {message}", ok)
        (self.log_message if ok else self.log_error)(f"{kind}: {message}")
        self._update_candidates()

    def _update_candidates(self) -> None:
        if self.ros is not None:
            self.acq_tab.set_candidates(*self.ros.candidate_progress())

    def _pose_context(self):
        """(T_base_target, T_flange_camera, board centre) from the current knowledge."""
        samples = self.session.active_samples()
        if not samples:
            raise RuntimeError("take at least one sample first")
        cfg = self.session.config
        mode = cfg.calibration_type
        result = self.report.result.T if self.report and self.report.mode == mode else None
        X = hand_eye_guess(cfg, result)
        return fixed_target_guess(samples, X, mode), X, BoardModel(cfg).center

    def random_pose(self) -> None:
        try:
            T_bt, X, center = self._pose_context()
        except (RuntimeError, ValueError) as exc:
            QMessageBox.information(self, "Random pose", str(exc))
            return
        T_tc = random_camera_pose(self._rng, center, self.session.config)
        T_bf = flange_goal(T_bt, T_tc, X, self.session.config.calibration_type)
        self.log_message("Planning to random pose " +
                         " ".join(f"{v:.3f}" for v in matrix_to_pose7(T_bf)))
        self._ros_call("plan_pose", T_bf, self.acq_tab.velocity(), self.acq_tab.acceleration())

    def hemisphere_poses(self) -> None:
        try:
            T_bt, X, center = self._pose_context()
        except (RuntimeError, ValueError) as exc:
            QMessageBox.information(self, "Hemisphere poses", str(exc))
            return
        from ..ros.motion_manager import Candidate
        cfg = self.session.config
        poses = fibonacci_hemisphere(cfg.hemisphere_n, cfg.hemisphere_distance,
                                     cfg.hemisphere_max_polar, center,
                                     alternate_roll_deg=0.5 * (cfg.random_roll_max - cfg.random_roll_min))
        cands = [Candidate(pose=flange_goal(T_bt, T, X, cfg.calibration_type),
                           label=f"hemisphere {i}")
                 for i, T in enumerate(poses)]
        self._ros_call("set_candidates", cands)

    def export_joints(self) -> None:
        if not any(s.joint_values for s in self.session.samples):
            QMessageBox.information(self, "Save joints", "No sample has joint values.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save joint states", "joint_states.yaml",
                                              "YAML (*.yaml)")
        if path:
            n = self.session.export_joint_states(path)
            self.log_message(f"{n} joint states written to {path}")

    def import_joints(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load joint states", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            names, values = read_joint_states(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load joint states", str(exc))
            return
        from ..ros.motion_manager import Candidate
        self._ros_call("set_candidates", [Candidate(names, v, label=f"joints {i}")
                                          for i, v in enumerate(values)])

    # -- calibration -------------------------------------------------------
    def compute(self) -> None:
        if len(self.session.active_samples()) < 3:
            QMessageBox.information(self, "Compute", "At least 3 samples (not excluded) are needed.")
            return
        if self._worker is not None:
            self._pending_compute = True
            return
        cfg = self.session.config
        T_oc = self.session.metadata.get("T_optical_camera")
        kwargs = dict(method=self.acq_tab.method_name(), ident_ratio=self.acq_tab.ident_ratio(),
                      seed=self.acq_tab.seed.value(), bootstrap_n=self.acq_tab.bootstrap_rounds(),
                      n_sigma=cfg.outlier_sigma,
                      T_optical_camera=pose7_to_matrix(T_oc) if T_oc else None,
                      distance_range=(cfg.random_z_min, cfg.random_z_max),
                      mode=cfg.calibration_type)
        self._worker = ComputeWorker(copy.deepcopy(self.session.samples), kwargs, self)
        self._worker.done.connect(self._on_computed)
        self._worker.failed.connect(self._on_compute_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self.acq_tab.set_computing(True)
        self.results_tab.set_status("computing...")
        self._worker.start()

    def _on_computed(self, report: Report) -> None:
        self.report = report
        self.session.results = report.to_dict()
        self._mark_dirty()
        self.results_tab.show_report(self.session.samples, report,
                                     self.session.config.max_reprojection_error)
        v = report.valid_stats or report.ident_stats
        self.results_tab.set_status(
            f"{report.result.method.name}: error {1e3 * v.translation_mean:.2f} mm / "
            f"{v.angle_mean:.3f} deg ({'validation' if report.valid_stats else 'identification'})")
        self.log_message("Calibration computed")

    def _on_compute_failed(self, message: str) -> None:
        self.results_tab.set_status(f"failed: {message}")
        self.log_error(f"calibration failed: {message}")

    def _on_worker_finished(self) -> None:
        self._worker = None
        self.acq_tab.set_computing(False)
        if self._pending_compute:
            self._pending_compute = False
            self.compute()

    def export_calibration(self) -> None:
        if self.report is None:
            QMessageBox.information(self, "Export", "Compute a calibration first.")
            return
        cfg = self.session.config
        parent = cfg.flange_frame if result_parent(self.report.mode) == "flange" else cfg.base_frame
        if self.report.T_camera_frame is not None:
            child, T = cfg.camera_frame, self.report.T_camera_frame
        else:
            child, T = self.session.metadata.get("image_frame", "camera_optical_frame"), self.report.result.T
        path, _ = QFileDialog.getSaveFileName(self, "Export calibration", "handeye_calibration.launch.py",
                                              "Launch file (*.py);;YAML (*.yaml)")
        if path:
            write_calibration(path, parent, child, matrix_to_pose7(T), self.report.mode)
            self.log_message(f"Calibration {parent} -> {child} written to {path}")

    # -- session -----------------------------------------------------------
    def _confirm_discard(self) -> bool:
        if not self._dirty or not self.session.samples:
            return True
        answer = QMessageBox.question(self, "Unsaved session", "Save the current session first?",
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if answer == QMessageBox.Cancel:
            return False
        if answer == QMessageBox.Save:
            return self.save_session()
        return True

    def _set_session(self, session: Session) -> None:
        self.autosaver.discard(self.session)    # the user saved or discarded it
        self.session = session
        self.report = None
        self._dirty = False
        self.config_tab.set_config(session.config)
        if self.ros is not None:
            self.ros.apply_config(session.config)
        self.results_tab.show_report(session.samples, None, session.config.max_reprojection_error)
        self._refresh_samples()
        self._update_motion_enabled()
        self.autosaver.claim(session)

    def new_session(self) -> None:
        if self._confirm_discard():
            self._set_session(Session(copy.deepcopy(self.session.config)))
            self.log_message("New session")

    def open_session(self) -> None:
        if not self._confirm_discard():
            return
        folder = QFileDialog.getExistingDirectory(self, "Open session folder")
        if not folder:
            return
        try:
            session = Session.load(folder)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open session", str(exc))
            return
        self._set_session(session)
        self.log_message(f"Session {folder} loaded: {len(session.samples)} samples")
        if len(session.active_samples()) >= 3:
            self.compute()

    def save_session(self) -> bool:
        if self.session.is_temporary:
            return self.save_session_as()
        return self._save_to(self.session.folder)

    def save_session_as(self) -> bool:
        folder = QFileDialog.getExistingDirectory(self, "Save session to folder")
        return self._save_to(Path(folder)) if folder else False

    def _save_to(self, folder: Path) -> bool:
        try:
            self.session.save(folder)
        except OSError as exc:
            QMessageBox.warning(self, "Save session", str(exc))
            return False
        self._dirty = False
        self.autosaver.discard(None)            # the user folder is up to date now
        self._update_title()
        self.log_message(f"Session saved to {folder}")
        return True

    def _mark_dirty(self) -> None:
        """Something changed: remember it and write the autosave."""
        self._dirty = True
        try:
            self.autosaver.save(self.session)
        except OSError as exc:
            self.log_error(f"autosave failed: {exc}")
        self._update_title()

    def _update_title(self) -> None:
        name = "unsaved session" if self.session.is_temporary else str(self.session.folder)
        self.setWindowTitle(f"Hand-eye calibration - {name}{' *' if self._dirty else ''}")

    def closeEvent(self, event):  # noqa: N802 (Qt naming)
        if not self._confirm_discard():
            event.ignore()
            return
        if self._worker is not None:
            self._worker.wait(10000)
        if self.ros is not None:
            self.ros.stop()
        self.autosaver.discard(self.session)    # clean exit: no crash to recover from
        event.accept()


def ask_restore() -> Optional[Session]:
    """Offer to restore the newest autosave left by a GUI that did not exit cleanly."""
    try:
        orphans = autosave.find_orphans()
    except OSError:
        return None
    if not orphans:
        return None
    newest = orphans[0]
    others = len(orphans) - 1
    box = QMessageBox(QMessageBox.Question, "Restore session",
                      "The calibration GUI did not exit cleanly. An autosaved session "
                      f"was found:\n\n{newest.describe()}\n\nRestore it?")
    box.setInformativeText(f"Folder: {newest.folder}" +
                           (f"\n{others} older autosave(s) are kept in {newest.folder.parent}."
                            if others else ""))
    restore_btn = box.addButton("Restore", QMessageBox.AcceptRole)
    delete_btn = box.addButton("Delete it", QMessageBox.DestructiveRole)
    box.addButton("Not now", QMessageBox.RejectRole)
    box.setDefaultButton(restore_btn)
    box.exec()
    if box.clickedButton() is restore_btn:
        try:
            return autosave.restore(newest)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(None, "Restore session", f"Cannot restore: {exc}")
            return None
    if box.clickedButton() is delete_btn:
        autosave.discard(newest)
    return None


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        from rclpy.utilities import remove_ros_args
        user_args = remove_ros_args(argv)
    except ImportError:
        user_args = argv
    parser = argparse.ArgumentParser(prog="handeye_gui", description="Hand-eye calibration GUI")
    parser.add_argument("--session", help="open this session folder")
    parser.add_argument("--config", help="start with this configuration file")
    parser.add_argument("--offline", action="store_true",
                        help="do not start ROS (review and recompute saved sessions)")
    args, _ = parser.parse_known_args(user_args[1:])

    signal.signal(signal.SIGINT, signal.SIG_DFL)   # Ctrl+C closes the application
    app = QApplication(user_args)
    app.setApplicationName("handeye_gui")

    session, restored = None, False
    if not args.session:
        session = ask_restore()
        restored = session is not None
    if session is None:
        if args.session:
            session = Session.load(args.session)
        else:
            session = Session(CalibConfig.load(args.config) if args.config else CalibConfig())
        if args.config and args.session:
            session.config = CalibConfig.load(args.config)
    window = MainWindow(session, use_ros=not args.offline, ros_args=argv, restored=restored)
    window.show()
    if args.session and len(session.active_samples()) >= 3:
        QTimer.singleShot(0, window.compute)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
