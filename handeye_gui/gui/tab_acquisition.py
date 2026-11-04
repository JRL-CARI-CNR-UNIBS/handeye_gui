# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Acquisition tab: live image, sampling, motion and compute controls.

The tab only shows state and emits requests; MainWindow does the work.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGridLayout,
                               QGroupBox, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter, QVBoxLayout,
                               QWidget)

from ..core.coverage import CoverageReport
from ..core.solvers import Method
from .widgets.image_view import ImageView

STABLE_STYLE = "QPushButton { background-color: black; color: white; font-weight: bold; padding: 8px; }"
SHAKY_STYLE = "QPushButton { background-color: #c62828; color: white; font-weight: bold; padding: 8px; }"


class AcquisitionTab(QWidget):
    take_sample_requested = Signal(bool)       # forced
    remove_sample_requested = Signal()
    plan_requested = Signal()
    execute_requested = Signal()
    next_requested = Signal()
    random_requested = Signal()
    hemisphere_requested = Signal()
    stop_requested = Signal()
    save_session_requested = Signal()
    load_session_requested = Signal()
    export_joints_requested = Signal()
    import_joints_requested = Signal()
    compute_requested = Signal()

    def __init__(self, velocity: float, acceleration: float, parent=None):
        super().__init__(parent)
        self._stable = False
        self.image = ImageView()

        # -- sampling ------------------------------------------------------
        self.take_btn = QPushButton("Take sample")
        self.take_btn.setMinimumHeight(48)
        self.take_btn.clicked.connect(self._on_take)
        self.remove_btn = QPushButton("Remove sample")
        self.remove_btn.setToolTip("Remove the sample selected in the Results table, or the last one.")
        self.remove_btn.clicked.connect(self.remove_sample_requested)
        self.count_label = QLabel("0 samples")
        self.stability_label = QLabel("waiting for TF...")
        self.stability_label.setWordWrap(True)
        save_s = QPushButton("Save samples...")
        save_s.clicked.connect(self.save_session_requested)
        load_s = QPushButton("Load samples...")
        load_s.clicked.connect(self.load_session_requested)
        save_j = QPushButton("Save joints...")
        save_j.setToolTip("Export the joint states of the samples (moveit_calibration format).")
        save_j.clicked.connect(self.export_joints_requested)
        load_j = QPushButton("Load joints...")
        load_j.setToolTip("Load joint states as candidate poses.")
        load_j.clicked.connect(self.import_joints_requested)
        sampling = QGroupBox("Sampling")
        g = QGridLayout(sampling)
        g.addWidget(self.take_btn, 0, 0, 1, 2)
        g.addWidget(self.remove_btn, 1, 0)
        g.addWidget(self.count_label, 1, 1)
        g.addWidget(self.stability_label, 2, 0, 1, 2)
        g.addWidget(save_s, 3, 0)
        g.addWidget(load_s, 3, 1)
        g.addWidget(save_j, 4, 0)
        g.addWidget(load_j, 4, 1)

        # -- motion --------------------------------------------------------
        self.vel = QDoubleSpinBox()
        self.vel.setRange(0.01, 1.0)
        self.vel.setSingleStep(0.05)
        self.vel.setValue(velocity)
        self.acc = QDoubleSpinBox()
        self.acc.setRange(0.01, 1.0)
        self.acc.setSingleStep(0.05)
        self.acc.setValue(acceleration)
        self.plan_btn = QPushButton("Plan")
        self.plan_btn.setToolTip("Plan to the current candidate pose.")
        self.plan_btn.clicked.connect(self.plan_requested)
        self.exec_btn = QPushButton("Execute")
        self.exec_btn.clicked.connect(self.execute_requested)
        self.next_btn = QPushButton("Next pose")
        self.next_btn.setToolTip("Plan and execute the next candidate pose.")
        self.next_btn.clicked.connect(self.next_requested)
        self.random_btn = QPushButton("Random pose")
        self.random_btn.setToolTip("Plan to a random camera pose around the target "
                                   "(distance and roll/pitch/yaw from the configuration). "
                                   "Needs at least one sample.")
        self.random_btn.clicked.connect(self.random_requested)
        self.hemi_btn = QPushButton("Hemisphere poses")
        self.hemi_btn.setToolTip("Generate candidate poses on a Fibonacci hemisphere around the "
                                 "target. Needs at least one sample.")
        self.hemi_btn.clicked.connect(self.hemisphere_requested)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet("QPushButton { color: #c62828; font-weight: bold; }")
        self.stop_btn.clicked.connect(self.stop_requested)
        self.candidate_label = QLabel("no candidate poses")
        self.motion_label = QLabel("")
        self.motion_label.setWordWrap(True)
        self.motion_block_label = QLabel("")
        self.motion_block_label.setWordWrap(True)
        self.motion_block_label.setStyleSheet("color: #c62828;")
        self.motion_block_label.setVisible(False)
        motion = QGroupBox("Motion")
        m = QGridLayout(motion)
        m.addWidget(QLabel("Velocity scaling"), 0, 0)
        m.addWidget(self.vel, 0, 1)
        m.addWidget(QLabel("Acceleration scaling"), 1, 0)
        m.addWidget(self.acc, 1, 1)
        m.addWidget(self.plan_btn, 2, 0)
        m.addWidget(self.exec_btn, 2, 1)
        m.addWidget(self.next_btn, 3, 0)
        m.addWidget(self.random_btn, 3, 1)
        m.addWidget(self.hemi_btn, 4, 0)
        m.addWidget(self.stop_btn, 4, 1)
        m.addWidget(self.candidate_label, 5, 0, 1, 2)
        m.addWidget(self.motion_label, 6, 0, 1, 2)
        m.addWidget(self.motion_block_label, 7, 0, 1, 2)

        # -- compute -------------------------------------------------------
        self.method = QComboBox()
        self.method.addItems([m.name for m in Method])
        self.ratio = QSlider(Qt.Horizontal)
        self.ratio.setRange(30, 100)
        self.ratio.setValue(70)
        self.ratio_label = QLabel()
        self.ratio.valueChanged.connect(self._update_ratio_label)
        self._update_ratio_label()
        self.bootstrap = QCheckBox("Bootstrap")
        self.bootstrap_n = QSpinBox()
        self.bootstrap_n.setRange(10, 5000)
        self.bootstrap_n.setValue(200)
        self.seed = QSpinBox()
        self.seed.setRange(0, 10**6)
        self.compute_btn = QPushButton("Compute")
        self.compute_btn.setMinimumHeight(36)
        self.compute_btn.clicked.connect(self.compute_requested)
        compute = QGroupBox("Calibration")
        c = QFormLayout(compute)
        c.addRow("OpenCV method", self.method)
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self.ratio)
        ratio_row.addWidget(self.ratio_label)
        c.addRow("Identification", ratio_row)
        boot_row = QHBoxLayout()
        boot_row.addWidget(self.bootstrap)
        boot_row.addWidget(self.bootstrap_n)
        c.addRow(boot_row)
        c.addRow("Seed", self.seed)
        c.addRow(self.compute_btn)

        # -- coverage ------------------------------------------------------
        self.coverage_bar = QProgressBar()
        self.coverage_bar.setRange(0, 100)
        self.coverage_label = QLabel("")
        self.coverage_label.setWordWrap(True)
        coverage = QGroupBox("Pose coverage")
        cv = QVBoxLayout(coverage)
        cv.addWidget(self.coverage_bar)
        cv.addWidget(self.coverage_label)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        for box in (sampling, motion, compute, coverage):
            side_layout.addWidget(box)
        side_layout.addStretch(1)
        # Scrollable, so the window can shrink below the panel height
        # (otherwise it cannot be maximised on a 1080p screen).
        side_scroll = QScrollArea()
        side_scroll.setWidget(side)
        side_scroll.setWidgetResizable(True)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        side_scroll.setMinimumWidth(side.sizeHint().width() + 20)
        side_scroll.setMaximumWidth(450)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.image)
        splitter.addWidget(side_scroll)
        splitter.setStretchFactor(0, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self.set_stability(False, {"reason": "waiting for ROS"})

    # -- getters -----------------------------------------------------------
    def velocity(self) -> float:
        return self.vel.value()

    def acceleration(self) -> float:
        return self.acc.value()

    def method_name(self) -> str:
        return self.method.currentText()

    def ident_ratio(self) -> float:
        return self.ratio.value() / 100.0

    def bootstrap_rounds(self) -> int:
        return self.bootstrap_n.value() if self.bootstrap.isChecked() else 0

    # -- setters -----------------------------------------------------------
    def set_stability(self, stable: bool, info: dict) -> None:
        self._stable = stable
        self.take_btn.setStyleSheet(STABLE_STYLE if stable else SHAKY_STYLE)
        parts = ["stable" if stable else f"not stable: {info.get('reason', '')}"]
        if "flange_pos" in info:
            parts.append(f"flange {1e3 * info['flange_pos']:.2f} mm / {info['flange_rot']:.3f} deg")
        if "target_pos" in info:
            parts.append(f"target {1e3 * info['target_pos']:.2f} mm / {info['target_rot']:.3f} deg")
        self.stability_label.setText("\n".join(parts))

    def set_sample_count(self, total: int, used: int) -> None:
        self.count_label.setText(f"{total} samples ({used} used)")

    def set_candidates(self, current: int, total: int) -> None:
        if total == 0:
            self.candidate_label.setText("no candidate poses")
        else:
            self.candidate_label.setText(f"candidate {current + 1}/{total}")

    def set_motion_status(self, text: str, ok: bool = True) -> None:
        self.motion_label.setStyleSheet("" if ok else "color: #c62828;")
        self.motion_label.setText(text)

    def set_coverage(self, report: CoverageReport) -> None:
        self.coverage_bar.setValue(int(round(100 * report.index)))
        d = ", ".join(str(int(v)) for v in report.distance_hist)
        t = ", ".join(str(int(v)) for v in report.tilt_hist)
        self.coverage_label.setText(
            f"views {report.view_occupied}/{report.view_total}, rolls {report.roll_occupied}/8, "
            f"distances {report.distance_occupied}/5\n"
            f"distance hist [{report.distance_edges[0]:.2f}-{report.distance_edges[-1]:.2f} m]: {d}\n"
            f"tilt hist [0-60 deg, 10 deg bins]: {t}")

    def set_motion_enabled(self, enabled: bool, reason: str = "") -> None:
        """Enable Plan/Execute/pose buttons only when the MoveIt names are valid (Stop always works)."""
        for b in (self.plan_btn, self.exec_btn, self.next_btn, self.random_btn, self.hemi_btn):
            b.setEnabled(enabled)
        self.motion_block_label.setVisible(not enabled)
        self.motion_block_label.setText(f"Motion disabled: {reason}" if reason else "")

    def set_computing(self, busy: bool) -> None:
        self.compute_btn.setEnabled(not busy)
        self.compute_btn.setText("Computing..." if busy else "Compute")

    # ----------------------------------------------------------------------
    def _on_take(self) -> None:
        if self._stable:
            self.take_sample_requested.emit(False)
            return
        answer = QMessageBox.question(
            self, "Pose not stable",
            "The robot or the target is not stable.\nTake the sample anyway?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            self.take_sample_requested.emit(True)

    def _update_ratio_label(self) -> None:
        v = self.ratio.value()
        self.ratio_label.setText(f"{v}% / {100 - v}%")
