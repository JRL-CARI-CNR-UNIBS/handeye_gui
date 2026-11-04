# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Configuration tab: a form generated from the CalibConfig dataclass fields.

* Fields that name ROS entities (``suggest`` metadata) are editable drop-downs
  filled from the ROS graph; names that do not exist are shown in red.
* Edits are not used until Apply: while the form differs from the applied
  configuration the Apply button is highlighted and ``pending_changed`` is
  emitted (MainWindow marks the tab title and warns when leaving the tab).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox, QVBoxLayout,
                               QWidget)

from ..core.config import CalibConfig
from ..core.ros_names import (ERROR, OK, UNKNOWN, WARN, GraphSnapshot, camera_info_for, check,
                              suggestions)

GROUP_COLUMNS = 2
FIELDS = {f.name: f for f in dataclasses.fields(CalibConfig)}
STATUS_STYLE = {ERROR: "background-color: #ffcdd2;", WARN: "background-color: #fff3c4;"}
STATUS_LABEL_STYLE = {ERROR: "color: #c62828; font-weight: bold;", WARN: "color: #b26a00;"}
APPLY_PENDING_STYLE = ("QPushButton { background-color: #ef6c00; color: white; "
                       "font-weight: bold; padding: 4px 12px; }")


class ConfigTab(QWidget):
    """Edits a CalibConfig. ``config_applied`` is emitted on Apply or Load."""

    config_applied = Signal(object)
    pending_changed = Signal(bool)
    refresh_requested = Signal()

    def __init__(self, config: CalibConfig, parent=None):
        super().__init__(parent)
        self._editors: Dict[str, QWidget] = {}
        self._labels: Dict[str, QLabel] = {}
        self._items: Dict[str, List[str]] = {}
        self._snapshot: Optional[GraphSnapshot] = None
        self._applied = config
        self._baseline: Optional[CalibConfig] = None
        self._pending = False
        groups: Dict[str, QFormLayout] = {}
        grid = QGridLayout()
        for f in dataclasses.fields(CalibConfig):
            meta = f.metadata
            group = meta.get("group", "Other")
            if group not in groups:
                box = QGroupBox(group)
                groups[group] = QFormLayout(box)
                n = len(groups) - 1
                grid.addWidget(box, n // GROUP_COLUMNS, n % GROUP_COLUMNS)
            editor = self._make_editor(f)
            editor.setToolTip(meta.get("help", ""))
            self._editors[f.name] = editor
            label = QLabel(meta.get("label", f.name))
            self._labels[f.name] = label
            groups[group].addRow(label, editor)
            self._connect_change(editor)
        self._editors["image_topic"].activated.connect(self._fill_camera_info)
        self._editors["calibration_type"].currentTextChanged.connect(self._update_enabled)

        inner = QWidget()
        inner.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setToolTip("Use these settings (re-subscribes topics, rebuilds the target model).")
        self.apply_btn.clicked.connect(self.apply)
        self.revert_btn = QPushButton("Revert")
        self.revert_btn.setToolTip("Discard the changes not applied yet.")
        self.revert_btn.clicked.connect(self.discard_changes)
        save_btn = QPushButton("Save config...")
        save_btn.clicked.connect(self._save)
        load_btn = QPushButton("Load config...")
        load_btn.clicked.connect(self._load)
        target_btn = QPushButton("View target...")
        target_btn.setToolTip("Show the ArUco grid described by the Target fields "
                              "(and export it for printing).")
        target_btn.clicked.connect(self._view_target)
        self.refresh_btn = QPushButton("Refresh ROS names")
        self.refresh_btn.setToolTip("Re-read topics, TF frames, actions and planning groups "
                                    "(also done automatically every few seconds).")
        self.refresh_btn.clicked.connect(self.refresh_requested)
        self.names_label = QLabel("")
        self.names_label.setWordWrap(True)
        self.pending_label = QLabel("")
        self.pending_label.setStyleSheet("color: #ef6c00; font-weight: bold;")
        buttons = QHBoxLayout()
        buttons.addWidget(target_btn)
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.names_label, 1)
        buttons.addWidget(self.pending_label)
        for b in (load_btn, save_btn, self.revert_btn, self.apply_btn):
            buttons.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(buttons)
        self.set_config(config)
        self.set_graph(None)

    # ------------------------------------------------------------------
    @staticmethod
    def _make_editor(f: dataclasses.Field) -> QWidget:
        meta = f.metadata
        default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
        if "choices" in meta:
            w = QComboBox()
            w.addItems(meta["choices"])
            w.setEditable(False)
            return w
        if "suggest" in meta:
            w = QComboBox()
            w.setEditable(True)
            w.setInsertPolicy(QComboBox.NoInsert)
            w.setMinimumContentsLength(24)
            w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            return w
        if isinstance(default, bool):
            return QCheckBox()
        if isinstance(default, int):
            w = QSpinBox()
            w.setRange(int(meta.get("min", -10**6)), int(meta.get("max", 10**6)))
            return w
        if isinstance(default, float):
            w = QDoubleSpinBox()
            w.setDecimals(int(meta.get("decimals", 2)))
            w.setRange(float(meta.get("min", -1e6)), float(meta.get("max", 1e6)))
            w.setSingleStep(10 ** -min(3, w.decimals()))
            return w
        return QLineEdit()

    def _connect_change(self, w: QWidget) -> None:
        if isinstance(w, QComboBox):
            w.currentTextChanged.connect(self._on_edit)
            if w.isEditable():
                w.editTextChanged.connect(self._on_edit)
        elif isinstance(w, QCheckBox):
            w.toggled.connect(self._on_edit)
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(self._on_edit)
        else:
            w.textChanged.connect(self._on_edit)

    def set_config(self, config: CalibConfig) -> None:
        """Show ``config`` and take it as the applied configuration."""
        for w in self._editors.values():
            w.blockSignals(True)
        try:
            for name, w in self._editors.items():
                value = getattr(config, name)
                if isinstance(w, QComboBox):
                    if w.isEditable():
                        w.setEditText(str(value))
                    else:
                        w.setCurrentText(str(value))
                elif isinstance(w, QCheckBox):
                    w.setChecked(bool(value))
                elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                    w.setValue(value)
                elif isinstance(value, list):
                    w.setText(", ".join(str(v) for v in value))
                else:
                    w.setText(str(value))
        finally:
            for w in self._editors.values():
                w.blockSignals(False)
        self._applied = config
        # Compare against what the form shows (spin boxes round to their decimals).
        try:
            self._baseline = self.config()
        except (ValueError, TypeError):
            self._baseline = None
        self._update_enabled()
        self._on_edit()

    def config(self) -> CalibConfig:
        data: Dict[str, Any] = {}
        for name, w in self._editors.items():
            if isinstance(w, QComboBox):
                data[name] = w.currentText().strip() if w.isEditable() else w.currentText()
            elif isinstance(w, QCheckBox):
                data[name] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                data[name] = w.value()
            else:
                data[name] = w.text()
        cfg = CalibConfig.from_dict(data)
        for name in ("initial_flange_to_camera", "initial_base_to_camera"):
            if len(getattr(cfg, name)) != 7:
                label = FIELDS[name].metadata["label"]
                raise ValueError(f"{label} needs 7 numbers: x y z qx qy qz qw")
        return cfg

    @property
    def applied_config(self) -> CalibConfig:
        return self._applied

    # -- pending changes ---------------------------------------------------
    def has_pending_changes(self) -> bool:
        return self._pending

    def _on_edit(self, *_args) -> None:
        try:
            cfg = self.config()
            pending = self._baseline is None or cfg != self._baseline
        except (ValueError, TypeError):
            cfg, pending = None, True
        self._update_status(cfg)
        if pending != self._pending:
            self._pending = pending
            self.apply_btn.setStyleSheet(APPLY_PENDING_STYLE if pending else "")
            self.pending_label.setText("Changes not applied" if pending else "")
            self.revert_btn.setEnabled(pending)
            self.pending_changed.emit(pending)
        elif not pending:
            self.revert_btn.setEnabled(False)

    def apply(self) -> bool:
        """Apply the form; False (with a message) if it is not valid."""
        try:
            cfg = self.config()
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))
            return False
        self.config_applied.emit(cfg)
        self.set_config(cfg)
        return True

    def discard_changes(self) -> None:
        self.set_config(self._applied)

    def _update_enabled(self) -> None:
        to_hand = self._editors["calibration_type"].currentText() == "eye_to_hand"
        for name, enabled in (("initial_flange_to_camera", not to_hand),
                              ("initial_base_to_camera", to_hand)):
            self._editors[name].setEnabled(enabled)
            self._labels[name].setEnabled(enabled)

    # -- ROS names ---------------------------------------------------------
    def set_graph(self, snapshot: Optional[GraphSnapshot]) -> None:
        """New view of the ROS graph (None = ROS not available)."""
        self._snapshot = snapshot
        self.refresh_btn.setEnabled(True)
        for name, w in self._editors.items():
            kind = FIELDS[name].metadata.get("suggest")
            if kind:
                self._set_items(name, w, suggestions(kind, snapshot))
        self._on_edit()

    def _set_items(self, name: str, w: QComboBox, items: List[str]) -> None:
        if self._items.get(name) == items or w.view().isVisible():
            return
        self._items[name] = items
        text = w.currentText()
        w.blockSignals(True)
        w.clear()
        w.addItems(items)
        w.setEditText(text)
        w.blockSignals(False)

    def check_config(self, cfg: CalibConfig) -> Dict[str, Tuple[str, str]]:
        """``field -> (status, message)`` for every ROS name of ``cfg``."""
        out = {}
        for name, f in FIELDS.items():
            kind = f.metadata.get("suggest")
            if kind:
                out[name] = check(kind, str(getattr(cfg, name)), self._snapshot)
        return out

    def _update_status(self, cfg: Optional[CalibConfig]) -> None:
        if cfg is None:
            return
        statuses = self.check_config(cfg)
        problems = []
        for name, (status, message) in statuses.items():
            w, label = self._editors[name], self._labels[name]
            w.lineEdit().setStyleSheet(STATUS_STYLE.get(status, ""))
            label.setStyleSheet(STATUS_LABEL_STYLE.get(status, ""))
            help_text = FIELDS[name].metadata.get("help", "")
            w.setToolTip((help_text + "\n\n" if help_text else "") + f"Check: {message}")
            if status in (ERROR, WARN):
                problems.append((status, FIELDS[name].metadata.get("label", name), message))
        if self._snapshot is None:
            self.names_label.setStyleSheet("color: gray;")
            self.names_label.setText("ROS names not checked")
        elif not problems:
            self.names_label.setStyleSheet("color: #2e7d32;")
            unknown = sum(1 for s, _ in statuses.values() if s == UNKNOWN)
            self.names_label.setText("ROS names found" +
                                     (f" ({unknown} not checkable yet)" if unknown else ""))
        else:
            worst = ERROR if any(p[0] == ERROR for p in problems) else WARN
            self.names_label.setStyleSheet(STATUS_LABEL_STYLE[worst])
            self.names_label.setText("Check: " + "; ".join(f"{label}: {msg}"
                                                           for _, label, msg in problems[:3])
                                     + (" ..." if len(problems) > 3 else ""))

    def _fill_camera_info(self, _index: int) -> None:
        """Image topic chosen from the list: fill the matching camera_info topic."""
        info = camera_info_for(self._editors["image_topic"].currentText(), self._snapshot)
        if info:
            self._editors["camera_info_topic"].setEditText(info)

    # ------------------------------------------------------------------
    def _view_target(self) -> None:
        from .target_dialog import TargetDialog
        try:
            cfg = self.config()
            dialog = TargetDialog(cfg, self)
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "View target", str(exc))
            return
        dialog.exec()

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save configuration", "config.yaml",
                                              "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.config().save(path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Save failed", str(exc))

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            cfg = CalibConfig.load(path)
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.config_applied.emit(cfg)
        self.set_config(cfg)


__all__ = ["ConfigTab", "OK", "ERROR", "WARN", "UNKNOWN"]
