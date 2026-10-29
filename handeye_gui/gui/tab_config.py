# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Configuration tab: a form generated from the CalibConfig dataclass fields."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGridLayout, QGroupBox, QHBoxLayout, QLineEdit, QMessageBox,
                               QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from ..core.config import CalibConfig

GROUP_COLUMNS = 2


class ConfigTab(QWidget):
    """Edits a CalibConfig. ``config_applied`` is emitted on Apply or Load."""

    config_applied = Signal(object)

    def __init__(self, config: CalibConfig, parent=None):
        super().__init__(parent)
        self._editors: Dict[str, QWidget] = {}
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
            groups[group].addRow(meta.get("label", f.name), editor)

        inner = QWidget()
        inner.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        apply_btn = QPushButton("Apply")
        apply_btn.setToolTip("Use these settings (re-subscribes topics, rebuilds the target model).")
        apply_btn.clicked.connect(self._apply)
        save_btn = QPushButton("Save config...")
        save_btn.clicked.connect(self._save)
        load_btn = QPushButton("Load config...")
        load_btn.clicked.connect(self._load)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        for b in (load_btn, save_btn, apply_btn):
            buttons.addWidget(b)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(buttons)
        self.set_config(config)

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

    def set_config(self, config: CalibConfig) -> None:
        for name, w in self._editors.items():
            value = getattr(config, name)
            if isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.setValue(value)
            elif isinstance(value, list):
                w.setText(", ".join(str(v) for v in value))
            else:
                w.setText(str(value))

    def config(self) -> CalibConfig:
        data: Dict[str, Any] = {}
        for name, w in self._editors.items():
            if isinstance(w, QComboBox):
                data[name] = w.currentText()
            elif isinstance(w, QCheckBox):
                data[name] = w.isChecked()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                data[name] = w.value()
            else:
                data[name] = w.text()
        cfg = CalibConfig.from_dict(data)
        if len(cfg.initial_flange_to_camera) != 7:
            raise ValueError("Initial flange->camera guess needs 7 numbers: x y z qx qy qz qw")
        return cfg

    # ------------------------------------------------------------------
    def _apply(self) -> None:
        try:
            self.config_applied.emit(self.config())
        except (ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid configuration", str(exc))

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
        self.set_config(cfg)
        self.config_applied.emit(cfg)
