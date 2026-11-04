# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Window that shows the ArUco grid described by the configuration."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtWidgets import (QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPushButton, QVBoxLayout)

from ..core.config import CalibConfig
from ..core.target_print import describe, save_target, target_figure


class TargetDialog(QDialog):
    """Preview of the configured target, with marker ids, and export at real size."""

    def __init__(self, config: CalibConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Target")
        self.config = config
        self.ids = QCheckBox("Show marker ids")
        self.ids.setChecked(True)
        self.axes = QCheckBox("Show target frame")
        self.axes.setChecked(True)
        self.ids.toggled.connect(self._redraw)
        self.axes.toggled.connect(self._redraw)
        info = QLabel(describe(config))
        info.setWordWrap(True)
        w, h, _, _ = config.marker_dimensions()
        note = ("Compare marker ids and positions with the printed sheet. "
                "Origin: top-left corner of the first marker, X right, Y down, Z into the board.")
        if abs(w - h) > 1e-9:
            note += " Markers are drawn non-square because width and height differ."
        hint = QLabel(note)
        hint.setWordWrap(True)
        export = QPushButton("Export for printing...")
        export.setToolTip("PDF, SVG or PNG at real size, on an A4/A3 page when it fits. "
                          "Marker ids are not printed on the markers.")
        export.clicked.connect(self._export)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)

        self.canvas = FigureCanvasQTAgg(target_figure(config, True, True, caption=False,
                                                      page=False))
        self.canvas.setMinimumSize(300, 300)
        options = QHBoxLayout()
        options.addWidget(self.ids)
        options.addWidget(self.axes)
        options.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(export)
        buttons.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(info)
        layout.addWidget(self.canvas, 1)
        layout.addLayout(options)
        layout.addWidget(hint)
        layout.addLayout(buttons)
        self.resize(640, 760)

    def _redraw(self) -> None:
        fig = target_figure(self.config, self.ids.isChecked(), self.axes.isChecked(),
                            caption=False, page=False)
        old = self.canvas
        self.canvas = FigureCanvasQTAgg(fig)
        self.canvas.setMinimumSize(300, 300)
        self.layout().replaceWidget(old, self.canvas)
        old.deleteLater()

    def _export(self) -> None:
        name = f"target_{self.config.aruco_dictionary}_{self.config.markers_x}x{self.config.markers_y}.pdf"
        path, _ = QFileDialog.getSaveFileName(self, "Export target", name,
                                              "PDF (*.pdf);;SVG (*.svg);;PNG 600 dpi (*.png)")
        if not path:
            return
        try:
            page = save_target(self.config, path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Export target", str(exc))
            return
        QMessageBox.information(self, "Export target",
                                f"Written {path} ({page}).\nPrint at 100 % (actual size, no "
                                "'fit to page') and measure the board width to check.")
