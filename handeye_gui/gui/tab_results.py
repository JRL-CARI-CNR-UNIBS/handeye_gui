# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Results tab: scatter plots, sample table, summary and depth check."""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from ..core.evaluation import Report, format_report
from ..core.sample import Sample
from .widgets.image_view import ImageView
from .widgets.sample_table import SampleTable
from .widgets.scatter_canvas import DepthCanvas, ScatterCanvas


class ResultsTab(QWidget):
    exclusion_changed = Signal(int, bool)
    export_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scatter = ScatterCanvas()
        self.depth = DepthCanvas()
        self.table = SampleTable()
        self.table.exclusion_changed.connect(self.exclusion_changed)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        font = QFont("Monospace")
        font.setStyleHint(QFont.TypeWriter)
        self.summary.setFont(font)
        self.summary.setPlaceholderText("Press Compute in the Acquisition tab.")
        self.preview = ImageView()
        self.preview.setText("select a sample")
        self.preview.setMinimumSize(240, 180)
        self.status = QLabel("")
        export_btn = QPushButton("Export calibration...")
        export_btn.setToolTip("Write the result as a static_transform_publisher launch file or YAML.")
        export_btn.clicked.connect(self.export_requested)

        top = QSplitter(Qt.Horizontal)
        top.addWidget(self.scatter)
        top.addWidget(self.depth)
        top.setStretchFactor(0, 3)
        top.setStretchFactor(1, 1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.summary, 3)
        rl.addWidget(self.preview, 2)
        bottom = QSplitter(Qt.Horizontal)
        bottom.addWidget(self.table)
        bottom.addWidget(right)
        bottom.setStretchFactor(0, 3)
        bottom.setStretchFactor(1, 2)

        main = QSplitter(Qt.Vertical)
        main.addWidget(top)
        main.addWidget(bottom)
        main.setStretchFactor(0, 2)
        main.setStretchFactor(1, 3)

        footer = QHBoxLayout()
        footer.addWidget(self.status, 1)
        footer.addWidget(export_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(main)
        layout.addLayout(footer)

    def show_report(self, samples: List[Sample], report: Optional[Report], max_reproj: float) -> None:
        self.table.fill(samples, report, max_reproj)
        self.scatter.plot(report)
        self.depth.plot(report)
        self.summary.setPlainText(format_report(report) if report else "")

    def show_samples(self, samples: List[Sample], max_reproj: float) -> None:
        """Refresh the table without a new calibration (keeps the old plots)."""
        self.table.fill(samples, None, max_reproj)

    def set_status(self, text: str) -> None:
        self.status.setText(text)
