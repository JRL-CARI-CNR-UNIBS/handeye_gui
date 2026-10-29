# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Table of samples with errors and an "Exclude" checkbox per row.

Rows above threshold (outliers or large reprojection error) are red,
excluded rows are grey.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from ...core.evaluation import Report
from ...core.sample import Sample

COLUMNS = ["Exclude", "#", "Set", "dX [mm]", "dY [mm]", "dZ [mm]", "|d| [mm]", "Angle [deg]",
           "Reproj [px]", "Dist [m]", "Tilt [deg]", "Depth dZ [mm]", "Forced"]
RED = QColor(255, 200, 200)
GREY = QColor(215, 215, 215)


class SampleTable(QTableWidget):
    exclusion_changed = Signal(int, bool)    # sample index, excluded
    sample_selected = Signal(int)            # sample index

    def __init__(self, parent=None):
        super().__init__(0, len(COLUMNS), parent)
        self.setHorizontalHeaderLabels(COLUMNS)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self._filling = False
        self._indices: List[int] = []
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self._on_selection)

    def selected_index(self) -> Optional[int]:
        rows = self.selectionModel().selectedRows()
        return self._indices[rows[0].row()] if rows else None

    def fill(self, samples: List[Sample], report: Optional[Report], max_reproj: float) -> None:
        self._filling = True
        self.setRowCount(len(samples))
        self._indices = [s.index for s in samples]
        outliers = set(report.outliers) if report else set()
        ident = set(report.ident_indices) if report else set()
        valid = set(report.valid_indices) if report else set()
        for row, s in enumerate(samples):
            err = report.error_of(s.index) if report else None
            if s.excluded:
                role = "excluded"
            elif s.index in ident:
                role = "ident"
            elif s.index in valid:
                role = "valid"
            else:
                role = ""
            dz = s.depth_dz
            dxyz = [f"{1e3 * v:+.2f}" for v in (err.dx, err.dy, err.dz)] if err else [""] * 3
            values = ["", str(s.index), role, *dxyz,
                      f"{1e3 * err.translation:.2f}" if err else "",
                      f"{err.angle:.3f}" if err else "",
                      f"{s.reprojection_error_mean:.2f}", f"{s.target_distance:.3f}",
                      f"{s.target_tilt:.1f}", f"{1e3 * dz:+.1f}" if dz is not None else "",
                      "yes" if s.forced else ""]
            bad = s.index in outliers or s.reprojection_error_mean > max_reproj
            color = GREY if s.excluded else (RED if bad else None)
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                if col == 0:
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if s.excluded else Qt.Unchecked)
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if color is not None:
                    item.setBackground(QBrush(color))
                self.setItem(row, col, item)
        self._filling = False

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != 0:
            return
        self.exclusion_changed.emit(self._indices[item.row()], item.checkState() == Qt.Checked)

    def _on_selection(self) -> None:
        idx = self.selected_index()
        if idx is not None:
            self.sample_selected.emit(idx)
