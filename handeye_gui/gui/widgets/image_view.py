# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Label that shows a numpy RGB image scaled to fit, keeping the aspect ratio."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    if rgb.ndim == 2:
        h, w = rgb.shape
        return QImage(rgb.data, w, h, w, QImage.Format_Grayscale8).copy()
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class ImageView(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #202020; color: #c0c0c0;")
        self.setText("waiting for images...")
        self._pixmap: QPixmap | None = None

    def set_image(self, rgb: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(rgb_to_qimage(rgb))
        self._rescale()

    def resizeEvent(self, event):  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))
