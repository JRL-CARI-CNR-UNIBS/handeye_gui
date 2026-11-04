# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""PySide6 user interface.

Importing this package prepares the environment for Qt:
* matplotlib is told to use PySide6;
* old matplotlib (< 3.6, e.g. 3.5.1 on Ubuntu 22.04 / Humble) is patched so it
  works with PySide6 >= 6.4, whose Qt enums no longer accept ``int()``;
* the Qt plugin path set by the pip ``opencv-python`` wheel (Qt5) is removed,
  otherwise PySide6 (Qt6) tries to load the wrong platform plugin.
"""

import os

os.environ.setdefault("QT_API", "pyside6")

import cv2  # noqa: E402,F401  (imported first on purpose, see above)

for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        del os.environ[_var]


def _patch_old_matplotlib_qt():
    """Make matplotlib < 3.6 convert PySide6 >= 6.4 enums to int.

    Must run before ``matplotlib.backends.backend_qt`` is imported, because
    that module copies ``_to_int`` from ``qt_compat`` at import time.
    """
    from matplotlib.backends import qt_compat

    def _to_int(x):
        try:
            return int(x)
        except TypeError:
            return x.value

    qt_compat._to_int = _to_int


_patch_old_matplotlib_qt()
