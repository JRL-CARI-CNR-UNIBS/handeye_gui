# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""PySide6 user interface.

Importing this package prepares the environment for Qt:
* matplotlib is told to use PySide6;
* the Qt plugin path set by the pip ``opencv-python`` wheel (Qt5) is removed,
  otherwise PySide6 (Qt6) tries to load the wrong platform plugin.
"""

import os

os.environ.setdefault("QT_API", "pyside6")

import cv2  # noqa: E402,F401  (imported first on purpose, see above)

for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        del os.environ[_var]
