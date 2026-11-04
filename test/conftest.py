# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
import pytest

from handeye_gui.core import session as session_module


@pytest.fixture(autouse=True)
def temp_root(tmp_path, monkeypatch):
    """Keep unsaved sessions and autosaves of the tests out of the real /tmp folder."""
    root = tmp_path / "autosave_root"
    monkeypatch.setattr(session_module, "TEMP_ROOT", root)
    return root
