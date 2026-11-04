# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Automatic backup of the current session, so that a crash loses nothing.

The session is written after every change (sample taken or removed,
exclusion, configuration, results):

* an unsaved session already lives in a folder under ``TEMP_ROOT``: its YAML
  files are simply rewritten there;
* a session saved to a user folder is mirrored to its own folder under
  ``TEMP_ROOT``, so that the user's folder changes only on an explicit save.

Each autosave folder holds ``autosave.yaml`` with the owner process id: a
folder whose process is gone was left by a crash and can be restored. On a
clean exit the GUI deletes its autosave folder.

``/tmp`` can be emptied at reboot: this does not replace saving the session.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

from . import session as _session
from .session import SAMPLES_FILE, Session, new_temp_folder, write_yaml_atomic

MARKER_FILE = "autosave.yaml"
EMPTY_MIN_AGE = 60.0   # s, empty orphan folders younger than this are left alone


@dataclass
class AutosaveInfo:
    folder: Path
    saved_at: float              # epoch seconds
    n_samples: int
    source_folder: Optional[str]  # user folder of the session, None if never saved

    def describe(self) -> str:
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.saved_at))
        origin = f"copy of {self.source_folder}" if self.source_folder else "unsaved session"
        return f"{when}, {self.n_samples} samples ({origin})"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_marker(folder: Path) -> dict:
    try:
        with open(folder / MARKER_FILE) as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _count_samples(folder: Path) -> int:
    try:
        with open(folder / SAMPLES_FILE) as stream:
            data = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return 0
    samples = data.get("samples") if isinstance(data, dict) else data
    return len(samples or [])


def find_orphans(root: Optional[Path] = None, cleanup_empty: bool = True) -> List[AutosaveInfo]:
    """Autosaves left by processes that are no longer running, newest first.

    Folders of dead processes without samples are deleted when
    ``cleanup_empty`` is set. Folders of running GUIs are ignored.
    """
    root = Path(root) if root is not None else _session.TEMP_ROOT
    if not root.is_dir():
        return []
    found = []
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        marker = _read_marker(folder)
        pid = int(marker.get("pid") or 0)
        if pid == os.getpid() or _pid_alive(pid):
            continue
        n = _count_samples(folder)
        if n == 0:
            # Skip very recent folders: another GUI may be starting and has
            # not written its marker yet.
            if cleanup_empty and time.time() - folder.stat().st_mtime > EMPTY_MIN_AGE:
                shutil.rmtree(folder, ignore_errors=True)
            continue
        saved_at = float(marker.get("saved_at") or folder.stat().st_mtime)
        found.append(AutosaveInfo(folder, saved_at, n, marker.get("source_folder")))
    return sorted(found, key=lambda a: a.saved_at, reverse=True)


def restore(info: AutosaveInfo) -> Session:
    """Load an autosave as an unsaved session (it must be saved to a user folder)."""
    session = Session.load(info.folder, temporary=True)
    write_yaml_atomic(info.folder / MARKER_FILE, _marker(None))   # now owned by this process
    return session


def discard(info: AutosaveInfo) -> None:
    shutil.rmtree(info.folder, ignore_errors=True)


def _marker(source_folder: Optional[str]) -> dict:
    return {"pid": os.getpid(), "saved_at": time.time(), "source_folder": source_folder}


class Autosaver:
    """Writes the current session to its autosave folder."""

    def __init__(self) -> None:
        self._mirror: Optional[Path] = None      # used when the session is not temporary

    def folder(self, session: Session) -> Path:
        if session.is_temporary:
            return session.folder
        if self._mirror is None or not self._mirror.is_dir():
            self._mirror = new_temp_folder()
        return self._mirror

    def save(self, session: Session) -> Path:
        folder = self.folder(session)
        session.write_files(folder)
        source = None if session.is_temporary else str(session.folder)
        write_yaml_atomic(folder / MARKER_FILE, _marker(source))
        return folder

    def claim(self, session: Session) -> None:
        """Mark a temporary session folder as owned by this process.

        Saved sessions are mirrored only after their first change.
        """
        if session.is_temporary:
            try:
                write_yaml_atomic(session.folder / MARKER_FILE, _marker(None))
            except OSError:
                pass

    def discard(self, session: Optional[Session]) -> None:
        """Delete the autosave (call after a save, or when the user discards the session).

        A temporary session folder is the autosave itself: it is deleted too.
        """
        if self._mirror is not None:
            shutil.rmtree(self._mirror, ignore_errors=True)
            self._mirror = None
        if (session is not None and session.is_temporary
                and _is_under(session.folder, _session.TEMP_ROOT)):
            shutil.rmtree(session.folder, ignore_errors=True)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
