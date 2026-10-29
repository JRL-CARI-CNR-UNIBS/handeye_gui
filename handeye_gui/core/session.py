# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Calibration session: configuration + samples, stored in one folder.

Folder layout::

    <folder>/
      config.yaml      CalibConfig
      samples.yaml     list of Sample (+ metadata)
      results.yaml     last evaluation report (optional)
      images/          sample_XXX.png and sample_XXX_decorated.png
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .config import CalibConfig
from .sample import Sample, _plain

CONFIG_FILE = "config.yaml"
SAMPLES_FILE = "samples.yaml"
RESULTS_FILE = "results.yaml"
IMAGES_DIR = "images"


class Session:
    """Holds the configuration and the list of samples.

    Images are written immediately to the working folder (a temporary folder
    until the session is saved), so that memory use stays low.
    """

    def __init__(self, config: Optional[CalibConfig] = None, folder: Optional[str | Path] = None):
        self.config = config or CalibConfig()
        self.samples: List[Sample] = []
        self.results: Optional[Dict[str, Any]] = None
        # Free-form data, e.g. the optical -> camera_frame transform.
        self.metadata: Dict[str, Any] = {}
        self._temporary = folder is None
        self.folder = Path(folder) if folder else Path(tempfile.mkdtemp(prefix="handeye_session_"))
        (self.folder / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        self._next_index = 0

    # -- samples -----------------------------------------------------------
    @property
    def is_temporary(self) -> bool:
        return self._temporary

    def get(self, index: int) -> Sample:
        for s in self.samples:
            if s.index == index:
                return s
        raise KeyError(f"no sample with index {index}")

    def active_samples(self) -> List[Sample]:
        return [s for s in self.samples if not s.excluded]

    def add_sample(self, sample: Sample, image: Optional[np.ndarray] = None,
                   decorated: Optional[np.ndarray] = None) -> Sample:
        """Append a sample, assign its index and write its images (BGR arrays)."""
        sample.index = self._next_index
        self._next_index += 1
        if image is not None:
            sample.image_path = f"{IMAGES_DIR}/sample_{sample.index:03d}.png"
            cv2.imwrite(str(self.folder / sample.image_path), image)
        if decorated is not None:
            sample.decorated_image_path = f"{IMAGES_DIR}/sample_{sample.index:03d}_decorated.png"
            cv2.imwrite(str(self.folder / sample.decorated_image_path), decorated)
        self.samples.append(sample)
        return sample

    def remove_sample(self, index: int) -> Sample:
        """Remove a sample by index. Its images are deleted only in a temporary folder."""
        sample = self.get(index)
        self.samples.remove(sample)
        if self._temporary:
            for rel in (sample.image_path, sample.decorated_image_path):
                if rel:
                    (self.folder / rel).unlink(missing_ok=True)
        return sample

    def set_excluded(self, index: int, value: bool) -> None:
        self.get(index).excluded = bool(value)

    def image_file(self, sample: Sample, decorated: bool = False) -> Optional[Path]:
        rel = sample.decorated_image_path if decorated else sample.image_path
        return self.folder / rel if rel else None

    # -- persistence -------------------------------------------------------
    def save(self, folder: str | Path) -> None:
        """Write config, samples, results and images to ``folder``."""
        folder = Path(folder)
        (folder / IMAGES_DIR).mkdir(parents=True, exist_ok=True)
        if folder.resolve() != self.folder.resolve():
            for s in self.samples:
                for rel in (s.image_path, s.decorated_image_path):
                    src = self.folder / rel
                    if rel and src.exists():
                        shutil.copy2(src, folder / rel)
        self.config.save(folder / CONFIG_FILE)
        with open(folder / SAMPLES_FILE, "w") as stream:
            yaml.safe_dump({"metadata": _plain(self.metadata),
                            "next_index": self._next_index,
                            "samples": [s.to_dict() for s in self.samples]},
                           stream, sort_keys=False)
        if self.results is not None:
            with open(folder / RESULTS_FILE, "w") as stream:
                yaml.safe_dump(_plain(self.results), stream, sort_keys=False)
        old_folder, was_temporary = self.folder, self._temporary
        self.folder = folder
        self._temporary = False
        if was_temporary and old_folder.resolve() != folder.resolve():
            shutil.rmtree(old_folder, ignore_errors=True)

    @classmethod
    def load(cls, folder: str | Path) -> "Session":
        folder = Path(folder)
        config_path = folder / CONFIG_FILE
        config = CalibConfig.load(config_path) if config_path.exists() else CalibConfig()
        session = cls(config, folder)
        samples_path = folder / SAMPLES_FILE
        if samples_path.exists():
            with open(samples_path) as stream:
                data = yaml.safe_load(stream) or {}
            if isinstance(data, list):  # tolerate a bare list
                data = {"samples": data}
            session.metadata = data.get("metadata") or {}
            session.samples = [Sample.from_dict(d) for d in data.get("samples") or []]
            session._next_index = int(data.get("next_index") or 0)
        results_path = folder / RESULTS_FILE
        if results_path.exists():
            with open(results_path) as stream:
                session.results = yaml.safe_load(stream)
        session._next_index = max(session._next_index,
                                  max((s.index for s in session.samples), default=-1) + 1)
        return session

    # -- joint states ------------------------------------------------------
    def export_joint_states(self, path: str | Path, include_excluded: bool = True) -> int:
        """Write the joint configurations of the samples (moveit_calibration format)."""
        samples = [s for s in self.samples if s.joint_values and (include_excluded or not s.excluded)]
        names = samples[0].joint_names if samples else []
        write_joint_states(path, names, [s.joint_values for s in samples])
        return len(samples)

    @staticmethod
    def import_joint_states(path: str | Path) -> Tuple[List[str], List[List[float]]]:
        return read_joint_states(path)


def write_joint_states(path: str | Path, names: Sequence[str],
                       values: Sequence[Sequence[float]]) -> None:
    """Write ``joint_names`` / ``joint_values`` like moveit_calibration does."""
    with open(path, "w") as stream:
        yaml.safe_dump({"joint_names": list(names),
                        "joint_values": [[float(v) for v in row] for row in values]},
                       stream, default_flow_style=None, sort_keys=False)


def read_joint_states(path: str | Path) -> Tuple[List[str], List[List[float]]]:
    """Read a moveit_calibration joint states file. Rows of the wrong size are skipped."""
    with open(path) as stream:
        data = yaml.safe_load(stream) or {}
    names = [str(n) for n in data.get("joint_names") or []]
    if not names:
        raise ValueError(f"'joint_names' missing in {path}")
    values = [[float(v) for v in row] for row in data.get("joint_values") or []
              if row is not None and len(row) == len(names)]
    return names, values
