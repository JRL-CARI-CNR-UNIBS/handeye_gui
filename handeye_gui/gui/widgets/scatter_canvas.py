# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Matplotlib canvases embedded in Qt: chain-closure scatter and depth check."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from ...core.evaluation import Report
from ...core.solvers import target_parent

STYLE = {
    "ident": dict(marker="o", color="tab:blue", label="identification"),
    "valid": dict(marker="^", color="tab:orange", label="validation"),
    "outlier": dict(marker="o", color="tab:red", label="outlier"),
    "excluded": dict(marker="x", color="0.6", label="excluded"),
}


class ScatterCanvas(FigureCanvasQTAgg):
    """Target position (mm, relative to the identification mean) on XY, XZ, YZ.

    The target is expressed in its fixed frame: base (eye-in-hand) or flange
    (eye-to-hand).
    """

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(9, 3), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.subplots(1, 3)
        self.clear()

    def clear(self, frame: str = "") -> None:
        self.figure.suptitle(f"Target position in the {frame} frame, relative to the mean"
                             if frame else "", fontsize=9)
        for ax, (a, b) in zip(self.axes, ("XY", "XZ", "YZ")):
            ax.clear()
            ax.set_xlabel(f"{a} [mm]")
            ax.set_ylabel(f"{b} [mm]")
            ax.set_title(f"{a}{b}")
            ax.grid(True, alpha=0.3)
        self.draw_idle()

    def plot(self, report: Optional[Report]) -> None:
        if report is None:
            self.clear()
            return
        self.clear(target_parent(report.mode))
        ref = report.reference[:3, 3]
        groups = {k: [] for k in STYLE}
        outliers, excluded = set(report.outliers), set(report.excluded_indices)
        ident = set(report.ident_indices)
        for e in report.errors:
            if e.index in excluded:
                key = "excluded"
            elif e.index in outliers:
                key = "outlier"
            elif e.index in ident:
                key = "ident"
            else:
                key = "valid"
            groups[key].append((e.index, 1e3 * (e.position - ref)))
        for ax, (i, j) in zip(self.axes, ((0, 1), (0, 2), (1, 2))):
            for key, items in groups.items():
                if not items:
                    continue
                pts = np.array([p for _, p in items])
                ax.scatter(pts[:, i], pts[:, j], s=28, **STYLE[key])
                for idx, p in items:
                    ax.annotate(str(idx), (p[i], p[j]), fontsize=7, alpha=0.7,
                                xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_aspect("equal", adjustable="datalim")
        self.axes[0].legend(fontsize=7, loc="best")
        self.draw_idle()


class DepthCanvas(FigureCanvasQTAgg):
    """Bar chart of z_PnP - z_depth per sample."""

    def __init__(self, parent=None):
        self.figure = Figure(figsize=(4, 3), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.plot(None)

    def plot(self, report: Optional[Report]) -> None:
        ax = self.ax
        ax.clear()
        ax.set_title("PnP - depth (Z)")
        ax.set_xlabel("sample")
        ax.set_ylabel("dZ [mm]")
        ax.grid(True, alpha=0.3)
        if report is not None and report.depth:
            idx = [i for i, _ in report.depth]
            dz = np.array([v for _, v in report.depth]) * 1e3
            excluded = set(report.excluded_indices)
            colors = ["0.6" if i in excluded else "tab:blue" for i in idx]
            ax.bar(idx, dz, color=colors)
            ax.axhline(dz.mean(), color="tab:red", lw=1, ls="--", label=f"mean {dz.mean():.1f} mm")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no depth data", ha="center", va="center", transform=ax.transAxes)
        self.draw_idle()
