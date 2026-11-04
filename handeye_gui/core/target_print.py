# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Picture of the configured ArUco grid, for checking and for printing at scale.

Markers are drawn as vector rectangles with the real dimensions (metres), so
a PDF or SVG printed at 100 % gives markers of the configured size. The page
is A4 or A3 when the board fits (printers handle standard sizes reliably),
otherwise exactly the board plus margins.

Uses matplotlib's object API only (no pyplot), so it works with or without Qt.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow, Rectangle

from .config import CalibConfig
from .target import BoardModel

INCH = 0.0254
MARGIN = 0.010                 # m, minimum white border around the board (quiet zone)
CAPTION = 0.012                # m, space for the caption under the board
PAGES = (("A4", (0.210, 0.297)), ("A3", (0.297, 0.420)))
PAGE_MARGIN = 0.010            # m, printers cannot print to the paper edge


def marker_bits(dictionary, marker_id: int) -> np.ndarray:
    """(n+2, n+2) bool array of a marker, True = black, black border included."""
    n = dictionary.markerSize + 2
    img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), n, borderBits=1)
    return img < 128


def page_size(content: Tuple[float, float]) -> Tuple[str, Tuple[float, float]]:
    """Smallest standard page (portrait or landscape) that holds ``content``, in metres."""
    cw, ch = content
    for name, (w, h) in PAGES:
        for pw, ph, orient in ((w, h, "portrait"), (h, w, "landscape")):
            if cw + 2 * PAGE_MARGIN <= pw and ch + 2 * PAGE_MARGIN <= ph:
                return f"{name} {orient}", (pw, ph)
    return "custom", (cw, ch)


def _runs(row: np.ndarray) -> List[Tuple[int, int]]:
    """(start, length) of the consecutive True cells of a row."""
    runs, start = [], None
    for i, v in enumerate(list(row) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - start))
            start = None
    return runs


def describe(config: CalibConfig) -> str:
    w, h, sx, sy = config.marker_dimensions()
    model = BoardModel(config)
    size = f"{1e3 * w:.1f}" if abs(w - h) < 1e-9 else f"{1e3 * w:.1f} x {1e3 * h:.1f}"
    sep = f"{1e3 * sx:.1f}" if abs(sx - sy) < 1e-9 else f"{1e3 * sx:.1f} / {1e3 * sy:.1f}"
    return (f"{config.aruco_dictionary}, {config.markers_x} x {config.markers_y} markers, "
            f"ids {int(model.ids.min())}-{int(model.ids.max())}, marker {size} mm, "
            f"separation {sep} mm, board {1e3 * model.size[0]:.1f} x {1e3 * model.size[1]:.1f} mm")


def target_figure(config: CalibConfig, show_ids: bool = False, show_axes: bool = False,
                  caption: bool = True, page: bool = True) -> Figure:
    """Figure with the board at real size (1 figure inch = 25.4 mm on paper).

    Args:
        show_ids: write each marker id over the marker (for checking only:
            a printed board with ids drawn on the markers is not detectable).
        show_axes: draw the target frame (origin, X right, Y down).
        caption: dictionary, layout and dimensions under the board.
        page: centre the board on an A4/A3 page when it fits.
    """
    model = BoardModel(config)
    w, h, sx, sy = config.marker_dimensions()
    bw, bh = model.size
    content = (bw + 2 * MARGIN, bh + 2 * MARGIN + (CAPTION if caption else 0.0))
    _, (pw, ph) = page_size(content) if page else ("custom", content)
    # Board origin (top-left of the first marker) on the page, Y down.
    ox = 0.5 * (pw - bw)
    oy = 0.5 * (ph - content[1]) + MARGIN

    fig = Figure(figsize=(pw / INCH, ph / INCH))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(-ox, pw - ox)
    ax.set_ylim(ph - oy, -oy)            # Y down, like the target frame
    ax.set_aspect("equal")
    ax.axis("off")

    nx = config.markers_x
    rects = []
    for k, marker_id in enumerate(model.ids):
        r, c = divmod(k, nx)
        x0, y0 = c * (w + sx), r * (h + sy)
        bits = marker_bits(model.dictionary, marker_id)
        n = bits.shape[0]
        cw, ch = w / n, h / n
        for i, row in enumerate(bits):
            for start, length in _runs(row):
                rects.append(Rectangle((x0 + start * cw, y0 + i * ch), length * cw, ch))
        if show_ids:
            pts = h / INCH * 72.0
            ax.text(x0 + 0.5 * w, y0 + 0.5 * h, str(int(marker_id)), ha="center", va="center",
                    fontsize=max(4.0, 0.28 * pts), color="#d32f2f", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#d32f2f", alpha=0.9))
    ax.add_collection(PatchCollection(rects, facecolor="black", edgecolor="none",
                                      linewidth=0, antialiased=False))
    if show_axes:
        L = 0.6 * min(w, h)
        for dx, dy, color, label in ((L, 0.0, "#d32f2f", "X"), (0.0, L, "#2e7d32", "Y")):
            ax.add_patch(FancyArrow(0.0, 0.0, dx, dy, width=0.03 * L, head_width=0.15 * L,
                                    head_length=0.2 * L, length_includes_head=True,
                                    color=color, zorder=5))
            ax.text(dx * 1.1 + (0.05 * L if dy else 0), dy * 1.1 + (0.05 * L if dx else 0),
                    label, color=color, fontsize=9, fontweight="bold", zorder=5)
    if caption:
        ax.text(0.0, bh + MARGIN + 0.25 * CAPTION,
                describe(config) + "\nPrint at 100 % (no scaling, no 'fit to page'), "
                f"then check that the board is {1e3 * bw:.1f} mm wide.",
                ha="left", va="top", fontsize=6.5, color="#333333", linespacing=1.3)
    return fig


def save_target(config: CalibConfig, path: str | Path, show_ids: bool = False,
                dpi: int = 600) -> str:
    """Write the board at real size (PDF, SVG or PNG); returns the page description."""
    path = Path(path)
    w, h, sx, sy = config.marker_dimensions()
    model = BoardModel(config)
    content = (model.size[0] + 2 * MARGIN, model.size[1] + 2 * MARGIN + CAPTION)
    name, _ = page_size(content)
    fig = target_figure(config, show_ids=show_ids, caption=True, page=True)
    fmt = path.suffix.lower().lstrip(".") or "pdf"
    fig.savefig(path, format=fmt, dpi=dpi)
    return name


def render_image(config: CalibConfig, px_per_mm: float = 4.0, **kwargs) -> np.ndarray:
    """RGB raster of ``target_figure`` (used by tests and previews)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = target_figure(config, **kwargs)
    canvas = FigureCanvasAgg(fig)
    fig.set_dpi(px_per_mm * 25.4)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()
