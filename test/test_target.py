# Copyright 2026 Manuel Beschi
# SPDX-License-Identifier: Apache-2.0
"""Detection + PnP + depth plane on a synthetic rendered board."""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from handeye_gui.core import CalibConfig
from handeye_gui.core.target import BoardModel, depth_plane

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
D = np.zeros(5)


def render(board: BoardModel, T_ct: np.ndarray, px_per_m=4000):
    """Render the board seen by a pinhole camera at T_ct (camera -> target)."""
    w_m, h_m = board.size
    margin = 0.01
    W, H = int((w_m + 2 * margin) * px_per_m), int((h_m + 2 * margin) * px_per_m)
    tex = np.full((H, W), 255, np.uint8)
    w, h, sx, sy = board.config.marker_dimensions()
    for k, mid in enumerate(board.ids):
        r, c = divmod(k, board.config.markers_x)
        m = cv2.aruco.generateImageMarker(board.dictionary, int(mid), int(w * px_per_m))
        x0 = int((margin + c * (w + sx)) * px_per_m)
        y0 = int((margin + r * (h + sy)) * px_per_m)
        tex[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    # texture pixel -> target plane point -> image
    S = np.array([[1 / px_per_m, 0, -margin], [0, 1 / px_per_m, -margin], [0, 0, 1]])
    Hm = K @ np.column_stack([T_ct[:3, 0], T_ct[:3, 1], T_ct[:3, 3]]) @ S
    img = cv2.warpPerspective(tex, Hm, (1280, 720), borderValue=128)
    return img


def test_pnp_on_rendered_board():
    cfg = CalibConfig(markers_x=3, markers_y=4, marker_size=0.04, marker_separation=0.01)
    board = BoardModel(cfg)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", [15, -10, 5], degrees=True).as_matrix()
    T[:3, 3] = [-0.05, -0.08, 0.45]
    img = render(board, T)
    det = board.detect(img)
    assert det.n_markers == 12
    pnp = board.estimate_pose(det, K, D)
    assert pnp is not None
    assert np.allclose(pnp.T[:3, 3], T[:3, 3], atol=2e-3)
    assert pnp.reprojection_error_mean < 1.0
    deco = board.draw(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), det, pnp, K, D)
    assert deco.shape == (720, 1280, 3)

    # synthetic depth of the board plane, aligned with the colour image
    n = T[:3, 2]
    d = -n @ T[:3, 3]
    v, u = np.mgrid[0:720, 0:1280]
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
    depth = (-d / (rays @ n)).astype(np.float32)
    res = depth_plane(depth, K, pnp.image_points, 0.85, 0.003, pnp.center)
    assert res is not None
    assert abs(res.center_z - pnp.center[2]) < 2e-3
