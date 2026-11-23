"""M8 — composition: stamp primitives into the scene height map and instance map.

Review Issue 2: `h = max_i(z_i + a_i * profile_i)` is only valid INSIDE each
primitive's support. Composition therefore stamps with
    candidate              = layer + amplitude * kernel     (inside the stamp support)
    candidate[~support]    = -inf
    h[support]             = maximum(h[support], candidate[support])
    inst[support & update] = inst_id                         (only valid pixels update)
so a positive layer offset can never leak over the whole canvas.
"""
from __future__ import annotations

import numpy as np

from .shapes import Stamp


def stamp_one(h: np.ndarray, inst: np.ndarray, s: Stamp) -> None:
    k = s.kernel
    kh, kw = k.shape
    H, W = h.shape
    x0, y0 = int(round(s.cx - kw / 2)), int(round(s.cy - kh / 2))
    x1, y1 = x0 + kw, y0 + kh
    cx0, cy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi_k = k[cy0 : cy0 + (y1 - y0), cx0 : cx0 + (x1 - x0)]
    support = roi_k > 1e-6
    if not support.any():
        return
    candidate = s.layer + s.amplitude * roi_k
    roi_h = h[y0:y1, x0:x1]
    update = support & (candidate > roi_h)
    roi_h[update] = candidate[update]
    inst[y0:y1, x0:x1][update] = s.inst_id


def compose(H: int, W: int, stamps: list[Stamp], bg_layer: float = 0.0, blur_sigma: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Height map (float32 [0, ~1]) and instance-ID map (0 = background)."""
    import cv2

    h = np.full((H, W), bg_layer, np.float32)
    inst = np.zeros((H, W), np.uint16)
    for s in stamps:
        stamp_one(h, inst, s)
    if bg_layer <= 0:
        h[h <= 1e-6] = 0.0
    if blur_sigma > 0:
        h = cv2.GaussianBlur(h, (0, 0), blur_sigma)
        h[inst == 0] = 0.0  # the blur is anti-aliasing only; the base plane stays exactly 0
    peak = float(h.max())
    if peak > 1e-6:
        h = h / peak
    return h.astype(np.float32), inst


def mirror_stamps(stamps: list[Stamp], W: int, mode: str, inst_offset: int = 0) -> list[Stamp]:
    """Left-right mirror and 4-fold symmetry at the layout level."""
    if mode == "none":
        return stamps
    out = list(stamps)
    if mode in ("mirror", "4fold"):
        out += [Stamp(W - s.cx, s.cy, s.kernel[:, ::-1].copy(), s.amplitude, s.layer, s.inst_id + (inst_offset if mode == "mirror" else 0)) for s in stamps]
    if mode == "4fold":
        out += [Stamp(s.cy, W - s.cx, s.kernel.T.copy(), s.amplitude, s.layer, s.inst_id) for s in stamps]
        out += [Stamp(W - s.cy, s.cx, s.kernel.T[:, ::-1].copy(), s.amplitude, s.layer, s.inst_id) for s in stamps]
    return out
