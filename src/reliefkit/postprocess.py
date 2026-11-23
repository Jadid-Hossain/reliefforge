"""M4 — height post-processing, fixed order, float32 until export.

1. reference region = dilated mask (background anchors the minimum, so a flat plateau
   stays a plateau instead of being stretched into noise); no mask -> whole image
2. percentiles 1 / 99.5 inside the reference region -> clip to [0, 1]
3. invert: raised <-> recessed RELATIVE TO THE BASE PLANE. The whole map flips and the
   level outside the relief becomes 1.0. (Review Issue 1: inverting only the dilated
   reference region raises a halo of background around the design.)
4. contrast curve
5. edge-preserving bilateral filter, then Gaussian smoothing
6. feather the mask edge and put the base level back outside it — exactly
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import dilate

REF_RADIUS_FRAC = 0.015
MIN_REF_RADIUS_PX = 8
PCT_LOW, PCT_HIGH = 1.0, 99.5
BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE = 9, 0.05, 3.0
FEATHER_SIGMA = 1.2  # ~3 px transition


def reference_region(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    r = max(MIN_REF_RADIUS_PX, int(round(REF_RADIUS_FRAC * max(shape))))
    return dilate(mask, r)


def finalize_height(
    h_raw: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    invert: bool = False,
    gamma: float = 1.0,
    smoothing: float = 1.0,
    use_bilateral: bool = True,
    feather_sigma: float = FEATHER_SIGMA,
) -> tuple[np.ndarray, dict]:
    """Raw model / classic height -> h01 in [0, 1]. Returns (h01 float32, info).

    Outside the feathered mask the result is exactly `base_level` (0.0, or 1.0 when
    inverted), so separate ornaments sit on one flat board.
    """
    h = np.asarray(h_raw, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError(f"h_raw must be 2D, got shape {h.shape}")
    info: dict = {"warnings": []}
    if not np.isfinite(h).all():
        info["warnings"].append("prediction contained NaN/inf (replaced by 0)")
        h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if smoothing < 0:
        raise ValueError("smoothing must be >= 0")

    mask = None if mask is None else mask.astype(bool)
    if mask is not None and not mask.any():
        info["warnings"].append("empty mask ignored")
        mask = None
    ref = reference_region(mask, h.shape)

    lo, hi = (float(v) for v in np.percentile(h[ref], [PCT_LOW, PCT_HIGH]))
    info.update(lo=lo, hi=hi)
    if hi - lo < 1e-6:
        info["warnings"].append("no relief detected (flat prediction)")
        return np.zeros(h.shape, np.float32), info

    h01 = np.clip((h - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    base_level = 1.0 if invert else 0.0
    if invert:
        h01 = 1.0 - h01
    if mask is not None:
        h01[~mask] = base_level
    if gamma != 1.0:
        h01 = np.power(h01, float(gamma)).astype(np.float32)
    if use_bilateral:
        h01 = cv2.bilateralFilter(np.ascontiguousarray(h01, dtype=np.float32), BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE)
    if smoothing > 0:
        h01 = cv2.GaussianBlur(np.ascontiguousarray(h01, dtype=np.float32), (0, 0), float(smoothing))
    h01 = np.clip(h01, 0.0, 1.0).astype(np.float32)

    if mask is not None:
        m_f = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(feather_sigma)) if feather_sigma > 0 else mask.astype(np.float32)
        m_f[m_f < 1e-6] = 0.0  # far from the design the feather is exactly 0 -> exactly base level
        h01 = (h01 * m_f + base_level * (1.0 - m_f)).astype(np.float32)

    info.update(base_level=base_level, peak=float(h01.max()), min=float(h01.min()))
    return h01, info
