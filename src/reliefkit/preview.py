"""Shaded previews of a height map (for the app, the report and the contact sheets)."""
from __future__ import annotations

import cv2
import numpy as np


def shade_height(h01: np.ndarray, kappa: float = 14.0, light_az_deg: float = 215.0, light_el_deg: float = 50.0, albedo: float = 0.75) -> np.ndarray:
    """Clay-style shaded render of a height map (uint8 RGB)."""
    h = np.asarray(h01, np.float32)
    gy, gx = np.gradient(h * kappa)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    az, el = np.deg2rad(light_az_deg), np.deg2rad(light_el_deg)
    L = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    sh = 0.35 + 0.65 * np.clip(n @ L, 0, 1)
    return np.stack([(np.clip(sh * albedo, 0, 1) * 255).astype(np.uint8)] * 3, -1)


def height_colormap(h01: np.ndarray) -> np.ndarray:
    """Viridis-like tint of the height map (uint8 RGB)."""
    cm = cv2.applyColorMap((np.clip(h01, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_PARULA)
    return cm[..., ::-1]


def preview_figure(rgb: np.ndarray | None, h01: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Side-by-side strip: input | shaded height | colour-mapped height (uint8 RGB)."""
    parts = []
    if rgb is not None:
        parts.append(("input", rgb))
    parts.append(("shaded", shade_height(h01)))
    parts.append(("height", height_colormap(h01)))
    if mask is not None:
        parts.append(("mask", np.stack([(mask.astype(np.uint8)) * 255] * 3, -1)))
    tiles = []
    for _, p in parts:
        p = np.asarray(p)
        if p.ndim == 2:
            p = np.stack([p] * 3, -1)
        tiles.append(p)
    H = max(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(round(t.shape[1] * H / t.shape[0])), H), interpolation=cv2.INTER_AREA) if t.shape[0] != H else t for t in tiles]
    return np.concatenate(tiles, axis=1)
