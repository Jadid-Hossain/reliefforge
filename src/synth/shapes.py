"""M8 — procedural relief primitives.

Every primitive is emitted as STAMPS: (cx, cy, kernel, amplitude, layer, inst_id) where
`kernel` is a small float32 HxW map in [0, 1] (already the profile). The composer
stamps them with `max` inside each stamp's support (kernel > 0), which is the review
Issue-2 fix: a primitive's offset can never leak outside its own support.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PROFILES = ("dome", "pillow", "bevel", "flat_round")


@dataclass
class Stamp:
    cx: float
    cy: float
    kernel: np.ndarray  # float32 [0,1], support = kernel > 0
    amplitude: float
    layer: float  # base offset z_i of this primitive
    inst_id: int


def _profile_of_distance(d_local: np.ndarray, profile: str, edge: float = 0.35) -> np.ndarray:
    """Height profile for a normalized inward distance in [0, 1]."""
    if profile == "dome":
        return np.sqrt(np.clip(1.0 - (1.0 - d_local) ** 2, 0.0, 1.0))
    if profile == "pillow":
        return np.sin(0.5 * np.pi * d_local)
    if profile == "bevel":
        return d_local
    if profile == "flat_round":
        t = np.clip(d_local / max(edge, 1e-3), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
    raise ValueError(f"unknown profile {profile}")


def _kernel_from_mask(mask_roi: np.ndarray, profile: str, r_frac: float = 1.0, edge: float = 0.35, max_radius: float | None = None) -> np.ndarray:
    """Profile kernel of a binary ROI mask (distance transform -> profile)."""
    d = cv2.distanceTransform(mask_roi.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    r = (max_radius if max_radius else d.max()) * float(r_frac)
    u = np.clip(d / max(r, 1e-6), 0.0, 1.0)
    return _profile_of_distance(u, profile, edge).astype(np.float32)


# ------------------------------------------------------------------ kernel builders
def disc_kernel(radius: float, profile: str = "dome") -> np.ndarray:
    r = max(1.0, float(radius))
    k = int(np.ceil(r)) * 2 + 3
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32)
    m = ((yy - k / 2 + 0.5) ** 2 + (xx - k / 2 + 0.5) ** 2 <= r * r)
    return _kernel_from_mask(m, profile, max_radius=r)


def ellipse_kernel(a: float, b: float, angle_deg: float, profile: str = "dome", tip: float = 0.0) -> np.ndarray:
    """Rotated ellipse; `tip` in [0,1] pulls one end into a teardrop (petal/leaf)."""
    a, b = max(1.0, float(a)), max(1.0, float(b))
    ka, kb = int(np.ceil(a * (1 + tip))) + 2, int(np.ceil(b)) + 2
    yy, xx = np.mgrid[0 : 2 * kb + 1, 0 : 2 * ka + 1].astype(np.float32)
    x = (xx - ka) / a
    y = (yy - kb) / b
    stretch = np.where(x >= 0, 1.0 + tip * np.clip(x, 0, 1), 1.0)
    m = (x / stretch) ** 2 + y * y <= 1.0
    k = _kernel_from_mask(m, profile)
    if abs(angle_deg) > 0.5:
        k = _rotate_kernel(k, angle_deg)
    return k


def _rotate_kernel(k: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = k.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(k, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0).astype(np.float32)


def tube_kernel(radius: float, profile: str = "dome") -> np.ndarray:
    """Cross-section kernel for vines/scrolls (same as a disc of the tube radius)."""
    return disc_kernel(radius, profile)


# ------------------------------------------------------------------ layout generators
def rosette_stamps(cx, cy, R, n_petals, petal_len, petal_wid, profile, amp, layer, inst, rng, boss_frac=0.35):
    """N petal stamps rotated around (cx, cy) plus a centre boss."""
    stamps = []
    for k in range(n_petals):
        ang = 360.0 * k / n_petals + float(rng.uniform(-6, 6))
        rad = R * float(rng.uniform(0.85, 1.0))
        px = cx + rad * np.cos(np.deg2rad(ang))
        py = cy + rad * np.sin(np.deg2rad(ang))
        stamps.append(Stamp(px, py, ellipse_kernel(petal_len, petal_wid, ang, profile, tip=float(rng.uniform(0, 0.5))), amp * float(rng.uniform(0.85, 1.0)), layer, inst))
    boss_r = R * boss_frac
    stamps.append(Stamp(cx, cy, disc_kernel(boss_r, "dome"), min(1.0, amp * 1.15), layer, inst))
    return stamps


def vine_stamps(cx, cy, size, length, base_width, turns, profile, amp, layer, inst, rng, wobble=0.35, leaves=None):
    """A spiral / sinusoidal vine: tube stamps along the path with tapering width."""
    stamps = []
    n = max(24, int(length / 3))
    direction = float(rng.uniform(0, 2 * np.pi))
    phase = float(rng.uniform(0, 2 * np.pi))
    for i in range(n):
        t = i / (n - 1)
        w = base_width * (1.0 - 0.75 * t) * float(rng.uniform(0.9, 1.0))
        ang = direction + turns * 2 * np.pi * t
        r = length * (0.15 + 0.85 * t)
        off = np.sin(t * 2 * np.pi * float(rng.uniform(1.5, 2.5)) + phase) * length * wobble * 0.3
        x = cx + r * np.cos(ang) + off * np.cos(direction + np.pi / 2)
        y = cy + r * np.sin(ang) + off * np.sin(direction + np.pi / 2)
        x = float(np.clip(x, w, size - w))
        y = float(np.clip(y, w, size - w))
        stamps.append(Stamp(x, y, tube_kernel(max(1.2, w), profile), amp * (0.7 + 0.3 * (1 - t)), layer, inst))
    if leaves:
        for _ in range(int(leaves)):
            i = int(rng.integers(n // 4, n))
            t = i / (n - 1)
            ang = direction + turns * 2 * np.pi * t
            r = length * (0.15 + 0.85 * t)
            x = cx + r * np.cos(ang)
            y = cy + r * np.sin(ang)
            la = float(rng.uniform(0, 360))
            stamps.append(Stamp(x, y, ellipse_kernel(base_width * float(rng.uniform(2.5, 4.5)), base_width * float(rng.uniform(1.0, 1.6)), la, "dome", tip=0.5), amp * 0.9, layer + 0.05, inst))
    return stamps


def ring_stamps(cx, cy, radius, width, profile, amp, layer, inst, segments: int = 96, rectangular: bool = False, hw=None, hh=None):
    """Annulus (or rectangular moulding) as tube stamps along a closed path."""
    stamps = []
    if rectangular:
        hw, hh = hw * 0.5, hh * 0.5
        pts = []
        for i in range(segments):
            t = i / segments
            per = 2 * (2 * (hw + hh))
            s = t * per
            if s < 2 * hw:
                p = (cx - hw + s, cy - hh)
            elif s < 2 * hw + 2 * hh:
                p = (cx + hw, cy - hh + (s - 2 * hw))
            elif s < 4 * hw + 2 * hh:
                p = (cx + hw - (s - 2 * hw - 2 * hh), cy + hh)
            else:
                p = (cx - hw, cy + hh - (s - 4 * hw - 2 * hh))
            pts.append(p)
    else:
        pts = [(cx + radius * np.cos(2 * np.pi * i / segments), cy + radius * np.sin(2 * np.pi * i / segments)) for i in range(segments)]
    for (x, y) in pts:
        stamps.append(Stamp(float(x), float(y), tube_kernel(width / 2, profile), amp, layer, inst))
    return stamps


def petal_stamps(cx, cy, r, angle_deg, profile, amp, layer, inst, rng):
    """A single leaf/petal with an optional central vein groove."""
    k = ellipse_kernel(r, r * float(rng.uniform(0.35, 0.6)), angle_deg, profile, tip=float(rng.uniform(0.2, 0.7)))
    stamps = [Stamp(cx, cy, k, amp, layer, inst)]
    if rng.random() < 0.5:  # vein: a thin recessed line along the petal axis
        n = 12
        dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        for i in range(n):
            t = (i / (n - 1) - 0.5) * 1.6 * r
            stamps.append(Stamp(cx + t * dx, cy + t * dy, tube_kernel(max(0.8, r * 0.05), "dome"), amp * 0.3, layer + 0.02, inst))
    return stamps
