"""M8 — procedural relief primitives.

Every primitive is emitted as STAMPS: (cx, cy, kernel, amplitude, layer, inst_id) where
`kernel` is a small float32 HxW map in [0, 1]. The composer stamps them with `max`
inside each stamp's support (review Issue 2).

Quality rules (learned from the first contact-sheet review 2026-11-24):
* tubes/rings are stamped with a step of ~0.35 x the tube radius -> CONTINUOUS ribbons,
  never beaded necklaces;
* petals are pointed teardrops, length ~2.5x width, with angular gaps -> they read as
  carved petals, not pills;
* frames are 4 straight continuous runs + corner blocks; the "panel" layout composes
  border + centre medallion + corner rosettes, the classic carved-door arrangement.
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


def _kernel_from_mask(mask_roi: np.ndarray, profile: str, max_radius: float | None = None) -> np.ndarray:
    d = cv2.distanceTransform(mask_roi.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    r = (max_radius if max_radius else d.max()) if (max_radius or d.max()) else 1.0
    u = np.clip(d / max(r, 1e-6), 0.0, 1.0)
    return _profile_of_distance(u, profile).astype(np.float32)


# ------------------------------------------------------------------ kernel builders
def disc_kernel(radius: float, profile: str = "dome") -> np.ndarray:
    r = max(1.5, float(radius))
    k = int(np.ceil(r)) * 2 + 3
    yy, xx = np.mgrid[0:k, 0:k].astype(np.float32)
    m = ((yy - k / 2 + 0.5) ** 2 + (xx - k / 2 + 0.5) ** 2 <= r * r)
    return _kernel_from_mask(m, profile, max_radius=r)


def _rotate_kernel(k: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = k.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(k, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0).astype(np.float32)


def petal_kernel(length: float, width: float, angle_deg: float, profile: str = "dome") -> np.ndarray:
    """A pointed teardrop: base (round) at -length/2, tip at +length/2 (the +x direction
    before rotation) - so a petal 'points' along its rotation angle."""
    L = max(4.0, float(length))
    Wd = max(2.5, float(width))
    if Wd > L:
        Wd = L
    k = int(np.ceil(L)) + int(np.ceil(Wd)) + 4
    yy, xx = np.mgrid[0:k, 0:2 * k].astype(np.float32)
    x = (xx - k) / (L / 2)  # x in [-1, 1] over the petal length
    y = (yy - k) / (Wd / 2)  # y in [-1, 1] over the petal width
    taper = 0.55 + 0.45 * np.clip(x, 0, 1)  # widest near the base, pointed at +1
    m = (x * x + (y / taper) ** 2) <= 1.0
    m &= x >= -0.98
    kern = _kernel_from_mask(m, profile)
    if abs(angle_deg) > 0.5:
        kern = _rotate_kernel(kern, angle_deg)
    return kern


ellipse_kernel = petal_kernel  # historical name: same pointed-leaf shape


def leaf_kernel(length: float, width: float, angle_deg: float) -> np.ndarray:
    return petal_kernel(length, width, angle_deg, "pillow")


def tube_kernel(radius: float, profile: str = "dome") -> np.ndarray:
    return disc_kernel(radius, profile)


# ------------------------------------------------------------------ tube stamping (continuous!)
def tube_along(path: np.ndarray, radius: np.ndarray | float, profile: str = "dome", inst_id: int = 1, amp: float = 1.0, layer: float = 0.0, amp_along: np.ndarray | None = None) -> list[Stamp]:
    """Stamp a continuous tube along a polyline.

    The step is ~0.35 x the local radius so consecutive kernels always overlap -> a
    ribbon, never beads.
    """
    path = np.asarray(path, np.float64)
    if path.ndim != 2 or len(path) < 2:
        return []
    seg = np.diff(path, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total <= 0:
        return []
    radii = np.broadcast_to(np.asarray(radius, np.float64), (len(path),)) if np.ndim(radius) == 0 else np.asarray(radius, np.float64)
    amps = np.broadcast_to(np.asarray(amp, np.float64), (len(path),)) if amp_along is None else np.asarray(amp_along, np.float64)
    stamps = []
    t = 0.0
    i = 0
    while t <= total:
        while i < len(s) - 2 and s[i + 1] < t:
            i += 1
        f = (t - s[i]) / max(s[i + 1] - s[i], 1e-9)
        x = path[i, 0] + f * seg[i, 0]
        y = path[i, 1] + f * seg[i, 1]
        r_i = radii[i] + f * (radii[min(i + 1, len(radii) - 1)] - radii[i])
        a_i = amps[i] + f * (amps[min(i + 1, len(amps) - 1)] - amps[i])
        stamps.append(Stamp(float(x), float(y), tube_kernel(max(1.2, r_i), profile), float(a_i), layer, inst_id))
        t += max(0.8, 0.35 * r_i)
    return stamps


def spiral_path(cx, cy, r_max, turns, direction=1.0, wobble=0.0, rng=None) -> np.ndarray:
    n = 400
    t = np.linspace(0.0, 1.0, n)
    r = r_max * (0.15 + 0.85 * t)
    ang = direction * turns * 2 * np.pi * t
    x = cx + r * np.cos(ang)
    y = cy + r * np.sin(ang)
    if wobble > 0:
        rng = rng or np.random.default_rng(0)
        x = x + wobble * np.sin(3 * np.pi * t + rng.uniform(0, 6))
        y = y + wobble * np.cos(2 * np.pi * t + rng.uniform(0, 6))
    return np.stack([x, y], 1)


def sine_path(p0, p1, amp, periods) -> np.ndarray:
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = 200
    t = np.linspace(0, 1, n)
    d = p1 - p0
    L = np.hypot(*d)
    if L < 1e-6:
        return np.stack([p0, p1], 1)
    nrm = np.array([-d[1], d[0]]) / L
    base = p0[None, :] + t[:, None] * d[None, :]
    off = (amp * np.sin(2 * np.pi * periods * t))[:, None] * nrm[None, :]
    return base + off


def circle_path(cx, cy, radius, n=None) -> np.ndarray:
    r = float(radius)
    n = n or max(64, int(2 * np.pi * r / 2.0))
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], 1)


def rect_path(cx, cy, hw, hh, inset=0.0) -> np.ndarray:
    """Closed rectangular path (optionally inset from the corners for rounder joints)."""
    x0, x1, y0, y1 = cx - hw + inset, cx + hw - inset, cy - hh + inset, cy + hh - inset
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return np.asarray(pts, np.float64)


# ------------------------------------------------------------------ ornament layouts
def rosette_stamps(cx, cy, R, n_petals, profile, amp, layer, inst, rng, petal_rings=1) -> list[Stamp]:
    """A carved rosette: centre boss + 1-2 rings of pointed petals.

    Petals start slightly INSIDE the boss and fill ~85% of their angular slot: they are
    joined at the base and separate toward the tips, like a real carved rosette."""
    stamps = []
    R = max(8.0, float(R))
    boss_r = max(3.0, R * 0.30)
    stamps.append(Stamp(cx, cy, disc_kernel(boss_r, "dome"), min(1.05, amp * 1.15), layer, inst))
    for ring in range(petal_rings):
        n = max(5, int(n_petals * (1.5 ** ring)))
        r_in = boss_r * 0.8 + ring * R * 0.45
        r_out = r_in + R * (0.6 if petal_rings == 1 else 0.5)
        span = 2 * np.pi / n
        for k in range(n):
            ang = span * k + (np.pi / n if ring else 0.0) + float(rng.uniform(-0.03, 0.03))
            mid = (r_in + r_out) / 2
            px = cx + mid * np.cos(ang)
            py = cy + mid * np.sin(ang)
            petal_len = r_out - r_in
            petal_w = min(0.62 * petal_len, 0.85 * span * mid)  # almost touching the neighbours
            petal_w = max(petal_len * 0.28, petal_w)
            stamps.append(Stamp(px, py, petal_kernel(petal_len, petal_w, np.degrees(ang), profile), amp * float(rng.uniform(0.85, 1.0)), layer + 0.02 * ring, inst))
    return stamps


def vine_stamps(cx, cy, size, length, base_width, turns, profile, amp, layer, inst, rng, leaves=None) -> list[Stamp]:
    """A continuous vine: spiral/sine ribbon + alternating pointed leaves."""
    path = spiral_path(cx, cy, length, turns, direction=float(rng.choice([-1.0, 1.0])), wobble=length * 0.08, rng=rng)
    radii = np.linspace(base_width, base_width * 0.35, len(path))
    stamps = tube_along(path, radii, profile, inst_id=inst, amp=amp, layer=layer)
    if leaves:
        n = int(leaves)
        for j in range(n):
            t = 0.15 + 0.8 * j / max(1, n - 1)
            idx = int(t * (len(path) - 1))
            px, py = path[idx]
            tang = np.degrees(np.arctan2(path[min(idx + 3, len(path) - 1), 1] - path[max(idx - 3, 0), 1], path[min(idx + 3, len(path) - 1), 0] - path[max(idx - 3, 0), 0]))
            side = 1.0 if j % 2 == 0 else -1.0
            ang = tang + side * float(rng.uniform(35, 55))
            ll = base_width * float(rng.uniform(4.0, 7.0))
            stamps.append(Stamp(float(px), float(py), leaf_kernel(ll, ll * float(rng.uniform(0.35, 0.5)), ang), amp * float(rng.uniform(0.75, 0.95)), layer + 0.04, inst))
    return stamps


def scroll_stamps(cx, cy, size, length, width, turns, profile, amp, layer, inst, rng) -> list[Stamp]:
    """A classic S/C-scroll: continuous tapering ribbon, no beads, one or two curls."""
    stamps = []
    n_scroll = int(rng.integers(1, 3))
    for s_i in range(n_scroll):
        ox = cx + (0 if n_scroll == 1 else (0.5 - s_i) * length * 1.4)
        direction = 1.0 if s_i % 2 == 0 else -1.0
        path = spiral_path(ox, cy, length * 0.5, turns * float(rng.uniform(0.8, 1.2)), direction=direction, wobble=length * 0.03, rng=rng)
        # open the spiral into an S with a CONTINUOUS proportional shift (no jump: a jump
        # in the path makes tube stamping leave beads across the gap)
        path = path.copy()
        t = (np.arange(len(path)) / max(1, len(path) - 1))[:, None]
        path += t * np.array([direction * length * 0.45, -length * 0.3])[None, :]
        radii = np.linspace(width, width * 0.3, len(path))
        stamps += tube_along(path, radii, profile, inst_id=inst, amp=amp, layer=layer)
    return stamps


def ring_stamps(cx, cy, radius, width, profile, amp, layer, inst, rectangular=False, hw=None, hh=None) -> list[Stamp]:
    """A continuous annulus / rectangular moulding (closed path, overlapping stamps)."""
    if rectangular:
        path = rect_path(cx, cy, hw, hh, inset=width * 0.4)
    else:
        path = circle_path(cx, cy, radius)
    return tube_along(np.vstack([path, path[:1]]), width / 2 * np.ones(len(path) + 1) if len(path) else 0, profile, inst_id=inst, amp=amp, layer=layer)


def border_stamps(W, H, margin, width, profile, amp, layer, inst, rng, motifs=True) -> list[Stamp]:
    """A furniture border: 4 continuous frame runs + corner blocks + inner motifs."""
    stamps = ring_stamps(W / 2, H / 2, 0, width, profile, amp, layer, inst, rectangular=True, hw=W / 2 - margin, hh=H / 2 - margin)
    if motifs:
        hw, hh = W / 2 - margin, H / 2 - margin
        n_top = max(2, int(hw * 2 / (width * 9)))
        for k in range(n_top):
            t = (k + 0.5) / n_top
            x = W / 2 - hw + t * 2 * hw
            for y in (H / 2 - hh, H / 2 + hh):
                if rng.random() < 0.75:
                    stamps.append(Stamp(x, y, disc_kernel(width * float(rng.uniform(0.9, 1.6)), "dome"), amp * float(rng.uniform(0.8, 1.0)), layer + 0.03, inst))
        n_side = max(2, int(hh * 2 / (width * 9)))
        for k in range(n_side):
            t = (k + 0.5) / n_side
            y = H / 2 - hh + t * 2 * hh
            for x in (W / 2 - hw, W / 2 + hw):
                if rng.random() < 0.75:
                    stamps.append(Stamp(x, y, disc_kernel(width * float(rng.uniform(1.0, 1.7)), "pillow"), amp * float(rng.uniform(0.8, 1.0)), layer + 0.03, inst))
    return stamps


def petal_stamps(cx, cy, r, angle_deg, profile, amp, layer, inst, rng) -> list[Stamp]:
    """A single leaf/petal with an optional central vein groove."""
    stamps = [Stamp(cx, cy, petal_kernel(r, r * float(rng.uniform(0.38, 0.55)), angle_deg, profile), amp, layer, inst)]
    if rng.random() < 0.5:
        n = 12
        dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        path = np.stack([np.linspace(cx - r * 0.8, cx + r * 0.8, n), np.linspace(cy - r * 0.8, cy + r * 0.8, n)], 1)
        stamps += tube_along(path, np.full(n, max(0.8, r * 0.045)), "dome", inst_id=inst, amp=amp * 0.4, layer=layer + 0.02)
    return stamps
