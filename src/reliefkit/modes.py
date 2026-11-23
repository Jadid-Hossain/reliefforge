"""M3 — classic relief modes.

Mode A (design -> relief): deterministic shape from a flat / B&W design.
Mode C (gray -> height):  luminance is the height (lithophane-style).

Mode A radius rule (review fix ৩ / Issue 5): the transition radius R is never a fixed
number of pixels. It comes from a physical width in mm when the user gives one
    R_px = transition_width_mm / pixel_pitch_mm
and otherwise scales with the image:  R_px = 0.03 * long_side.
"""
from __future__ import annotations

import cv2
import numpy as np

from .io import luminance
from .preprocess import estimate_background, fill_holes, foreground_from_background, keep_components

PROFILES = ("dome", "pillow", "bevel", "flat_round")
AUTO_RADIUS_FRAC = 0.03
MIN_RADIUS_PX = 2.0


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def profile_values(u: np.ndarray, profile: str = "dome", edge: float = 0.5) -> np.ndarray:
    """Height (0..1) for a normalized inward distance u in [0, 1]."""
    if profile == "dome":
        return np.sqrt(np.clip(1.0 - (1.0 - u) ** 2, 0.0, 1.0))
    if profile == "pillow":
        return np.sin(0.5 * np.pi * u)
    if profile == "bevel":
        return u.astype(np.float32)
    if profile == "flat_round":
        return smoothstep(0.0, float(np.clip(edge, 0.05, 1.0)), u).astype(np.float32)
    raise ValueError(f"unknown profile '{profile}', expected one of {PROFILES}")


def resolve_radius(
    r_px: float | None = None,
    transition_width_mm: float | str | None = "auto",
    pixel_pitch_mm: float | None = None,
    long_side: int | None = None,
) -> tuple[float, str]:
    """R in pixels for Mode A and a label saying where it came from."""
    if transition_width_mm not in (None, "auto"):
        if not pixel_pitch_mm or pixel_pitch_mm <= 0:
            raise ValueError("pixel_pitch_mm is required when transition_width_mm is set")
        width_mm = float(transition_width_mm)
        if width_mm <= 0:
            raise ValueError("transition_width_mm must be positive")
        return max(width_mm / float(pixel_pitch_mm), MIN_RADIUS_PX), "transition_width_mm"
    if r_px is not None:
        return max(float(r_px), MIN_RADIUS_PX), "r_px"
    if long_side is None:
        raise ValueError("give r_px, transition_width_mm or long_side")
    return max(AUTO_RADIUS_FRAC * float(long_side), MIN_RADIUS_PX), "auto(0.03*long_side)"


def design_to_height(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    profile: str = "dome",
    edge: float = 0.5,
    r_px: float | None = None,
    transition_width_mm: float | str | None = "auto",
    pixel_pitch_mm: float | None = None,
    smooth_sigma: float = 0.8,
    fill: bool = False,
) -> tuple[np.ndarray, dict]:
    """Mode A: threshold the design and give every region a rounded profile."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H x W x 3")
    info: dict = {"profile": profile, "warnings": []}
    if mask is None:
        bg, share, _frame = estimate_background(rgb)
        fg, thr = foreground_from_background(rgb, bg)
        info.update(foreground_source="background-colour", threshold=float(thr), bg_color=[int(v) for v in bg])
    else:
        fg = mask.astype(bool)
        info["foreground_source"] = "mask"
    fg = keep_components(fg)
    if fill:
        fg = fill_holes(fg)
    H, W = fg.shape
    if not fg.any():
        info["warnings"].append("no foreground found: the output is flat")
        return np.zeros((H, W), np.float32), info

    dist = cv2.distanceTransform(fg.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    dist = np.maximum(dist - 0.5, 0.0)  # the true outline lies half a pixel outside the last foreground pixel
    R, r_source = resolve_radius(r_px, transition_width_mm, pixel_pitch_mm, max(H, W))
    u = np.clip(dist / R, 0.0, 1.0)
    h = profile_values(u, profile, edge)
    if smooth_sigma > 0:
        h = cv2.GaussianBlur(np.ascontiguousarray(h), (0, 0), float(smooth_sigma))
    support = cv2.dilate(fg.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    h = np.where(support, np.clip(h, 0.0, 1.0), 0.0).astype(np.float32)

    info.update(r_px=float(R), r_source=r_source, foreground_frac=float(fg.mean()), peak=float(h.max()))
    return h, info


def gray_to_height(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    gamma: float = 1.0,
    invert_luminance: bool = False,
) -> tuple[np.ndarray, dict]:
    """Mode C: h = luminance^gamma inside the mask (bright = raised unless inverted)."""
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    lum = luminance(rgb.astype(np.float32) / 255.0)
    if invert_luminance:
        lum = 1.0 - lum
    h = np.power(lum, float(gamma)).astype(np.float32)
    if mask is not None:
        h = h * mask.astype(np.float32)
    return h, {"gamma": float(gamma), "invert_luminance": bool(invert_luminance)}
