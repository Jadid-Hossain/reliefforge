"""M8 — render styles: turn (h, inst) into input images, with a physically defined kappa.

kappa rule (review Issue 3):  kappa = relief_depth_mm * px_per_mm_native * s
where s is the view scale factor: full view -> S / long_side, crop view -> the actual
resize factor of the crop. The same s feeds the normal loss during training.
"""
from __future__ import annotations

import cv2
import numpy as np

STYLES = ("wood", "clay", "flat", "bw")
STYLE_WEIGHTS = {"wood": 0.40, "clay": 0.35, "flat": 0.15, "bw": 0.10}

# oak / walnut / painted / gold (run 1 uses albedo only - review point A)
TINTS = ((1.00, 0.86, 0.62), (0.85, 0.66, 0.45), (0.78, 0.72, 0.64), (1.0, 0.83, 0.45), (0.92, 0.90, 0.86), (0.70, 0.60, 0.52))
PALETTE = ((0.72, 0.18, 0.15), (0.16, 0.35, 0.60), (0.85, 0.62, 0.15), (0.20, 0.55, 0.35), (0.55, 0.30, 0.60), (0.90, 0.45, 0.20))


def kappa_of(meta: dict, view_scale: float, jitter: float = 1.0) -> float:
    return float(meta["relief_depth_mm_scene"]) * float(meta["px_per_mm_native"]) * float(view_scale) * float(jitter)


def shade(h: np.ndarray, kappa: float, rng: np.random.Generator, ambient: float | None = None) -> np.ndarray:
    """Lambert + cheap AO of a height field -> float shading in [0, 1]."""
    gy, gx = np.gradient(h * kappa)
    n = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
    az, el = np.deg2rad(rng.uniform(0, 360)), np.deg2rad(rng.uniform(30, 75))
    L = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], np.float32)
    amb = rng.uniform(0.25, 0.5) if ambient is None else ambient
    sh = amb + (1 - amb) * np.clip(n @ L, 0, 1)
    alpha = rng.uniform(1, 4)
    blur = cv2.GaussianBlur(h, (0, 0), max(2.0, 0.01 * max(h.shape)))
    ao = np.clip(1 - alpha * np.maximum(0, blur - h), 0.4, 1.0)
    return (sh * ao).astype(np.float32)


def procedural_wood(h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fallback wood albedo when no CC0 texture files are available."""
    H, W = h.shape
    f = max(1, min(H, W) // 220)
    small = cv2.resize(h, (max(2, W // f), max(2, H // f)), interpolation=cv2.INTER_AREA)
    yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]].astype(np.float32)
    ang = rng.uniform(0, np.pi)
    u = xx * np.cos(ang) + yy * np.sin(ang)
    freq = rng.uniform(0.05, 0.25)
    grain = 0.5 + 0.5 * np.sin(u * freq * 2 * np.pi + 3.0 * np.sin(u * freq * 0.23 * 2 * np.pi + rng.uniform(0, 6)))
    rings = 0.5 + 0.5 * np.sin(u * freq * 0.11 * 2 * np.pi + rng.uniform(0, 6))
    grain = cv2.resize(grain, (W, H), interpolation=cv2.INTER_LINEAR)
    rings = cv2.resize(rings, (W, H), interpolation=cv2.INTER_LINEAR)
    base = 0.45 + 0.35 * grain + 0.12 * rings
    tex = np.clip(base + np.random.default_rng(rng.integers(1 << 30)).normal(0, 0.03, (H, W)).astype(np.float32), 0, 1)
    return np.clip(tex, 0.05, 1.0).astype(np.float32)


def render_view(
    h: np.ndarray,
    inst: np.ndarray,
    meta: dict,
    view_scale: float,
    style: str,
    rng: np.random.Generator,
    texture: np.ndarray | None = None,
    micro_normal: np.ndarray | None = None,
) -> np.ndarray:
    """One rendered view of a scene (uint8 RGB)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style}")
    kappa = kappa_of(meta, view_scale, jitter=float(rng.uniform(0.7, 1.5)))
    H, W = h.shape
    if style == "flat":
        ids = sorted(int(v) for v in np.unique(inst) if v > 0)
        albedo = np.zeros((H, W), np.float32)
        for i, iid in enumerate(ids):
            c = PALETTE[(iid + int(rng.integers(0, 6))) % len(PALETTE)]
            m = inst == iid
            v = 0.45 + 0.5 * float(c[0] + c[1] + c[2]) / 3
            albedo[m] = v * (1.0 if rng.random() < 0.5 else 0.9)
        if ids:  # dark outlines at instance borders
            edge = cv2.dilate((inst > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) - (inst > 0).astype(np.uint8) + cv2.dilate((inst > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) * 0
            borders = cv2.morphologyEx((inst > 0).astype(np.uint8), cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
            outline = (borders > 0) & cv2.erode(borders, np.ones((3, 3), np.uint8)) == 0
            grad = cv2.morphologyEx(inst, cv2.MORPH_GRADIENT, np.ones((2 + int(rng.integers(0, 2)), 2 + int(rng.integers(0, 2))), np.uint8)) > 0
            albedo = np.where(grad, albedo * 0.35, albedo)
        img = np.stack([albedo] * 3, -1)
    elif style == "bw":
        fg = inst > 0 if rng.random() < 0.6 else h > 0.02
        if rng.random() < 0.5:  # silhouette
            img = np.where(fg, 0.12, 0.95).astype(np.float32)
            img = np.stack([img] * 3, -1)
        else:  # outline only
            grad = cv2.morphologyEx((fg.astype(np.uint8)), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            v = np.where(grad, 0.10, 0.95).astype(np.float32)
            img = np.stack([v] * 3, -1)
    else:
        sh = shade(h, kappa, rng)
        if style == "clay":
            albedo = float(rng.uniform(0.55, 0.75))
            img = np.stack([sh * albedo] * 3, -1)
        else:  # wood
            if texture is not None:
                tex = _crop_resize(texture, H, W, rng)
            else:
                tex = procedural_wood(h, rng)
            tint = np.array(TINTS[int(rng.integers(0, len(TINTS)))], np.float32)
            if rng.random() < 0.3:
                tint = np.ones(3, np.float32) * rng.uniform(0.6, 1.0)  # painted
            img = tex[..., None] * sh[..., None] * tint[None, None, :] if tex.ndim == 2 else tex * sh[..., None] * tint[None, None, :]
    img = _degrade(img, rng)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _crop_resize(texture: np.ndarray, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    tex = texture
    if rng.random() < 0.5:
        tex = np.rot90(tex, int(rng.integers(1, 4)))
    if rng.random() < 0.5:
        tex = tex[:, ::-1]
    th, tw = tex.shape[:2]
    s = max(H / th, W / tw)
    nh, nw = int(np.ceil(th * s)), int(np.ceil(tw * s))
    tex = cv2.resize(tex, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    th, tw = tex.shape[:2]
    y0, x0 = int(rng.integers(0, max(1, th - H + 1))), int(rng.integers(0, max(1, tw - W + 1)))
    return tex[y0 : y0 + H, x0 : x0 + W]


def _degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Web-image degradations applied AFTER any downscale (PRD_v2 §5.5)."""
    out = img.astype(np.float32)
    out = out * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)
    out = np.clip(out, 0, 1)
    out = cv2.GaussianBlur(out, (0, 0), float(rng.uniform(0, 1.2)))
    out = np.clip(out + np.random.default_rng(rng.integers(1 << 30)).normal(0, rng.uniform(0.0, 0.02), out.shape).astype(np.float32), 0, 1)
    q = int(rng.integers(60, 99))
    ok, buf = cv2.imencode(".jpg", (out * 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255.0
    return cv2.GaussianBlur(dec, (0, 0), float(rng.uniform(0, 0.6)))
