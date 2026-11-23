"""M2 — preprocessing: background mask, components, letterbox, mode suggestion, inference size.

Rules that came out of the PRD review (docs/architecture.md §5):
* keep EVERY component above a small area — never only the largest one;
* never stretch an image: letterbox and remember the geometry;
* suggest_mode does not count unique grey levels (anti-aliasing and JPEG create hundreds
  of them) — it measures how much of the image is covered by a few dominant colours and
  how much of the tone variation comes from broad shading;
* the inference size follows the LONG side of the panel in mm and is capped by the
  real resolution of the source image.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_COMPONENT_AREA_FRAC = 0.0005  # components below 0.05 % of the image are dust
MIN_S, MAX_S = 518, 1036  # ViT patch 14 -> multiples of 14
PIXELS_PER_TOOL_DIAMETER = 3.0


# ------------------------------------------------------------------ small helpers
def round14(x: float) -> int:
    return max(14, int(round(float(x) / 14.0)) * 14)


def ceil14(x: float) -> int:
    return max(14, int(np.ceil(float(x) / 14.0 - 1e-9)) * 14)


def resize_arr(arr: np.ndarray, w: int, h: int, interpolation: int | None = None) -> np.ndarray:
    """Resize HxW or HxWxC; INTER_AREA when shrinking, Lanczos when enlarging."""
    w, h = max(1, int(w)), max(1, int(h))
    shrink = w <= arr.shape[1] and h <= arr.shape[0]
    interp = interpolation if interpolation is not None else (cv2.INTER_AREA if shrink else cv2.INTER_LANCZOS4)
    if arr.dtype == bool:
        return cv2.resize(arr.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST) > 127
    return cv2.resize(arr, (w, h), interpolation=interp)


def resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Area-resample a boolean mask and keep it boolean."""
    w, h = max(1, int(w)), max(1, int(h))
    if mask.shape == (h, w):
        return mask.copy()
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    r = int(max(r, 0))
    if r == 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def keep_components(mask: np.ndarray, min_area_frac: float = MIN_COMPONENT_AREA_FRAC) -> np.ndarray:
    """Drop only dust: every component >= min_area_frac of the image area is kept."""
    mask = mask.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    min_area = max(1, int(round(min_area_frac * mask.size)))
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    return ndimage.binary_fill_holes(mask.astype(bool))


# ------------------------------------------------------------------ background / mask
def _dominant_color(pixels: np.ndarray) -> tuple[np.ndarray, float]:
    """Most frequent colour of a set of pixels (3-bit bins) and the share of pixels near it."""
    q = pixels // 32
    code = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    dominant = int(np.argmax(np.bincount(code, minlength=512)))
    color = np.median(pixels[code == dominant], axis=0)
    share = float((np.abs(pixels - color).max(axis=1) <= 24).mean())
    return color, share


def _distance_to_border(H: int, W: int) -> np.ndarray:
    r = np.minimum(np.arange(H), np.arange(H)[::-1])[:, None]
    c = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :]
    return np.minimum(r, c)


def estimate_background(rgb: np.ndarray, band_frac: float = 0.06) -> tuple[np.ndarray, float, bool]:
    """Background colour, how much of the outer band is close to it, and whether a frame was seen through.

    The band (not a 1-px ring) gives the dominant colour. If that colour forms only a
    thin rim against the image edge (a frame cropped flush with the border), the colour
    behind the rim is the real background.
    """
    H, W = rgb.shape[:2]
    b = max(2, int(round(min(H, W) * band_frac)))
    band = np.concatenate(
        [
            rgb[:b].reshape(-1, 3),
            rgb[-b:].reshape(-1, 3),
            rgb[b : H - b, :b].reshape(-1, 3),
            rgb[b : H - b, W - b :].reshape(-1, 3),
        ]
    ).astype(np.int32)
    # The design background always touches the image edge, so the outermost few pixels
    # get the vote: a large plate that fills most of the band must not steal it.
    dist0 = _distance_to_border(H, W)
    edge = rgb[dist0 <= 3].reshape(-1, 3).astype(np.int32)
    bg_edge, share_edge = _dominant_color(edge)
    bg, share = _dominant_color(band)
    if share_edge >= 0.5 and not np.array_equal(bg_edge, bg):
        bg = bg_edge
        share = float((np.abs(band - bg).max(axis=1) <= 24).mean())

    dist = _distance_to_border(H, W)
    cover = (np.abs(rgb.astype(np.int16) - bg.astype(np.int16)).max(axis=2) <= 24).astype(np.float64)
    d_flat = dist.ravel()
    d_max = int(min(3 * b, dist.max()))
    inside = d_flat <= d_max
    at_dist = np.bincount(d_flat[inside], minlength=d_max + 1).astype(np.float64)
    covered = np.bincount(d_flat[inside], weights=cover.ravel()[inside], minlength=d_max + 1)
    frac = np.divide(covered, np.maximum(at_dist, 1))[: d_max + 1]
    solid = np.flatnonzero(frac >= 0.5)
    rim = int(solid[solid == np.arange(len(solid))][-1]) if len(solid) and solid[0] == 0 else 0
    if 0 < rim < 0.08 * min(H, W) and rim + 2 + b <= dist.max():
        ring = rgb[(dist > rim + 2) & (dist <= rim + 2 + b)].reshape(-1, 3).astype(np.int32)
        bg2, share2 = _dominant_color(ring)
        if share2 >= 0.5:
            return bg2.astype(np.uint8), share2, True
    return bg.astype(np.uint8), share, False


def foreground_from_background(rgb: np.ndarray, bg_color: np.ndarray, min_dist: float = 24.0) -> tuple[np.ndarray, float]:
    """Foreground = pixels far enough from the background colour. Works for dark-on-light,
    light-on-dark and coloured-on-white designs alike (no luminance polarity guess)."""
    d = np.abs(rgb.astype(np.int16) - np.asarray(bg_color, np.int16)).max(axis=2).astype(np.float32)
    t, _ = cv2.threshold(np.clip(d, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t), float(min_dist))
    return d > t, t


# ------------------------------------------------------------------ flat-design diagnostics
DEFAULT_SUGGEST_CFG = {
    "small_long_side": 256,
    "blur_sigma": 1.0,
    "bits_per_channel": 4,
    "top_colors": 6,
    "cov_min": 0.85,  # share of the image that the dominant colours must cover
    "jump_min": 8.0,  # |cell diff| on the 8x-downscaled luma that means a hard design edge
    "jump_min_share": 0.03,  # downscale-jump density above which the image reads as a design
    "neutral_min": 0.80,
    "dark_bright_min": 0.85,
}


def analyze_image(rgb: np.ndarray, cfg: dict | None = None) -> dict:
    """Flat-design diagnostics (see docs/architecture.md §5, M2).

    cov_top      share of the image covered by the few dominant colours (after a blur
                 that removes anti-aliasing and JPEG noise) -> high for flat designs
    jump_share    share of cells of the 8x-downscaled luminance that neighbour a hard
                 edge. Line art and flat ornaments are dense in hard edges; a smoothly
                 shaded (carved) panel is not. Used to decide whether a colour mask is
                 safe: thresholding a shaded panel punches holes where its tone equals
                 the background
    dark_bright  share of very dark + very bright pixels (B/W line art)
    """
    c = {**DEFAULT_SUGGEST_CFG, **(cfg or {})}
    H, W = rgb.shape[:2]
    f = min(1.0, c["small_long_side"] / max(H, W))
    small = cv2.resize(rgb, (max(1, round(W * f)), max(1, round(H * f))), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(small.astype(np.float32), (0, 0), float(c["blur_sigma"]))
    bits = int(c["bits_per_channel"])
    levels = 1 << bits
    step = 256 // levels
    q = np.clip(sm // step, 0, levels - 1).astype(np.int32)
    code = (q[..., 0] << (2 * bits)) | (q[..., 1] << bits) | q[..., 2]
    hist = np.sort(np.bincount(code.ravel(), minlength=levels**3))[::-1]
    cov = np.cumsum(hist) / code.size
    k = int(c["top_colors"])
    lum = 0.299 * sm[..., 0] + 0.587 * sm[..., 1] + 0.114 * sm[..., 2]
    gy, gx = np.gradient(lum)
    g = np.hypot(gx, gy)
    small = cv2.resize(lum, (max(2, lum.shape[1] // 8), max(2, lum.shape[0] // 8)), interpolation=cv2.INTER_AREA)
    d = np.zeros(small.shape, np.float32)
    dxx = np.abs(np.diff(small, axis=1))
    dyy = np.abs(np.diff(small, axis=0))
    d[:, :-1] = np.maximum(d[:, :-1], dxx)
    d[:, 1:] = np.maximum(d[:, 1:], dxx)
    d[:-1] = np.maximum(d[:-1], dyy)
    d[1:] = np.maximum(d[1:], dyy)
    return {
        "cov_top": float(cov[min(k, len(cov)) - 1]),
        "top_colors": k,
        "k90": int(np.searchsorted(cov, 0.90) + 1),
        "jump_share": float((d > float(c["jump_min"])).mean()),
        "neutral": float(((sm.max(axis=2) - sm.min(axis=2)) < 24).mean()),
        "dark_bright": float((lum < 50).mean() + (lum > 205).mean()),
    }


def is_flat_design(a: dict, cfg: dict | None = None) -> bool:
    """True when the diagnostics say "flat colour / B&W design"."""
    c = {**DEFAULT_SUGGEST_CFG, **(cfg or {})}
    few_colours = a["cov_top"] >= c["cov_min"]
    bw = a["neutral"] >= c["neutral_min"] and a["dark_bright"] >= c["dark_bright_min"]
    return bool(few_colours or bw)


def is_edge_dense_design(a: dict, cfg: dict | None = None) -> bool:
    """True when the image is dense in hard edges (line art / flat ornament). Only such
    images are safe to colour-threshold; smooth shaded panels are not."""
    c = {**DEFAULT_SUGGEST_CFG, **(cfg or {})}
    return bool(a["jump_share"] >= c["jump_min_share"])


def suggest_mode(rgb: np.ndarray, cfg: dict | None = None) -> tuple[str, dict]:
    """'A' (classic design -> relief) for flat colour / B&W designs, 'B' (AI) otherwise."""
    c = {**DEFAULT_SUGGEST_CFG, **(cfg or {})}
    a = analyze_image(rgb, c)
    return ("A" if is_flat_design(a, c) else "B"), a


def get_mask(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    remove_bg: str = "auto",
    bg_share_min: float = 0.5,
    analysis: dict | None = None,
    suggest_cfg: dict | None = None,
) -> tuple[np.ndarray | None, dict]:
    """Mask of the design (True = foreground) or None when the whole image is the relief.

    remove_bg: "auto" | "on" | "off" | "rembg" (optional plug-in).

    "auto" uses alpha, or the plain background of a FLAT design. A shaded render or a
    photo gets no mask: thresholding its colours leaves holes where the shading equals
    the background (the flat top of a dome), and those holes would be flattened. Without
    a mask the plain background still becomes the base plane through the normalization.
    """
    info: dict = {"source": None}
    if remove_bg == "off":
        info["reason"] = "background removal off"
        return None, info
    if alpha is not None and float(alpha.min()) < 250.0:
        return alpha > 127, {**info, "source": "alpha"}
    if remove_bg == "rembg":
        try:
            from rembg import remove  # optional dependency, licence of the weights must be checked

            out = remove(rgb[..., ::-1], only_mask=True)
            return np.asarray(out) > 127, {**info, "source": "rembg"}
        except ImportError:
            info["rembg_fallback"] = "rembg is not installed; background-colour method used"
    bg, share, frame = estimate_background(rgb)
    info.update(bg_color=[int(v) for v in bg], bg_share=round(share, 4))
    if frame:
        info["frame_seen_through"] = True
    a = analysis if analysis is not None else analyze_image(rgb, suggest_cfg)
    flat = is_flat_design(a, suggest_cfg)
    edge_dense = is_edge_dense_design(a, suggest_cfg)
    info.update(flat_design=bool(flat), edge_dense=bool(edge_dense), jump_share=round(a["jump_share"], 4))
    if remove_bg == "auto" and (share < bg_share_min or not flat or not edge_dense):
        info["reason"] = "no plain background of an edge-dense flat design detected - whole image used"
        return None, info
    d = np.abs(rgb.astype(np.int16) - np.asarray(bg, np.int16)).max(axis=2).astype(np.float32)
    mask, t = foreground_from_background(rgb, bg)
    info.update(source="background-colour", threshold=float(t))
    # A colour mask is only meaningful when the distance-from-background map is BIMODAL:
    # background near 0, objects clearly away. A smoothly shaded panel fills the valley
    # between the modes and thresholding it splits the shading, not the design.
    valley = float(((d >= 0.5 * t) & (d < t)).mean())
    info["valley_share"] = round(valley, 4)
    if remove_bg == "auto" and valley > 0.05:
        info["reason"] = "background-to-design transition is graded, not a colour mask case - whole image used"
        return None, info
    if not edge_dense or valley > 0.05:  # forced on a shaded image: close the holes shading leaves
        from scipy import ndimage

        r = max(2, int(round(0.02 * min(mask.shape))))
        mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, structure=np.ones((2 * r + 1, 2 * r + 1), bool)))
        info["closed"] = True
    mask = keep_components(mask)
    if not mask.any():
        info["reason"] = "no foreground found"
        return None, info
    return mask, info


# ------------------------------------------------------------------ letterbox
@dataclass(frozen=True)
class LetterboxMeta:
    orig_h: int
    orig_w: int
    S: int
    new_h: int
    new_w: int
    pad_top: int
    pad_left: int

    @property
    def scale(self) -> float:
        """View scale factor s = pixels of the model view per pixel of the source grid."""
        return self.S / max(self.orig_h, self.orig_w)


def letterbox_meta(h: int, w: int, S: int) -> LetterboxMeta:
    scale = S / max(int(h), int(w))
    new_w = int(min(S, max(1, round(w * scale))))
    new_h = int(min(S, max(1, round(h * scale))))
    return LetterboxMeta(int(h), int(w), int(S), new_h, new_w, (S - new_h) // 2, (S - new_w) // 2)


def letterbox(rgb: np.ndarray, S: int) -> tuple[np.ndarray, LetterboxMeta]:
    """Aspect-preserving resize to long side S + replicate padding to S x S. Never stretches."""
    meta = letterbox_meta(rgb.shape[0], rgb.shape[1], S)
    img = resize_arr(rgb, meta.new_w, meta.new_h)
    pad = cv2.copyMakeBorder(
        img,
        meta.pad_top,
        S - meta.new_h - meta.pad_top,
        meta.pad_left,
        S - meta.new_w - meta.pad_left,
        borderType=cv2.BORDER_REPLICATE,
    )
    return np.ascontiguousarray(pad), meta


def unletterbox(arr: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    """Crop the padding and go back to the original grid (float32 stays float32)."""
    if arr.shape[:2] != (meta.S, meta.S):
        raise ValueError(f"expected a {meta.S}x{meta.S} map, got {arr.shape[:2]}")
    crop = arr[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w]
    if crop.shape[:2] == (meta.orig_h, meta.orig_w):
        return crop.copy()
    interp = cv2.INTER_CUBIC if arr.dtype.kind == "f" else None
    return resize_arr(crop, meta.orig_w, meta.orig_h, interpolation=interp)


def letterbox_mask(meta: LetterboxMeta) -> np.ndarray:
    """Valid-pixel mask of a letterboxed tensor (False on the padding)."""
    m = np.zeros((meta.S, meta.S), dtype=bool)
    m[meta.pad_top : meta.pad_top + meta.new_h, meta.pad_left : meta.pad_left + meta.new_w] = True
    return m


# ------------------------------------------------------------------ inference size
@dataclass(frozen=True)
class InferenceSizeInfo:
    S: int
    target_px: float
    capped_by_source: bool
    capped_by_device: bool


def choose_inference_size(
    src_long_side: int,
    width_mm: float,
    height_mm: float,
    tool_diameter_mm: float,
    *,
    device: str = "cpu",
    override: int | str = "auto",
    mode: str = "B",
) -> InferenceSizeInfo:
    """Working resolution (long side in px).

    target_px = 3 * long_side_mm / tool_diameter_mm  -> rounded to a multiple of 14,
    clamped to [518, 1036], never larger than the source resolution (upscaling adds no
    detail) and 518 on CPU unless the user overrides.
    """
    src_long_side = max(1, int(src_long_side))
    if override not in (None, "auto"):
        S = int(override)
        if S % 14 or S < MIN_S or S > MAX_S * 2:
            raise ValueError(f"inference size must be a multiple of 14 in [{MIN_S}, {MAX_S * 2}], got {S}")
        return InferenceSizeInfo(S, float(S), False, False)
    if mode in ("A", "C"):  # classic modes have no patch-size constraint: keep what the source has
        return InferenceSizeInfo(int(min(max(src_long_side, 2), MAX_S)), float(src_long_side), False, False)
    long_mm = max(float(width_mm), float(height_mm))
    target_px = PIXELS_PER_TOOL_DIAMETER * long_mm / float(tool_diameter_mm)
    S = int(min(max(round14(target_px), MIN_S), MAX_S))
    source_cap = max(MIN_S, ceil14(src_long_side))
    capped_src = S > source_cap
    S = int(min(S, source_cap))
    capped_dev = False
    if device == "cpu" and S > MIN_S:
        S, capped_dev = MIN_S, True
    return InferenceSizeInfo(S, target_px, capped_src, capped_dev)
