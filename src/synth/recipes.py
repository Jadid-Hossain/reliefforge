"""M8 — scene recipes: families, symmetry, background plate, rejection rules.

Families (furniture-carving oriented, v2 after the contact-sheet review 2026-11-24):
floral, botanical, scroll, border, medallion, panel — "panel" is the classic carved
door layout: border frame + centre medallion + corner rosettes (+ optional plate).
"""
from __future__ import annotations

import numpy as np

from . import shapes as sh
from .compose import compose, mirror_stamps

FAMILIES = ("floral", "botanical", "scroll", "border", "medallion", "panel")
SYMMETRIES = ("none", "mirror", "4fold")


def _canvas(aspect_key: str) -> tuple[int, int]:
    return {"1:1": (1036, 1036), "3:2": (1260, 840), "2:3": (840, 1260), "2:1": (1470, 728), "1:2": (728, 1470)}[aspect_key]


def pick_canvas(rng: np.random.Generator) -> tuple[str, int, int]:
    r = rng.random()
    aspect = "1:1" if r < 0.35 else rng.choice(["3:2", "2:3"]) if r < 0.65 else rng.choice(["2:1", "1:2"])
    W, H = _canvas(str(aspect))
    return str(aspect), W, H


def _family_stamps(family: str, W: int, H: int, rng: np.random.Generator):
    size = min(W, H)
    stamps = []
    inst = 1
    if family == "floral":
        # a main rosette + a few clearly separated flowers (gaps, pointed petals)
        cx, cy = W / 2, H / 2
        R = size * rng.uniform(0.10, 0.22)
        stamps += sh.rosette_stamps(cx, cy, R, int(rng.choice([5, 6, 8, 12])), str(rng.choice(["dome", "pillow"])), rng.uniform(0.55, 1.0), rng.uniform(0, 0.2), inst, rng, petal_rings=int(rng.integers(1, 3)))
        inst += 1
        for _ in range(int(rng.integers(2, 4))):
            for _try in range(8):
                fx, fy = rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H
                fr = size * rng.uniform(0.06, 0.11)
                if np.hypot(fx - cx, fy - cy) > R + fr * 1.6:  # clear separation, but real size
                    stamps += sh.rosette_stamps(fx, fy, fr, int(rng.choice([5, 6, 8])), "dome", rng.uniform(0.45, 0.8), rng.uniform(0, 0.1), inst, rng)
                    inst += 1
                    break
    elif family == "botanical":
        # 1-2 vines with leaves, plus a central rosette
        for _ in range(int(rng.integers(1, 3))):
            stamps += sh.vine_stamps(rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H, size, size * rng.uniform(0.22, 0.4), size * rng.uniform(0.015, 0.03), float(rng.uniform(0.6, 1.4)), str(rng.choice(["dome", "pillow"])), rng.uniform(0.5, 0.95), rng.uniform(0, 0.15), inst, rng, leaves=int(rng.integers(4, 10)))
            inst += 1
        if rng.random() < 0.7:
            stamps += sh.rosette_stamps(W / 2, H / 2, size * rng.uniform(0.08, 0.15), int(rng.choice([6, 8])), "dome", rng.uniform(0.6, 0.95), 0.1, inst, rng)
            inst += 1
    elif family == "scroll":
        for _ in range(int(rng.integers(1, 3))):
            stamps += sh.scroll_stamps(W / 2, H / 2, size, size * rng.uniform(0.24, 0.34), size * rng.uniform(0.018, 0.035), float(rng.uniform(0.8, 1.6)), "dome", rng.uniform(0.5, 0.95), rng.uniform(0, 0.15), inst, rng)
            inst += 1
        if rng.random() < 0.5:  # a leaf accent at the scroll centre
            stamps += sh.petal_stamps(W / 2 + size * 0.05, H / 2, size * rng.uniform(0.05, 0.09), rng.uniform(0, 360), "pillow", 0.8, 0.1, inst, rng)
            inst += 1
    elif family == "border":
        m = size * rng.uniform(0.05, 0.12)
        stamps += sh.border_stamps(W, H, m, size * rng.uniform(0.015, 0.035), str(rng.choice(["dome", "pillow", "flat_round"])), rng.uniform(0.5, 0.9), rng.uniform(0, 0.15), inst, rng)
        inst += 1
        if rng.random() < 0.7:  # centre ornament inside the frame
            hw, hh = W / 2 - m, H / 2 - m
            stamps += sh.rosette_stamps(W / 2, H / 2, min(hw, hh) * rng.uniform(0.45, 0.8), int(rng.choice([6, 8])), "dome", rng.uniform(0.6, 1.0), 0.12, inst, rng)
            inst += 1
    elif family == "medallion":
        cx, cy = W / 2, H / 2
        r = size * rng.uniform(0.3, 0.42)
        rings = int(rng.integers(2, 4))
        for i in range(rings):
            stamps += sh.ring_stamps(cx, cy, r, size * rng.uniform(0.018, 0.05), str(rng.choice(["dome", "pillow", "bevel", "flat_round"])), rng.uniform(0.4, 0.85), rng.uniform(0, 0.15), inst)
            r *= float(rng.uniform(0.64, 0.8))
            inst += 1
        stamps += sh.rosette_stamps(cx, cy, r * 1.35, int(rng.choice([6, 8, 12])), "dome", rng.uniform(0.7, 1.0), 0.15, inst, rng, petal_rings=1)
        inst += 1
    elif family == "panel":
        # the carved-door layout: plate + border + centre medallion + 4 corner rosettes
        m = size * rng.uniform(0.045, 0.1)
        if rng.random() < 0.7:  # background plate
            plate_mask = np.zeros((H, W), np.uint8)
            cv2_rect = None
            import cv2 as _cv2

            _cv2.rectangle(plate_mask, (int(m * 2), int(m * 2)), (int(W - m * 2), int(H - m * 2)), 1, -1)
            stamps.append(sh.Stamp(W / 2, H / 2, sh._kernel_from_mask(plate_mask, "flat_round") * 0.001 + _cv2.GaussianBlur(sh._kernel_from_mask(plate_mask, "flat_round"), (0, 0), 0.001), 1.0, 0.0, 0))
        stamps += sh.border_stamps(W, H, m, size * rng.uniform(0.015, 0.032), str(rng.choice(["dome", "pillow"])), rng.uniform(0.5, 0.85), 0.0, inst, rng)
        inst += 1
        stamps += sh.rosette_stamps(W / 2, H / 2, min(W - 2 * m, H - 2 * m) * rng.uniform(0.2, 0.32), int(rng.choice([6, 8])), "dome", rng.uniform(0.75, 1.0), 0.1, inst, rng, petal_rings=1)
        inst += 1
        hw, hh = W / 2 - m, H / 2 - m
        for sx in (-1, 1):
            for sy in (-1, 1):
                stamps += sh.rosette_stamps(W / 2 + sx * hw * rng.uniform(0.6, 0.75), H / 2 + sy * hh * rng.uniform(0.6, 0.75), size * rng.uniform(0.05, 0.09), int(rng.choice([5, 6, 8])), "dome", rng.uniform(0.45, 0.75), 0.08, inst, rng)
                inst += 1
    else:
        raise ValueError(f"unknown family {family}")
    return stamps, inst


def make_scene(seed: int, family: str | None = None) -> dict:
    """Generate one scene: height, instance map, metadata."""
    rng = np.random.default_rng(seed)
    family = family or str(rng.choice(FAMILIES))
    aspect, W, H = pick_canvas(rng)
    symmetry = str(rng.choice(SYMMETRIES, p=[0.5, 0.35, 0.15])) if family != "panel" else "mirror"
    stamps, n_inst = _family_stamps(family, W, H, rng)
    stamps = mirror_stamps(stamps, W, symmetry, inst_offset=n_inst)
    h, inst = compose(H, W, stamps, bg_layer=0.0)
    fg = h > 0.02
    meta = {
        "seed": int(seed),
        "family": family,
        "symmetry": symmetry,
        "aspect": aspect,
        "W": int(W),
        "H": int(H),
        "width_mm_scene": float(rng.uniform(200, 600)),
        "relief_depth_mm_scene": float(rng.uniform(4, 15)),
        "fg_frac": float(fg.mean()),
        "h_std": float(h.std()),
    }
    meta["px_per_mm_native"] = meta["width_mm_scene"] / W
    return {"h": h, "inst": inst, "meta": meta}


def scene_ok(meta: dict) -> bool:
    """Rejection rules (PRD_v2 §5.3)."""
    return 0.03 <= meta["fg_frac"] <= 0.95 and meta["h_std"] >= 0.02


def make_scene_checked(seed: int, family: str | None = None, max_tries: int = 8) -> dict:
    for attempt in range(max_tries):
        scene = make_scene(seed * 1000 + attempt, family)
        if scene_ok(scene["meta"]):
            scene["meta"]["seed"] = int(seed * 1000 + attempt)
            return scene
    return scene
