"""M8 — scene recipes: the five families, symmetry, background plate, rejection rules."""
from __future__ import annotations

import numpy as np

from . import shapes as sh
from .compose import compose, mirror_stamps

FAMILIES = ("floral", "botanical", "scroll", "border", "medallion")
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
        n = int(rng.integers(1, 4))
        for _ in range(n):
            cx, cy = rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H
            R = size * rng.uniform(0.08, 0.22)
            petals = int(rng.choice([4, 6, 8, 12]))
            prof = str(rng.choice(["dome", "pillow"]))
            stamps += sh.rosette_stamps(cx, cy, R, petals, R * 0.75, R * 0.38, prof, rng.uniform(0.5, 1.0), rng.uniform(0, 0.25), inst, rng)
            inst += 1
        for _ in range(int(rng.integers(2, 8))):  # scattered petals
            stamps += sh.petal_stamps(rng.uniform(0.1, 0.9) * W, rng.uniform(0.1, 0.9) * H, size * rng.uniform(0.03, 0.09), rng.uniform(0, 360), str(rng.choice(sh.PROFILES)), rng.uniform(0.25, 0.7), rng.uniform(0, 0.15), inst, rng)
            inst += 1
    elif family == "botanical":
        n = int(rng.integers(1, 3))
        for _ in range(n):
            stamps += sh.vine_stamps(rng.uniform(0.25, 0.75) * W, rng.uniform(0.25, 0.75) * H, min(W, H), size * rng.uniform(0.2, 0.42), size * rng.uniform(0.015, 0.035), float(rng.uniform(0.5, 1.2)), str(rng.choice(["dome", "pillow"])), rng.uniform(0.5, 1.0), rng.uniform(0, 0.2), inst, rng, wobble=0.35, leaves=int(rng.integers(3, 9)))
            inst += 1
    elif family == "scroll":
        n = int(rng.integers(1, 3))
        for _ in range(n):
            stamps += sh.vine_stamps(rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H, min(W, H), size * rng.uniform(0.22, 0.45), size * rng.uniform(0.012, 0.03), float(rng.uniform(1.0, 2.2)), "dome", rng.uniform(0.45, 0.95), rng.uniform(0, 0.2), inst, rng, wobble=0.15, leaves=0)
            inst += 1
    elif family == "border":
        m = size * rng.uniform(0.03, 0.09)  # border margin
        hw, hh = W - 2 * m, H - 2 * m
        stamps += sh.ring_stamps(W / 2, H / 2, 0, size * rng.uniform(0.012, 0.03), str(rng.choice(["bead" if False else "dome", "pillow", "flat_round"])), rng.uniform(0.5, 0.9), rng.uniform(0, 0.15), inst, segments=180, rectangular=True, hw=hw, hh=hh)
        inst += 1
        for _ in range(int(rng.integers(4, 20))):  # repeated motifs along the edges
            side = int(rng.integers(0, 4))
            t = rng.uniform(0.1, 0.9)
            if side == 0:
                cx, cy = m + t * hw, m
            elif side == 1:
                cx, cy = m + hw, m + t * hh
            elif side == 2:
                cx, cy = m + t * hw, m + hh
            else:
                cx, cy = m, m + t * hh
            if rng.random() < 0.5:
                stamps += sh.petal_stamps(cx, cy, size * rng.uniform(0.025, 0.06), rng.uniform(0, 360), "dome", rng.uniform(0.4, 0.8), rng.uniform(0.05, 0.25), inst, rng)
            else:
                stamps.append(sh.Stamp(cx, cy, sh.disc_kernel(size * rng.uniform(0.015, 0.04), "dome"), rng.uniform(0.4, 0.8), rng.uniform(0.05, 0.2), inst))
            inst += 1
        if rng.random() < 0.6:  # centre ornament
            stamps += sh.rosette_stamps(W / 2, H / 2, min(hw, hh) * rng.uniform(0.3, 0.55), int(rng.choice([4, 6, 8])), min(hw, hh) * 0.2, min(hw, hh) * 0.1, "dome", rng.uniform(0.6, 1.0), 0.1, inst, rng)
            inst += 1
    elif family == "medallion":
        cx, cy = W / 2, H / 2
        rings = int(rng.integers(2, 5))
        r = size * rng.uniform(0.28, 0.42)
        for i in range(rings):
            stamps += sh.ring_stamps(cx, cy, r, size * rng.uniform(0.015, 0.045), str(rng.choice(["dome", "pillow", "bevel", "flat_round"])), rng.uniform(0.4, 0.9), rng.uniform(0, 0.2), inst)
            r *= float(rng.uniform(0.62, 0.8))
            inst += 1
        stamps += sh.rosette_stamps(cx, cy, r * 1.1, int(rng.choice([6, 8, 12])), r * 0.7, r * 0.35, "dome", rng.uniform(0.7, 1.0), 0.15, inst, rng)
        inst += 1
    else:
        raise ValueError(f"unknown family {family}")
    return stamps, inst


def make_scene(seed: int, family: str | None = None) -> dict:
    """Generate one scene: height, instance map, metadata."""
    rng = np.random.default_rng(seed)
    family = family or str(rng.choice(FAMILIES))
    aspect, W, H = pick_canvas(rng)
    symmetry = str(rng.choice(SYMMETRIES, p=[0.5, 0.35, 0.15]))
    stamps, n_inst = _family_stamps(family, W, H, rng)
    stamps = mirror_stamps(stamps, W, symmetry, inst_offset=n_inst)
    bg_layer = float(rng.uniform(0.05, 0.3)) if rng.random() < 0.5 else 0.0
    if bg_layer > 0:  # a background plate with a moulded edge
        stamps = [sh.Stamp(W / 2, H / 2, sh._kernel_from_mask(np.ones((H - 8, W - 8), np.uint8), "flat_round", r_frac=0.12), 1.0, 0.0, 0)] + stamps
        bg_layer = 0.0  # the plate IS the layer via the stamp
    h, inst = compose(H, W, stamps, bg_layer=0.0)
    fg = (h > 0.02)
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
    return scene  # last attempt wins (QC will flag it on the contact sheet)
