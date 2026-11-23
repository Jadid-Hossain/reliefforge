"""M8 — unit tests for the generator: support-mask composition, kappa, views, QC."""
import json

import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from synth import compose as C
from synth import recipes as R
from synth import render as RD
from synth import shapes as SH
from synth.generate import contact_sheet, generate_dataset, h_to_u16, qc_assertions


# ------------------------------------------------------------------ composition (Issue 2)
def test_layer_offset_never_leaks_outside_the_primitive_support():
    rng = np.random.default_rng(3)
    H = W = 200
    stamps = [
        SH.Stamp(50, 100, SH.disc_kernel(30, "dome"), 0.4, layer=0.3, inst_id=1),
        SH.Stamp(150, 100, SH.disc_kernel(20, "dome"), 0.6, layer=0.0, inst_id=2),
    ]
    h, inst = C.compose(H, W, stamps, blur_sigma=0.0)
    far = np.zeros((H, W), bool)
    far[:40, :] = True  # 40 px above the first stamp: far outside every support
    assert (h[far] == 0.0).all(), "a positive layer offset leaked outside its support"
    assert h[100, 50] >= 0.3 and h[100, 150] >= 0.6
    assert inst[100, 50] == 1 and inst[100, 150] == 2 and inst[0, 0] == 0


def test_instance_map_only_updates_on_valid_pixels():
    stamps = [SH.Stamp(60, 60, SH.disc_kernel(40, "dome"), 1.0, 0.0, 1), SH.Stamp(100, 60, SH.disc_kernel(30, "dome"), 0.2, 0.0, 2)]
    h, inst = C.compose(120, 120, stamps, blur_sigma=0.0)
    # stamp 2 is lower: where they overlap, stamp 1 must keep the instance id
    overlap = (inst == 1)
    assert overlap[60, 80]
    assert (inst != 0).sum() > 0 and set(np.unique(inst)) <= {0, 1, 2}


def test_compose_peak_is_one_and_background_zero():
    rng = np.random.default_rng(0)
    scene = R.make_scene_checked(1)
    h, meta = scene["h"], scene["meta"]
    assert h.max() == pytest.approx(1.0, abs=1e-4)
    assert h.min() >= 0.0
    assert (h[scene["inst"] == 0] <= 0.02).all()


# ------------------------------------------------------------------ canvases + rejection
@pytest.mark.parametrize("aspect,W,H", [("1:1", 1036, 1036), ("3:2", 1260, 840), ("2:3", 840, 1260), ("2:1", 1470, 728), ("1:2", 728, 1470)])
def test_canvases_are_multiples_of_14(aspect, W, H):
    assert R._canvas(aspect) == (W, H)
    assert W % 14 == 0 and H % 14 == 0


def test_crop_scale_lock_is_respected_by_every_canvas():
    # review fix ১: s_min = max(0.55, 518 / min(H, W)) must make a 518 crop possible
    for aspect, (W, H) in {"1:1": (1036, 1036), "3:2": (1260, 840), "2:1": (1470, 728)}.items():
        s_min = max(0.55, 518 / min(H, W))
        assert round(H * s_min) >= 518 and round(W * s_min) >= 518, aspect


def test_every_family_generates_and_passes_rejection():
    for fam in R.FAMILIES:
        scene = R.make_scene_checked(100 + hash(fam) % 500, fam)
        assert scene["meta"]["family"] == fam
        assert R.scene_ok(scene["meta"])


def test_scenes_are_deterministic():
    a = R.make_scene_checked(42)
    b = R.make_scene_checked(42)
    assert np.array_equal(a["h"], b["h"]) and a["meta"]["family"] == b["meta"]["family"]


# ------------------------------------------------------------------ rendering
def test_kappa_is_physical_and_view_scaled():
    meta = {"relief_depth_mm_scene": 8.0, "px_per_mm_native": 1036 / 300.0}
    k_native = RD.kappa_of(meta, view_scale=1.0)
    assert k_native == pytest.approx(8.0 * 1036 / 300.0, rel=1e-6)
    assert RD.kappa_of(meta, view_scale=0.5) == pytest.approx(k_native / 2, rel=1e-6)


def test_all_four_styles_render():
    scene = R.make_scene_checked(7)
    h, inst, meta = scene["h"], scene["inst"], scene["meta"]
    small_h = cv2.resize(h, (518, int(518 * h.shape[0] / h.shape[1])), interpolation=cv2.INTER_AREA)
    small_i = cv2.resize(inst, small_h.shape[::-1], interpolation=cv2.INTER_NEAREST)
    for style in RD.STYLES:
        rng = np.random.default_rng(1)
        img = RD.render_view(small_h, small_i, meta, 518 / max(meta["W"], meta["H"]), style, rng)
        assert img.shape[:2] == small_h.shape and img.dtype == np.uint8
        assert 5 <= int(img.std()) <= 200  # not flat, not noise


def test_render_is_deterministic_given_the_seed():
    scene = R.make_scene_checked(9)
    h, inst, meta = scene["h"], scene["inst"], scene["meta"]
    a = RD.render_view(h, inst, meta, 1.0, "wood", np.random.default_rng(5))
    b = RD.render_view(h, inst, meta, 1.0, "wood", np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_lighting_polarity():
    """A bump lit from the left is bright on its left flank (PRD_v2 §5.7 polarity test)."""
    yy, xx = np.mgrid[0:200, 0:200].astype(np.float32)
    h = np.exp(-(((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 45**2)))
    kappa = 12.0
    gy, gx = np.gradient(h * kappa)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array([-1.0, 0.0, 0.5], np.float32)  # light sits to the left: L points toward it
    L /= np.linalg.norm(L)
    sh = 0.35 + 0.65 * np.clip(n @ L, 0, 1)
    assert sh[100, 60] > sh[100, 140]  # left flank brighter when the light comes from the left
    img = RD.render_view(h, (h > 0.1).astype(np.uint16), {"relief_depth_mm_scene": 8, "px_per_mm_native": 1.5}, 1.0, "clay", np.random.default_rng(2))
    assert img.std() > 3  # gentle bump; real scenes use kappa 14-31


# ------------------------------------------------------------------ dataset build + QC
def _tiny_scene(seed, family):
    # cheap stand-in for pipeline tests: 518x518 canvas
    rng = np.random.default_rng(seed)
    stamps = SH.rosette_stamps(259, 259, 120, 6, 80, 38, "dome", 0.9, 0.1, 1, rng)
    h, inst = C.compose(518, 518, stamps)
    return {"h": h, "inst": inst, "meta": {"seed": seed, "family": family, "symmetry": "none", "aspect": "1:1", "W": 518, "H": 518, "width_mm_scene": 300.0, "relief_depth_mm_scene": 8.0, "fg_frac": float((h > 0.02).mean()), "h_std": float(h.std())} | {"px_per_mm_native": 300.0 / 518}}


def test_generate_dataset_and_qc(tmp_path):
    root = generate_dataset(tmp_path / "ds", n_scenes=12, val_frac=0.25, test_frac=0.25, scene_fn=_tiny_scene, make_previews=True, log_every=0)
    rep = qc_assertions(root)
    assert rep["n_scenes"] == 12 and rep["splits"] == {"train": 6, "val": 3, "test": 3}
    rows = [json.loads(l) for l in open(root / "scenes.jsonl")]
    for r in rows:
        if r["split"] in ("val", "test"):
            for style in ("wood", "clay", "flat"):
                assert (root / "renders" / r["split"] / f"{r['id']:06d}_{style}.png").exists()
    sheet = contact_sheet(root, n=4)
    assert sheet.exists() and sheet.stat().st_size > 10_000


def test_train_val_test_never_share_a_scene(tmp_path):
    root = generate_dataset(tmp_path / "ds", n_scenes=8, val_frac=0.25, test_frac=0.25, scene_fn=_tiny_scene, log_every=0)
    rows = [json.loads(l) for l in open(root / "scenes.jsonl")]
    by_split = {s: {r["seed"] for r in rows if r["split"] == s} for s in ("train", "val", "test")}
    assert not (by_split["train"] & by_split["val"]) and not (by_split["train"] & by_split["test"])
