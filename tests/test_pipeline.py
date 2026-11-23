"""M3/M4 + pipeline tests. Mode B is tested with the real cached zero-shot model once."""
import cv2
import numpy as np
import pytest

from reliefkit import modes as A
from reliefkit import pipeline as P
from reliefkit import postprocess as F


def _flat_design(size=240, r=80):
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.circle(img, (size // 2, size // 2), r, (30, 30, 160), -1, lineType=cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ Mode A
def test_mode_a_gives_a_dome_that_scales_with_the_image():
    small, info_s = A.design_to_height(_flat_design(240, 80))
    big, info_b = A.design_to_height(cv2.resize(_flat_design(240, 80), (960, 960), interpolation=cv2.INTER_AREA))
    for h, size in ((small, 240), (big, 960)):
        assert h.shape == (size, size)
        assert h.max() == pytest.approx(1.0, abs=0.05)
        assert h[0, 0] == 0.0  # background flat
    # review fix ৩: the bevel width scales with the long side, not with pixels
    assert info_b["r_px"] == pytest.approx(info_s["r_px"] * 4, rel=0.05)


def test_mode_a_transition_width_mm_overrides_the_auto_radius():
    img = _flat_design(300, 100)
    h1, i1 = A.design_to_height(img, profile="bevel", transition_width_mm=20.0, pixel_pitch_mm=1.0)
    h2, i2 = A.design_to_height(img, profile="bevel", transition_width_mm=4.0, pixel_pitch_mm=1.0)
    assert i1["r_source"] == i2["r_source"] == "transition_width_mm"
    assert i1["r_px"] == pytest.approx(20.0) and i2["r_px"] == pytest.approx(4.0)
    # 10 px inside the outline the wide bevel is only at half height, the narrow one is a plateau
    yy, xx = np.mgrid[0:300, 0:300]
    d = np.hypot(yy - 150, xx - 150)
    ring = (d > 88) & (d < 92)  # ~10 px inside the radius-100 circle
    assert h1[ring].mean() == pytest.approx(0.5, abs=0.08)
    assert h2[ring].mean() > 0.97


def test_mode_a_profiles():
    img = _flat_design(220, 70)
    for profile in ("dome", "pillow", "bevel", "flat_round"):
        h, info = A.design_to_height(img, profile=profile)
        assert 0.95 < float(h.max()) <= 1.0 and float(h.min()) >= 0.0
    with pytest.raises(ValueError):
        A.design_to_height(img, profile="wave")


def test_mode_a_separate_ornaments_stay_separate():
    img = np.full((200, 400, 3), 255, np.uint8)
    cv2.circle(img, (100, 100), 40, (0, 0, 0), -1)
    cv2.rectangle(img, (260, 60), (340, 140), (0, 0, 0), -1)
    h, info = A.design_to_height(img)
    assert h[100, 100] > 0.8 and h[100, 320] > 0.8 and h[100, 200] == 0.0


# ------------------------------------------------------------------ Mode C
def test_mode_c_luminance_is_height():
    yy = np.linspace(0, 1, 64, dtype=np.float32)[:, None].repeat(64, 1)
    grey = (np.stack([yy] * 3, -1) * 255).astype(np.uint8)
    h, _ = A.gray_to_height(grey, gamma=1.0)
    assert h[0, 0] == pytest.approx(0.0, abs=0.01) and h[-1, 0] == pytest.approx(1.0, abs=0.01)
    h2, _ = A.gray_to_height(grey, invert_luminance=True)
    assert h2[0, 0] == pytest.approx(1.0, abs=0.01)


# ------------------------------------------------------------------ M4 postprocess
def _bump(size=200, r=70):
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    return np.exp(-((yy - size / 2) ** 2 + (xx - size / 2) ** 2) / (2 * r * r / 4)).astype(np.float32)


def test_normalization_anchors_background_and_keeps_plateaus():
    raw = _bump() * 0.4  # a bump whose max is only 0.4
    mask = raw > 0.05
    h, info = F.finalize_height(raw, mask)
    assert info["peak"] == pytest.approx(1.0, abs=0.03)  # normalized inside the reference
    assert h[0, 0] == 0.0 and h[-1, -1] == 0.0
    # a flat plateau is not stretched into noise (review fix, PRD_v2 §0.1 #4)
    plateau = np.where(mask, 0.5, 0.0).astype(np.float32)
    h2, info2 = F.finalize_height(plateau, mask)
    assert info2["peak"] == pytest.approx(1.0, abs=0.05)
    assert float(np.median(h2[mask])) > 0.9


def test_invert_is_relative_to_the_base_plane():
    raw = _bump()
    mask = raw > 0.05
    h, _ = F.finalize_height(raw, mask, invert=False)
    hi, _ = F.finalize_height(raw, mask, invert=True)
    assert h[100, 100] > 0.9 and hi[100, 100] == pytest.approx(0.0, abs=0.03)
    # outside the (feathered) mask: base plane = 1.0 when inverted, 0.0 otherwise (Issue 1)
    assert hi[2, 2] == 1.0 and h[2, 2] == 0.0
    assert hi[-2, -2] == 1.0
    # no raised halo: just inside the dilated reference ring but outside the mask
    ring = F.reference_region(mask, (200, 200)) & ~mask
    assert (hi[ring] > 0.97).mean() > 0.999 or hi[ring].max() <= 1.0


def test_flat_prediction_returns_zeros_with_warning():
    h, info = F.finalize_height(np.full((64, 64), 3.0, np.float32), None)
    assert h.max() == 0.0 and any("no relief" in w for w in info["warnings"])


def test_nan_input_is_sanitized_not_fatal():
    raw = _bump(64, 20)
    raw[10, 10] = np.nan
    h, info = F.finalize_height(raw, None)
    assert np.isfinite(h).all() and any("NaN" in w for w in info["warnings"])


def test_outside_feathered_mask_is_exactly_base_level():
    raw = _bump(300, 100)
    mask = raw > 0.05
    for invert in (False, True):
        h, _ = F.finalize_height(raw, mask, invert=invert, smoothing=1.0)
        far = np.zeros_like(mask)
        far[:20, :] = True  # 20 px away from the design: beyond the ~3 px feather
        base = 1.0 if invert else 0.0
        assert (h[far] == base).all()


def test_gamma_darkens_or_brightens():
    raw = _bump(100, 35)
    mask = raw > 0.05
    h1, _ = F.finalize_height(raw, mask, gamma=1.0)
    h2, _ = F.finalize_height(raw, mask, gamma=2.0)
    mid = (h1 > 0.3) & (h1 < 0.7)
    assert (h2[mid] < h1[mid]).all()


# ------------------------------------------------------------------ pipeline (classic modes)
def test_pipeline_mode_a_end_to_end(tmp_path):
    img = _flat_design(300, 100)
    params = P.ReliefParams(mode="A", width_mm=300, relief_depth_mm=8, base_thickness_mm=6)
    res = P.generate_relief(img, None, params, model=None)
    assert res.validation["is_valid"]
    assert res.meta["mode"] == "A"
    exported = res.save(tmp_path)
    for key in ("heightmap_tiff", "stl", "obj", "3mf", "meta", "preview", "zip"):
        assert key in exported
    from reliefkit.io import load_height

    back = load_height(exported["heightmap_tiff"])
    assert back.shape == res.h01.shape
    assert float(back.max()) == pytest.approx(float(res.h01.max()), abs=1 / 65535)


def test_pipeline_rejects_invalid_params():
    img = _flat_design(64, 20)
    with pytest.raises(ValueError):
        P.generate_relief(img, None, P.ReliefParams(mode="A", width_mm=9999))
    with pytest.raises(ValueError):
        P.generate_relief(img, None, P.ReliefParams(mode="A", base_thickness_mm=0.1))
    with pytest.raises(ValueError):
        P.generate_relief(img, None, P.ReliefParams(mode="B", remove_bg="sometimes"))


def test_pipeline_tool_and_slope_warnings():
    img = _flat_design(300, 100)
    res = P.generate_relief(img, None, P.ReliefParams(mode="A", width_mm=900, tool_diameter_mm=0.5))
    assert any("tool" in w or "pitch" in w for w in res.warnings)


def test_pipeline_mode_b_with_fake_model(tmp_path):
    class FakeModel:
        device = "cpu"
        model_id = "fake"

        def predict(self, rgb, S=518, tta=False):
            yy, xx = np.mgrid[0 : rgb.shape[0], 0 : rgb.shape[1]].astype(np.float32)
            h = np.exp(-(((yy - rgb.shape[0] / 2) ** 2 + (xx - rgb.shape[1] / 2) ** 2) / (2 * (rgb.shape[0] / 3) ** 2)))
            return h.astype(np.float32)

    img = _flat_design(260, 90)
    res = P.generate_relief(img, None, P.ReliefParams(mode="B", width_mm=300), model=FakeModel())
    assert res.validation["is_valid"] and res.meta["mode"] == "B" and res.info["S"] == 518
    assert res.meta["model_version"] == "fake"
    res.save(tmp_path)
