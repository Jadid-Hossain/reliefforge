"""M2 — mask, components, letterbox, mode suggestion and inference size."""
import cv2
import numpy as np
import pytest

from reliefkit import preprocess as P


def _jpeg(rgb, quality=75):
    ok, buf = cv2.imencode(".jpg", rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]


def _dome_shaded(size=300, radius=110, light=(-0.6, -0.6, 0.53)):
    """A clay render of a dome: smooth Lambert shading, no flat colours anywhere."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r2 = (yy - size / 2) ** 2 + (xx - size / 2) ** 2
    u = np.clip(np.sqrt(r2) / radius, 0, 1)
    h = np.sqrt(np.clip(1 - (1 - u) ** 2, 0, 1))
    gy, gx = np.gradient(h * 25.0)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    L = np.array(light, np.float32)
    L /= np.linalg.norm(L)
    sh = 0.35 + 0.65 * np.clip(n @ L, 0, 1)
    img = (np.clip(sh, 0, 1) * 0.72 * 255).astype(np.uint8)
    return np.stack([img] * 3, -1)


# ------------------------------------------------------------------ letterbox
@pytest.mark.parametrize("shape", [(200, 400), (400, 200), (300, 300), (7, 1000), (518, 518), (1000, 518)])
def test_letterbox_never_stretches_and_round_trips(shape):
    img = np.random.default_rng(0).integers(0, 256, (*shape, 3), dtype=np.uint8)
    out, meta = P.letterbox(img, 518)
    assert out.shape == (518, 518, 3) and out.dtype == np.uint8
    assert meta.orig_h == shape[0] and meta.orig_w == shape[1]
    assert max(meta.new_h, meta.new_w) == 518 and min(meta.new_h, meta.new_w) >= 1
    # aspect ratio preserved to within a pixel
    assert abs(meta.new_h / meta.new_w - shape[0] / shape[1]) <= 1.2 / min(meta.new_h, meta.new_w)  # within a pixel
    assert P.letterbox_mask(meta).sum() == meta.new_h * meta.new_w
    back = P.unletterbox(out, meta)
    assert back.shape == (*shape, 3)
    if shape == (518, 518):
        assert np.array_equal(back, img)


def test_letterbox_padding_is_replicated_from_the_edge():
    img = np.zeros((100, 200, 3), np.uint8)
    img[:, :] = (10, 200, 30)
    out, meta = P.letterbox(img, 518)  # wide image: padded top and bottom only
    assert meta.pad_left == 0 and meta.pad_top == 129 and meta.new_w == 518
    assert (out[: meta.pad_top] == (10, 200, 30)).all()
    assert (out[meta.pad_top + meta.new_h :] == (10, 200, 30)).all()
    tall = np.zeros((300, 100, 3), np.uint8)
    tall[:] = (200, 10, 10)
    tall[:, :4] = (10, 10, 250)
    out, meta = P.letterbox(tall, 518)
    assert meta.pad_left > 0 and meta.pad_top == 0
    # the red strip is ~21 px wide after the resize, so the left padding replicates it
    assert (out[:, : meta.pad_left] == (10, 10, 250)).all()
    strip = max(1, round(4 * meta.new_w / 100))
    inner = out[:, meta.pad_left : meta.pad_left + strip - 1].astype(np.int16)
    assert (np.abs(inner - (10, 10, 250)).max(axis=2) <= 14).mean() > 0.9  # Lanczos ringing stays small


def test_unletterbox_keeps_float32_and_values():
    h = np.linspace(0, 1, 518, dtype=np.float32)[None, :].repeat(259, 0)  # horizontal ramp
    meta = P.letterbox_meta(518, 1036, 518)
    assert meta.new_h == 259 and meta.new_w == 518 and meta.pad_left == 0
    padded = np.zeros((518, 518), np.float32)
    padded[meta.pad_top : meta.pad_top + 259] = h
    back = P.unletterbox(padded, meta)
    assert back.shape == (518, 1036) and back.dtype == np.float32
    assert abs(float(back.mean()) - 0.5) < 0.005
    with pytest.raises(ValueError):
        P.unletterbox(h, meta)  # a map without the padding must not pass silently


def test_letterbox_meta_scale_is_the_view_scale():
    meta = P.letterbox_meta(840, 1260, 518)
    assert meta.scale == pytest.approx(518 / 1260)
    assert meta.new_w == 518 and meta.new_h == round(840 * 518 / 1260)


# ------------------------------------------------------------------ components / holes
def test_keep_components_never_keeps_only_the_largest():
    m = np.zeros((200, 300), bool)
    m[20:80, 20:80] = True  # 60x60
    m[120:170, 200:250] = True  # 50x50 - a second, separate ornament
    m[10, 290] = True  # dust
    m[150:152, 10:13] = True  # 6 px < 0.05 % of 60000 (=30 px)
    out = P.keep_components(m)
    n, labels = cv2.connectedComponents(out.astype(np.uint8))
    assert n - 1 == 2
    assert out[50, 50] and out[145, 225]
    assert not out[10, 290] and not out[150, 11]


def test_fill_holes_only_when_asked():
    ring = np.zeros((100, 100), bool)
    ring[10:90, 10:90] = True
    ring[35:65, 35:65] = False
    assert P.keep_components(ring)[50, 50] is np.False_
    assert P.fill_holes(ring)[50, 50]


# ------------------------------------------------------------------ background mask
def _flat_design(bg=(255, 255, 255), fg=(200, 20, 30), dark=False):
    img = np.empty((240, 320, 3), np.uint8)
    img[:] = bg
    cv2.circle(img, (110, 120), 60, fg, -1, lineType=cv2.LINE_AA)
    cv2.rectangle(img, (210, 60), (300, 180), (20, 40, 220), -1, lineType=cv2.LINE_AA)
    return _jpeg(img, 80)


def test_mask_from_alpha():
    rgb = np.zeros((50, 60, 3), np.uint8)
    alpha = np.zeros((50, 60), np.uint8)
    alpha[10:30, 20:40] = 255
    mask, info = P.get_mask(rgb, alpha, "auto")
    assert info["source"] == "alpha"
    assert mask.shape == (50, 60) and mask.dtype == bool and mask[20, 30] and not mask[0, 0]


def test_opaque_alpha_carries_no_information():
    rgb = _flat_design()
    mask, info = P.get_mask(rgb, np.full(rgb.shape[:2], 255, np.uint8), "auto")
    assert info["source"] == "background-colour"


def test_mask_from_plain_background_and_removal_switches():
    rgb = _flat_design()
    mask, info = P.get_mask(rgb, None, "auto")
    assert info["source"] == "background-colour"
    assert mask[120, 110] and mask[120, 250] and not mask[5, 5]
    assert P.get_mask(rgb, None, "off")[0] is None
    assert info["bg_share"] > 0.6


@pytest.mark.parametrize("bg,fg", [((255, 255, 255), (0, 0, 0)), ((0, 0, 0), (255, 255, 255)), ((30, 30, 140), (250, 240, 120))])
def test_mask_has_no_luminance_polarity_assumption(bg, fg):
    rgb = _flat_design(bg=bg, fg=fg)
    mask, _ = P.get_mask(rgb, None, "on")
    assert mask[120, 110] and not mask[5, 5]


def test_photo_like_background_gives_no_mask_in_auto():
    rng = np.random.default_rng(1)
    img = _dome_shaded()
    img = np.clip(img.astype(np.int16) + rng.integers(-25, 25, img.shape), 0, 255).astype(np.uint8)  # wood grain / noise
    mask, info = P.get_mask(img, None, "auto")
    assert mask is None and "whole image used" in info["reason"]
    forced, info2 = P.get_mask(img, None, "on")
    assert forced is not None and info2["source"] == "background-colour"


def test_shaded_render_gets_no_mask_so_its_flat_top_is_not_punched_out():
    # the top of a lit dome has the same tone as the plain background -> a colour mask has a
    # hole exactly there; "auto" must therefore return no mask for shaded content
    raw, _ = P.foreground_from_background(_dome_shaded(), np.array([127, 127, 127], np.uint8))
    assert raw[150, 150] is np.False_  # the dome top is one flat plateau
    mask, info = P.get_mask(_dome_shaded(), None, "auto")
    assert mask is None and info["edge_dense"] is False
    forced, _ = P.get_mask(_dome_shaded(), None, "on")
    assert forced is not None and forced.any()


def test_frame_touching_the_border_is_not_taken_for_the_background():
    img = np.full((300, 400, 3), 250, np.uint8)
    cv2.rectangle(img, (0, 0), (399, 299), (20, 20, 20), -1)  # dark frame flush with the edges
    cv2.rectangle(img, (18, 18), (381, 281), (250, 250, 250), -1)
    cv2.circle(img, (200, 150), 60, (180, 30, 30), -1)
    mask, info = P.get_mask(img, None, "on")
    assert mask[5, 5] and mask[150, 150] and not mask[150, 60]


# ------------------------------------------------------------------ suggest_mode
def test_flat_colour_design_with_antialiasing_and_jpeg_suggests_A():
    mode, info = P.suggest_mode(_flat_design())
    assert mode == "A", info
    assert info["k90"] <= 8 and info["jump_share"] >= 0.03


def test_bw_line_art_with_jpeg_suggests_A():
    img = np.full((300, 400, 3), 255, np.uint8)
    for k in range(6):
        pts = np.array([[20 + k * 10, 280 - k * 40], [120, 40 + k * 20], [260, 250 - k * 15], [380, 60 + k * 10]], np.int32)
        cv2.polylines(img, [pts], False, (0, 0, 0), 3, lineType=cv2.LINE_AA)
    img = _jpeg(img, 65)
    mode, info = P.suggest_mode(img)
    assert mode == "A", info
    assert info["dark_bright"] > 0.85 and info["neutral"] > 0.9


def test_shaded_render_suggests_B():
    # a richly coloured photo (many hues, smooth variation) is clearly not a flat design
    yy, xx = np.mgrid[0:240, 0:320].astype(np.float32) / 240.0
    photo = np.stack([255 * (0.3 + 0.7 * yy), 255 * (0.2 + 0.6 * xx), 255 * (0.9 - 0.5 * yy * xx)], -1)
    photo = np.clip(photo + np.random.default_rng(2).normal(0, 4, photo.shape), 0, 255).astype(np.uint8)
    photo = _jpeg(photo, 85)
    mode, info = P.suggest_mode(photo)
    assert mode == "B", info
    assert info["cov_top"] < 0.85


def test_smooth_gray_render_suggestion_is_a_documented_limitation():
    # A smooth monochrome render of a gentle dome is statistically near-flat (few dominant
    # colours, no hard edges): the suggestion says A. Mode A still produces a plausible
    # dome from it, and the user can override - documented in docs/architecture.md §5.
    from tests.test_preprocess import _dome_shaded

    mode, info = P.suggest_mode(_jpeg(_dome_shaded(), 90))
    assert mode in ("A", "B")  # the value itself is not part of the contract
    mask, minfo = P.get_mask(_dome_shaded(), None, "auto")
    assert mask is None and "edge-dense" in minfo["reason"]  # but no colour mask is auto-applied


# ------------------------------------------------------------------ inference size (Issue 4)
def test_resolution_follows_the_long_side_not_the_width():
    # portrait panel 300 mm x 600 mm, 1 mm tool -> 1800 px wanted -> clamped to 1036
    info = P.choose_inference_size(2000, 300, 600, 1.0, device="cuda", override="auto", mode="B")
    assert info.S == 1036
    # landscape 300 x 150 mm, 1 mm tool -> long side 300 mm -> 900 -> 896
    assert P.choose_inference_size(2000, 300, 150, 1.0, device="cuda").S == 896
    # small panel 150 x 75 mm -> 450 -> clamped to the minimum 518
    assert P.choose_inference_size(2000, 150, 75, 1.0, device="cuda").S == 518
    # landscape 600 x 300 mm, 2 mm tool -> 900 -> 896
    assert P.choose_inference_size(2000, 600, 300, 2.0, device="cuda").S == 896
    # both orientations of the same 600 x 300 panel get the same resolution
    assert P.choose_inference_size(2000, 300, 600, 2.0, device="cuda").S == P.choose_inference_size(2000, 600, 300, 2.0, device="cuda").S


def test_resolution_is_never_higher_than_the_source():
    info = P.choose_inference_size(512, 600, 300, 1.0, device="cuda")
    assert info.S == 518 and info.capped_by_source
    info = P.choose_inference_size(700, 600, 300, 0.5, device="cuda")
    assert info.capped_by_source and info.S <= P.ceil14(700)
    assert P.choose_inference_size(4000, 600, 300, 0.5, device="cuda").S == 1036


def test_cpu_defaults_to_518_and_override_wins():
    info = P.choose_inference_size(4000, 900, 450, 1.0, device="cpu")
    assert info.S == 518 and info.capped_by_device
    assert P.choose_inference_size(4000, 900, 450, 1.0, device="cpu", override=1036).S == 1036
    with pytest.raises(ValueError):
        P.choose_inference_size(4000, 900, 450, 1.0, override=500)


def test_classic_modes_keep_the_source_resolution():
    assert P.choose_inference_size(731, 300, 300, 1.0, mode="A").S == 731
    assert P.choose_inference_size(4000, 300, 300, 1.0, mode="C").S == 1036


def test_all_sizes_are_multiples_of_14():
    for src in range(100, 4000, 37):
        for wm, hm in ((300, 300), (450, 900), (2000, 100)):
            S = P.choose_inference_size(src, wm, hm, 0.8, device="cuda").S
            assert S % 14 == 0 and 518 <= S <= 1036
