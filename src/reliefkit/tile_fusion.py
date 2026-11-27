from __future__ import annotations
import numpy as np
import cv2


def _gauss(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def _resize(a: np.ndarray, w: int, h: int, up: bool) -> np.ndarray:
    interp = cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA
    return cv2.resize(a, (int(w), int(h)), interpolation=interp)


def _window(t: int, feather: float = 0.5) -> np.ndarray:
    f = max(2, int(t * feather * 0.5))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, f, dtype=np.float32))
    w1 = np.ones(t, np.float32)
    w1[:f] = ramp
    w1[-f:] = ramp[::-1]
    w1 = np.maximum(w1, 1e-3)
    return np.outer(w1, w1)


def _positions(total: int, tile: int, step: int) -> list[int]:
    if total <= tile:
        return [0]
    pos = list(range(0, total - tile + 1, step))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def _fit_affine(p: np.ndarray, g: np.ndarray, sigma: float) -> tuple[float, float, float]:
    lp, lg = _gauss(p, sigma), _gauss(g, sigma)
    mp, mg = lp.mean(), lg.mean()
    vp = float(((lp - mp) ** 2).mean())
    vg = float(((lg - mg) ** 2).mean())
    if vp < 1e-10 or vg < 1e-10:
        return 1.0, float(mg - mp), 0.0
    cov = float(((lp - mp) * (lg - mg)).mean())
    corr = cov / np.sqrt(vp * vg)
    s = cov / vp
    if corr >= -0.3:
        s = max(s, 0.05)
    t = mg - s * mp
    return float(s), float(t), float(corr)


def tile_fusion(
    img_rgb: np.ndarray,
    predict_fn,
    global_height: np.ndarray,
    long_side: int = 1536,
    tile: int = 518,
    overlap: float = 0.45,
    flip_tta: bool = False,
    match_sigma_frac: float = 0.06,
) -> dict:
    h0, w0 = img_rgb.shape[:2]
    scale = long_side / max(h0, w0)
    W, H = int(round(w0 * scale)), int(round(h0 * scale))
    W, H = max(W, tile), max(H, tile)
    img_f = _resize(img_rgb, W, H, up=scale >= 1.0)
    g_f = _resize(global_height.astype(np.float32), W, H, up=True)

    step = max(1, int(tile * (1.0 - overlap)))
    xs, ys = _positions(W, tile, step), _positions(H, tile, step)
    win = _window(tile)
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    corrs = []
    sig = max(2.0, tile * match_sigma_frac)

    for y in ys:
        for x in xs:
            crop = img_f[y:y + tile, x:x + tile]
            p = predict_fn(crop).astype(np.float32)
            if flip_tta:
                pf = predict_fn(np.ascontiguousarray(crop[:, ::-1])).astype(np.float32)[:, ::-1]
                p = 0.5 * (p + pf)
            if p.shape != (tile, tile):
                p = _resize(p, tile, tile, up=True)
            g = g_f[y:y + tile, x:x + tile]
            s, t, c = _fit_affine(p, g, sig)
            corrs.append(c)
            acc[y:y + tile, x:x + tile] += win * (s * p + t)
            wsum[y:y + tile, x:x + tile] += win

    height = acc / np.maximum(wsum, 1e-6)
    return {
        "height": height.astype(np.float32),
        "shape": (H, W),
        "n_tiles": len(xs) * len(ys),
        "corr_mean": float(np.mean(corrs)) if corrs else 0.0,
        "corr_min": float(np.min(corrs)) if corrs else 0.0,
    }


def flatten_trend(h: np.ndarray, sigma_frac: float = 0.15, alpha: float = 0.8,
                  mask: np.ndarray | None = None) -> np.ndarray:
    if alpha <= 0:
        return h
    ls = max(h.shape)
    sigma = max(2.0, sigma_frac * ls)
    f = max(1, int(ls // 256))
    hs = cv2.resize(h, (h.shape[1] // f, h.shape[0] // f), interpolation=cv2.INTER_AREA)
    if mask is not None:
        ms = cv2.resize(mask.astype(np.float32), (hs.shape[1], hs.shape[0]), interpolation=cv2.INTER_AREA)
        num = _gauss(hs * ms, sigma / f)
        den = _gauss(ms, sigma / f)
        trend = num / np.maximum(den, 1e-3)
    else:
        trend = _gauss(hs, sigma / f)
    trend = cv2.resize(trend, (h.shape[1], h.shape[0]), interpolation=cv2.INTER_CUBIC)
    return (h - alpha * (trend - trend.mean())).astype(np.float32)


def luminance_detail(img_rgb: np.ndarray, h: np.ndarray, lam: float = 0.05,
                     sigma_hp: float = 2.0, sigma_illum_frac: float = 0.03) -> np.ndarray:
    if lam <= 0:
        return h
    H, W = h.shape
    im = img_rgb if img_rgb.shape[:2] == (H, W) else _resize(img_rgb, W, H, up=True)
    lum = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    illum = _gauss(lum, sigma_illum_frac * max(H, W))
    ln = lum / np.maximum(illum, 1e-3)
    hp = ln - _gauss(ln, sigma_hp)
    hp = np.clip(hp / (3.0 * hp.std() + 1e-6), -1.0, 1.0)
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3)
    gm = np.sqrt(gx * gx + gy * gy)
    gate = np.clip(gm / (np.percentile(gm, 95) + 1e-6), 0.0, 1.0)
    gate = _gauss(cv2.dilate(gate, np.ones((9, 9), np.uint8)), 2.5)
    rng = float(np.percentile(h, 99) - np.percentile(h, 1)) + 1e-6
    return (h + lam * rng * gate * hp).astype(np.float32)


def boost_relief(
    img_rgb: np.ndarray,
    predict_fn,
    global_predict_fn=None,
    long_side: int = 1536,
    tile: int = 518,
    overlap: float = 0.45,
    flip_tta: bool = False,
    flatten_alpha: float = 0.95,
    flatten_sigma_frac: float = 0.08,
    detail_lam: float = 0.22,
    mask: np.ndarray | None = None,
) -> dict:
    gfn = global_predict_fn or predict_fn
    g = gfn(img_rgb).astype(np.float32)
    if g.shape != img_rgb.shape[:2]:
        g = _resize(g, img_rgb.shape[1], img_rgb.shape[0], up=True)
    out = tile_fusion(
        img_rgb, predict_fn, g, long_side=long_side, tile=tile,
        overlap=overlap, flip_tta=flip_tta,
    )
    h = out["height"]
    m = None
    if mask is not None:
        m = _resize(mask.astype(np.float32), h.shape[1], h.shape[0], up=False)
    h = flatten_trend(h, sigma_frac=flatten_sigma_frac, alpha=flatten_alpha, mask=m)
    lo, hi = np.percentile(h, [2.0, 98.0])
    h = np.clip(h, lo, hi).astype(np.float32)
    h = luminance_detail(img_rgb, h, lam=detail_lam)
    out["height"] = h
    out["global_height"] = g
    return out
