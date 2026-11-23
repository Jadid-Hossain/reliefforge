"""M11 — evaluation metrics, computed AFTER the production normalization (§M4)."""
from __future__ import annotations

import cv2
import numpy as np


def normalize_production(h_raw: np.ndarray, ref_mask: np.ndarray | None) -> np.ndarray:
    """The same normalization the user's file goes through (percentiles 1 / 99.5)."""
    h = np.asarray(h_raw, np.float32)
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    ref = ref_mask if ref_mask is not None else np.ones(h.shape, bool)
    lo, hi = np.percentile(h[ref], [1.0, 99.5])
    if hi - lo < 1e-6:
        return np.zeros_like(h)
    return np.clip((h - lo) / (hi - lo), 0, 1)


def mae_rmse(pred: np.ndarray, gt: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    d = pred[ref] - gt[ref]
    return float(np.abs(d).mean()), float(np.sqrt((d**2).mean()))


def ssim(a: np.ndarray, b: np.ndarray, ref: np.ndarray, win: int = 7) -> float:
    """SSIM over the valid region: crop to the reference bounding box, then compare."""
    a8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    b8 = (np.clip(b, 0, 1) * 255).astype(np.uint8)
    ys, xs = np.where(ref)
    a8, b8 = a8[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], b8[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    from skimage.metrics import structural_similarity

    w = win if min(a8.shape) >= win else (min(a8.shape) - 1 if min(a8.shape) % 2 == 1 else min(a8.shape) - 2)
    return float(structural_similarity(a8, b8, win_size=w, data_range=255))


def normal_error_deg(pred: np.ndarray, gt: np.ndarray, ref: np.ndarray, kappa: float) -> float:
    def normals(h):
        gy, gx = np.gradient(h * kappa)
        n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)
        return n

    dot = np.clip((normals(pred) * normals(gt)).sum(-1), -1, 1)
    return float(np.degrees(np.arccos(dot[ref]).mean()))


def detail_retention(pred: np.ndarray, gt: np.ndarray, ref: np.ndarray, sigma: float = 2.0) -> float:
    """R_detail = |h - G(h)|_1 / |gt - G(gt)|_1 ~ 1 keeps detail, << 1 is blur (PRD_v2 §6.5)."""

    def hp(h):
        return np.abs(h - cv2.GaussianBlur(h, (0, 0), sigma))

    denom = hp(gt)[ref].sum()
    return float(hp(pred)[ref].sum() / max(denom, 1e-9))


def pearson(pred: np.ndarray, gt: np.ndarray, ref: np.ndarray) -> float:
    a, b = pred[ref].astype(np.float64), gt[ref].astype(np.float64)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def all_metrics(pred_raw: np.ndarray, gt: np.ndarray, ref: np.ndarray, kappa: float) -> dict:
    pred = normalize_production(pred_raw, ref)
    mae, rmse = mae_rmse(pred, gt, ref)
    return {
        "MAE_norm": mae,
        "RMSE_norm": rmse,
        "SSIM": ssim(pred, gt, ref),
        "normal_err_deg": normal_error_deg(pred, gt, ref, kappa),
        "R_detail": detail_retention(pred, gt, ref),
        "pearson": pearson(pred, gt, ref),
    }
