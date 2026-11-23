"""M10 — losses (PRD_v2 §7.2). Everything runs in float32, OUTSIDE autocast.

1. MiDaS-style normalization: median / mean-absolute-deviation -> the scale is positive
   by construction, so a flipped polarity cannot hide behind a negative scale.
2. Weighted L1 on the normalized maps (foreground weighted x2).
3. Multi-scale gradient matching (keeps edges sharp).
4. Normal loss with a PHYSICAL per-sample kappa = depth_mm * px_per_mm * view_scale.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def masked_median_mad(x: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample median and mean absolute deviation over the valid mask (float32)."""
    x = x.float()
    m = m.float()
    n = m.sum(dim=(1, 2)).clamp_min(1.0)[:, None, None]
    w = m / n
    med = (x * w).sum(dim=(1, 2))
    mad = ((x - med[:, None, None]).abs() * w).sum(dim=(1, 2))
    return med, mad


def normalize_batch(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    med, mad = masked_median_mad(x, m)
    return (x - med[:, None, None]) / mad[:, None, None].clamp_min(EPS)


def height_loss(p: torch.Tensor, g: torch.Tensor, valid: torch.Tensor, fg: torch.Tensor | None, fg_weight: float = 2.0) -> torch.Tensor:
    """Weighted L1 between the normalized prediction and target."""
    w = valid.float()
    if fg is not None and fg_weight != 1.0:
        w = w * (1.0 + (fg_weight - 1.0) * fg.float())
    pn, gn = normalize_batch(p, valid), normalize_batch(g, valid)
    return ((pn - gn).abs() * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(EPS)


def gradient_loss(p: torch.Tensor, g: torch.Tensor, valid: torch.Tensor, levels: int = 4) -> torch.Tensor:
    """Multi-scale |d residual | on the normalized maps (area downsampling by 2^k)."""
    pn, gn = normalize_batch(p, valid), normalize_batch(g, valid)
    r_k = (pn - gn).unsqueeze(1)  # B x 1 x H x W
    m_k = valid.float().unsqueeze(1)
    total = p.new_zeros(())
    for _ in range(levels):
        dr_x = r_k[..., 1:] - r_k[..., :-1]
        dr_y = r_k[:, :, 1:, :] - r_k[:, :, :-1, :]
        wx = (m_k[..., 1:] * m_k[..., :-1]).clamp_min(1e-20)
        wy = (m_k[:, :, 1:, :] * m_k[:, :, :-1, :]).clamp_min(1e-20)
        lx = (dr_x.abs() * wx).sum(dim=(1, 2, 3)) / wx.sum(dim=(1, 2, 3)).clamp_min(EPS)
        ly = (dr_y.abs() * wy).sum(dim=(1, 2, 3)) / wy.sum(dim=(1, 2, 3)).clamp_min(EPS)
        total = total + (lx + ly).mean()
        r_k = F.avg_pool2d(r_k * m_k, 2)  # mask BEFORE pooling: padding garbage must not leak into coarser levels
        m_k = F.avg_pool2d(m_k, 2)
    return total / levels


def _normals(h: torch.Tensor, kappa: torch.Tensor, fill: torch.Tensor | None = None, valid: torch.Tensor | None = None) -> torch.Tensor:
    """n = normalize(-kappa * dh/dx, -kappa * dh/dy, 1), kappa per sample.

    Invalid pixels are filled with the per-sample median first, so a gradient at the
    last valid row never sees padding garbage.
    """
    h = h.float()
    if valid is not None and fill is not None:
        h = h * valid.float() + fill[:, None, None] * (1.0 - valid.float())
    k = kappa.float()[:, None, None]
    gy, gx = torch.gradient(h, dim=(1, 2))
    n = torch.stack([-k * gx, -k * gy, torch.ones_like(gx)], dim=1)
    return F.normalize(n, dim=1, eps=EPS)


def normal_loss(p: torch.Tensor, g: torch.Tensor, valid: torch.Tensor, kappa: torch.Tensor, fg: torch.Tensor | None = None, fg_weight: float = 2.0) -> torch.Tensor:
    """1 - n(p̂)·n(g) on de-normalized heights (physical units)."""
    med_p, mad_p = masked_median_mad(p, valid)
    med_g, mad_g = masked_median_mad(g, valid)
    p_hat = (p.float() - med_p[:, None, None]) / mad_p[:, None, None].clamp_min(EPS) * mad_g[:, None, None] + med_g[:, None, None]
    np_ = _normals(p_hat, kappa, fill=med_g, valid=valid)
    ng = _normals(g.float(), kappa, fill=med_g, valid=valid)
    w = valid.float()
    if fg is not None and fg_weight != 1.0:
        w = w * (1.0 + (fg_weight - 1.0) * fg.float())
    dot = (np_ * ng).sum(dim=1)
    return ((1.0 - dot) * w).sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp_min(EPS)


def relief_loss(
    p: torch.Tensor,
    g: torch.Tensor,
    valid: torch.Tensor,
    kappa: torch.Tensor,
    fg: torch.Tensor | None = None,
    w_height: float = 1.0,
    w_grad: float = 0.5,
    w_normal: float = 0.1,
    fg_weight: float = 2.0,
) -> tuple[torch.Tensor, dict]:
    """Total loss + components for logging. All inputs B x H x W (kappa: B)."""
    l_h = height_loss(p, g, valid, fg, fg_weight)
    l_g = gradient_loss(p, g, valid)
    l_n = normal_loss(p, g, valid, kappa, fg, fg_weight)
    total = w_height * l_h.mean() + w_grad * l_g + w_normal * l_n.mean()
    return total, {"L_h": float(l_h.mean()), "L_g": float(l_g), "L_n": float(l_n.mean())}


def pearson_corr(p: torch.Tensor, g: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-sample Pearson correlation over the valid mask (polarity gate)."""
    pn = normalize_batch(p, valid)
    gn = normalize_batch(g, valid)
    w = valid.float()
    n = w.sum(dim=(1, 2)).clamp_min(1.0)
    cov = ((pn * gn * w).sum(dim=(1, 2)) - (pn * w).sum(dim=(1, 2)) * (gn * w).sum(dim=(1, 2)) / n) / n
    return cov
