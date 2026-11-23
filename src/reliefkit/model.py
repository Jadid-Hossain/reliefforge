"""M3 Mode B — the fine-tuned / zero-shot Depth Anything V2 Small wrapper.

Contract: `predict(rgb) -> h_raw` on the ORIGINAL grid of the input image (the
letterboxing is handled here and undone before returning). h_raw is relative height
("larger = more raised") — the sign is fixed once (Day-1 polarity check) and stored
with the checkpoint.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .preprocess import letterbox, unletterbox

DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SIZES = (518, 756, 1036)


def pick_device() -> str:
    import torch

    if os.environ.get("RELIEFFORGE_DEVICE"):
        return os.environ["RELIEFFORGE_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


class ReliefModel:
    """Depth Anything V2 Small as a relief-height predictor."""

    def __init__(
        self,
        model_id_or_path: str | Path = DEFAULT_MODEL_ID,
        *,
        device: str | None = None,
        sign: float = 1.0,
        fp16: bool | None = None,
    ):
        import torch
        from transformers import DepthAnythingForDepthEstimation

        self.torch = torch
        self.device = device or pick_device()
        self.sign = float(sign)
        self.model_id = str(model_id_or_path)
        self.model = DepthAnythingForDepthEstimation.from_pretrained(self.model_id)
        self.model.to(self.device).eval()
        self.fp16 = bool((self.device.startswith("cuda")) if fp16 is None else fp16)
        if self.fp16:
            self.model.half()

    # ------------------------------------------------------------------ internals
    def _forward(self, rgb: np.ndarray, S: int):
        """Letterboxed forward pass -> depth map on the S x S grid (float32 numpy)."""
        import torch

        t, meta = self._to_tensor(rgb, S)
        with torch.no_grad():
            out = self.model(pixel_values=t).predicted_depth
        if out.dtype != torch.float32:
            out = out.float()
        if tuple(out.shape[-2:]) != (S, S):  # defensive interpolate (§7.1 shape guard)
            out = torch.nn.functional.interpolate(out[None, None], size=(S, S), mode="bilinear", align_corners=False)[0, 0]
        depth = out.squeeze(0).cpu().numpy().astype(np.float32)
        return depth * self.sign, meta

    def _to_tensor(self, rgb: np.ndarray, S: int):
        import torch

        img, meta = letterbox(rgb, S)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        if self.fp16:
            t = t.half()
        return t, meta

    # ------------------------------------------------------------------ public API
    def predict(self, rgb: np.ndarray, S: int = 518, tta: bool = False) -> np.ndarray:
        """Relative height on the original image grid (float32, un-normalized)."""
        h, meta = self._forward(rgb, S)
        if tta:
            h_flip, _ = self._forward(rgb[:, ::-1], S)
            h = 0.5 * (h + h_flip[:, ::-1])
        return unletterbox(h, meta)

    def polarity_sign(self, rgb_samples: list[np.ndarray], S: int = 518) -> float:
        """Day-1 check: Depth Anything says 'larger = nearer'. For a raised ornament on
        a plain background, the foreground must be the maximum. Returns +1 or -1 for the
        given samples (majority vote over images that have a clear foreground)."""
        from .preprocess import get_mask

        votes = []
        for rgb in rgb_samples:
            mask, info = get_mask(rgb)
            if mask is None or info.get("source") != "background-colour":
                continue
            h = self.predict(rgb, S)
            if float(np.median(h[mask])) > float(np.median(h[~mask])):
                votes.append(1.0)
            else:
                votes.append(-1.0)
        if not votes:
            return 1.0
        return float(np.sign(sum(votes)))
