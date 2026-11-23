"""M9/M10 — torch datasets: training views (full + crop) and deterministic eval views.

View sampling (PRD_v2 §6.2, review fix ১):
* full view  (40%): the whole scene area-downscaled to long side 518 + letterbox;
                   the padding is masked out of the loss
* crop view  (60%): scale s ~ U(s_min, 1) with s_min = max(0.55, 518 / min(H, W))
                   -> a 518x518 window is ALWAYS possible on every native canvas
Both are rendered on the fly (new light/texture/noise every epoch = free augmentation).
kappa returned for the normal loss = depth_mm * px_per_mm_native * s (no jitter:
the GT heights are fixed; the jitter only varies the shading).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.render import STYLES, STYLE_WEIGHTS, render_view  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_scenes(root: str | Path, split: str) -> list[dict]:
    rows = [json.loads(l) for l in open(Path(root) / "scenes.jsonl", encoding="utf-8")]
    return [r for r in rows if r["split"] == split]


def load_geometry(root: str | Path, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    gid = f"{meta['id']:06d}"
    base = Path(root) / "geometry" / meta["split"]
    h = cv2.imread(str(base / f"{gid}_h.png"), cv2.IMREAD_UNCHANGED)
    inst = cv2.imread(str(base / f"{gid}_inst.png"), cv2.IMREAD_UNCHANGED)
    if h is None or inst is None:
        raise FileNotFoundError(f"missing geometry for scene {gid}")
    if h.ndim == 3:
        h = h[..., 0]
    if inst.ndim == 3:
        inst = inst[..., 0]
    return h.astype(np.float32) / 65535.0, inst.astype(np.uint16)


def pick_style(rng: np.random.Generator) -> str:
    r = rng.random()
    acc = 0.0
    for s, w in STYLE_WEIGHTS.items():
        acc += w
        if r <= acc:
            return s
    return "clay"


def crop_scale_min(H: int, W: int, size: int = 518, base_min: float = 0.55) -> float:
    """Review fix ১: the crop window must fit on EVERY canvas."""
    return max(base_min, size / min(H, W))


def _resize_h(h: np.ndarray, s: float) -> np.ndarray:
    return cv2.resize(h, (max(1, int(round(h.shape[1] * s))), max(1, int(round(h.shape[0] * s)))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def _augment(img: np.ndarray, gt: np.ndarray, valid: np.ndarray, rng: np.random.Generator):
    k = int(rng.integers(0, 4))
    if k:
        img = np.rot90(img, k)
        gt = np.rot90(gt, k)
        valid = np.rot90(valid, k)
    if rng.random() < 0.5:
        img = img[:, ::-1]
        gt = gt[:, ::-1]
        valid = valid[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(gt), np.ascontiguousarray(valid)


class ReliefTrainDataset(Dataset):
    def __init__(self, root: str | Path, size: int = 518, full_frac: float = 0.4, base_crop_min: float = 0.55, augment: bool = True, textures: list[np.ndarray] | None = None, seed: int = 0):
        self.root = Path(root)
        self.metas = load_scenes(root, "train")
        self.size = size
        self.full_frac = full_frac
        self.base_crop_min = base_crop_min
        self.augment = augment
        self.textures = textures or []
        self.base_seed = seed

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, i: int) -> dict:
        S = self.size
        meta = self.metas[i]
        h, inst = load_geometry(self.root, meta)
        H, W = h.shape
        rng = np.random.default_rng(self.base_seed * 1_000_003 + i * 7919 + self.epoch)
        full = rng.random() < self.full_frac
        if full:
            s = S / max(H, W)
            hv = _resize_h(h, s)
            iv = cv2.resize(inst, (hv.shape[1], hv.shape[0]), interpolation=cv2.INTER_NEAREST)
            view_scale = s
        else:
            s_min = crop_scale_min(H, W, S, self.base_crop_min)
            s = float(rng.uniform(s_min, 1.0))
            hw, ww = int(round(H * s)), int(round(W * s))
            hc, wc = max(S, 1), max(S, 1)
            y0 = int(rng.integers(0, max(1, hw - S + 1))) if hw >= S else 0
            x0 = int(rng.integers(0, max(1, ww - S + 1))) if ww >= S else 0
            hs = _resize_h(h, s)
            is_ = cv2.resize(inst, (hs.shape[1], hs.shape[0]), interpolation=cv2.INTER_NEAREST)
            hv = np.zeros((S, S), np.float32)
            iv = np.zeros((S, S), np.uint16)
            vh, vw = min(S, hs.shape[0]), min(S, hs.shape[1])
            hv[:vh, :vw] = hs[y0 : y0 + vh, x0 : x0 + vw]
            iv[:vh, :vw] = is_[y0 : y0 + vh, x0 : x0 + vw]
            view_scale = s
            valid = np.zeros((S, S), bool)
            valid[:vh, :vw] = True
            meta = dict(meta)
            img = None  # render below from the crop
            img = self._render(hv, iv, meta, view_scale, rng)
            return self._package(img, hv, valid, meta, view_scale)
        # full view: letterbox to S x S
        img = self._render(hv, iv, meta, view_scale, rng)
        valid = np.zeros((S, S), bool)
        vh, vw = hv.shape
        y0, x0 = (S - vh) // 2, (S - vw) // 2
        padded_img = np.zeros((S, S, 3), np.uint8)
        padded_img[y0 : y0 + vh, x0 : x0 + vw] = img
        padded_gt = np.zeros((S, S), np.float32)
        padded_gt[y0 : y0 + vh, x0 : x0 + vw] = hv
        valid[y0 : y0 + vh, x0 : x0 + vw] = True
        return self._package(padded_img, padded_gt, valid, meta, view_scale)

    epoch = 0

    def set_epoch(self, e: int):
        self.epoch = int(e)

    def _render(self, hv, iv, meta, view_scale, rng):
        style = pick_style(rng)
        texture = self.textures[int(rng.integers(0, len(self.textures)))] if self.textures and style == "wood" else None
        return render_view(hv, iv, meta, view_scale, style, rng, texture=texture)

    def _package(self, img, gt, valid, meta, view_scale):
        if self.augment:
            img, gt, valid = _augment(img, gt, valid, np.random.default_rng(meta["seed"] + self.epoch * 31))
        x = (img.astype(np.float32) / 255.0 - MEAN) / STD
        kappa = float(meta["relief_depth_mm_scene"]) * float(meta["px_per_mm_native"]) * float(view_scale)
        fg = (gt > 0.02).astype(np.float32)
        return {
            "img": torch.from_numpy(x.transpose(2, 0, 1)).float(),
            "gt": torch.from_numpy(gt).float(),
            "valid": torch.from_numpy(valid),
            "fg": torch.from_numpy(fg),
            "kappa": torch.tensor(kappa, dtype=torch.float32),
            "meta": meta,
        }


class ReliefEvalDataset(Dataset):
    """Deterministic evaluation views: fixed style per scene, rendered at the requested S."""

    STYLES_EVAL = ("wood", "clay", "flat")

    def __init__(self, root: str | Path, split: str = "test", size: int = 518, textures: list[np.ndarray] | None = None):
        self.root = Path(root)
        self.size = size
        self.textures = textures or []
        self.items = []
        for meta in load_scenes(root, split):
            for si, style in enumerate(self.STYLES_EVAL):
                self.items.append((meta, style, si))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        meta, style, si = self.items[i]
        S = self.size
        h, inst = load_geometry(self.root, meta)
        H, W = h.shape
        s = S / max(H, W)
        hv = _resize_h(h, s)
        iv = cv2.resize(inst, (hv.shape[1], hv.shape[0]), interpolation=cv2.INTER_NEAREST)
        rng = np.random.default_rng(90_000 + int(meta["id"]) * 10 + si * 3 + S)
        texture = self.textures[int(meta["id"]) % len(self.textures)] if self.textures and style == "wood" else None
        img = render_view(hv, iv, meta, s, style, rng, texture=texture)
        vh, vw = hv.shape
        y0, x0 = (S - vh) // 2, (S - vw) // 2
        padded_img = np.zeros((S, S, 3), np.uint8)
        padded_img[y0 : y0 + vh, x0 : x0 + vw] = img
        padded_gt = np.zeros((S, S), np.float32)
        padded_gt[y0 : y0 + vh, x0 : x0 + vw] = hv
        valid = np.zeros((S, S), bool)
        valid[y0 : y0 + vh, x0 : x0 + vw] = True
        x = (padded_img.astype(np.float32) / 255.0 - MEAN) / STD
        kappa = float(meta["relief_depth_mm_scene"]) * float(meta["px_per_mm_native"]) * float(s)
        return {
            "img": torch.from_numpy(x.transpose(2, 0, 1)).float(),
            "gt": torch.from_numpy(padded_gt).float(),
            "valid": torch.from_numpy(valid),
            "fg": torch.from_numpy((padded_gt > 0.02).astype(np.float32)),
            "kappa": torch.tensor(kappa, dtype=torch.float32),
            "meta": meta,
            "style": style,
        }


def collate(batch: list[dict]) -> dict:
    out = {k: [] for k in ("img", "gt", "valid", "fg", "kappa")}
    metas, styles = [], []
    for b in batch:
        for k in out:
            out[k].append(b[k])
        metas.append(b["meta"])
        styles.append(b.get("style", ""))
    return {
        "img": torch.stack(out["img"]),
        "gt": torch.stack(out["gt"]),
        "valid": torch.stack(out["valid"]),
        "fg": torch.stack(out["fg"]),
        "kappa": torch.stack(out["kappa"]),
        "meta": metas,
        "style": styles,
    }
