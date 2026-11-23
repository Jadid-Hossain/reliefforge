"""M10 — the training loop: two stages, AMP, EMA, resume, MLflow, history.csv.

Parameter groups are built from named_parameters() prefixes (review Issue 6):
    encoder = backbone.*   (DINOv2 ViT-S)
    decoder = neck.* + head.*   (DPT reassemble/fusion + depth head)
The assignment is ASSERTED to cover every parameter - no generic 'decoder' guess.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import tracking
from ml.dataset import ReliefEvalDataset, ReliefTrainDataset, collate
from ml.ema import ModelEMA
from ml.losses import normal_loss, pearson_corr, relief_loss


def param_groups(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    enc, dec = [], []
    for name, p in model.named_parameters():
        if name.startswith("backbone."):
            enc.append(p)
        elif name.startswith(("neck.", "head.")):
            dec.append(p)
        else:
            raise RuntimeError(f"unassigned parameter (update param_groups): {name}")
    if not enc or not dec:
        raise RuntimeError(f"empty param group: encoder={len(enc)} decoder={len(dec)}")
    return enc, dec


def freeze(module: torch.nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)


def build_lr_lambda(warmup: int, total: int):
    def fn(step: int):
        if step < warmup:
            return step / max(1, warmup)
        t = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    return fn


class Trainer:
    def __init__(self, model, cfg: dict, out_dir: str | Path, device: str | None = None, run=None):
        self.model = model
        self.cfg = cfg
        self.out = Path(out_dir)
        (self.out / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.run = run
        self.amp = bool(cfg.get("amp", True)) and self.device.startswith("cuda")
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        enc, dec = param_groups(self.model)
        self.enc, self.dec = enc, dec
        stage = cfg.get("stage", "b")
        if stage == "a":
            freeze(self.model.backbone)
            self.opt = torch.optim.AdamW(
                [{"params": dec, "lr": float(cfg["lr"]["decoder"])}],
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
            )
        else:
            self.opt = torch.optim.AdamW(
                [
                    {"params": enc, "lr": float(cfg["lr"]["encoder"])},
                    {"params": dec, "lr": float(cfg["lr"]["decoder"])},
                ],
                weight_decay=float(cfg.get("weight_decay", 1e-4)),
            )
        total = int(cfg.get("steps", 2000))
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, build_lr_lambda(int(cfg.get("warmup_steps", 300)), total))
        self.ema = ModelEMA(self.model, float(cfg.get("ema_decay", 0.999)))
        self.step = 0
        self.history_path = self.out / "history.csv"

    # ------------------------------------------------------------------ training
    def train(self, train_ds, val_ds=None) -> dict:
        cfg = self.cfg
        loader = DataLoader(
            train_ds,
            batch_size=int(cfg.get("batch_size", 8)),
            shuffle=True,
            num_workers=int(cfg.get("num_workers", 2)),
            collate_fn=collate,
            drop_last=True,
            persistent_workers=int(cfg.get("num_workers", 2)) > 0,
            pin_memory=self.device.startswith("cuda"),
        )
        total = int(cfg.get("steps", 2000))
        log_every, save_every = int(cfg.get("log_every", 100)), int(cfg.get("save_every", 500))
        lw = cfg.get("loss", {})
        accum = int(cfg.get("grad_accum", 1))
        it = iter(loader)
        t0 = time.time()
        imgs_per_s = 0.0
        while self.step < total:
            self.model.train()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
                train_ds.set_epoch(getattr(train_ds, "epoch", 0) + 1)
            x = batch["img"].to(self.device, non_blocking=True)
            gt = batch["gt"].to(self.device, non_blocking=True)
            valid = batch["valid"].to(self.device, non_blocking=True)
            fg = batch["fg"].to(self.device, non_blocking=True)
            kappa = batch["kappa"].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                pred = self.model(pixel_values=x).predicted_depth
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            loss, parts = relief_loss(
                pred.float(), gt, valid, kappa, fg,
                w_height=float(lw.get("height", 1.0)), w_grad=float(lw.get("grad", 0.5)),
                w_normal=float(lw.get("normal", 0.1)), fg_weight=float(lw.get("fg_weight", 2.0)),
            )
            (loss / accum).backward()
            if (self.step + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get("grad_clip", 1.0)))
                self.scaler.step(self.opt)
                self.scaler.update()
                self.sched.step()
                self.opt.zero_grad(set_to_none=True)
            self.ema.update(self.model)
            self.step += 1
            dt = time.time() - t0
            imgs_per_s = self.step * int(cfg.get("batch_size", 8)) * accum / max(dt, 1e-9) if dt > 0 else 0.0
            if self.step % log_every == 0:
                lr_e = self.opt.param_groups[0]["lr"]
                lr_d = self.opt.param_groups[-1]["lr"]
                row = {"step": self.step, "stage": self.cfg.get("stage", "b"), "lr_enc": f"{lr_e:.2e}", "lr_dec": f"{lr_d:.2e}", "train_L": f"{float(loss):.4f}", "L_h": f"{parts['L_h']:.4f}", "L_g": f"{parts['L_g']:.4f}", "L_n": f"{parts['L_n']:.4f}", "images_per_s": f"{imgs_per_s:.1f}"}
                self._append_history(row)
                if self.run is not None:
                    tracking.log_metrics_safe(self.run, {"train_loss": float(loss), **parts, "lr_encoder": lr_e, "lr_decoder": lr_d, "images_per_s": imgs_per_s}, self.step)
                print(f"[{self.cfg.get('stage','b')}] step {self.step}/{total} L={float(loss):.4f} Lh={parts['L_h']:.3f} Lg={parts['L_g']:.3f} Ln={parts['L_n']:.3f} ({imgs_per_s:.1f} img/s)", flush=True)
            if self.step % save_every == 0 or self.step == total:
                self.save_checkpoint()
        self.save_checkpoint()
        return {"final_step": self.step, "images_per_s": imgs_per_s}

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, val_ds, max_batches: int | None = None, use_ema: bool = True) -> dict:
        model = (self.ema.module if use_ema else self.model).to(self.device).eval()
        loader = DataLoader(val_ds, batch_size=int(self.cfg.get("batch_size", 8)), shuffle=False, num_workers=0, collate_fn=collate)
        maes, norms, corrs = [], [], []
        n = 0
        for batch in loader:
            x = batch["img"].to(self.device)
            gt = batch["gt"].to(self.device)
            valid = batch["valid"].to(self.device)
            kappa = batch["kappa"].to(self.device)
            fg = batch["fg"].to(self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                pred = model(pixel_values=x).predicted_depth
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=gt.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            pred = pred.float()
            from ml.losses import normalize_batch

            pn, gn = normalize_batch(pred, valid), normalize_batch(gt, valid)
            mae = ((pn - gn).abs() * valid.float()).sum(dim=(1, 2)) / valid.float().sum(dim=(1, 2)).clamp_min(1)
            maes += mae.tolist()
            ln = normal_loss(pred, gt, valid, kappa, fg, 1.0)
            norms += ln.tolist()
            corrs += pearson_corr(pred, gt, valid).tolist()
            n += 1
            if max_batches and n >= max_batches:
                break
        out = {"val_MAE_norm": float(np.mean(maes)), "val_normal_err_deg": float(np.degrees(np.arccos(np.clip(1 - np.mean(norms) / 1, -1, 1)))), "val_pearson": float(np.mean(corrs)), "val_polarity_positive_frac": float(np.mean([c > 0 for c in corrs]))}
        if self.run is not None:
            tracking.log_metrics_safe(self.run, out, self.step)
        return out

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, name: str | None = None) -> Path:
        name = name or f"ckpt_{self.cfg.get('stage', 'b')}_{self.step:06d}.pt"
        path = self.out / "checkpoints" / name
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scaler": self.scaler.state_dict(),
                "sched": self.sched.state_dict(),
                "step": self.step,
                "stage": self.cfg.get("stage", "b"),
                "cfg": self.cfg,
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ck["model"])
        self.ema.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["optimizer"])
        self.scaler.load_state_dict(ck["scaler"])
        self.sched.load_state_dict(ck["sched"])
        self.step = int(ck["step"])

    def export_best(self, path: str | Path) -> Path:
        """EMA weights in HF format (config + safetensors) for reliefkit.ReliefModel."""
        from transformers import DepthAnythingForDepthEstimation

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model = DepthAnythingForDepthEstimation.from_pretrained(self.cfg.get("model", "depth-anything/Depth-Anything-V2-Small-hf"))
        model.load_state_dict(self.ema.state_dict())
        model.save_pretrained(path, safe_serialization=True)
        (path / "reliefforge_info.json").write_text(json.dumps({"stage": self.cfg.get("stage", "b"), "step": self.step, "cfg": {k: v for k, v in self.cfg.items() if k != "textures"}}, indent=2, default=str), encoding="utf-8")
        return path

    def _append_history(self, row: dict) -> None:
        new = not self.history_path.exists()
        with open(self.history_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)
        if self.run is not None:
            tracking.log_artifact_safe(self.run, self.history_path)
