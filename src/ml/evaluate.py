"""M11 — run the comparison: zero-shot vs fine-tuned (vs + postprocessing).

Uses deterministic eval views (fixed seed per scene/style/S) and the production
normalization, and reports per family / style / S breakdowns (PRD_v2 §8).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml import tracking
from ml.dataset import ReliefEvalDataset, collate
from ml.metrics import all_metrics


def load_model(model_id_or_path: str, device: str):
    from transformers import DepthAnythingForDepthEstimation

    m = DepthAnythingForDepthEstimation.from_pretrained(model_id_or_path)
    return m.to(device).eval()


@torch.no_grad()
def evaluate_model(
    model,
    root: str | Path,
    split: str = "test",
    sizes=(518,),
    device: str = "cpu",
    batch_size: int = 4,
    max_items: int | None = None,
    label: str = "model",
) -> list[dict]:
    rows = []
    for S in sizes:
        ds = ReliefEvalDataset(root, split=split, size=S)
        if max_items:
            ds.items = ds.items[:max_items]
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
        for batch in loader:
            x = batch["img"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                pred = model(pixel_values=x).predicted_depth.float().cpu().numpy()
            gt = batch["gt"].numpy()
            valid = batch["valid"].numpy()
            kappa = batch["kappa"].numpy()
            for b in range(pred.shape[0]):
                ref = valid[b]
                if ref.sum() < 100:
                    continue
                m = all_metrics(pred[b], gt[b], ref, float(kappa[b]))
                meta = batch["meta"][b]
                rows.append({"model": label, "S": S, "family": meta["family"], "style": batch["style"][b], "scene": int(meta["id"]), **m})
    return rows


def write_report(rows: list[dict], out_dir: str | Path, run=None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # summary table: model x S -> mean metrics
    summary: dict = {}
    for r in rows:
        key = (r["model"], r["S"])
        summary.setdefault(key, []).append({k: r[k] for k in ("MAE_norm", "RMSE_norm", "SSIM", "normal_err_deg", "R_detail", "pearson")})
    table = []
    for (model, S), ms in sorted(summary.items()):
        agg = {k: round(float(np.mean([m[k] for m in ms])), 4) for k in ms[0]}
        table.append({"model": model, "S": S, "n": len(ms), **agg})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    # per family / style breakdown for the primary model
    breakdown = {}
    for r in rows:
        breakdown.setdefault((r["model"], r["family"]), {"MAE_norm": [], "normal_err_deg": []})
        breakdown[(r["model"], r["family"])]["MAE_norm"].append(r["MAE_norm"])
        breakdown[(r["model"], r["family"])]["normal_err_deg"].append(r["normal_err_deg"])
    fam_rows = [{"model": m, "family": f, "MAE_norm": round(float(np.mean(v["MAE_norm"])), 4), "normal_err_deg": round(float(np.mean(v["normal_err_deg"])), 4)} for (m, f), v in sorted(breakdown.items())]
    with open(out / "by_family.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "family", "MAE_norm", "normal_err_deg"])
        w.writeheader()
        w.writerows(fam_rows)
    if run is not None:
        for t in table:
            tracking.log_metrics_safe(run, {f"{t['model']}/S{t['S']}/{k}": v for k, v in t.items() if isinstance(v, (int, float))}, 0)
        tracking.log_artifact_safe(run, out / "summary.csv")
        tracking.log_artifact_safe(run, out / "results.csv")
        tracking.log_artifact_safe(run, out / "by_family.csv")
    return {"table": table, "n_rows": len(rows)}


@torch.no_grad()
def qualitative_figure(
    models: dict[str, torch.nn.Module],
    root: str | Path,
    out_path: str | Path,
    split: str = "test",
    size: int = 518,
    n: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> Path:
    """input | GT | one column per model, for the report (PRD_v2 §8)."""
    import cv2

    from reliefkit.preview import shade_height

    ds = ReliefEvalDataset(root, split=split, size=size)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    tiles = []
    for i in pick:
        item = ds[int(i)]
        x = item["img"][None].to(device)
        gt = item["gt"].numpy()
        valid = item["valid"].numpy()
        img = ((item["img"].numpy().transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255).astype(np.uint8)
        parts = [img, shade_height(gt)]
        for name, model in models.items():
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                pred = model(pixel_values=x).predicted_depth.float().cpu().numpy()[0]
            from ml.metrics import normalize_production

            parts.append(shade_height(normalize_production(pred, valid)))
        tiles.append(np.concatenate(parts, axis=1))
    sheet = np.concatenate(tiles, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet[..., ::-1])
    return out_path
