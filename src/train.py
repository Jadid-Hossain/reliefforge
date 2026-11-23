"""Training entry point: python src/train.py --config configs/stage_a.yaml [--smoke]

MLflow is always on (course requirement): params, metrics, artifacts land in ./mlruns.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ml import tracking


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(paths: list[str]) -> dict:
    cfg: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    master = ROOT / "configs" / "config.yaml"
    if str(master) not in [str(Path(p).resolve()) for p in paths]:
        with open(master, encoding="utf-8") as f:
            cfg = deep_update(yaml.safe_load(f) or {}, cfg)
    return cfg


def load_textures(textures_dir: str | None):
    import cv2
    import numpy as np

    if not textures_dir:
        return []
    paths = sorted(Path(textures_dir).glob("*.png")) + sorted(Path(textures_dir).glob("*.jpg"))
    out = []
    for p in paths[:80]:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and min(img.shape[:2]) >= 256:
            out.append(img[..., ::-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/stage_a.yaml"])
    ap.add_argument("--smoke", action="store_true", help="tiny step budget for a wiring test")
    ap.add_argument("--out", default=None, help="output dir (default artifacts/train_<stage>)")
    ap.add_argument("--data", default=None, help="override data.root")
    ap.add_argument("--stage-a-dir", default=None, help="Stage A export dir for Stage B")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data:
        cfg["data"]["root"] = args.data
    stage = cfg.get("stage", "a")
    if args.smoke:
        cfg["steps"] = 8
        cfg["warmup_steps"] = 2
        cfg["log_every"] = 2
        cfg["save_every"] = 8
        cfg["train"]["num_workers"] = 0
        cfg["train"]["batch_size"] = 2
    out_dir = Path(args.out) if args.out else ROOT / "artifacts" / f"train_{stage}"
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import DepthAnythingForDepthEstimation

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = cfg.get("model", "depth-anything/Depth-Anything-V2-Small-hf")
    if stage == "b" and cfg.get("use_stage_a_checkpoint"):
        stage_a_dir = args.stage_a_dir or cfg.get("stage_a_dir") or str(ROOT / "artifacts" / "stage_a_export")
        if Path(stage_a_dir).exists():
            model_id = stage_a_dir
        else:
            print(f"warning: stage A export not found at {stage_a_dir}; starting from the base model")
    print(f"loading model: {model_id} on {device}")
    model = DepthAnythingForDepthEstimation.from_pretrained(model_id)

    from ml.dataset import ReliefEvalDataset, ReliefTrainDataset
    from ml.trainer import Trainer

    root = cfg["data"]["root"]
    if not Path(root, "scenes.jsonl").exists():
        raise FileNotFoundError(f"dataset not found at {root} - run src/data_download.py or src/generate_dataset.py first")
    textures = load_textures(cfg["data"].get("textures_dir"))
    if textures:
        print(f"loaded {len(textures)} CC0 textures")
    train_ds = ReliefTrainDataset(root, size=int(cfg["data"].get("size", 518)), full_frac=float(cfg["views"]["full_frac"]), base_crop_min=float(cfg["views"]["crop_min"]), textures=textures)
    val_ds = ReliefEvalDataset(root, split="val", size=int(cfg["data"].get("size", 518)), textures=textures)
    print(f"dataset: {len(train_ds)} train scenes, {len(val_ds)} val views")

    run_name = (cfg.get("mlflow", {}).get("run_name") or f"{stage}_{'smoke_' if args.smoke else ''}{int(time.time())}")
    run = tracking.start_run(cfg["mlflow"]["experiment"], run_name, tags={"stage": stage, "smoke": str(args.smoke)})
    tcfg = {**cfg.get("train", {}), "lr": cfg["lr"], "steps": cfg.get("steps", 2000), "stage": stage, "warmup_steps": cfg.get("warmup_steps", 300), "loss": cfg["loss"], "model": model_id, "ema_decay": cfg.get("train", {}).get("ema_decay", 0.999)}
    tracking.log_params_flat(run, tcfg)
    tracking.log_dict_json(run, tcfg, "config_effective.json")

    trainer = Trainer(model, tcfg, out_dir=out_dir, device=device, run=run)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed from {args.resume} at step {trainer.step}")
    t0 = time.time()
    stats = trainer.train(train_ds)
    print(f"training done in {(time.time() - t0) / 60:.1f} min")
    val = trainer.validate(val_ds, max_batches=3)
    print("validation:", val)
    if stage == "a":
        gate = val.get("val_polarity_positive_frac", 0)
        print(f"polarity gate (Pearson > 0 on val): {'PASS' if gate >= 0.9 else 'CHECK MANUALLY'} ({gate:.0%})")
    export_dir = out_dir / "export"
    trainer.export_best(export_dir)
    tracking.log_artifact_safe(run, trainer.history_path)
    # save the export as an MLflow artifact too, but weights stay out of git (.gitignore)
    for f in export_dir.iterdir():
        tracking.log_artifact_safe(run, f, artifact_path="model")
    tracking.log_dict_json(run, {"val": val, "train_stats": stats, "export": str(export_dir)}, "run_info.json")
    try:
        run.set_terminated()
    except Exception:
        pass
    print(f"export saved to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
