"""Evaluation entry point: python src/evaluate.py [--model path|--zero-shot] [--sizes 518 756]

Writes artifacts/eval/{results,summary,by_family}.csv and a qualitative figure;
logs everything to MLflow (experiment ReliefForge-Eval).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "src"))

from ml import tracking
from ml.evaluate import evaluate_model, load_model, qualitative_figure, write_report
from train import load_config, load_textures  # noqa: E402  (src/ is on sys.path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["configs/config.yaml"])
    ap.add_argument("--model", default=None, help="fine-tuned export dir; omit with --zero-shot")
    ap.add_argument("--zero-shot", action="store_true")
    ap.add_argument("--label", default=None)
    ap.add_argument("--sizes", nargs="+", type=int, default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--data", default=None, help="override data.root")
    ap.add_argument("--real", default=None, help="Relief-Depth converted root (real test set)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.data or cfg["data"]["root"]
    if args.data:
        cfg["data"]["root"] = args.data
    if not Path(root, "scenes.jsonl").exists():
        raise FileNotFoundError(f"dataset not found at {root}")
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from ml.evaluate import evaluate_model, load_model, qualitative_figure, write_report

    textures = load_textures(cfg["data"].get("textures_dir"))
    if args.zero_shot:
        model = load_model(cfg.get("model", "depth-anything/Depth-Anything-V2-Small-hf"), device)
        label = args.label or "zeroshot"
    else:
        model_dir = args.model or str(ROOT / "artifacts" / "train_b" / "export")
        if not Path(model_dir).exists():
            model_dir = str(ROOT / "artifacts" / "train_a" / "export")
        model = load_model(model_dir, device)
        label = args.label or Path(model_dir).parent.name + "_export"
    sizes = tuple(args.sizes or cfg["eval"].get("sizes", [518]))
    split = args.split or cfg["eval"].get("split", "test")
    max_items = args.max_items or cfg["eval"].get("max_items")
    out_dir = Path(cfg["eval"].get("out_dir", "artifacts/eval"))

    run = tracking.start_run(cfg["mlflow"]["eval_experiment"], f"{label}_{int(time.time())}", tags={"label": label, "split": split})
    tracking.log_params_flat(run, {"model": label, "sizes": list(sizes), "split": split, "max_items": max_items})
    if args.real:
        from ml.evaluate import evaluate_reliefdepth

        rows = []
        for S in sizes:
            rows += evaluate_reliefdepth(model, args.real, split=split, size=S, device=device, max_items=max_items, label=label)
    else:
        rows = evaluate_model(model, root, split=split, sizes=sizes, device=device, max_items=max_items, label=label)
    rep = write_report(rows, out_dir, run)
    if not args.real:
        fig = qualitative_figure({"pred": model}, root, out_dir / f"qualitative_{label}.png", split=split, size=sizes[0], n=5, device=device)
        tracking.log_artifact_safe(run, fig)
    print("| model | S | n | MAE | RMSE | SSIM | normal deg | R_detail | pearson |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in rep["table"]:
        print(f"| {t['model']} | {t['S']} | {t['n']} | {t['MAE_norm']:.4f} | {t['RMSE_norm']:.4f} | {t['SSIM']:.4f} | {t['normal_err_deg']:.2f} | {t['R_detail']:.3f} | {t['pearson']:.4f} |")
    try:
        run.set_terminated()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
