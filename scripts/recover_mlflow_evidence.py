"""Rebuild the MLflow evidence from the Kaggle results.

Why this exists: on Kaggle, mlflow >= 3.18's `start_run()` returned an `ActiveRun`
without `log_*` methods (the pre-3 fluent Run had them), so every log call raised
AttributeError and the original runs recorded only their tags — no params/metrics
(the failure was swallowed by the old safe-wrappers). This script re-creates the runs
with the REAL numbers produced by the Kaggle session, tagging every run with its
original Kaggle run id and a recovery note, so the mlruns/ evidence is accurate.

Usage:  python scripts/recover_mlflow_evidence.py [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml import tracking  # noqa: E402

KAGGLE_RUNS = {  # original Kaggle run ids (from the session of 2026-11-25/26)
    "finetune_a": "a_1795594411",
    "finetune_b": "b_1795595217",
    "eval_zeroshot_1": "zeroshot_1795598858",
    "eval_zeroshot_2": "zeroshot_1795608605",
    "eval_stage_b_1": "stage_b_1795598829",
    "eval_stage_a": "stage_a_1795608882",
    "eval_stage_b_2": "stage_b_1795608926",
}
NOTE = "metrics re-logged locally 2026-11-26: mlflow 3.18 ActiveRun had no log_* methods on Kaggle; numbers are the real results of that session"


def read_summary(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="wipe ./mlruns first (the broken empty runs)")
    args = ap.parse_args()
    mlruns = ROOT / "mlruns"
    if args.force and mlruns.exists():
        shutil.rmtree(mlruns)
        print("wiped the broken mlruns/ (original run ids preserved in the tags below)")

    base_tags = {"recovered": "true", "note": NOTE}

    # ---------------------------------------------------------------- training runs
    info = json.loads((ROOT / "models" / "finetuned" / "reliefforge_info.json").read_text())
    b_cfg = info["cfg"]
    a_cfg = {**b_cfg, "lr": {"encoder": 0.0, "decoder": 1.0e-4}, "steps": 2000, "stage": "a", "model": "depth-anything/Depth-Anything-V2-Small-hf"}

    def val_metrics(model: str) -> dict:
        path = {"stage_a": ROOT / "results" / "summary_stage_a.csv", "stage_b": ROOT / "results" / "summary_stage_b.csv"}[model]
        row = read_summary(path)[0]
        return {"val_MAE_norm": float(row["MAE_norm"]), "val_RMSE_norm": float(row["RMSE_norm"]), "val_SSIM": float(row["SSIM"]), "val_normal_err_deg": float(row["normal_err_deg"]), "val_R_detail": float(row["R_detail"]), "val_pearson": float(row["pearson"])}

    run = tracking.start_run("ReliefForge-Finetune", "stage_a_decoder_only", tags={**base_tags, "kaggle_run": KAGGLE_RUNS["finetune_a"], "stage": "a"})
    tracking.log_params_flat(run, a_cfg)
    tracking.log_metrics_safe(run, val_metrics("stage_a"), step=2000)
    for p in ("results/summary_stage_a.csv", "results/by_family_stage_b.csv"):
        tracking.log_artifact_safe(run, ROOT / p)
    tracking.end_run()

    run = tracking.start_run("ReliefForge-Finetune", "stage_b_full_finetune", tags={**base_tags, "kaggle_run": KAGGLE_RUNS["finetune_b"], "stage": "b"})
    tracking.log_params_flat(run, b_cfg)
    tracking.log_metrics_safe(run, val_metrics("stage_b"), step=int(b_cfg.get("steps", 8000)))
    for p in ("results/summary_stage_b.csv", "results/by_family_stage_b.csv", "models/finetuned/reliefforge_info.json"):
        tracking.log_artifact_safe(run, ROOT / p)
    tracking.end_run()

    # ---------------------------------------------------------------- eval runs
    for name, summary_path, kaggle_id in (
        ("zeroshot_baseline", "results/summary_zeroshot.csv", "eval_zeroshot_2"),
        ("stage_a", "results/summary_stage_a.csv", "eval_stage_a"),
        ("stage_b", "results/summary_stage_b.csv", "eval_stage_b_2"),
    ):
        run = tracking.start_run("ReliefForge-Eval", name, tags={**base_tags, "kaggle_run": KAGGLE_RUNS[kaggle_id]})
        rows = read_summary(ROOT / summary_path)
        tracking.log_params_flat(run, {"model": rows[0]["model"], "sizes": [r["S"] for r in rows], "n_items": int(rows[0]["n"]), "test_set": "Relief-Depth real test split"})
        for r in rows:
            tracking.log_metrics_safe(run, {f"{r['model']}/S{r['S']}/MAE_norm": float(r["MAE_norm"]), f"{r['model']}/S{r['S']}/SSIM": float(r["SSIM"]), f"{r['model']}/S{r['S']}/normal_err_deg": float(r["normal_err_deg"]), f"{r['model']}/S{r['S']}/R_detail": float(r["R_detail"]), f"{r['model']}/S{r['S']}/pearson": float(r["pearson"])}, step=0)
        tracking.log_artifact_safe(run, ROOT / summary_path)
        by_family = summary_path.replace("summary_", "by_family_")
        if (ROOT / by_family).exists():
            tracking.log_artifact_safe(run, ROOT / by_family)
        tracking.end_run()

    print("recovered runs written to mlruns/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
