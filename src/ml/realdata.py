"""Relief-Depth (KarimT/Relief-Depth) — download + convert into our unified format.

Facts (audited 2026-11-24):
* Source: 3D relief models from Sketchfab (British Museum, MET, ...), rendered from the
  front under 5 lightings (indoor_museum / outdoor_noon / outdoor_overcast /
  raking_left / raking_right). One rendered DEPTH per scene shared by all lightings
  -> real geometry labels, not pseudo-labels.
* 768x768, RGB uint8 + depth uint16; 807 scenes with files, 8,070 pairs.
* model_annotations.csv gives split (train/val/test/exclude), relief_type (bas/sunk/high),
  culture, material. attribution_ledger.csv gives per-model source + license
  (mostly CC-BY-NC from museums).
* Decision (user, 2026-11-24): use it as the PRIMARY training dataset for the course
  project. Licenses are non-commercial -> fine for research/course, NOT for the
  client's commercial product (re-licensing needed there).

Conversion to our height convention h in [0,1], 0 = base plane:
    d01     = depth / 65535
    bg      = median(d01 over a border ring)      # the flat slab plane
    h       = |d01 - bg|                          # bas: figure nearer than the slab;
                                                 # sunk: figure farther - both become
                                                 # "raised", which is the CNC convention
    h      /= percentile(h, 99.5)                 # robust peak normalization
All lightings of one scene stay in the SAME split (the CSV is per scene).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

DATASET_ID = "KarimT/Relief-Depth"
LIGHTINGS = ("indoor_museum", "outdoor_noon", "outdoor_overcast", "raking_left", "raking_right")
RING_PX = 24
PEAK_PCT = 99.5
MIN_H_STD = 0.008  # nearly flat scenes carry no signal
# Nominal physical scale for the normal-loss kappa only (the real physical depth of a
# museum relief is unknown; kappa only weights the normal term, see docs).
NOMINAL_WIDTH_MM = 300.0
NOMINAL_DEPTH_MM = 8.0


def download_reliefdepth(raw_dir: str | Path, max_scenes: int | None = None) -> Path:
    """Pull rgb_d_pairs + the metadata CSVs from the HF Hub (skips qualitative_images)."""
    from huggingface_hub import snapshot_download

    raw = Path(raw_dir)
    patterns = ["rgb_d_pairs/*", "model_annotations.csv", "attribution_ledger.csv", "README.md"]
    snapshot_download(DATASET_ID, repo_type="dataset", local_dir=str(raw), allow_patterns=patterns)
    return raw


def load_annotations(raw_dir: str | Path) -> dict[str, dict]:
    ann = {}
    p = Path(raw_dir) / "model_annotations.csv"
    if not p.exists():
        from huggingface_hub import hf_hub_download

        p = hf_hub_download(DATASET_ID, "model_annotations.csv", repo_type="dataset", cache_dir=str(Path(raw_dir) / "_hf"))
    for r in csv.DictReader(open(p, encoding="utf-8")):
        ann[r["Dataset_ID"]] = r
    return ann


def height_from_depth(depth_u16: np.ndarray) -> tuple[np.ndarray, dict]:
    """uint16 rendered depth -> h01 following the convention in the module docstring."""
    d01 = depth_u16.astype(np.float32) / 65535.0
    H, W = d01.shape
    ring = np.concatenate([d01[:RING_PX].ravel(), d01[-RING_PX:].ravel(), d01[:, :RING_PX].ravel(), d01[:, -RING_PX:].ravel()])
    bg = float(np.median(ring))
    h = np.abs(d01 - bg)
    peak = float(np.percentile(h, PEAK_PCT))
    info = {"bg": bg, "peak": peak, "flat": peak < 1e-4 or float(h.std()) < MIN_H_STD}
    if info["flat"]:
        return np.zeros_like(h), info
    h = np.clip(h / max(peak, 1e-6), 0.0, 1.0)
    return h, info


def convert_reliefdepth(
    raw_dir: str | Path,
    out_root: str | Path,
    *,
    splits=("train", "val", "test"),
    max_scenes: int | None = None,
    log_every: int = 100,
) -> Path:
    """Convert the downloaded dataset into data/<out_root>/{split}/{rgb,h} + reliefdepth.jsonl."""
    raw = Path(raw_dir)
    out = Path(out_root)
    ann = load_annotations(raw)
    pairs_dir = raw / "rgb_d_pairs"
    rows: list[dict] = []
    scenes = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1] for p in pairs_dir.glob("RLF_*_rgb.png")})
    if max_scenes:
        scenes = scenes[:max_scenes]
    kept = 0
    for n, scene in enumerate(scenes):
        a = ann.get(scene)
        if a is None or a["split"] not in splits:
            continue
        split = a["split"]
        (out / split / "rgb").mkdir(parents=True, exist_ok=True)
        (out / split / "h").mkdir(parents=True, exist_ok=True)
        for lighting in LIGHTINGS:
            rgb_p = pairs_dir / f"{scene}_{lighting}_rgb.png"
            dep_p = pairs_dir / f"{scene}_{lighting}_depth.png"
            if not (rgb_p.exists() and dep_p.exists()):
                continue
            rgb = cv2.imread(str(rgb_p), cv2.IMREAD_COLOR)
            dep = cv2.imread(str(dep_p), cv2.IMREAD_UNCHANGED)
            if rgb is None or dep is None or dep.ndim == 3:
                continue
            if dep.dtype == np.uint8:
                dep = (dep.astype(np.float32) / 255.0 * 65535).astype(np.uint16)
            h, hinfo = height_from_depth(dep)
            if hinfo["flat"]:
                continue
            gid = f"{scene}_{lighting}"
            cv2.imwrite(str(out / split / "rgb" / f"{gid}.png"), rgb)
            cv2.imwrite(str(out / split / "h" / f"{gid}.png"), (h * 65535.0).round().astype(np.uint16))
            rows.append(
                {
                    "id": gid,
                    "scene": scene,
                    "split": split,
                    "lighting": lighting,
                    "relief_type": a.get("relief_type", "unknown"),
                    "culture": a.get("culture", "Unknown"),
                    "material": a.get("material", "unknown"),
                    "W": int(dep.shape[1]),
                    "H": int(dep.shape[0]),
                    "width_mm_scene": NOMINAL_WIDTH_MM,
                    "relief_depth_mm_scene": NOMINAL_DEPTH_MM,
                    "px_per_mm_native": NOMINAL_WIDTH_MM / dep.shape[1],
                    "source": "relief-depth",
                }
            )
            kept += 1
        if log_every and (n + 1) % log_every == 0:
            print(f"  {n + 1}/{len(scenes)} scenes, {kept} pairs kept", flush=True)
    with open(out / "reliefdepth.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out


def qc_reliefdepth(out_root: str | Path) -> dict:
    """Numeric QC: split counts, no scene in two splits, height stats."""
    out = Path(out_root)
    rows = [json.loads(l) for l in open(out / "reliefdepth.jsonl", encoding="utf-8")]
    assert rows, "empty conversion"
    by_split: dict[str, set[str]] = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["scene"])
    assert not (by_split.get("train", set()) & by_split.get("val", set())), "scene leaked train->val"
    assert not (by_split.get("train", set()) & by_split.get("test", set())), "scene leaked train->test"
    rng = np.random.default_rng(0)
    for r in rng.choice(rows, size=min(25, len(rows)), replace=False):
        h = cv2.imread(str(out / r["split"] / "h" / f"{r['id']}.png"), cv2.IMREAD_UNCHANGED)
        assert h is not None and h.dtype == np.uint16 and not np.isnan(h).any()
        assert int(h.max()) > 30000, "peak too low"
        assert float(h.std()) > MIN_H_STD * 65535
    counts = {k: len(v) for k, v in by_split.items()}
    return {"n_pairs": len(rows), "scenes_per_split": counts}
