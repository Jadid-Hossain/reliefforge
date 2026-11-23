"""Fetch the dataset (and optional CC0 textures) into data/ — the course rule:
the repo never contains data, only the code that fetches/builds it.

Priority:
1. HF Hub dataset (if RELIEFFORGE_DATASET or the public repo exists) -> data/relief_dataset
2. Local generation with the shipped generator (deterministic given the seed)
Textures: CC0 wood/plaster/stone albedos from ambientCG (URL list below) -> data/textures
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# A small curated CC0 set (ambientCG.com, CC0 1.0). Falls back to procedural wood.
AMBIENTCG_TEXTURES = [
    "https://ambientcg.com/get?file=WoodSidingWeathered001_Flat_4K_JPG.jpg",
    "https://ambientcg.com/get?file=WoodFineWood001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=WoodBirch001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=WoodPlywood001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=WoodPlanksOiled002_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=WoodTable001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=PlasterPainted001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=PlasterPainted004_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=StoneGranite001_Color_4K_JPG.jpg",
    "https://ambientcg.com/get?file=StoneMarble001_Color_4K_JPG.jpg",
]


def try_download_textures(out_dir: Path, max_files: int = 10) -> int:
    import urllib.request

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for url in AMBIENTCG_TEXTURES[:max_files]:
        name = url.split("file=")[-1].replace("%20", "_")
        dest = out_dir / name
        if dest.exists():
            n += 1
            continue
        try:
            print(f"  downloading {name} ...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "ReliefForge/0.1 (course project)"})
            with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
                f.write(r.read())
            n += 1
        except Exception as exc:
            print(f"    failed ({exc}); the procedural wood generator will cover it")
            if dest.exists():
                dest.unlink()
    return n


def try_hf_dataset(repo_id: str, out_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return False
    try:
        snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(out_dir), local_dir_use_symlinks=False)
        return (out_dir / "scenes.jsonl").exists()
    except Exception as exc:
        print(f"  HF dataset '{repo_id}' not available ({type(exc).__name__}); generating locally")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/relief_dataset")
    ap.add_argument("--n-scenes", type=int, default=4000)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--textures", action="store_true", help="also fetch the CC0 texture pack")
    ap.add_argument("--hf-repo", default=None, help="try this HF dataset repo first")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    if (out / "scenes.jsonl").exists():
        print(f"dataset already present: {out}")
    else:
        done = bool(args.hf_repo) and try_hf_dataset(args.hf_repo, out)
        if not done:
            print(f"generating {args.n_scenes} scenes locally (seed0={args.seed0}) ...")
            from synth.generate import generate_dataset

            generate_dataset(out, n_scenes=args.n_scenes, seed0=args.seed0, workers=args.workers)
    from synth.generate import qc_assertions

    rep = qc_assertions(out)
    print("QC:", rep)
    if args.textures:
        n = try_download_textures(ROOT / "data" / "textures")
        print(f"textures: {n} files in data/textures (licence: CC0, source ambientcg.com)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
