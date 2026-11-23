"""M8/M9 — mass generation of the geometry dataset + fixed val/test renders.

Layout on disk (PRD_v2 §5.4):
    <root>/geometry/{split}/{id:06d}_h.png     uint16 height (peak = 1, bg = 0)
    <root>/geometry/{split}/{id:06d}_inst.png  uint16 instance IDs
    <root>/scenes.jsonl                         per-scene metadata
    <root>/renders/{val,test}/{id:06d}_{style}.png   fixed pre-renders
    <root>/previews/{id:06d}.jpg                one preview per train scene
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

from . import recipes
from .render import STYLES, STYLE_WEIGHTS, render_view

TRAIN_PREVIEW_STYLE = "clay"


def h_to_u16(h: np.ndarray) -> np.ndarray:
    return (np.clip(h, 0, 1) * 65535.0).round().astype(np.uint16)


def generate_dataset(
    out_root: str | Path,
    n_scenes: int = 4000,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed0: int = 0,
    families: tuple[str, ...] | None = None,
    pre_render_styles: tuple[str, ...] = ("wood", "clay", "flat"),
    make_previews: bool = True,
    scene_fn=None,
    log_every: int = 200,
    workers: int = 1,
) -> Path:
    """Generate the geometry dataset. `scene_fn` allows tests to inject a cheap generator.

    The (id, split, family, seed) plan is fixed in the parent; workers only build the
    scenes, so parallel output can never duplicate ids or leak scenes across splits.
    """
    out = Path(out_root)
    (out / "geometry").mkdir(parents=True, exist_ok=True)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    if make_previews:
        (out / "previews").mkdir(parents=True, exist_ok=True)
    make_scene = scene_fn or recipes.make_scene_checked
    fams = list(families or recipes.FAMILIES)

    n_val, n_test = int(n_scenes * val_frac), int(n_scenes * test_frac)
    n_train = n_scenes - n_val - n_test
    plan = []
    idx = 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        (out / "geometry" / split).mkdir(parents=True, exist_ok=True)
        (out / "renders" / split).mkdir(parents=True, exist_ok=True)
        for _ in range(n):
            plan.append((idx, split, seed0 + idx, fams[idx % len(fams)]))
            idx += 1

    scenes = None
    if workers > 1 and scene_fn is None:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            scenes = list(ex.map(_scene_job, [(seed, fam) for (_i, _s, seed, fam) in plan], chunksize=8))

    rows = []
    for k, (i, split, seed, fam) in enumerate(plan):
        scene = scenes[k] if scenes is not None else make_scene(seed, fam)
        meta = scene["meta"]
        meta.update({"id": i, "split": split})
        gid = f"{i:06d}"
        cv2.imwrite(str(out / "geometry" / split / f"{gid}_h.png"), h_to_u16(scene["h"]))
        cv2.imwrite(str(out / "geometry" / split / f"{gid}_inst.png"), scene["inst"].astype(np.uint16))
        if split in ("val", "test"):
            for style in pre_render_styles:
                rng = np.random.default_rng(10_000 + i * 10)
                img = render_view(scene["h"], scene["inst"], meta, view_scale=518 / max(meta["W"], meta["H"]), style=style, rng=rng)
                cv2.imwrite(str(out / "renders" / split / f"{gid}_{style}.png"), img[..., ::-1])
        elif make_previews:
            rng = np.random.default_rng(i)
            img = render_view(scene["h"], scene["inst"], meta, view_scale=518 / max(meta["W"], meta["H"]), style=TRAIN_PREVIEW_STYLE, rng=rng)
            cv2.imwrite(str(out / "previews" / f"{gid}.jpg"), img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(meta)
        if log_every and (k + 1) % log_every == 0:
            print(f"  {k + 1}/{n_scenes} scenes", flush=True)
    with open(out / "scenes.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return out


def _scene_job(args):
    seed, fam = args
    return recipes.make_scene_checked(seed, fam)


def contact_sheet(out_root: str | Path, n: int = 24, seed: int = 0, out_path: str | Path | None = None) -> Path:
    """image | height | instance triples for QC (look at them - PRD_v2 §5.7)."""
    from reliefkit.preview import shade_height

    root = Path(out_root)
    rows = [json.loads(l) for l in open(root / "scenes.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    tiles = []
    for i in pick:
        m = rows[int(i)]
        gid = f"{m['id']:06d}"
        style = "clay"
        img = cv2.imread(str(root / "renders" / m["split"] / f"{gid}_{style}.png"))
        if img is None:
            img = cv2.imread(str(root / "previews" / f"{gid}.jpg"))
        h = cv2.imread(str(root / "geometry" / m["split"] / f"{gid}_h.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        inst = cv2.imread(str(root / "geometry" / m["split"] / f"{gid}_inst.png"), cv2.IMREAD_UNCHANGED)
        hh = cv2.resize(h, (256, 256), interpolation=cv2.INTER_AREA)
        ii = (cv2.resize(inst.astype(np.float32) / max(inst.max(), 1), (256, 256), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
        im = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        sh = shade_height(hh)
        tiles.append(np.concatenate([im, sh, np.stack([ii] * 3, -1)], axis=1))
    cols = 4
    grid = [np.concatenate(tiles[k : k + cols], axis=1) for k in range(0, len(tiles), cols)]
    sheet = np.concatenate(grid, axis=0)
    out_path = Path(out_path) if out_path else (root / "contact_sheet.png")
    cv2.imwrite(str(out_path), sheet[..., ::-1])
    return out_path


def qc_assertions(out_root: str | Path) -> dict:
    """Numeric QC gate (PRD_v2 §5.7)."""
    root = Path(out_root)
    rows = [json.loads(l) for l in open(root / "scenes.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate scene ids"
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    for r in rows:
        assert r["W"] % 14 == 0 and r["H"] % 14 == 0, f"canvas not multiple of 14: {r['W']}x{r['H']}"
        assert 0.03 <= r["fg_frac"] <= 0.95, f"fg_frac out of range: {r['fg_frac']}"
        assert r["h_std"] >= 0.02, "flat scene slipped through"
    rng = np.random.default_rng(0)
    for r in rng.choice(rows, size=min(20, len(rows)), replace=False):
        u16 = cv2.imread(str(root / "geometry" / r["split"] / f"{r['id']:06d}_h.png"), cv2.IMREAD_UNCHANGED)
        assert u16 is not None and u16.dtype == np.uint16 and not np.isnan(u16).any()
        assert int(u16.max()) > 65000, "peak must be ~1"
    return {"n_scenes": len(rows), "splits": counts}
