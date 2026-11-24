"""Client data — hand-made 3D reliefs / depth maps from the client's own workshop.

This is the highest-value data we can get (exact domain + client-owned licence, so
weights trained on it are commercially clean, unlike the CC-BY-NC Relief-Depth).

Accepted inputs (any mix, dropped into one folder):
  A) design image + height map pairs:   {name}.png/.jpg (rgb) + {name}_h.png/.tif (height)
     or explicitly paired by a manifest.csv (rgb,height)
  B) 3D meshes:                          {name}.stl / .obj / .3mf  (Z-up, flat base,
     relief above it - the normal CNC relief layout)
       -> height map = orthographic max-Z projection (exact triangle rasterization),
          base plane = the slab level (15th percentile of the maxima) - our h
          convention: 0 = base, raised = positive
       -> an input image is rendered from the height map with the clay recipe (if the
          client also has the original design image, prefer the pair form A)

Output: the SAME unified real-data layout Relief-Depth uses
  {out}/{split}/rgb/{id}.png + {out}/{split}/h/{id}.png + rows appended to
  reliefdepth.jsonl (source="client"), so MixedReliefDataset picks it up unchanged.

Height direction check: for pairs we correlate the height map with the design image
luminance; CNC height exports are usually bright=raised - if the correlation is
strongly negative we warn (and can flip with auto_flip=True).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

RGB_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
HEIGHT_HINT = ("_h", "_height", "_depth", "_d")
MESH_EXT = (".stl", ".obj", ".3mf")
MAX_RES = 1036
NOMINAL_WIDTH_MM = 300.0
NOMINAL_DEPTH_MM = 8.0


def _find_pairs(folder: Path) -> list[tuple[Path, Path | None]]:
    """Heuristic pairing: foo.png + foo_h.png, or meshes on their own."""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    meshes = [p for p in files if p.suffix.lower() in MESH_EXT]
    rgbs = [p for p in files if p.suffix.lower() in RGB_EXT and not any(p.stem.lower().endswith(h) for h in HEIGHT_HINT)]
    heights = [p for p in files if p.suffix.lower() in (".png", ".tif", ".tiff") and any(p.stem.lower().endswith(h) for h in HEIGHT_HINT)]
    pairs = []
    used = set()
    for h in heights:
        stem = h.stem
        base = next((s for s in HEIGHT_HINT if stem.lower().endswith(s)), "_h")
        rgb_stem = stem[: -len(base)]
        rgb = next((r for r in rgbs if r.stem == rgb_stem and r not in used), None)
        if rgb is not None:
            used.add(rgb)
        pairs.append((rgb, h))
    for r in rgbs:
        if r not in used:
            pairs.append((r, None))  # image without height: skipped at import, reported
    return [(rgb, h) for rgb, h in pairs if h is not None] + [(m, m) for m in meshes]


def load_height_u16(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"could not read {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        arr = (arr.astype(np.float32) / 255.0 * 65535).round().astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = np.clip(arr.astype(np.float32), 0, 65535).round().astype(np.uint16)
    return arr


def height_from_heightmap(u16: np.ndarray, flip: bool = False) -> np.ndarray:
    """uint16 height export -> h01. Bright = raised (CNC convention); `flip` inverts."""
    h = u16.astype(np.float32) / 65535.0
    if flip:
        h = 1.0 - h
    return np.clip(h, 0.0, 1.0)


def direction_check(rgb: np.ndarray | None, u16: np.ndarray) -> float | None:
    """Pearson(height, luminance(rgb)) - strongly negative may mean dark=raised."""
    if rgb is None:
        return None
    lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    h = u16.astype(np.float32)
    if lum.std() < 1e-6 or h.std() < 1e-6:
        return None
    return float(np.corrcoef(lum.ravel(), h.ravel())[0, 1])


def _rasterize_max_z(tris: np.ndarray, W_px: int, H_px: int, scale_x, scale_y, off_x, off_y) -> np.ndarray:
    """Exact z-buffer: for every pixel centre, the max z over covering triangles."""
    zbuf = np.full((H_px, W_px), -np.inf, np.float64)
    for tri in tris:
        px = (tri[:, 0] - off_x) * scale_x
        py = (tri[:, 1] - off_y) * scale_y
        x0, x1 = int(np.floor(px.min())), int(np.ceil(px.max()))
        y0, y1 = int(np.floor(py.min())), int(np.ceil(py.max()))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, W_px - 1), min(y1, H_px - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        gx, gy = np.meshgrid(xs, ys)
        d = ((px[1] - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (py[1] - py[0]))
        if abs(d) < 1e-12:
            continue
        b1 = ((gx - px[0]) * (py[2] - py[0]) - (px[2] - px[0]) * (gy - py[0])) / d
        b2 = ((px[1] - px[0]) * (gy - py[0]) - (gx - px[0]) * (py[1] - py[0])) / d
        b0 = 1.0 - b1 - b2
        inside = (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9)
        if not inside.any():
            continue
        z = b0 * tri[0, 2] + b1 * tri[1, 2] + b2 * tri[2, 2]
        iy, ix = gy[inside], gx[inside]
        zbuf[iy, ix] = np.maximum(zbuf[iy, ix], z[inside])
    return zbuf


def mesh_to_height(mesh_path: str | Path, out_long_side: int = MAX_RES) -> np.ndarray:
    """Orthographic max-Z projection of a Z-up relief mesh -> h01 (base = slab plane).

    Rasterized exactly (z-buffer over triangles, no stochastic sampling) and undercuts
    collapse to the visible height field - what a 3-axis CNC can carve anyway. The
    background plane is a low percentile of the rasterized maxima: the relief surface
    dominates the projection of a real CNC file, so the slab level is that mode.
    """
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=True)
    if mesh.is_empty or len(mesh.faces) == 0:
        raise ValueError(f"empty mesh: {mesh_path}")
    mins, maxs = mesh.bounds
    W_m, D_m = maxs[0] - mins[0], maxs[1] - mins[1]
    if W_m <= 0 or D_m <= 0:
        raise ValueError(f"degenerate mesh bounds: {mesh.bounds}")
    long_m = max(W_m, D_m)
    n = max(2, int(round(out_long_side * min(W_m, D_m) / long_m)))
    m = max(2, int(round(out_long_side * max(W_m / long_m, D_m / long_m))))
    W_px, H_px = (m, n) if W_m >= D_m else (n, m)
    zbuf = _rasterize_max_z(np.asarray(mesh.triangles, np.float64), W_px, H_px, (W_px - 1) / W_m, (H_px - 1) / D_m, mins[0], mins[1])
    covered = zbuf > -np.inf
    if not covered.any():
        raise ValueError(f"mesh projects to nothing: {mesh_path}")
    base = float(np.percentile(zbuf[covered], 15))
    if float(zbuf[covered].max()) - base <= 1e-9:
        # no slab: a free-standing object whose visible height field is one level
        # (e.g. an undercut mushroom). Base = the mesh bottom, as if it sat on a board.
        base = float(mins[2])
    zbuf[~covered] = base
    h = np.clip(zbuf - base, 0.0, None)
    peak = h.max()
    if peak <= 1e-9:
        raise ValueError(f"flat mesh (zero relief height): {mesh_path}")
    return np.clip(h / peak, 0.0, 1.0).astype(np.float32)


def render_input(h01: np.ndarray, seed: int = 0) -> np.ndarray:
    """Clay-style input render of a client height map (used when only a mesh exists)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from synth.render import render_view

    meta = {"relief_depth_mm_scene": NOMINAL_DEPTH_MM, "px_per_mm_native": NOMINAL_WIDTH_MM / max(h01.shape)}
    rng = np.random.default_rng(seed)
    inst = (h01 > 0.02).astype(np.uint16)
    return render_view(h01, inst, meta, view_scale=1.0, style="clay", rng=rng)


def import_client_data(
    src_dir: str | Path,
    out_root: str | Path,
    *,
    split: str = "train",
    auto_flip: bool = False,
    render_missing_inputs: bool = True,
) -> dict:
    """Import a folder of client files into the unified real-data format."""
    src = Path(src_dir)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "reliefdepth.jsonl"
    rows = []
    if jsonl.exists():
        rows = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    report = {"pairs": 0, "meshes": 0, "skipped": [], "flipped": [], "direction": {}}
    existing_ids = {r["id"] for r in rows}
    entries = _find_pairs(src)
    if not entries:
        raise FileNotFoundError(f"nothing importable found in {src} (need rgb+_h pairs or stl/obj/3mf)")
    k = 0
    for rgb_path, other in entries:
        is_mesh = other.suffix.lower() in MESH_EXT
        try:
            if is_mesh:
                h = mesh_to_height(other)
                rgb = None
                if render_missing_inputs:
                    rgb = render_input(h, seed=k)
                report["meshes"] += 1
                sid = other.stem
            else:
                u16 = load_height_u16(other)
                corr = direction_check(cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)[..., ::-1] if rgb_path else None, u16)
                flip = bool(auto_flip and corr is not None and corr < -0.2)
                h = height_from_heightmap(u16, flip=flip)
                if flip:
                    report["flipped"].append(other.name)
                if corr is not None:
                    report["direction"][other.name] = round(corr, 3)
                rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if rgb is None:
                    report["skipped"].append(f"{rgb_path.name}: unreadable rgb")
                    continue
                rgb = rgb[..., ::-1]
                if rgb.shape[:2] != h.shape:
                    h = cv2.resize(h, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
                report["pairs"] += 1
                sid = rgb_path.stem
            if float(h.std()) < 0.01:
                report["skipped"].append(f"{sid}: nearly flat height")
                continue
            gid = f"CLIENT_{sid}"
            if gid in existing_ids:
                gid = f"CLIENT_{sid}_{k}"
            (out / split / "rgb").mkdir(parents=True, exist_ok=True)
            (out / split / "h").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / split / "rgb" / f"{gid}.png"), rgb)
            cv2.imwrite(str(out / split / "h" / f"{gid}.png"), (h * 65535.0).round().astype(np.uint16))
            rows.append(
                {
                    "id": gid,
                    "scene": f"CLIENT_{sid}",
                    "split": split,
                    "lighting": "client",
                    "relief_type": "client",
                    "culture": "client",
                    "material": "unknown",
                    "W": int(h.shape[1]),
                    "H": int(h.shape[0]),
                    "width_mm_scene": NOMINAL_WIDTH_MM,
                    "relief_depth_mm_scene": NOMINAL_DEPTH_MM,
                    "px_per_mm_native": NOMINAL_WIDTH_MM / h.shape[1],
                    "source": "client",
                }
            )
            k += 1
        except Exception as exc:
            report["skipped"].append(f"{other.name}: {exc}")
    with open(jsonl, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    report["imported"] = k
    return report


def client_qc(out_root: str | Path) -> dict:
    """Contact sheet + stats of the client rows (look at the sheet before training)."""
    out = Path(out_root)
    rows = [json.loads(l) for l in open(out / "reliefdepth.jsonl", encoding="utf-8")]
    client = [r for r in rows if r["source"] == "client"]
    if not client:
        return {"n": 0}
    tiles = []
    from reliefkit.preview import shade_height

    for r in client[:24]:
        rgb = cv2.imread(str(out / r["split"] / "rgb" / f"{r['id']}.png"))
        h = cv2.imread(str(out / r["split"] / "h" / f"{r['id']}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535.0
        rgb = cv2.resize(rgb, (256, 256))
        hh = cv2.resize(h, (256, 256))
        tiles.append(np.concatenate([rgb, shade_height(hh)], axis=1))
    sheet = np.concatenate(tiles, axis=0)
    cv2.imwrite(str(out / "client_contact_sheet.png"), sheet)
    return {"n": len(client), "splits": {s: sum(1 for r in client if r["split"] == s) for s in {r["split"] for r in client}}}
