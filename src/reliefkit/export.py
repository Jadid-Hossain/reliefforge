"""M6 — exporters: 16-bit TIFF, STL/OBJ/3MF mesh files, JSON sidecar, ZIP bundle.

Files (docs/architecture.md §M6):
    relief_heightmap_16bit.tiff   uint16, full working resolution
    relief_heightmap_8bit.png     grayscale preview copy for CAM tools that read 8-bit
    relief.stl / relief.obj / relief.3mf   authored in millimetres
    relief_meta.json              physical scale + validation + warnings
    preview.png
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from . import io as rk_io
from . import mesh as rk_mesh
from .preview import shade_height


def build_meta(
    *,
    width_mm: float,
    height_mm: float,
    relief_depth_mm: float,
    base_mm: float,
    h_shape: tuple[int, int],
    mesh_info: dict,
    validation: dict,
    mode: str,
    model_version: str,
    invert: bool,
    infer_long_side: int,
    warnings: list[str],
    extra: dict | None = None,
) -> dict:
    H, W = int(h_shape[0]), int(h_shape[1])
    meta = {
        "units": "mm",
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "relief_depth_mm": float(relief_depth_mm),
        "base_thickness_mm": float(base_mm),
        "pixel_pitch_mm": float(width_mm / W),
        "mesh_pitch_mm": mesh_info.get("mesh_pitch_x_mm"),
        "grid_w": W,
        "grid_h": H,
        "mesh_grid_w": mesh_info.get("grid_w"),
        "mesh_grid_h": mesh_info.get("grid_h"),
        "mesh_triangles": validation.get("n_triangles"),
        "mesh_downsampled": mesh_info.get("downsampled"),
        "detail": mesh_info.get("detail"),
        "infer_long_side": int(infer_long_side),
        "mode": mode,
        "model_version": model_version,
        "invert": bool(invert),
        "mesh_valid": validation.get("is_valid"),
        "z_min_mm": validation.get("z_min_mm"),
        "z_max_mm": validation.get("z_max_mm"),
        "warnings": warnings,
        "note": "STL/OBJ carry no units: set the import unit to mm. For the TIFF scale Z to relief_depth_mm (0 = base plane).",
    }
    if extra:
        meta.update(extra)
    return meta


def export_all(
    h01: np.ndarray,
    mesh: "object",
    mesh_info: dict,
    validation: dict,
    out_dir: str | Path,
    meta: dict,
    *,
    make_zip: bool = True,
    skip_3mf: bool = False,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["heightmap_tiff"] = rk_io.save_height_tiff(h01, out / "relief_heightmap_16bit.tiff")
    import cv2

    ok = cv2.imwrite(str(out / "relief_heightmap_8bit.png"), (np.clip(h01, 0, 1) * 255).round().astype(np.uint8))
    if not ok:
        raise OSError("could not write the 8-bit heightmap PNG")
    paths["heightmap_png8"] = out / "relief_heightmap_8bit.png"

    if not validation.get("is_valid"):
        raise ValueError(f"refusing to export an invalid mesh: {validation}")

    paths["stl"] = out / "relief.stl"
    mesh.export(paths["stl"], file_type="stl")

    paths["obj"] = out / "relief.obj"
    paths["obj"].write_bytes(rk_mesh.mesh_to_obj_bytes(mesh))

    if not skip_3mf:
        paths["3mf"] = out / "relief.3mf"
        mesh.export(paths["3mf"], file_type="3mf")

    paths["meta"] = out / "relief_meta.json"
    paths["meta"].write_text(json.dumps(meta, indent=2), encoding="utf-8")

    paths["preview"] = out / "preview.png"
    rk_io.save_rgb(shade_height(h01), paths["preview"])

    if make_zip:
        zpath = out / "relief_package.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for key, p in paths.items():
                if key != "zip":
                    zf.write(p, p.name)
        paths["zip"] = zpath
    return paths
