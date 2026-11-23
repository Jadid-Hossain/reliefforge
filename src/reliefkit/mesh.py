"""M5 — watertight 2.5D relief mesh builder and validator.

Conventions (docs/architecture.md §4):
    image column i -> x = i * pitch_x          (left of the image = x = 0)
    image row j   -> y = (H-1-j) * pitch_y     (top of the image = back = max Y)
    z = base_thickness_mm + h * relief_depth_mm

A relief is a height field over a rectangle, so the solid is always:
    * a top surface (2 triangles per grid cell),
    * four walls that close the perimeter down to z = 0,
    * a flat bottom made of one centre vertex and a triangle fan.
No undercuts are possible, and no triangle is degenerate as long as base >= 0.5 mm.
"""
from __future__ import annotations

import numpy as np
import trimesh

MIN_BASE_MM = 0.5
TRIANGLE_BUDGETS = {"draft": 250_000, "standard": 600_000, "high": 2_000_000}
BBOX_TOL_MM = 0.01
MIN_FACE_AREA_MM2 = 1e-9


class MeshError(ValueError):
    """Raised when a relief cannot be turned into a valid solid."""


def perimeter_loop(W: int, H: int) -> np.ndarray:
    """Grid vertex indices of the perimeter, counter-clockwise seen from above.

    In grid coordinates row j is y = j * pitch (the height map is flipped before
    this is called, so row H-1 is the back of the panel).
    Order: front edge left->right, right edge front->back, back edge right->left,
    left edge back->front. Corners appear once. With this direction the wall
    faces (T_k, B_k, B_k+1), (T_k, B_k+1, T_k+1) and the bottom fan (C, B_k+1, B_k)
    all point outward.
    """
    idx = np.arange(W * H).reshape(H, W)
    return np.concatenate(
        [
            idx[0, :],
            idx[1:H, W - 1],
            idx[H - 1, W - 2 :: -1],
            idx[H - 2 : 0 : -1, 0],
        ]
    )


def count_triangles(W: int, H: int) -> int:
    """Triangles of the solid built on a W x H grid (W, H >= 2)."""
    ring = len(perimeter_loop(W, H))
    return 2 * (W - 1) * (H - 1) + 3 * ring


def fit_grid(W: int, H: int, budget: int) -> tuple[int, int]:
    """Largest grid (same aspect) whose solid stays inside the triangle budget."""
    W, H = max(int(W), 2), max(int(H), 2)
    if count_triangles(W, H) <= budget:
        return W, H
    f = 1.0
    for _ in range(400):
        f *= 1.03
        W2 = max(2, int(round(W / f)))
        H2 = max(2, int(round(H / f)))
        if count_triangles(W2, H2) <= budget:
            return W2, H2
    return 2, 2


def _downsample(h: np.ndarray, Wm: int, Hm: int) -> np.ndarray:
    import cv2  # local import: mesh building itself does not need OpenCV

    return cv2.resize(h, (Wm, Hm), interpolation=cv2.INTER_AREA)


def build_relief_mesh(
    h01: np.ndarray,
    width_mm: float,
    relief_depth_mm: float,
    base_mm: float,
    detail: str = "standard",
    max_triangles: int | None = None,
) -> tuple[trimesh.Trimesh, dict]:
    """Build a watertight solid from a height map.

    Returns (mesh, info). The mesh is authored in millimetres.
    """
    h = np.asarray(h01, dtype=np.float32)
    if h.ndim != 2 or min(h.shape) < 2:
        raise MeshError(f"height map must be 2D with both sides >= 2, got {h.shape}")
    if not np.isfinite(h).all():
        raise MeshError("height map contains NaN or inf")
    if h.min() < -1e-4 or h.max() > 1 + 1e-4:
        raise MeshError(f"height map must be in [0, 1], got [{h.min():.4f}, {h.max():.4f}]")
    h = np.clip(h, 0.0, 1.0)
    if not (width_mm > 0 and relief_depth_mm > 0):
        raise MeshError("width_mm and relief_depth_mm must be positive")
    if base_mm < MIN_BASE_MM:
        raise MeshError(f"solid mode needs base >= {MIN_BASE_MM} mm (got {base_mm:g} mm)")
    if max_triangles is None:
        if detail not in TRIANGLE_BUDGETS:
            raise MeshError(f"unknown detail '{detail}', expected one of {sorted(TRIANGLE_BUDGETS)}")
        max_triangles = TRIANGLE_BUDGETS[detail]

    H, W = h.shape
    height_mm = width_mm * H / W
    Wm, Hm = fit_grid(W, H, int(max_triangles))
    hm = h if (Wm, Hm) == (W, H) else _downsample(h, Wm, Hm)

    # Image row 0 is the top of the picture -> y = height_mm (back of the job).
    hm = hm[::-1, :]
    x = np.linspace(0.0, width_mm, Wm, dtype=np.float64)
    y = np.linspace(0.0, height_mm, Hm, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    Z = base_mm + hm.astype(np.float64) * relief_depth_mm
    top = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    idx = np.arange(Hm * Wm).reshape(Hm, Wm)
    a = idx[:-1, :-1].ravel()
    b = idx[:-1, 1:].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[1:, :-1].ravel()
    top_faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)], 0)

    loop = perimeter_loop(Wm, Hm)
    n_top = Hm * Wm
    n_ring = len(loop)
    ring_xy = top[loop, :2]
    ring_bottom = np.column_stack([ring_xy, np.zeros(n_ring)])
    centre = np.array([[width_mm / 2.0, height_mm / 2.0, 0.0]])
    bottom_idx = n_top + np.arange(n_ring)
    centre_idx = n_top + n_ring

    k = np.arange(n_ring)
    k1 = (k + 1) % n_ring
    T, Tn = loop, loop[k1]
    B, Bn = bottom_idx, bottom_idx[k1]
    wall_faces = np.concatenate([np.stack([T, B, Bn], 1), np.stack([T, Bn, Tn], 1)], 0)
    bottom_faces = np.stack([np.full(n_ring, centre_idx), Bn, B], 1)

    vertices = np.vstack([top, ring_bottom, centre])
    faces = np.concatenate([top_faces, wall_faces, bottom_faces], 0).astype(np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    info = {
        "detail": detail,
        "max_triangles": int(max_triangles),
        "grid_w": int(Wm),
        "grid_h": int(Hm),
        "downsampled": (Wm, Hm) != (W, H),
        "mesh_pitch_x_mm": width_mm / (Wm - 1),
        "mesh_pitch_y_mm": height_mm / (Hm - 1),
        "expected_z_max_mm": base_mm + float(hm.max()) * relief_depth_mm,
    }
    return mesh, info


def validate_mesh(
    mesh: trimesh.Trimesh,
    width_mm: float,
    height_mm: float,
    expected_z_max_mm: float,
    tol: float = BBOX_TOL_MM,
) -> dict:
    """Check everything a CAM program needs from the file. `is_valid` is the AND of all checks."""
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    dims = bounds[1] - bounds[0]
    expected = np.array([width_mm, height_mm, expected_z_max_mm], dtype=np.float64)
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    z_min = float(bounds[0, 2])
    z_max = float(bounds[1, 2])
    volume = float(mesh.volume)
    report = {
        "is_watertight": bool(mesh.is_watertight),
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        "volume_mm3": volume,
        "volume_positive": bool(volume > 0),
        "bbox_dims_mm": [round(float(v), 5) for v in dims],
        "bbox_dims_ok": bool(np.all(np.abs(dims[:2] - expected[:2]) <= tol)),
        "z_min_mm": round(z_min, 5),
        "z_max_mm": round(z_max, 5),
        "z_ok": bool(abs(z_min) <= tol and abs(z_max - expected_z_max_mm) <= tol),
        "origin_ok": bool(np.all(np.abs(bounds[0, :2]) <= tol)),
        "has_nan": bool(not np.isfinite(np.asarray(mesh.vertices)).all()),
        "min_face_area_mm2": float(areas.min()),
        "min_face_area_ok": bool(areas.min() >= MIN_FACE_AREA_MM2),
        "n_vertices": int(len(mesh.vertices)),
        "n_triangles": int(len(mesh.faces)),
    }
    report["is_valid"] = bool(
        report["is_watertight"]
        and report["is_winding_consistent"]
        and report["volume_positive"]
        and report["bbox_dims_ok"]
        and report["z_ok"]
        and report["origin_ok"]
        and not report["has_nan"]
        and report["min_face_area_ok"]
    )
    return report


def mesh_to_obj_bytes(mesh: trimesh.Trimesh, precision: int = 6) -> bytes:
    """Fast Wavefront OBJ writer (v + f only). Coordinates in mm, no units in the file."""
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64) + 1
    fmt = "v {0:.%df} {1:.%df} {2:.%df}\n" % (precision, precision, precision)
    v_lines = [fmt.format(p[0], p[1], p[2]) for p in v]
    f_lines = ["f %d %d %d\n" % (int(t[0]), int(t[1]), int(t[2])) for t in f]
    return ("".join(["# ReliefForge relief, coordinates in millimetres\n"] + v_lines + f_lines)).encode("ascii")
