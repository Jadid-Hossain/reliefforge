"""M5 — the solid must be watertight and exactly the requested size, every time."""
import numpy as np
import pytest
import trimesh

from reliefkit import mesh as M


def _gauss(shape, cy=None, cx=None, sigma=None):
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cy = (H - 1) / 2 if cy is None else cy
    cx = (W - 1) / 2 if cx is None else cx
    sigma = min(H, W) / 8 if sigma is None else sigma
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)).astype(np.float32)


def _build(h, width_mm=300.0, depth=8.0, base=6.0, detail="standard", **kw):
    mesh, info = M.build_relief_mesh(h, width_mm, depth, base, detail=detail, **kw)
    H, W = h.shape
    rep = M.validate_mesh(mesh, width_mm, width_mm * H / W, info["expected_z_max_mm"])
    return mesh, info, rep


# ---------------------------------------------------------------- boxes
def test_flat_height_is_an_exact_box():
    h = np.zeros((48, 80), np.float32)
    mesh, info, rep = _build(h, width_mm=300, depth=8, base=6)
    assert rep["is_valid"], rep
    assert rep["bbox_dims_mm"] == pytest.approx([300.0, 180.0, 6.0], abs=1e-4)
    assert rep["volume_mm3"] == pytest.approx(300 * 180 * 6, rel=1e-6)
    assert not info["downsampled"]


def test_full_plateau_is_an_exact_taller_box():
    h = np.ones((48, 80), np.float32)
    mesh, info, rep = _build(h, width_mm=300, depth=8, base=6)
    assert rep["is_valid"], rep
    assert rep["bbox_dims_mm"] == pytest.approx([300.0, 180.0, 14.0], abs=1e-4)
    assert rep["volume_mm3"] == pytest.approx(300 * 180 * 14, rel=1e-6)


def test_half_height_plateau_scales_with_relief_depth():
    for depth in (2.0, 8.0, 20.0):
        _, _, rep = _build(np.full((20, 30), 0.5, np.float32), width_mm=120, depth=depth, base=3)
        assert rep["is_valid"]
        assert rep["z_max_mm"] == pytest.approx(3 + 0.5 * depth, abs=1e-4)


# ---------------------------------------------------------------- geometry
def test_bump_volume_grows_with_relief_depth_and_matches_integral():
    h = _gauss((96, 128))
    vols = []
    for depth in (2.0, 4.0, 8.0, 16.0):
        _, _, rep = _build(h, width_mm=256, depth=depth, base=5)
        assert rep["is_valid"], rep
        vols.append(rep["volume_mm3"])
    assert np.all(np.diff(vols) > 0)
    # volume = base slab + relief_depth * integral of h over the panel (piecewise-linear surface)
    W, H = 256, 192
    cell = (W / 127) * (H / 95)
    integral = h.sum() * cell  # each node owns one cell-area unit
    for depth, v in zip((2.0, 4.0, 8.0, 16.0), vols):
        assert v == pytest.approx(W * H * 5 + depth * integral, rel=0.01)


def test_volume_is_linear_in_relief_depth():
    h = _gauss((64, 64))
    _, _, r1 = _build(h, width_mm=100, depth=5, base=2)
    _, _, r2 = _build(h, width_mm=100, depth=10, base=2)
    assert (r2["volume_mm3"] - 100 * 100 * 2) == pytest.approx(2 * (r1["volume_mm3"] - 100 * 100 * 2), rel=1e-5)


@pytest.mark.parametrize("seed", range(50))
def test_random_heightmaps_are_always_watertight(seed):
    rng = np.random.default_rng(seed)
    H, W = int(rng.integers(4, 70)), int(rng.integers(4, 90))
    kind = seed % 3
    if kind == 0:
        h = rng.random((H, W)).astype(np.float32)
    elif kind == 1:  # blocky
        h = (rng.random((H, W)) > 0.5).astype(np.float32) * rng.random()
    else:  # smooth
        h = rng.random((H, W)).astype(np.float32)
        for _ in range(3):
            h = (h + np.roll(h, 1, 0) + np.roll(h, 1, 1) + np.roll(h, -1, 0) + np.roll(h, -1, 1)) / 5
    width = float(rng.uniform(20, 2000))
    _, _, rep = _build(h, width_mm=width, depth=float(rng.uniform(0.5, 100)), base=float(rng.uniform(0.5, 100)))
    assert rep["is_valid"], (seed, rep)


@pytest.mark.parametrize("shape", [(518, 518), (6, 518), (518, 6), (3, 1036), (1036, 3), (2, 2), (2, 900)])
def test_extreme_aspect_ratios_are_watertight(shape):
    h = rng_height(shape, 0)
    _, _, rep = _build(h, width_mm=2000, depth=10, base=1)
    assert rep["is_valid"], (shape, rep)


def test_every_vertex_is_used_and_unique():
    h = _gauss((30, 40))
    mesh, _, rep = _build(h, width_mm=80)
    assert rep["is_valid"]
    used = np.unique(mesh.faces)
    assert len(used) == len(mesh.vertices) == 30 * 40 + len(M.perimeter_loop(40, 30)) + 1
    # no duplicate positions (two vertices at the same spot would mean a cracked seam)
    v = np.round(mesh.vertices, 7)
    assert len(np.unique(v, axis=0)) == len(v)


# ---------------------------------------------------------------- orientation
def test_top_of_image_is_back_of_the_panel():
    # bump placed on the first image rows (top of the picture)
    h = _gauss((64, 64), cy=4, sigma=6)
    mesh, info, rep = _build(h, width_mm=100)
    assert rep["is_valid"]
    top_z = mesh.vertices[: 64 * 64, 2].reshape(64, 64)  # mesh row j <-> y = j * pitch
    j, i = np.unravel_index(int(np.argmax(top_z)), top_z.shape)
    assert j == 63 - 4  # image row 4 is near the top of the picture -> back of the panel
    pitch = 100.0 / 63
    assert mesh.vertices[j * 64 + i][1] == pytest.approx(59 * pitch, abs=1e-9)
    assert abs(mesh.vertices[j * 64 + i][0] - 31.5 * pitch) <= pitch  # column is not mirrored
    assert mesh.vertices[0][1] == 0 and mesh.vertices[63 * 64][1] == pytest.approx(100.0)
    assert mesh.vertices[0][0] == 0 and mesh.vertices[63][0] == pytest.approx(100.0)


def test_normals_point_outward():
    h = _gauss((20, 20), sigma=6)
    mesh, _, rep = _build(h, width_mm=50)
    assert rep["is_valid"] and rep["volume_positive"]
    n_top = 20 * 20
    bottom = mesh.faces[mesh.faces[:, 0] == n_top + len(M.perimeter_loop(20, 20))]
    assert len(bottom) == len(M.perimeter_loop(20, 20))
    assert (mesh.face_normals[mesh.faces[:, 0] == n_top + len(M.perimeter_loop(20, 20))][:, 2] < -0.99).all()


# ---------------------------------------------------------------- budget
@pytest.mark.parametrize("detail,budget", [("draft", 250_000), ("standard", 600_000), ("high", 2_000_000)])
def test_triangle_budget_is_respected_and_bbox_stays_exact(detail, budget):
    h = rng_height((1036, 1036), 1)
    _, info, rep = _build(h, width_mm=400, detail=detail)
    assert rep["is_valid"], rep
    assert rep["n_triangles"] <= budget
    assert rep["bbox_dims_mm"][:2] == pytest.approx([400.0, 400.0], abs=1e-6)
    assert info["downsampled"] == (detail != "high" or rep["n_triangles"] < 2_100_000)


def test_custom_budget():
    h = rng_height((400, 300), 2)
    mesh, info, rep = _build(h, width_mm=300, max_triangles=20_000)
    assert rep["is_valid"] and rep["n_triangles"] <= 20_000
    assert info["grid_w"] < 400 and info["downsampled"]


# ---------------------------------------------------------------- file round trips
def test_stl_round_trip(tmp_path):
    h = _gauss((90, 120))
    mesh, info, rep = _build(h, width_mm=240)
    assert rep["is_valid"]
    p = tmp_path / "relief.stl"
    mesh.export(p, file_type="stl")
    assert p.stat().st_size == 84 + 50 * len(mesh.faces)  # binary STL
    back = trimesh.load(p, process=True, force="mesh")
    rep2 = M.validate_mesh(back, 240.0, 180.0, info["expected_z_max_mm"])
    assert rep2["is_valid"], rep2


def test_obj_round_trip(tmp_path):
    h = _gauss((90, 120))
    mesh, info, rep = _build(h, width_mm=240)
    assert rep["is_valid"]
    p = tmp_path / "relief.obj"
    p.write_bytes(M.mesh_to_obj_bytes(mesh))
    back = trimesh.load(p, process=True, force="mesh")
    rep2 = M.validate_mesh(back, 240.0, 180.0, info["expected_z_max_mm"], tol=1e-4)
    assert rep2["is_valid"], rep2
    assert rep2["volume_mm3"] == pytest.approx(rep["volume_mm3"], rel=1e-6)


def test_3mf_round_trip_has_millimetre_units(tmp_path):
    h = _gauss((40, 50))
    mesh, info, rep = _build(h, width_mm=100)
    assert rep["is_valid"]
    p = tmp_path / "relief.3mf"
    mesh.export(p, file_type="3mf")
    loaded = trimesh.load(p, force="mesh")
    rep2 = M.validate_mesh(loaded, 100.0, 80.0, info["expected_z_max_mm"], tol=1e-4)
    assert rep2["is_valid"], rep2
    assert "millimeter" in str(loaded.units).lower() or loaded.units is None
    import zipfile

    with zipfile.ZipFile(p) as z:
        model = [n for n in z.namelist() if n.endswith(".model")][0]
        assert 'unit="millimeter"' in z.read(model).decode("utf8")


# ---------------------------------------------------------------- guards
def test_base_below_half_mm_is_rejected():
    with pytest.raises(M.MeshError, match="base"):
        M.build_relief_mesh(np.zeros((10, 10), np.float32), 100, 8, 0.0)


@pytest.mark.parametrize("bad", [np.full((10, 10), np.nan, np.float32), np.full((10, 10), 1.5, np.float32), np.full((10, 10), -0.2, np.float32)])
def test_invalid_height_maps_are_rejected(bad):
    with pytest.raises(M.MeshError):
        M.build_relief_mesh(bad, 100, 8, 2)


def test_invalid_dimensions_are_rejected():
    h = np.zeros((10, 10), np.float32)
    with pytest.raises(M.MeshError):
        M.build_relief_mesh(h, 0, 8, 2)
    with pytest.raises(M.MeshError):
        M.build_relief_mesh(h, 100, 0, 2)
    with pytest.raises(M.MeshError):
        M.build_relief_mesh(h, 100, 8, 2, detail="ultra")
    with pytest.raises(M.MeshError):
        M.build_relief_mesh(np.zeros((10,), np.float32), 100, 8, 2)


def test_perimeter_loop_is_a_closed_ring():
    for W, H in ((5, 4), (2, 2), (2, 9), (9, 2), (40, 30)):
        loop = M.perimeter_loop(W, H)
        assert len(loop) == 2 * (W - 1) + 2 * (H - 1) == len(np.unique(loop))
        pts = np.unravel_index(loop, (H, W))
        pts = np.stack(pts, 1)
        # every consecutive pair (and the wrap-around) is a unit step
        d = np.abs(np.diff(np.vstack([pts, pts[:1]]), axis=0)).sum(1)
        assert (d == 1).all(), (W, H, d)
        assert M.count_triangles(W, H) == 2 * (W - 1) * (H - 1) + 3 * len(loop)


def rng_height(shape, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    h = 0.5 + 0.5 * np.sin(xx / 9.0 + seed) * np.cos(yy / 7.0)
    return np.clip(h + 0.05 * rng.random(shape, dtype=np.float32), 0, 1).astype(np.float32)
