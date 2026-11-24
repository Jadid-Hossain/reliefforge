"""Client-data importer tests: mesh->height projection, pair import, direction QC."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import clientdata as CD


def _bump_mesh(path: Path, base=20.0, bump_r=0.25, bump_h=8.0, size=1.0):
    """A Z-up square slab with a dome on top (the classic CNC relief layout)."""
    import trimesh

    slab = trimesh.creation.box(extents=(size, size, base))
    slab.apply_translation((0, 0, base / 2))
    dome = trimesh.creation.icosphere(subdivisions=3, radius=bump_r)
    dome.apply_translation((0, 0, base))  # half inside the slab, dome sticks out
    mesh = trimesh.util.concatenate([slab, dome])
    path.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    return path


def test_mesh_to_height_projects_max_z_with_base_at_zero(tmp_path):
    mesh_p = _bump_mesh(tmp_path / "relief1.stl")
    h = CD.mesh_to_height(mesh_p, out_long_side=256)
    assert h.shape == (256, 256)
    assert h.min() == pytest.approx(0.0, abs=1e-6)  # the flat base plane
    assert h.max() == pytest.approx(1.0, abs=0.05)  # the dome peak
    H, W = h.shape
    assert h[H // 2, W // 2] > 0.5 and h[5, 5] < 0.05  # dome centre raised, corner flat


def test_mesh_with_undercut_collapses_to_height_field(tmp_path):
    import trimesh

    # an undercut mushroom: max-Z projection must only see the visible height field
    stem = trimesh.creation.cylinder(radius=0.1, height=0.5)
    stem.apply_translation((0, 0, 0.25))
    cap = trimesh.creation.cylinder(radius=0.4, height=0.2)
    cap.apply_translation((0, 0, 0.55))
    mesh = trimesh.util.concatenate([stem, cap])
    p = tmp_path / "mush.stl"
    p.write_bytes(trimesh.exchange.stl.export_stl(mesh))
    h = CD.mesh_to_height(p, out_long_side=128)
    H, W = h.shape
    assert h.max() == pytest.approx(1.0, abs=0.05)
    assert h[H // 2, W // 2] == pytest.approx(1.0, abs=0.05)      # cap centre is the peak
    assert h[3, 3] < 0.1 and h[H - 4, W - 4] < 0.1                 # corners: far outside cap+stem
    assert 0.0 <= h.min() < 1e-6


def test_import_pairs_and_direction_check(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    h = np.exp(-(((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 20**2)))
    rgb = (np.stack([0.2 + 0.7 * h] * 3, -1) * 255).astype(np.uint8)  # bright = raised
    cv2.imwrite(str(src / "design1.png"), rgb[..., ::-1])
    cv2.imwrite(str(src / "design1_h.png"), (h * 65535).astype(np.uint16))
    out = tmp_path / "unified"
    rep = CD.import_client_data(src, out)
    assert rep["imported"] == 1 and rep["pairs"] == 1
    row = [json.loads(l) for l in open(out / "reliefdepth.jsonl")]
    assert row[0]["source"] == "client" and row[0]["id"] == "CLIENT_design1"
    saved = cv2.imread(str(out / "train" / "h" / "CLIENT_design1.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    assert float(np.corrcoef(saved.ravel(), h.ravel())[0, 1]) > 0.99
    assert rep["direction"]["design1_h.png"] > 0.5  # luminance agrees with height


def test_import_dark_raised_gets_flagged_and_can_flip(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    h = np.exp(-(((yy - 64) ** 2 + (xx - 64) ** 2) / (2 * 20**2)))
    rgb = (np.stack([0.9 - 0.7 * h] * 3, -1) * 255).astype(np.uint8)  # dark = raised
    cv2.imwrite(str(src / "dark.png"), rgb[..., ::-1])
    cv2.imwrite(str(src / "dark_h.png"), (h * 65535).astype(np.uint16))
    out = tmp_path / "u2"
    rep = CD.import_client_data(src, out, auto_flip=False)
    assert rep["direction"]["dark_h.png"] < -0.5 and rep["flipped"] == []
    rep2 = CD.import_client_data(src, tmp_path / "u3", auto_flip=True)
    assert rep2["flipped"] == ["dark_h.png"]
    saved = cv2.imread(str(tmp_path / "u3" / "train" / "h" / "CLIENT_dark.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535
    assert float(np.corrcoef(saved.ravel(), h.ravel())[0, 1]) < -0.99  # inverted export, now dark=raised like the image


def test_import_meshes_renders_input(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    _bump_mesh(src / "ornament.stl")
    out = tmp_path / "u4"
    rep = CD.import_client_data(src, out, render_missing_inputs=True)
    assert rep["imported"] == 1 and rep["meshes"] == 1
    rgb = cv2.imread(str(out / "train" / "rgb" / "CLIENT_ornament.png"))
    h = cv2.imread(str(out / "train" / "h" / "CLIENT_ornament.png"), cv2.IMREAD_UNCHANGED)
    assert rgb is not None and rgb.std() > 3  # a shaded render, not a flat image
    assert int(h.max()) > 30000


def test_import_appends_to_existing_reliefdepth_rows(tmp_path):
    out = tmp_path / "u5"
    (out / "train" / "rgb").mkdir(parents=True)
    (out / "train" / "h").mkdir(parents=True)
    with open(out / "reliefdepth.jsonl", "w") as f:
        f.write(json.dumps({"id": "RLF_1", "scene": "RLF_1", "split": "train", "lighting": "x", "relief_type": "bas", "culture": "x", "material": "x", "W": 64, "H": 64, "width_mm_scene": 300.0, "relief_depth_mm_scene": 8.0, "px_per_mm_native": 1.0, "source": "relief-depth"}) + "\n")
    src = tmp_path / "in"
    src.mkdir()
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    h = np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 15**2)))
    cv2.imwrite(str(src / "extra.png"), (np.stack([0.2 + 0.7 * h] * 3, -1) * 255).astype(np.uint8)[..., ::-1])
    cv2.imwrite(str(src / "extra_h.png"), (h * 65535).astype(np.uint16))
    rep = CD.import_client_data(src, out)
    rows = [json.loads(l) for l in open(out / "reliefdepth.jsonl")]
    assert rep["imported"] == 1 and len(rows) == 2
    assert {r["source"] for r in rows} == {"relief-depth", "client"}
    from ml.dataset import ReliefDepthDataset

    ds = ReliefDepthDataset(out, split="train", size=64, augment=False)
    assert len(ds) == 2  # both usable by the same training pipeline


def test_flat_height_is_skipped(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    cv2.imwrite(str(src / "flat.png"), np.full((64, 64, 3), 200, np.uint8))
    cv2.imwrite(str(src / "flat_h.png"), np.full((64, 64), 30000, np.uint16))
    rep = CD.import_client_data(src, tmp_path / "u6")
    assert rep["imported"] == 0 and any("flat" in s for s in rep["skipped"])
