"""Tests for the Relief-Depth converter and the mixed/real datasets."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import realdata as RD
from ml.dataset import MixedReliefDataset, ReliefDepthDataset


def _fake_raw(tmp_path: Path, n_scenes=4):
    """A miniature replica of the downloaded dataset (2 scenes per split)."""
    raw = tmp_path / "raw"
    (raw / "rgb_d_pairs").mkdir(parents=True)
    rng = np.random.default_rng(0)
    ann = ["Dataset_ID,culture,relief_type,material,split,tags_raw"]
    for i in range(n_scenes):
        scene = f"RLF_{i:04d}"
        split = ["train", "train", "val", "test"][i % 4]
        rel_type = ["bas", "sunk"][i % 2]
        ann.append(f"{scene},Ancient Egypt,{rel_type},marble,{split},tag")
        H = W = 96
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d01 = np.full((H, W), 0.6, np.float32)  # slab plane
        bump = 0.25 * np.exp(-(((yy[24:72, 24:72] - 48) ** 2 + (xx[24:72, 24:72] - 48) ** 2) / (2 * 12**2)))
        d01[24:72, 24:72] += (bump if rel_type == "bas" else -bump)
        depth = (np.clip(d01, 0, 1) * 65535).astype(np.uint16)
        lum = (0.4 + 0.5 * (d01 - d01.min())).clip(0, 1)
        for lighting in RD.LIGHTINGS[:2]:
            noise = rng.normal(0, 0.02, (H, W, 1)).astype(np.float32)
            rgb = np.clip(lum[..., None] * (1 + 0.1 * (lighting == "indoor_museum")) + noise, 0, 1)
            cv2.imwrite(str(raw / "rgb_d_pairs" / f"{scene}_{lighting}_rgb.png"), (rgb * 255).astype(np.uint8)[..., ::-1])
            cv2.imwrite(str(raw / "rgb_d_pairs" / f"{scene}_{lighting}_depth.png"), depth)
    (raw / "model_annotations.csv").write_text("\n".join(ann), encoding="utf-8")
    return raw


def test_height_from_depth_handles_bas_and_sunk(tmp_path):
    # bas: figure ABOVE the slab (nearer, larger depth); sunk: below
    d_bas = np.full((64, 64), 40000, np.uint16)
    d_bas[16:48, 16:48] = 55000
    h_bas, info_bas = RD.height_from_depth(d_bas)
    d_sunk = np.full((64, 64), 55000, np.uint16)
    d_sunk[16:48, 16:48] = 40000
    h_sunk, _ = RD.height_from_depth(d_sunk)
    assert h_bas[32, 32] > 0.8 and h_sunk[32, 32] > 0.8  # both map to raised figures
    assert abs(h_bas[32, 32] - h_sunk[32, 32]) < 0.05     # same magnitude, opposite sign
    assert h_bas[2, 2] == pytest.approx(0.0, abs=1e-4)    # slab stays the base plane
    flat, finfo = RD.height_from_depth(np.full((32, 32), 30000, np.uint16))
    assert finfo["flat"] and flat.max() == 0


def test_convert_respects_splits_and_lightings(tmp_path):
    raw = _fake_raw(tmp_path)
    out = RD.convert_reliefdepth(raw, tmp_path / "out", log_every=0)
    rows = [json.loads(l) for l in open(out / "reliefdepth.jsonl")]
    assert len(rows) == 4 * 2  # 4 scenes x 2 lightings
    by_split = {}
    for r in rows:
        by_split.setdefault(r["split"], set()).add(r["scene"])
    assert by_split["train"] == {"RLF_0000", "RLF_0001"}
    for r in rows:
        assert (out / r["split"] / "rgb" / f"{r['id']}.png").exists()
        h = cv2.imread(str(out / r["split"] / "h" / f"{r['id']}.png"), cv2.IMREAD_UNCHANGED)
        assert h.dtype == np.uint16 and int(h.max()) > 30000
    assert RD.qc_reliefdepth(out)["n_pairs"] == 8


def test_convert_excludes_the_exclude_split(tmp_path):
    raw = _fake_raw(tmp_path)
    ann = (raw / "model_annotations.csv").read_text(encoding="utf-8").replace(",train,tag", ",exclude,tag")
    (raw / "model_annotations.csv").write_text(ann, encoding="utf-8")
    out = RD.convert_reliefdepth(raw, tmp_path / "out2", log_every=0)
    rows = [json.loads(l) for l in open(out / "reliefdepth.jsonl")]
    assert all(r["split"] != "train" for r in rows)


def _make_converted(tmp_path, n=3):
    raw = _fake_raw(tmp_path)
    return RD.convert_reliefdepth(raw, tmp_path / "rd", log_every=0)


def _make_synth(tmp_path):
    from tests.test_ml import _make_tiny_ds

    return _make_tiny_ds(tmp_path / "synth", n=6)


def test_reliefdepth_dataset_shapes_and_determinism(tmp_path):
    out = _make_converted(tmp_path)
    ds = ReliefDepthDataset(out, split="test", size=64, augment=False)
    assert len(ds) == 2
    a, b = ds[0], ReliefDepthDataset(out, split="test", size=64, augment=False)[0]
    assert np.array_equal(a["img"].numpy(), b["img"].numpy())
    assert a["img"].shape == (3, 64, 64) and a["gt"].shape == (64, 64)
    assert a["valid"].all() and float(a["kappa"]) > 0
    assert a["meta"]["source"] == "relief-depth"


def test_reliefdepth_train_augmentation_keeps_geometry(tmp_path):
    out = _make_converted(tmp_path)
    ds = ReliefDepthDataset(out, split="train", size=64, augment=True)
    for i in range(len(ds)):
        item = ds[i]
        assert item["img"].shape == (3, 64, 64)
        # after rot90/flip/crop, bright regions must still sit high in gt (isotropy check)
        assert float(item["gt"].max()) > 0.5


def test_mixed_dataset_samples_both_sources(tmp_path):
    out = _make_converted(tmp_path)
    synth = _make_synth(tmp_path)
    ds = MixedReliefDataset(synth, out, size=64, synthetic_frac=0.5, seed=1)
    assert len(ds) == len(ds.real) + len(ds.synth)
    seen = {"synth": 0, "real": 0}
    for i in range(30):
        item = ds[i]
        key = "synth" if item["meta"].get("source", "synth") == "synth" else "real"
        seen[key] += 1
        assert item["img"].shape == (3, 64, 64)
    assert seen["synth"] > 3 and seen["real"] > 3


def test_mixed_dataset_fails_cleanly_without_real_data(tmp_path):
    synth = _make_synth(tmp_path)
    with pytest.raises(FileNotFoundError):
        MixedReliefDataset(synth, tmp_path / "missing", size=64)
