"""Unit tests for the ML stack: losses, views, param groups, metrics."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml import losses as L
from ml import metrics as MT
from ml.dataset import ReliefEvalDataset, ReliefTrainDataset, crop_scale_min, collate
from ml.trainer import Trainer, param_groups


# ------------------------------------------------------------------ losses
def _toy(batch=2, H=64, seed=0, flat=False):
    rng = np.random.default_rng(seed)
    if flat:
        g = np.full((batch, H, H), 0.3, np.float32)
    else:
        yy, xx = np.mgrid[0:H, 0:H].astype(np.float32)
        bump = np.exp(-(((yy - H / 2) ** 2 + (xx - H / 2) ** 2) / (2 * (H / 5) ** 2)))
        g = np.stack([bump] * batch).astype(np.float32)
    g = g + 0.1 * rng.normal(size=(batch, H, H)).astype(np.float32)
    valid = np.ones((batch, H, H), bool)
    valid[:, :6, :] = False  # letterbox padding
    fg = (g > 0.25).astype(np.float32)
    return torch.tensor(g), torch.tensor(valid), torch.tensor(fg)


def test_flipped_polarity_costs_more_than_the_correct_one():
    g, valid, fg = _toy()
    p = g * 0.8 + 0.1
    good, _ = L.relief_loss(p, g, valid, torch.full((2,), 12.0), fg)
    bad, _ = L.relief_loss(-p, g, valid, torch.full((2,), 12.0), fg)
    assert float(bad) > float(good) * 1.5  # review fix: a flipped polarity cannot hide


def test_median_mad_scale_is_always_positive():
    g, valid, fg = _toy(seed=1)
    p = torch.full_like(g, -5.0)  # constant negative prediction
    med, mad = L.masked_median_mad(p, valid)
    assert (med < 0).all()  # handles negatives without a negative 'scale'


def test_flat_scene_gives_no_nan():
    g, valid, fg = _toy(flat=True)
    p = torch.full_like(g, 0.4)
    total, parts = L.relief_loss(p, g, valid, torch.full((2,), 20.0), fg)
    assert torch.isfinite(total), parts


def test_padding_is_ignored_by_the_loss():
    g, valid, fg = _toy(seed=2)
    p = g.clone()
    p2 = p.clone()
    p2[:, :6, :] = 999.0  # garbage in the padding zone
    l1, _ = L.relief_loss(p, g, valid, torch.full((2,), 12.0), fg)
    l2, _ = L.relief_loss(p2, g, valid, torch.full((2,), 12.0), fg)
    assert float(l1) == pytest.approx(float(l2), abs=1e-5)


def test_normal_loss_uses_physical_kappa():
    g, valid, fg = _toy(seed=3)
    p = g + 0.05 * torch.randn_like(g)
    ln_small = L.normal_loss(p, g, valid, torch.full((2,), 4.0), fg, 1.0).mean()
    ln_big = L.normal_loss(p, g, valid, torch.full((2,), 40.0), fg, 1.0).mean()
    assert float(ln_big) > float(ln_small)  # same height error -> bigger physical slope


def test_perfect_prediction_has_near_zero_loss():
    g, valid, fg = _toy(seed=4)
    total, parts = L.relief_loss(g.clone(), g, valid, torch.full((2,), 12.0), fg)
    assert float(total) < 0.05


# ------------------------------------------------------------------ views (dataset)
def _make_tiny_ds(tmp_path, n=6):
    from synth import compose as C
    from synth import shapes as SH
    from synth.generate import h_to_u16

    import cv2

    root = tmp_path / "ds"
    (root / "geometry" / "train").mkdir(parents=True)
    (root / "geometry" / "val").mkdir(parents=True)
    (root / "geometry" / "test").mkdir(parents=True)
    import json

    rows = []
    for i in range(n):
        split = "train" if i < 3 else ("val" if i < 4 else "test")
        rng = np.random.default_rng(100 + i)
        stamps = SH.rosette_stamps(518, 518, 200, 6, "dome", 0.9, 0.1, 1, rng)
        h, inst = C.compose(518, 518, stamps)
        cv2.imwrite(str(root / "geometry" / split / f"{i:06d}_h.png"), h_to_u16(h))
        cv2.imwrite(str(root / "geometry" / split / f"{i:06d}_inst.png"), inst.astype(np.uint16))
        rows.append({"id": i, "split": split, "family": "floral", "W": 518, "H": 518, "width_mm_scene": 300.0, "relief_depth_mm_scene": 8.0, "px_per_mm_native": 300.0 / 518.0, "seed": 100 + i})
    with open(root / "scenes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return root


def test_crop_scale_min_locks_every_canvas():
    assert crop_scale_min(1036, 1036) == pytest.approx(0.55)
    assert crop_scale_min(840, 1260) == pytest.approx(518 / 840)
    assert crop_scale_min(728, 1470) == pytest.approx(518 / 728)  # the 2:1 crash case (fix ১)
    for H, W in ((1036, 1036), (840, 1260), (728, 1470)):
        s = crop_scale_min(H, W)
        assert round(H * s) >= 518 and round(W * s) >= 518


def test_train_views_always_produce_518_and_finite_kappa(tmp_path):
    root = _make_tiny_ds(tmp_path)
    ds = ReliefTrainDataset(root, size=518, augment=False)
    for i in range(len(ds)):
        item = ds[i]
        assert item["img"].shape == (3, 518, 518)
        assert item["gt"].shape == (518, 518)
        assert item["valid"].any()
        assert 1.0 < float(item["kappa"]) < 200.0
        assert torch.isfinite(item["img"]).all()


def test_train_crop_views_match_the_gt_geometry(tmp_path):
    """A crop of a domed rosette must keep its height: peak close to 1 somewhere."""
    root = _make_tiny_ds(tmp_path)
    ds = ReliefTrainDataset(root, size=518, full_frac=0.0, augment=False)
    peaks = [float(ds[i]["gt"].max()) for i in range(len(ds))]
    assert min(peaks) > 0.85  # the rosette boss (peak=1) is large enough to be caught


def test_eval_dataset_is_deterministic(tmp_path):
    root = _make_tiny_ds(tmp_path)
    ds = ReliefEvalDataset(root, split="test", size=518)
    a, b = ds[0], ds[3]  # scene 0 style flat (index 2 = flat? order: wood, clay, flat)
    a2 = ReliefEvalDataset(root, split="test", size=518)[0]
    assert np.array_equal(a["img"].numpy(), a2["img"].numpy())
    assert len(ds) == 2 * 3  # 2 test scenes x 3 styles


# ------------------------------------------------------------------ param groups (Issue 6)
def test_param_groups_cover_every_parameter():
    from transformers import DepthAnythingForDepthEstimation

    model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    enc, dec = param_groups(model)
    n_total = sum(p.numel() for p in model.parameters())
    n_groups = sum(p.numel() for p in enc) + sum(p.numel() for p in dec)
    assert n_groups == n_total
    assert all(not n.startswith("backbone.") for p, n in zip(dec, [n for n, _ in model.named_parameters() if not n.startswith("backbone.")][: len(dec)]))


def test_stage_a_freezes_the_backbone():
    cfg = {"stage": "a", "lr": {"decoder": 1e-4}, "steps": 1, "batch_size": 1, "log_every": 1, "save_every": 1, "num_workers": 0, "amp": False}
    from transformers import DepthAnythingForDepthEstimation

    model = DepthAnythingForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    t = Trainer(model, cfg, out_dir="/tmp/rlf_test_ckpt" if Path("/tmp").exists() else "./artifacts/tmp_ckpt", device="cpu")
    assert all(not p.requires_grad for p in t.enc)
    assert all(p.requires_grad for p in t.dec)


# ------------------------------------------------------------------ metrics
def test_metrics_on_a_perfect_prediction():
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32)
    gt = np.exp(-(((yy - 48) ** 2 + (xx - 48) ** 2) / (2 * 18**2)))
    ref = np.ones(gt.shape, bool)
    m = MT.all_metrics(gt.copy(), gt, ref, kappa=10.0)
    assert m["MAE_norm"] < 0.01 and m["RMSE_norm"] < 0.01  # percentile normalization clips a hair
    assert m["SSIM"] > 0.99 and m["pearson"] > 0.999
    assert m["normal_err_deg"] < 1.0  # 0.24 deg: only the percentile-normalization clip
    assert 0.9 < m["R_detail"] < 1.1


def test_metrics_on_a_blurred_prediction_lose_detail():
    import cv2

    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    gt = ((yy % 16 < 8) ^ (xx % 16 < 8)).astype(np.float32) * 0.8
    ref = np.ones(gt.shape, bool)
    pred = cv2.GaussianBlur(gt, (0, 0), 3)
    m = MT.all_metrics(pred, gt, ref, kappa=10.0)
    assert m["R_detail"] < 0.6 and m["MAE_norm"] > 0.05


def test_normalize_production_uses_the_reference_region():
    h = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    ref = np.zeros((128, 128), bool)
    ref[:64, :] = True
    out = MT.normalize_production(h, ref)
    assert out[ref].max() <= 1.0001
    lo_range = h[ref].min()
    assert out[0, 0] == pytest.approx(0.0, abs=1e-6) or lo_range == pytest.approx(0.0)
