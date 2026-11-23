# ReliefForge — Fine-Tuned 2D Image → 2.5D/3D CNC Relief File Generation

Final project for the **ML & DL course (SICIP, BRAC University)** — category: *Other*.

**What it does:** you upload a 2D decorative design (a flat ornament, B/W line art, or a
photo/render of a carved panel), set the physical size in millimetres, and the system
produces a **CAM-importable 2.5D relief**:

- a 16-bit heightmap (`relief_heightmap_16bit.tiff`, full resolution, 0 = base plane),
- a **watertight solid mesh**: `relief.stl`, `relief.obj`, `relief.3mf` (authored in mm),
- `relief_meta.json` with the physical scale, mesh validation report and warnings,
- a shaded preview and a 3D viewer in the web app.

The ML core is a **fine-tuned Depth Anything V2 Small** (ViT-S, 24.8 M parameters,
Apache-2.0) trained on a paired dataset we generate ourselves: a procedural relief
generator renders (design image → exact height map) pairs at native resolution, so the
supervision is exact. Everything runs on free Kaggle GPUs.

> Height from a single image is **relative and plausible**, not measured geometry. The
> output is a height field `Z = f(X, Y)` — exactly what a 3-axis CNC router carves from
> above. No undercuts, no back side. "AI draft + human review" is the intended workflow.

---

## 1. Install

```bash
git clone https://github.com/<YOUR_USERNAME>/reliefforge.git
cd reliefforge
python -m venv .venv && .venv/Scripts/activate    # Windows; source .venv/bin/activate on Linux
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -r requirements.txt
pytest tests/ -q    # 154 tests (downloads the zero-shot weights on first use)
```

## 2. Dataset (never committed — fetched/built into `data/`)

**Hybrid data plan (v3):** the *primary* training data is the real
**[KarimT/Relief-Depth](https://huggingface.co/datasets/KarimT/Relief-Depth)** dataset —
807 relief scenes (Sketchfab 3D models from the British Museum, MET, …) rendered under 5
lightings each with **depth rendered from the real geometry** (8,070 RGB–depth pairs,
768×768). Our converter turns it into the project's height convention
(`h = |depth − slab plane| / p99.5`, per-scene splits respected: 689 train / 89 val /
84 test). An auxiliary synthetic set (1,500 procedural scenes with exact labels) covers
flat/BW styles and anchors the height convention. Training samples ~65 % real + ~35 % synthetic.

```bash
python src/data_download.py --out data/relief_dataset --n-scenes 1500 --workers 4 \
    --relief-depth --real-out data/relief_depth --textures
```

- `--relief-depth` downloads (~11 GB) and converts the real dataset; `--max-scenes N` for a local test subset.
- `--hf-repo <user>/cnc-relief-depth` pulls a published copy from the Hugging Face Hub instead of generating the synthetic part.
- `--textures` also downloads ~10 CC0 albedo textures (ambientCG, CC0 1.0) into `data/textures`.
- Output: `data/relief_depth/{train,val,test}/{rgb,h}` + `reliefdepth.jsonl` (primary) and
  `data/relief_dataset/{geometry,renders,previews,scenes.jsonl}` (auxiliary).
- **Licence note:** the real dataset's sources are mostly museum CC-BY-NC — fine for this
  course/research project; the client's *commercial* product would need different data.
- Explore the data with `notebooks/exploration.ipynb`.

## 3. Training (MLflow-tracked, two stages)

```bash
python src/train.py --config configs/stage_a.yaml --data data/relief_dataset \
    --real-root data/relief_depth                       # decoder-only warm-up (hybrid data)
python src/train.py --config configs/stage_b.yaml --data data/relief_dataset \
    --real-root data/relief_depth --stage-a-dir artifacts/train_a/export   # full fine-tune
```

Training samples ~65 % real Relief-Depth pairs (native-resolution 518 crops, rot90/flip
augmentation) and ~35 % synthetic views; **validation runs on the real held-out scenes**.

- Stage A freezes the DINOv2 `backbone.*` and trains the DPT `neck.*`/`head.*` (groups are
  built from `named_parameters()` and asserted to cover every parameter).
- Stage B unfreezes everything with a small encoder LR. AMP (fp16), EMA (0.999),
  warmup+cosine, grad clipping, resume checkpoints, `history.csv`.
- Loss: median/MAD scale-shift-invariant normalization (scale is positive by construction),
  weighted L1 + multi-scale gradient matching + a normal loss with a **physical**
  `κ = depth_mm × px_per_mm × view_scale` per sample. All in float32 outside autocast.
- On Kaggle: use the notebooks in `notebooks/kaggle/` (`nb01_generate_dataset` →
  `nb02_train` → `nb03_evaluate`), accelerator **GPU T4 x2 (one GPU used)**, and copy the
  resulting `mlruns/` folder into the repo root.

## 4. Evaluation

```bash
# primary: the real held-out Relief-Depth test split
python src/evaluate.py --data data/relief_dataset --real data/relief_depth --zero-shot
python src/evaluate.py --data data/relief_dataset --real data/relief_depth --model artifacts/train_b/export --label stage_b
# secondary: synthetic exact-label test + the resolution study
python src/evaluate.py --data data/relief_dataset --model artifacts/train_b/export --sizes 518 756 1036
```

Metrics (computed after the production normalization): MAE/RMSE of normalized height,
SSIM, normal angular error, detail retention `R_detail`, Pearson (polarity), with per
family / render style / inference-size breakdowns (`artifacts/eval*/summary.csv`,
`by_family.csv`) and qualitative figures (`input | GT | zero-shot | fine-tuned`).

## 5. Prediction (CLI)

```bash
python src/predict.py design.png out/ --mode B --width 300 --depth 8 --base 6 --tool 1.0
# Mode A (classic, deterministic) for flat/B&W designs:
python src/predict.py logo.png out/ --mode A --profile dome --transition-width 6
```

## 6. Prediction app (Gradio) — served with Docker

```bash
docker build -t reliefforge-app:1.0 .
docker run -p 7860:7860 reliefforge-app:1.0            # zero-shot weights are baked in
# with the fine-tuned export:
docker run -p 7860:7860 -v /path/to/export:/models/finetuned reliefforge-app:1.0
```

then open <http://localhost:7860>. Upload an image, set the sliders, **Generate relief**,
inspect the 3D preview and download the ZIP. The same UI runs locally without Docker:
`python app/app.py`. Screenshots: `screenshots/prediction_ui.png`, `screenshots/docker_app.png`.

## 7. MLflow experiment tracking

`mlruns/` **is committed** to the repository as evidence (only weight binaries are
git-ignored; they are reproducible and exported as MLflow artifacts).

```bash
export MLFLOW_ALLOW_FILE_STORE=true    # needed by mlflow >= 3.18 for the file store
mlflow ui --backend-store-uri file:///$(pwd)/mlruns   # http://127.0.0.1:5000
```

Experiments: `ReliefForge-Finetune` (`a_smoke…`, `b_smoke…`, `stage_a…`, `stage_b…`) and
`ReliefForge-Eval` (`zeroshot…`, `stage_b…`). Each run logs the full config, per-100-step
train metrics (`train_loss, L_h, L_g, L_n, lr_*, images_per_s`), validation metrics
(`val_MAE_norm, val_normal_err_deg, val_pearson`), `history.csv`, the eval tables and the
model export. Screenshot: `screenshots/mlflow_runs.png`.

## 8. Course techniques used (and where)

| Technique | Where |
|---|---|
| Transfer learning / fine-tuning a pretrained ViT | `src/ml/trainer.py` (Stage A decoder-only → Stage B full fine-tune of Depth Anything V2 Small) |
| Transformer backbone (DINOv2 ViT-S) + DPT head | `src/ml/trainer.py`, `src/reliefkit/model.py` |
| Custom loss design (scale-shift-invariant regression, multi-scale gradients, geometry-aware normal loss) | `src/ml/losses.py` |
| Real-data fine-tuning + synthetic augmentation (hybrid) | `src/ml/realdata.py`, `src/ml/dataset.py` (Relief-Depth primary, generator auxiliary) |
| Synthetic data generation / domain randomization | `src/synth/` (5 families, 4 render styles, on-the-fly augmentation) |
| Data pipeline & augmentation (aspect-preserving views, crop/full sampling, letterboxing) | `src/ml/dataset.py` |
| AMP mixed precision, EMA, warmup + cosine schedule, resume | `src/ml/trainer.py` |
| MLflow experiment tracking | `src/ml/tracking.py` (used by `src/train.py`, `src/evaluate.py`) |
| Evaluation & error analysis (metrics, per-family/style/resolution breakdowns) | `src/ml/metrics.py`, `src/ml/evaluate.py` |
| Dockerized prediction app with upload UI | `Dockerfile`, `app/app.py` (Gradio) |
| Geometric validation / engineering QA (watertight meshes, exact mm dimensions) | `src/reliefkit/mesh.py`, `tests/test_mesh.py` |

## 9. Repository structure

```
README.md  final_report.md  requirements.txt  .gitignore  .gitattributes  Dockerfile
configs/            config.yaml, stage_a.yaml, stage_b.yaml
src/
  data_download.py  generate_dataset.py  train.py  evaluate.py  predict.py
  reliefkit/        io, preprocess, modes, model, postprocess, mesh, export, pipeline, preview
  synth/            shapes, compose, recipes, render, generate
  ml/               dataset, losses, ema, tracking, trainer, metrics, evaluate
app/app.py          Gradio prediction UI (served with Docker)
notebooks/          exploration.ipynb, kaggle/{nb01,nb02,nb03}
tests/              154 pytest tests (mesh, preprocess, modes, postprocess, pipeline, synth, ml)
mlruns/             MLflow evidence (committed)
screenshots/        mlflow_runs.png, prediction_ui.png, docker_app.png
models/  data/  artifacts/   git-kept empty; weights/data are never committed
report/             SICIP_NAME_ID.docx + SICIP_NAME_ID.pdf (filled faculty template)
docs/               architecture.md, memory.md, plan.md, log.md
```

## 10. Reproduce from a fresh clone

1. install dependencies (§1), 2. fetch/build the dataset (§2), 3. train Stage A then
Stage B (§3), 4. inspect the MLflow runs (§7), 5. evaluate (§4), 6. build and run the
Dockerized prediction app (§6). Expected compute: dataset ~10 min CPU, training
~2–3 h on one Kaggle T4, evaluation ~20 min GPU.

## 11. Units (important for CAD)

STL and OBJ **do not carry units** — ReliefForge authors the coordinates in millimetres;
set the import unit to mm. 3MF carries `unit="millimeter"` (written by trimesh). The TIFF
has no physical Z range: scale Z to `relief_depth_mm` using `relief_meta.json`
(`0 = base plane, 65535 = highest point`).

## 12. Known limitations

- Single-image height is **relative and plausible**, not measured. Sliders (depth,
  invert, smoothing, transition width) plus human review are part of the workflow.
- Supported input domain: ornamental panel styles (floral, botanical, scroll, border,
  medallion), flat designs and B/W line art, near-frontal photos/renders of carved
  panels. **Faces, figures, animals, lettering and general 3D-scene photos are out of
  distribution** for the fine-tuned model — use Mode B zero-shot or Mode C there and
  expect weaker results.
- Strongly perspective photos (a phone shot from an angle) are not rectified yet —
  near-orthogonal / front-facing inputs work best.
- No undercuts (height field by construction), no G-code/toolpaths, no proprietary CAM
  formats (`.rlf`, `.3dClip`, `.jdp`), no STEP export.
- Not carving-validated: no physical CNC test was run for this course project.

## 13. Licences / data sources

| Item | Licence | Use |
|---|---|---|
| Depth Anything V2 Small | Apache-2.0 | base + fine-tuned weights (commercial OK) |
| Our procedural dataset | generated by `src/synth/` (this repo) | train/val/test (ours) |
| ambientCG textures | CC0 1.0 | render albedos only |
| Procedural wood fallback | generated by `src/synth/render.py` | render albedos |
| Relief-Depth (KarimT) | source models mostly CC-BY-NC (museums, via Sketchfab) | ✅ training for this course/research project — ❌ NOT for the client's commercial product |
| RRD Borobudur, GrabCAD | non-commercial / per-model | not used (optional qualitative test only) |

## 14. Report

`final_report.md` is the report source; the filled faculty template and the PDF live in
`report/SICIP_NAME_ID.pdf`.
