# ReliefForge — Architecture and Technical Specification

**Status:** living document — the implementation-level source of truth.
**Precedence:** `resources/instructions.pdf` (faculty) → `PRD_v2.md` + review fixes (2026-11-23) → this file.
**Change rule:** if code and this file disagree, fix whichever is wrong and add a line to `log.md`.

---

## 1. What the system is

```
 2D decorative design (JPG/PNG/WEBP)  ──►  relief height map h∈[0,1]  ──►  watertight 2.5D solid
                                         (Mode A classic | Mode B AI  |        STL / OBJ / 3MF
                                          Mode C gray→height)                + 16-bit TIFF + JSON sidecar
```

* Output is a **height field** `Z = f(X, Y)` — exactly what a 3-axis CNC router carves from above. No undercuts, no back side.
* The ML core is a **fine-tuned Depth Anything V2 Small** (ViT-S, 24.8 M params, Apache-2.0) trained on a paired dataset we generate ourselves (design image → exact relief height).
* Terminology: the output is **CAM-importable**, not "CNC-ready" (no toolpaths, no carving validation yet).

### 1.1 Requirement traceability (faculty instructions → where satisfied)

| Requirement (instructions.pdf) | Where |
|---|---|
| Valid domain / "Other" category | Form text: `docs/plan.md` §7. User must confirm "Other" is accepted |
| Public GitHub repo, meaningful commits | Local git repo, small commits per module (`docs/log.md`) |
| Complete README.md | `README.md` |
| No dataset in repo, `data/` ignored, `data/.gitkeep` | `.gitignore` |
| Code fetches data automatically into `data/` | `src/data_download.py` |
| MLflow: params, metrics, model artifacts, plots | `src/ml/tracking.py`, used by `src/train.py`, `src/evaluate.py` |
| ≥ 2 runs with different settings | Stage A (decoder-only) and Stage B (full fine-tune); optional ablation run |
| `mlruns/` committed, not ignored | `.gitignore` (only weight binaries excluded, see §8.5) |
| Dockerized prediction app with upload UI | `Dockerfile`, `app/app.py` (Gradio) |
| Screenshots (MLflow, prediction, Docker) | `screenshots/` |
| Course techniques explicitly listed | `README.md` §Course techniques |
| Final report in the faculty template + PDF | `final_report.md`, `report/SICIP_NAME_ID.docx` → `SICIP_NAME_ID.pdf` |
| Repo structure | §3 below |

---

## 2. System overview

```
            ┌──────────────────────────── INFERENCE (reliefkit) ────────────────────────────┐
 image ──►  M2 preprocess ──► M3 mode ──────────────► M4 postprocess ──► M5 mesh ──► M6 export ──► M7 Gradio
            mask, components,   A design→height          normalize, invert,   watertight   STL OBJ 3MF    upload → preview
            letterbox,          B fine-tuned DAv2-S      gamma, bilateral,    + validate    TIFF JSON     → 3D view → ZIP
            suggest_mode        C gray→height            smoothing, feather
            └──────────────┬─────────────────────────────────────────────────────────────────┘
                           │ same library used by CLI (src/predict.py), tests and evaluation
            ┌──────────────┴─────────────── OFFLINE (synth + ml) ────────────────────────────┐
            M8 generator ──► M9 dataset ──► M10 train (MLflow) ──► M11 evaluate (MLflow) ──► fine-tuned weights
            4 primitives,    native geometry,   Stage A decoder-only,   proc_test metrics,
            composition,     on-the-fly render, Stage B full fine-tune  tables, figures
            4 render styles  val/test pre-rendered
```

All randomness is seeded. Every module has unit tests; every pipeline output is validated before export.

---

## 3. Repository layout

```
.
├── README.md                      # required by the course (all sections)
├── final_report.md                # report (markdown source of the docx/PDF)
├── report/                        # SICIP_TEMPLATE filled docx + SICIP_NAME_ID.pdf
├── requirements.txt
├── Dockerfile  .dockerignore
├── .gitignore  .gitattributes
├── .github/workflows/ci.yml       # pytest + Docker build/run smoke test on every push
├── configs/
│   ├── config.yaml                # master config: data, views, loss, stages, eval, app
│   ├── stage_a.yaml               # decoder-only warm-up (overrides config.yaml)
│   ├── stage_b.yaml               # full fine-tune (overrides config.yaml)
│   └── ablation_micro_normal.yaml # optional run 3 (nuisance augmentation on)
├── src/
│   ├── data_download.py           # fetch textures + published dataset into data/ (or generate locally)
│   ├── generate_dataset.py        # run the generator (local or Kaggle)
│   ├── train.py                   # MLflow-tracked training entry point
│   ├── evaluate.py                # metrics, tables, figures entry point
│   ├── predict.py                 # CLI: image → relief files
│   ├── reliefkit/                 # inference library (M1–M6)
│   │   ├── io.py  preprocess.py  modes.py  model.py  postprocess.py
│   │   ├── mesh.py  export.py  pipeline.py  preview.py
│   ├── synth/                     # procedural generator + renderer (M8–M9)
│   │   ├── shapes.py  compose.py  recipes.py  render.py  textures.py
│   │   ├── generate.py  qc.py
│   └── ml/                         # training and evaluation (M10–M11)
│       ├── dataset.py  losses.py  ema.py  tracking.py
│       ├── trainer.py  evaluate.py  metrics.py
├── app/app.py                     # Gradio prediction UI (M7)
├── notebooks/
│   ├── exploration.ipynb          # dataset exploration (runs locally on a small set)
│   └── kaggle/                    # nb01_generate_dataset, nb02_train, nb03_evaluate
├── tests/                         # pytest (one file per module)
├── mlruns/                        # MLflow evidence — COMMITTED
├── screenshots/
├── models/  data/  artifacts/     # .gitkeep only
└── docs/                          # memory.md, plan.md, architecture.md, log.md, figures/
```

---

## 4. Conventions (used everywhere)

| Item | Convention |
|---|---|
| Height | `h ∈ [0, 1]` float32. **0 = base plane / background, 1 = highest point** |
| Physical Z | `Z_mm = base_thickness_mm + h × relief_depth_mm` |
| Base thickness | solid mode requires `base_thickness_mm ≥ 0.5 mm` |
| Units | STL/OBJ carry no units — coordinates are **authored in mm**. 3MF is written with `unit="millimeter"`. TIFF has no Z range: scale Z to `relief_depth_mm` using the sidecar |
| Image → mesh axes | image column `i` → `x = i·pitch` (left = 0); image row `j` → `y = (H−1−j)·pitch` (**top of the image = back / max Y**, the usual CAM height-map convention) |
| Mesh validity | watertight, consistent winding, volume > 0, bbox within 0.01 mm, no NaN, no face area < 1e-9 mm² |
| Model input | letterboxed to `S×S`, `S` multiple of 14 (518 / 756 / 1036), never stretched |
| Quantization | float32 until export; TIFF = uint16 `round(h·65535)` |
| Reproducibility | every random step takes a seed; splits by `scene_id`; val/test renders pre-rendered with fixed seeds |

---

## 5. Inference pipeline (`src/reliefkit`)

### M2 — Preprocess (`preprocess.py`)

1. **Load** (`io.py`): JPG/PNG/WEBP/BMP, any bit depth → RGB uint8 + optional alpha uint8; EXIF orientation applied.
2. **Mask** — `get_mask(rgb, alpha, remove_bg)`:
   * alpha present → `mask = alpha > 127`;
   * `remove_bg="auto"`: if the image border ring (2 px) is uniform → flat design on a plain background → background colour = median of the border ring, foreground = pixels whose max channel distance from that colour exceeds an Otsu threshold; otherwise (photo) → no mask;
   * `remove_bg="on"`: force the border-colour method; `"off"`: never.
   * `rembg` is an optional plug-in (`remove_bg="rembg"`) and is **not** a dependency; its weights licence must be checked before commercial use.
3. **Components**: `keep_components(mask, min_area_frac=0.0005)` removes only dust (components < 0.05 % of the image area). **Never** keep only the largest component. `fill_holes` only when the user turns it on.
4. **Mode suggestion** — `suggest_mode(rgb)` returns `("A"|"B", diagnostics)`. Diagnosis from the review is right: `np.unique()`-style counting fails on anti-aliased / JPEG images. The rule used:
   * downscale (long side 256), blur σ = 1 (kills anti-alias fringe), quantize to 4 bits/channel;
   * `k90` = number of quantized colours needed to cover 90 % of pixels;
   * `neutral` = fraction of near-grey pixels; `dark_bright` = fraction of pixels with luminance < 50 or > 205 (the reviewer's bimodal test, kept as a B/W indicator);
   * **A** if `k90 ≤ 6` (flat colour or B/W design), else **B**. Thresholds live in `configs/config.yaml` → `suggest_mode`.
5. **Letterbox** — `letterbox(rgb, S)` keeps the aspect ratio (INTER_AREA when shrinking, Lanczos when enlarging), pads to `S×S` with **replicate** padding and returns `LetterboxMeta(scale, pad_left, pad_top, new_h, new_w, S)`. `unletterbox(arr, meta)` crops and resizes back to the working grid.
6. **Inference size** — `choose_inference_size()` (fix of §6.3 of the PRD):
   ```
   height_mm   = width_mm × H / W
   long_mm     = max(width_mm, height_mm)                 # the model sees the LONG side at S px
   target_px   = 3.0 × long_mm / tool_diameter_mm         # 3 px per tool diameter
   S           = clamp(round14(target_px), 518, 1036)
   S           = min(S, max(518, source_long_side rounded up to 14))   # upscaling adds no detail
   S           = 518 if device == cpu and the user left "auto"
   ```

### M3 — Relief modes (`modes.py`, `model.py`)

* **Mode B (AI)** — `ReliefModel.predict(rgb, S, tta)`:
  letterbox → normalize (ImageNet mean/std) → `DepthAnythingForDepthEstimation` → `predicted_depth` (S×S) → `× sign` → `unletterbox`. `sign` is fixed once (Day-1 polarity check) and stored in the checkpoint; default `+1`. Flip-TTA averages the prediction with its horizontal flip. A `model` selector chooses fine-tuned or zero-shot weights; if the fine-tuned weights are missing the app **falls back to zero-shot and says so**.
* **Mode A (design → relief, classic)** — `design_to_height(rgb, mask, profile, R_px)`:
  1. foreground = mask, or Otsu on the distance from the border colour (auto polarity);
  2. `dist = distanceTransform(fg)` (sub-pixel: `dist − 0.5`), `u = clip(dist / R_px, 0, 1)`;
  3. profile: `dome = √(1−(1−u)²)`, `pillow = sin(πu/2)`, `bevel = u`, `flat_round = smoothstep(0, e, u)`;
  4. light Gaussian (σ = 0.8).
  **`R_px` is never a fixed pixel count** (review fix ৩ / Issue 5):
  `R_px = transition_width_mm / pixel_pitch_mm` when the user sets a transition width in mm, otherwise `R_px = max(2, 0.03 × long_side)` ("auto"). `pixel_pitch_mm = width_mm / W_work`.
* **Mode C (gray → height)** — `h = luminance^γ` inside the mask (lithophane-style).

### M4 — Height post-processing (`postprocess.finalize_height`) — fixed order, float32

Work on the un-letterboxed grid (long side = S).

1. **Reference region** `ref = dilate(mask, r)`, `r = max(8, 0.015 × long_side)`; no mask → whole image.
2. `lo = pct(h[ref], 1)`, `hi = pct(h[ref], 99.5)`; if `hi − lo < 1e-6` → zeros + warning "no relief detected".
3. `h01 = clip((h − lo) / (hi − lo), 0, 1)`.
4. **Invert (corrected — Issue 1):** `invert` means *raised ↔ recessed relative to the base plane*, so
   `h01 = 1 − h01` on the **whole working area** and the base level outside the relief becomes `base_level = 1.0` (normally `0.0`).
   The dilated ring of background must **not** be inverted in place — that creates a raised halo — and the area outside the mask must be *reset* to `base_level`, not left inverted. With no mask the whole image flips, so the masked and unmasked cases agree on what "invert" means.
5. Contrast curve `h01^γ`.
6. Edge-preserving bilateral (`d=9, σ_color=0.05, σ_space=3`) then Gaussian σ = `smoothing`.
7. **Feather the mask** (`m_f = GaussianBlur(mask, σ=1.2)`, ≈ 3 px): `h = h01·m_f + base_level·(1 − m_f)`. Far outside the mask `m_f = 0` exactly, so the base level is preserved. Unit test: outside the feathered mask the value equals `base_level` bit-exactly, inverted or not.

### M5 — Mesh builder (`mesh.py`)

Grid `W×H` spans `width_mm × height_mm` (`height_mm = width_mm × H / W`), `z_ij = base + h_ij × depth` with rows flipped (§4).

* **Top:** 2 triangles per cell, normals +z.
* **Walls:** the perimeter ring (counter-clockwise seen from above) duplicated at `z = 0`, joined by quads; faces `(T_k, B_k, B_{k+1})`, `(T_k, B_{k+1}, T_{k+1})` point outward.
* **Bottom:** one centre vertex at `z = 0`, fan over the bottom ring — no zero-area triangles even for long thin panels.
* **Triangle budget** (area-average downsample of the height grid, the TIFF keeps full resolution): draft 250 k / standard 600 k / high 2 M.
* `base_thickness_mm < 0.5` → error `solid mode needs base ≥ 0.5 mm`.
* `validate_mesh()` returns `is_watertight, is_winding_consistent, volume_mm3 > 0, bbox dims vs request (±0.01 mm), has_nan, min_face_area, z_min, z_max, n_triangles, is_valid`. **The app never offers a file whose mesh is invalid.**

Tests (`tests/test_mesh.py`): flat → exact box; plateau → box of `base + depth`; Gaussian bump → volume grows with depth; 50 random heightmaps → watertight; long thin panel → watertight; budget downsample keeps the bbox exact; exported-and-reloaded STL passes validation.

### M6 — Export (`export.py`)

```
out/
  relief_heightmap_16bit.tiff   uint16, full working resolution
  relief_heightmap_8bit.png     grayscale preview copy for CAM tools that read 8-bit images
  relief.stl  relief.obj  relief.3mf
  relief_meta.json              width_mm, height_mm, relief_depth_mm, base_mm, pixel_pitch_mm,
                                mesh_pitch_mm, infer_long_side, mode, model_version, invert, validation, warnings
  preview.png                   shaded preview of the height map
```

### M7 — Gradio app (`app/app.py`) + Docker

* Inputs: image, mode (auto/A/B/C), model (fine-tuned/zero-shot), width, relief depth, base thickness, smoothing, gamma, invert, remove_bg, fill_holes, transition width (Mode A), detail, tool diameter, inference size.
* Outputs: mask + shaded height preview, **3D preview (`gr.Model3D` with OBJ — Gradio 6 accepts .obj/.glb/.stl/.gltf/.ply, **not .3mf**; 3MF is in the ZIP only)**, ZIP download, validation report, warnings.
* Served with Docker: `docker build -t reliefforge-app:1.0 .` → `docker run -p 7860:7860 reliefforge-app:1.0`. Weights are pulled from HF Hub (zero-shot baked at build time; fine-tuned from the public model repo through `RELIEFFORGE_FINETUNED`).
* Warnings: pitch > tool/3; max slope > 70°; mesh downsampled; S capped by the source resolution; fallback to zero-shot.

### Pipeline entry points

`reliefkit.pipeline.generate_relief(rgb, alpha, params) → ReliefResult` is the only function the CLI (`src/predict.py`) and the app call.

---

## 6. Data (M8–M9, `src/synth`)

### 6.1 Why synthetic

No public dataset pairs decorative 2D designs with exact CNC-style height. Our generator owns 100 % of the supervised labels; real CC0 textures are used only to **render** the inputs.

### 6.2 Primitives (height-field safe)

| Primitive | Shape | Profile |
|---|---|---|
| Petal / leaf | outline from a half-width function along a (possibly curved) axis | dome of inward distance; optional central vein groove |
| Radial rosette | N ∈ {4, 6, 8, 12} petals around a centre + boss (spherical cap) | layered petals |
| Vine / scroll | spiral or sinusoidal centreline with tapering width | round tube, `u = d / (d + d_centre)` so it tapers correctly |
| Frame / medallion | concentric rings and rectangular mouldings | bead, pillow, flat_round |

Profiles of normalized inward distance `u`: `dome`, `pillow`, `bevel`, `flat_round`.

### 6.3 Composition — corrected (Issue 2)

`h = max_i (z_i + a_i · profile_i)` is **only valid on each primitive's support**. Every primitive is rasterized into its own bounding box with a support mask:

```
candidate                 = z_i + a_i · profile_i          (inside the primitive's ROI)
candidate[~support_i]     = -inf
update                    = support_i & (candidate > h_roi)
h_roi[update]             = candidate[update]
inst_roi[update]          = instance_id_i                  # only valid pixels update the ID map
h[~foreground]            = 0.0                            # background stays the base plane
```

Without this, any primitive with `z_i > 0` raises the whole canvas to `z_i` and the ground truth is wrong.

### 6.4 Scenes

* Families (recipes): `floral`, `botanical`, `scroll`, `border`, `medallion`.
* Symmetry: none, left-right, 4-fold (layout-level mirroring; overlaps merge through `max`).
* Optional background plate (flat_round, `z ∈ [0.05, 0.3]`).
* Per-scene physical metadata: `width_mm_scene ~ U(200, 600)`, `relief_depth_mm_scene ~ U(4, 15)`, `px_per_mm_native = W_native / width_mm_scene`.
* Peak normalization: `h /= h.max()` → peak = 1, background = 0. Anti-alias blur σ = 0.6 px.
* Rejection: foreground coverage < 3 % or > 95 %, or height std < 0.02 → regenerate.
* Native canvases (sides multiple of 14, ≈ 1.07 M px): 1:1 → 1036×1036 (35 %); 3:2 → 1260×840 / 840×1260 (30 %); 2:1 → 1470×728 / 728×1470 (35 %).
* **4,000 scenes**, split by `scene_id`: 3,200 / 400 / 400.

### 6.5 Stored vs rendered

Stored: `geometry/{split}/{id}_h.png` (uint16), `{id}_inst.png` (uint16 instance IDs), `scenes.jsonl`.
Rendered on the fly during training (new light / texture / tint / noise each time). Val/test are pre-rendered once (wood, clay, flat) with fixed seeds.

### 6.6 Render styles

| Style | Share | Recipe |
|---|---|---|
| Wood render | 40 % | CC0 wood albedo crop (ambientCG), rotate/flip/tint (oak, walnut, painted, gold) × shading × AO. **Albedo only in run 1** (see 6.8) |
| Clay render | 35 % | neutral grey albedo, same shading |
| Flat colour design | 15 % | one palette colour per instance, dark outlines at instance borders, no shading |
| B/W silhouette / line | 10 % | binary silhouette or outline only |

Shading (κ defined — Issue 3):

```
κ_view = relief_depth_mm_scene × px_per_mm_native × s      # s = view scale factor (below)
κ_r    = κ_view × U(0.7, 1.5)
n      = normalize(−κ_r ∂x h, −κ_r ∂y h, 1)
L      = unit vector, azimuth U(0°, 360°), elevation U(30°, 75°)
sh     = ambient + (1 − ambient)·max(0, n·L),  ambient U(0.25, 0.5)
AO     = clip(1 − α·max(0, G_σ(h) − h), 0.4, 1),  α U(1, 4)
img    = albedo · sh · AO → gamma jitter, noise, JPEG 60–98, blur σ ∈ [0, 1.2], brightness/contrast
```

`s` is defined once and used by both the renderer and the loss:
* **full view:** `s = S / native_long_side` (letterboxing does not resample);
* **crop view:** the actual resize factor of the crop (`s = 1` → no resampling).

### 6.7 Training views (every sample 518×518)

* 40 % **full view** — whole scene area-downscaled to long side 518, letterboxed, padding masked out of the loss.
* 60 % **crop view** — scale `s ~ U(s_min, 1.0)` and a random 518×518 crop; retry up to 5 times if the crop has almost no relief.
* **Crop-scale lock (fix ১):** `s_min = max(0.55, 518 / min(H, W))`. For the 2:1 canvas (1470×728) `s_min = 0.7115`; `0.55` would need a 400 px side to fill a 518 px crop and crash. Unit test: every canvas size yields exactly 518×518 for 200 random draws.

### 6.8 Textures and licences

* Wood / plaster / stone albedos: ambientCG (CC0), downloaded by `src/data_download.py` with a credits CSV. A procedural wood generator is the fallback when no textures are available.
* **Micro-normal maps are OFF in run 1** (review point A): they add surface variation that is not in the target height and would teach the model to hallucinate grain. Added only as a controlled nuisance augmentation in the optional ablation run.
* Relief-Depth: inspection only (no provenance / licence). RRD Borobudur, GrabCAD: never in training or in the published dataset.

### 6.9 QC gate

Contact sheet of 100 samples (image | height | instance map); asserts: no NaN, peak = 1, split counts, unique IDs, aspect mix, every dimension a multiple of 14; lighting polarity test (a bump is lit consistently with the light direction); DataLoader throughput ≥ 3× the GPU's images/s.

---

## 7. Training (M10, `src/ml`)

### 7.1 Model and parameter groups (Issue 6)

`depth-anything/Depth-Anything-V2-Small-hf` → `DepthAnythingForDepthEstimation` with `backbone` (DINOv2 ViT-S), `neck` (DPT reassemble + fusion), `head` (3 convs, ReLU output for relative depth). Groups are built from **`named_parameters()` prefixes** and asserted to cover every parameter:

| Group | Prefixes | Stage A | Stage B |
|---|---|---|---|
| encoder | `backbone.*` | frozen | LR 1e-5 (all layers, or the last N) |
| decoder | `neck.*`, `head.*` | LR 1e-4 | LR 5e-5 |

The code fails loudly if a group is empty or if parameters are unassigned.

### 7.2 Loss (float32, outside autocast)

For each image with valid mask `v`, prediction `p`, target `g`:

1. **MiDaS-style normalization (scale > 0 by construction):** `p̃ = (p − med(p)) / max(mad(p), 1e-6)`, same for `g̃`; medians/means over `v`.
2. `L_h = Σ w·|p̃ − g̃| / Σ w`, `w = v · (1 + 2·fg_dilated)`.
3. `L_g = Σ_{k=0..3} mean(|∂x R_k| + |∂y R_k|)`, `R = p̃ − g̃` area-downsampled by `2^k`.
4. **Normal loss with physical κ:** `p̂ = p̃·mad(g) + med(g)`; `n(h) = normalize(−κ ∂x h, −κ ∂y h, 1)`; **`κ = relief_depth_mm_scene × px_per_mm_native × s`** (per sample); `L_n = mean_w (1 − n(p̂)·n(g)) / 2`.

`L = 1.0·L_h + 0.5·L_g + 0.1·L_n`. Unit tests: flipped polarity costs more than the correct polarity; flat scene gives no NaN; changing padded pixels does not change the loss.

### 7.3 Recipe

| Item | Value |
|---|---|
| Input | 518×518, ImageNet mean/std |
| Precision | `autocast(float16)` forward; loss in float32; `GradScaler`; grad-clip 1.0 |
| Optimizer | AdamW, weight decay 1e-4, cosine schedule, 300 warm-up steps |
| Batch | 8 (4 × accumulation 2 if OOM) |
| EMA | decay 0.999; evaluate and export the EMA weights |
| Stage A | decoder only, ~2,000 steps |
| Stage B | all weights, ~8,000 steps |
| Checkpoint | every 500 steps `{model, ema, optimizer, scaler, step, stage}`; best by validation; early stop patience 4 validations |
| Gate | `Pearson(pred, gt) > 0` on val after Stage A |
| Step budget | measured from a 200-step smoke test, then rescaled |

Resume is mandatory; checkpoints can be pushed to a private HF Hub repo (`HF_TOKEN` as a Kaggle secret).

### 7.4 Kaggle runbook

* Accelerator **GPU T4 ×2, use one GPU** (`cuda:0`). Avoid P100. Internet on.
* `nb01_generate_dataset` (CPU) → `/kaggle/working/dataset` → Kaggle Dataset (and/or HF dataset repo).
* `nb02_train` (GPU) → `/kaggle/working/mlruns`, checkpoints → download the notebook output, copy `mlruns/` to the repo root.
* `nb03_evaluate` → tables + figures.
* Sessions up to 12 h; weekly quota is shown in the account. Budget ≤ 8–10 GPU-hours.

---

## 8. MLflow design (`src/ml/tracking.py`)

### 8.1 Experiments and runs

| Experiment | Runs |
|---|---|
| `ReliefForge-Finetune` | `stage_a_decoder_only`, `stage_b_full_finetune`, optional `ablation_*`, local `smoke_*` runs |
| `ReliefForge-Eval` | `zeroshot_baseline`, `stage_a`, `stage_b_ema`, resolution study |

### 8.2 What is logged

* **Params:** the whole flattened config + stage, git commit, GPU name, versions.
* **Metrics (every 100 steps):** `train_loss, train_L_h, train_L_g, train_L_n, lr_encoder, lr_decoder, images_per_s`.
* **Metrics (every validation):** `val_MAE_norm, val_RMSE_norm, val_SSIM, val_normal_err_deg, val_R_detail, val_pearson` and `val_MAE_norm/<family>`, `val_MAE_norm/<style>`.
* **Artifacts:** `config.yaml`, `history.csv`, `eval_samples/step_XXXXX.png` (input | GT | prediction), `loss_curves.png`, `model/` (EMA weights, fp16 safetensors + config), `run_info.json`.

### 8.3 Portability

`mlruns/` produced on Kaggle is copied into the repo. `scripts/make_mlruns_portable.py` rewrites absolute `artifact_location`/`artifact_uri` paths so `mlflow ui` works on any machine.

### 8.4 Screenshots

`mlflow ui` run from the repo root → `screenshots/mlflow_runs.png`, `screenshots/mlflow_run_detail.png`.

### 8.5 What is committed

Everything in `mlruns/` **except weight binaries** (`*.safetensors, *.pt, *.pth`): two fp16 checkpoints are ≈ 100 MB, above what belongs in git. The weights are published on the HF model repo and are reproducible with `python src/train.py --config configs/config.yaml` — documented in the README, as the instructions allow for large artifacts.

---

## 9. Evaluation (M11, `src/ml/evaluate.py`)

* **Test sets (reported separately):** `proc_test` = 400 held-out scenes × {wood, clay, flat} = 1,200 fixed renders; optional qualitative real examples (RRD Borobudur) only where alignment and direction were verified.
* **Metrics** after the production normalization (M4) with the GT foreground as reference region for both prediction and GT: MAE, RMSE (normalized height), SSIM, normal angular error (κ from 300 mm / 8 mm at the tested S), detail retention `R_detail = ‖h − G_σ(h)‖₁ / ‖h_gt − G_σ(h_gt)‖₁` (σ = 2 px).
* **Breakdowns:** per family, per render style, per inference size S ∈ {518, 756, 1036}.
* **Comparison table:** zero-shot | Stage A | Stage B (EMA) | final + post-processing | Mode A / Mode C on the flat & B/W subset. Plus 8–10 qualitative figures (input | GT | zero-shot | fine-tuned).
* Results are reported honestly: an improvement on synthetic data does not prove real-carving quality.

---

## 10. Docker and CI

* `Dockerfile`: `python:3.11-slim`, CPU-only PyTorch wheel, `requirements.txt`, code, `EXPOSE 7860`. The zero-shot weights are downloaded during the build so the container starts offline-capable.
* `.dockerignore` keeps data, caches, venv, mlruns, artifacts out of the image.
* `.github/workflows/ci.yml`: on push → `pytest` (CPU) → `docker build` → `docker run` → health check. This is the build proof while the development laptop has no free disk for Docker.

---

## 11. Review-fix adjudication (2026-11-23)

| # | Fix request | Verdict | Where |
|---|---|---|---|
| ১ | 2:1 canvas crop scale crashes (`728 × 0.55 < 518`) | **Accepted** | `ml/dataset.py` `s_min`, `tests/test_dataset.py` |
| ২ | `suggest_mode`: count histogram extremes, not `np.unique` | **Accepted with extension** — dark+bright > 85 % only detects B/W; multi-colour flat designs also need Mode A, so the rule is quantized-colour coverage (`k90 ≤ 6`) plus neutrality/bimodal diagnostics | `reliefkit/preprocess.py` |
| ৩ | Mode A `R` must scale with the image | **Accepted** | `reliefkit/modes.py`, default `0.03 × long_side` |
| 1 | Invert raises background (halo) | **Accepted, refined** — outside the mask must be `base_level`, which is **1.0 when inverted**, otherwise masked and unmasked images disagree on what "invert" means | `reliefkit/postprocess.py`, `tests/test_postprocess.py` |
| 2 | Generator composition leaks `z_i` over the canvas | **Accepted (critical)** | `synth/compose.py`, `tests/test_synth.py` |
| 3 | `κ_view` undefined; define `s` | **Accepted** | `synth/render.py`, `ml/losses.py`, `ml/dataset.py` |
| 4 | Auto resolution must use the long side and the source resolution | **Accepted** | `reliefkit/preprocess.choose_inference_size` |
| 5 | Mode A `R` undefined → user control in mm | **Accepted** | `transition_width_mm` parameter, app slider |
| 6 | Stage A layer names (DINOv2 backbone / DPT neck / head) | **Accepted** — verified in transformers 5.18: `backbone`, `neck`, `head` | `ml/trainer.py` |
| A | Micro-normal texture off in run 1, ablation later | **Accepted** | `configs/`, `synth/render.py` |
| B | State the perspective-photo limitation | **Accepted** | README §Limitations |
| C | `gr.Model3D` has no .3mf → preview OBJ/STL, 3MF in ZIP | **Accepted** — verified in Gradio 6.32 | `reliefkit/export.py`, `app/app.py` |

---

## 12. Out of scope (MVP)

"100 % perfect" output; proprietary formats (`.rlf`, `.3dClip`, `.jdp`); G-code / toolpaths; STEP; accounts / queues; brush editing; SAM 2; perspective correction of phone photos; faces, figures, animals and lettering (out of distribution).
