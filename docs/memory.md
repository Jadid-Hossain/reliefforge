# ReliefForge — Project Memory

Read this first in any new session. Keep it short: facts, decisions, rules.
Details live in `architecture.md` (spec), `plan.md` (tasks), `log.md` (history).

## 1. What this is

- Final project of the **SICIP "ML & DL" course (BRAC University)**.
  Title: *Fine-Tuned 2D Image to 2.5D/3D CNC Relief File Generation System* (submitted under the form's **"Other"** category).
- System: a 2D decorative design image → relief height map → watertight 2.5D solid (STL/OBJ/3MF) + 16-bit TIFF + JSON sidecar, usable in CAD/CAM.
- ML core: **Depth Anything V2 Small** (Apache-2.0) fine-tuned on a paired dataset we generate procedurally.
- Origin: a client wants a tool like <https://cncrelief.com/> (design in → relief files the CNC can carve: furniture panels, faces, ornaments).

## 2. Hard constraints (faculty `resources/instructions.pdf` — binding)

- **Deadline 28 Nov 2026, 23:59, no extension.** Working target: everything done by **27 Nov**, 28 Nov = buffer.
- Public GitHub repo, **meaningful commit history** (not one last-minute upload), complete `README.md`.
- **No dataset in the repo**: `data/` ignored except `data/.gitkeep`; `src/data_download.py` fetches data into `data/` automatically.
- **MLflow**: log params, metrics, model artifacts, plots; **≥ 2 runs with different settings**; `mlruns/` is **committed, never ignored**; README has MLflow screenshots.
- **Dockerized prediction app** with an upload UI; README has `docker build` / `docker run` + screenshots of the Docker-served app.
- Expected structure: `README.md, final_report.md, requirements.txt, .gitignore, Dockerfile, src/{data_download,train,evaluate,predict}.py, app/app.py, notebooks/exploration.ipynb, configs/config.yaml, mlruns/, screenshots/, models/, data/, artifacts/`.
- README must explicitly list the **course techniques** used and where.
- Final report in the faculty template → PDF named **`SICIP_NAME_ID.pdf`**, kept in the repo. Template outline: 1 Introduction (1.1–1.3), 2 Dataset (2.1–2.3), 3 Methodology (3.1 course techniques table, 3.2 architecture, 3.3 training), 4 Experiments (4.1 MLflow, 4.2 results), 5 Prediction App & Docker (5.1–5.3), 6 Repository & Reproducibility (6.1–6.3), 7 Limitations & Future Work, 8 Conclusion, Appendix A/B. (The template's June 2026 dates are stale; the PDF's 28 Nov 2026 applies.)
- Submit the GitHub link via <https://forms.gle/MZv5CtUh55LvuG8X9>.
- ⚠ Risk to confirm with faculty: "Minimum acceptance: a valid domain selected from the provided list" — this project is not on that list.

## 3. Source documents and precedence

1. `resources/instructions.pdf` — overrides everything.
2. `resources/PRD_v2.md` + the review fixes (3 + 6 + 3 points, 2026-11-23) — product and ML spec.
3. `docs/architecture.md` — implementation-level truth (merges 1 + 2, adjudicates every review fix in §11).

`resources/PRD.md` (v1) is superseded — ignore except for history. `resources/` is local only (git-ignored).

## 4. Environment facts (verified 2026-11-23)

- User's laptop: Windows 11, i5-8265U (4C/8T), 8 GB RAM, **no NVIDIA GPU**.
  **Disk almost full**: C: ≈ 1.5 GB free, G: ≈ 5.3 GB free → never create big files locally, never download a second torch.
- System Python 3.12 has torch 2.12.0+cpu, transformers 5.18.0, mlflow 3.18.0, opencv, scikit-image, scipy, datasets.
- Project venv: `.venv` (created with `--system-site-packages`) → run **`.venv/Scripts/python.exe`**. Install with `pip --no-cache-dir` (keeps C: free).
  Added in the venv: trimesh 5.2.2, gradio 6.32.0.
- HF cache: export `HF_HOME="G:/Code Playground/3D Master/.hf_cache"` (keeps C: free). Zero-shot model `depth-anything/Depth-Anything-V2-Small-hf` is already cached there.
- Docker Desktop 29.5 is installed but the **daemon is not running** and there is no disk for an image build → `.github/workflows/ci.yml` builds the image as proof; local Docker run needs the user to free ≥ 8 GB and start Docker Desktop.
- Kaggle CLI is not configured locally (no `~/.kaggle`); the user uploads notebooks through the Kaggle web UI.
- **Training and full dataset generation run on Kaggle free GPU** (T4 ×2 → use `cuda:0`; avoid P100). Locally only tiny smoke sets.
- Model facts (transformers 5.18): `DepthAnythingForDepthEstimation` has `backbone` (Dinov2Backbone: embeddings, encoder, layernorm), `neck`, `head`; 24.79 M params; relative depth → head output is ReLU (≥ 0); output shape = input shape. CPU forward ≈ 4 s at 518.

## 5. Decisions (stable — do not re-litigate without a reason)

| Topic | Decision |
|---|---|
| Output | height field `h ∈ [0,1]`, 0 = base, 1 = peak; mm authored; image top row = max Y |
| Mesh | top grid + perimeter walls + centre-fan bottom; triangle budget 250k/600k/2M; base ≥ 0.5 mm; never export an invalid mesh |
| Modes | A classic (flat designs, deterministic), B AI, C gray→height; `suggest_mode` = quantized-colour coverage (`k90 ≤ 6` → A) |
| Invert | raised ↔ recessed **relative to the base plane**: outside the relief the level is `1.0` when inverted (not 0) |
| Mode A radius | `R_px = transition_width_mm / pixel_pitch_mm`, else `0.03 × long_side` |
| Inference size | from the **long side** in mm and the tool, then capped by the source resolution, clamp 518–1036 |
| Data | **Relief-Depth (KarimT/Relief-Depth) = PRIMARY training data** (user decision 2026-11-24; rendered-from-3D labels, 8,070 pairs, splits respected, CC-BY-NC → course OK / commercial NOT); own generator = auxiliary 35 % mix; textures (CC0 ambientCG) only for rendering |
| Composition | `max` over `z_i + a_i·profile_i` **inside each primitive's support only** |
| κ | `κ = relief_depth_mm × px_per_mm_native × s`; `s` = full view `S/long_side`, crop view = real resize factor |
| Crop scale | `s_min = max(0.55, 518 / min(H, W))` |
| Loss | MiDaS-style median/MAD normalization, weighted L1 + multi-scale gradient + physical normal loss; float32 |
| Training | Stage A decoder only (LR 1e-4, ~2k steps) → Stage B all weights (enc 1e-5 / dec 5e-5, ~8k steps); EMA 0.999; AMP fp16; resume |
| Param groups | by `named_parameters()` prefix: `backbone.*` vs `neck.*`/`head.*`, asserted |
| MLflow | experiments `ReliefForge-Finetune`, `ReliefForge-Eval`; weights saved as run artifacts but **weight binaries are git-ignored** (published on HF Hub) |
| 3D preview | OBJ for `gr.Model3D` (Gradio 6 has no .3mf); 3MF only inside the ZIP |
| Real datasets | Relief-Depth (no licence) inspection only; RRD Borobudur / GrabCAD test only, never in the published dataset |
| Language | user writes Bangla/Banglish → reply in Bangla; docs, code, README in English |

## 6. Owned by the user (not the assistant)

- GitHub repo (create public repo, add remote, push) — commits are made locally on `main`.
- Kaggle: phone-verified account, run `nb01 → nb02 → nb03`, save outputs, download `mlruns/`.
- Hugging Face: account + write token (dataset, checkpoints, final weights).
- Docker: free disk, start Docker Desktop, build/run, screenshots.
- Report: student name + ID (`SICIP_NAME_ID.pdf`), docx from the template, PDF.
- Links to the datasets the user found (not provided yet).
- Confirm with faculty that the "Other" category is accepted.

## 7. Pitfalls learned

- Windows + Git Bash: Python does not understand `/tmp/...` paths — use `cygpath -w`.
- HF cache on this machine has no symlink support (warning only).
- Gradio 6: `Model3D` accepts .obj/.glb/.stl/.gltf/.ply only.
- `mlruns` moved between machines keeps absolute artifact paths → run `scripts/make_mlruns_portable.py` before `mlflow ui`.
