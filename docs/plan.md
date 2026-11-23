# ReliefForge — Execution Plan

Deadline: **28 Nov 2026 23:59** (hard). Working target: **27 Nov**. Tick items here and add a line to `log.md` when a gate passes or a decision changes.

## 1. Day plan

| Day | Date | Goal | Gate |
|---|---|---|---|
| 1 | Mon 23 Nov | Repo + docs; full inference pipeline (mesh → export → Modes A/B/C → CLI); generator; dataset/training code ready to ship to Kaggle | **G1:** one real image → validated STL from the CLI; `pytest` green |
| 2 | Tue 24 Nov | Kaggle: generate dataset (nb01), launch training (nb02); app + Docker + README draft | **G2:** dataset QC passes; Stage A/B started |
| 3 | Wed 25 Nov | Training done; evaluation (nb03); plug weights into the app; zero-shot vs fine-tuned table | **G3:** fine-tuned beats zero-shot on `proc_test` (or the gap is analysed honestly) |
| 4 | Thu 26 Nov | Docker run + screenshots; MLflow screenshots; README complete; report draft | **G4:** README checklist complete |
| — | Fri 27 Nov | Report (template → PDF), final polish, **scope freeze 12:00**, push, form | **G5:** submitted |
| — | Sat 28 Nov | Buffer only | — |

## 2. Work breakdown

### Phase 0 — Scaffold and documents
- [x] Read instructions, PRDs and review fixes; adjudicate every fix (`architecture.md` §11)
- [x] Environment check (Python, torch, Docker, git, disk, CPU)
- [x] `docs/architecture.md`, `docs/memory.md`, `docs/plan.md`, `docs/log.md`
- [ ] `.gitignore`, `.gitattributes`, `requirements.txt`, `configs/config.yaml`, git init + first commit

### Phase 1 — Inference pipeline (Day-1 safety net)
- [ ] `reliefkit/io.py`, `preprocess.py` (mask, components, letterbox, `suggest_mode`, inference size)
- [ ] `reliefkit/mesh.py` + `tests/test_mesh.py` (watertight, budget, validation)
- [ ] `reliefkit/modes.py` (A, C), `postprocess.py` (M4 with the invert fix) + tests
- [ ] `reliefkit/model.py` (zero-shot, sign, TTA), `export.py`, `preview.py`, `pipeline.py`
- [ ] `src/predict.py` CLI; end-to-end test with a fake model + optional real model
- [ ] **Gate G1**

### Phase 2 — Data
- [ ] `synth/shapes.py`, `compose.py` (support-mask composition) + tests
- [ ] `synth/recipes.py` (5 families, symmetry, plate), `textures.py`, `render.py` (κ defined, 4 styles, degradations)
- [ ] `synth/generate.py`, `qc.py` (contact sheet, asserts, polarity test)
- [ ] `src/data_download.py` (CC0 textures + credits; published dataset or local generation)
- [ ] `src/generate_dataset.py`

### Phase 3 — Training and evaluation
- [ ] `ml/dataset.py` (full/crop views, `s_min` lock, fixed val/test) + tests
- [ ] `ml/losses.py` + tests; `ml/ema.py`; `ml/tracking.py` (MLflow)
- [ ] `ml/trainer.py` (named param groups, Stage A/B, AMP, EMA, resume, history.csv, HF Hub push)
- [ ] `ml/evaluate.py`, `ml/metrics.py` (MAE, RMSE, SSIM, normal error, R_detail; per family/style/S)
- [ ] `src/train.py`, `src/evaluate.py`; local smoke run (tiny set, CPU) → `mlruns/`
- [ ] Kaggle notebooks `nb01_generate_dataset`, `nb02_train`, `nb03_evaluate`

### Phase 4 — App and delivery
- [ ] `app/app.py` (Gradio), `Dockerfile`, `.dockerignore`, CI workflow
- [ ] `notebooks/exploration.ipynb`
- [ ] `README.md` (all required sections, course-technique table, screenshots)
- [ ] `final_report.md` → docx from the template → `SICIP_NAME_ID.pdf`

## 3. User checklist (only the user can do these)

| # | Action | When |
|---|---|---|
| 1 | Create the **public GitHub repo**, add the remote, push `main` | after Phase 1 |
| 2 | Kaggle: phone-verified account, GPU + internet on; check weekly quota | before Day 2 |
| 3 | Kaggle: upload and run `nb01_generate_dataset` → save output as a Kaggle Dataset | Day 2 |
| 4 | Kaggle: upload and run `nb02_train` → download output → copy `mlruns/` to the repo root → commit | Day 2–3 |
| 5 | Kaggle: run `nb03_evaluate` → copy `artifacts/eval/` and `models/` outputs | Day 3 |
| 6 | Hugging Face: account + write token; publish dataset + weights (optional but recommended) | Day 2–3 |
| 7 | Free ≥ 8 GB on C: (or move Docker's data), start Docker Desktop, `docker build` / `docker run`, screenshots | Day 4 |
| 8 | `mlflow ui` → screenshots of the runs | Day 4 |
| 9 | Report: name + student ID, fill the template, export `SICIP_NAME_ID.pdf` | Day 4–5 |
| 10 | Confirm with faculty that the "Other" category is accepted; submit the form | before Day 5 |

## 4. Cut ladder (drop in this order if behind)

1. Optional real tests (RRD / GrabCAD / Relief-Depth viewing)
2. Ablation run (micro-normal)
3. Higher inference sizes (756 / 1036) → keep 518 + Lanczos
4. Normal loss (keep L1 + gradient)
5. Flip-TTA, auto mode suggestion
6. Stage B improvement run (ship Stage A + first Stage B)

**Never cut:** mesh validation, one fine-tuned model, the zero-shot-vs-fine-tuned table, MLflow runs + screenshots, Dockerized app, dataset fetch script, README sections.

## 5. Risks

| Risk | Mitigation |
|---|---|
| "Other" domain not accepted | user confirms with faculty on Day 1 |
| Kaggle quota / session loss | resume from checkpoints, HF Hub push, Colab fallback |
| Synthetic-to-real gap | rich rendering randomisation, model selector (zero-shot fallback), honest real examples, Modes A/C |
| Laptop disk / Docker | CI builds the image; user frees space for the local Docker run |
| Time | scope freeze 27 Nov 12:00; cut ladder above |
