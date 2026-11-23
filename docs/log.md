# ReliefForge — Work Log

One entry per work session. Record what was done, test results, decisions and deviations.
Format: date → summary → tests → decisions.

---

## 2026-11-23 (Day 1) — analysis, adjudication, scaffold

### Inputs read
- `resources/instructions.pdf` (faculty, binding), `resources/PRD.md` (v1, superseded), `resources/PRD_v2.md`, review fixes (3 minor + 6 issues + 3 improvements).
- `resources/SICIP_TEMPLATE_final_report.docx` outline (sections 1–8 + Appendix A/B).

### Findings
1. The PRDs do not contain four mandatory course requirements: **Docker**, **MLflow** (+ `mlruns/` committed, ≥ 2 runs, screenshots), the **expected repo structure + meaningful commit history**, and the **report template**. All four are now designed into `architecture.md` (§1.1 traceability).
2. The real deadline is **28 Nov 2026 23:59** (PDF). `PRD_v2.md` §0.3 wrongly dismisses a deadline date; the PDF wins.
3. Acceptance criteria say "a valid domain selected from the provided list"; the project is submitted under "Other" → the user confirms with the faculty.
4. Machine: 4C/8T laptop, 8 GB RAM, no GPU, C: 1.5 GB and G: 5.3 GB free, Docker daemon off → local work is limited to code + tests + tiny smoke sets; training on Kaggle.

### Review-fix verdicts (details in `architecture.md` §11)
- All 12 points accepted. Two refinements:
  - **Fix ২ (`suggest_mode`):** "dark + bright > 85 %" detects only B/W designs; flat *multi-colour* designs also need Mode A → rule = quantized-colour coverage `k90 ≤ 6`, with the dark/bright share kept as a B/W diagnostic.
  - **Issue 1 (invert):** the proposed "outside mask → 0" makes a masked image disagree with an unmasked one. Invert = raised ↔ recessed relative to the base plane, so the area outside the relief is `1.0` when inverted.
- Verified facts: transformers 5.18 names are `backbone` / `neck` / `head` (Issue 6); Gradio 6.32 `Model3D` has no `.3mf` (point C); trimesh writes `unit="millimeter"` in 3MF.

### Done
- Repo skeleton, lightweight venv on top of the system torch, zero-shot model cached under `.hf_cache`.
- `docs/architecture.md`, `docs/memory.md`, `docs/plan.md`, this log.

### Tests
- Not yet (no code).

### Decisions
- Work locally on `main` with small commits per module; pushing is done by the user.
- Weight binaries are not committed; `mlruns/` metadata, metrics and plots are.


---

## 2026-11-23 (Day 1, evening) — full pipeline, generator, training stack, app, delivery layer

### Done (commits 68092ba…bcbe442)
- **reliefkit**: mesh (78 tests), io/preprocess/modes/postprocess/model/export/pipeline/CLI.
  All 12 review fixes implemented (architecture.md §11 has the mapping).
- **synth**: procedural generator (5 families, 4 styles, native canvases) with
  support-mask composition; QC + contact sheet; 0.36 s/scene.
- **ml**: losses (median/MAD + multi-scale grad + physical-κ normal), on-the-fly view
  datasets (s_min lock), EMA, MLflow tracking, two-stage trainer (asserted param
  groups), metrics + evaluation with breakdowns; 20 tests.
- **entry points**: train/evaluate/data_download/generate_dataset + configs; local
  smoke chain green (zeroshot MAE .366 -> stage_b_smoke .364 on 3 views; 4 MLflow runs).
- **app**: Gradio UI + Dockerfile + CI (pytest + docker build/run); Kaggle notebooks
  nb01–nb03; README; final_report.md (template sections, TODO slots for Kaggle numbers).

### Bugs found and fixed by the tests
1. Mesh winding was inverted (row flip) -> negative volume.
2. Colour masks punched the flat top out of shaded renders and inverted the mask when a
   plate touched the border band -> background colour now taken from the outermost
   pixels AND masks are only applied when the distance map is bimodal (valley < 5%).
   Zero-shot on a realistic render: corr 0.289 -> 0.822.
3. suggest_mode: five feature probes showed smooth monochrome renders are statistically
   near-flat; final rule = dominant-colour coverage + B/W bimodality (review fix 2),
   suggestion is overrideable; limitation documented.
4. Parallel dataset generation duplicated ids (per-shard split) -> parent-fixed plan.
5. MLflow 3.18 file store needs MLFLOW_ALLOW_FILE_STORE + file:/// URI.
6. gradient-loss leaked padding garbage into coarser scales -> mask residual before pooling.

### Status vs plan
Phase 0-4 complete locally. Gate G1 passed (real image -> validated STL, corr 0.822).
Remaining is the user's part: GitHub push, Kaggle runs, Docker/screenshots, report fill.
