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
