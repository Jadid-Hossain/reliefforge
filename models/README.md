# models/

Weight binaries are never committed (`.gitignore`). Where they live instead:

- `models/finetuned/` — put the fine-tuned export here (from `artifacts/train_b/export`,
  or the Kaggle output) so the app and CLI use it. Falls back to zero-shot if missing.
- Zero-shot model: `depth-anything/Depth-Anything-V2-Small-hf` (Hugging Face, Apache-2.0),
  cached under `HF_HOME`.
- Optional: publish the fine-tuned export to the HF Hub and set
  `RELIEFFORGE_FINETUNED_HF=<user>/reliefforge-da2-small`.
