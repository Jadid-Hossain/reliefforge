"""CLI: python src/predict.py design.png out/ --mode B --width 300

Runs the full pipeline (mask -> mode -> postprocess -> watertight mesh -> files).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reliefkit import (  # noqa: E402
    APP_VERSION, ImageLoadError, MeshError, ReliefModel, ReliefParams, generate_relief, load_image,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reliefforge", description=f"2D design -> CNC relief files ({APP_VERSION})")
    p.add_argument("input", help="input image (jpg/png/webp/bmp)")
    p.add_argument("out_dir", help="output directory")
    p.add_argument("--mode", default="auto", choices=["auto", "A", "B", "C"], help="A design->relief, B AI, C gray->height")
    p.add_argument("--model", default="zero-shot", choices=["zero-shot", "fine-tuned"], help="Mode B weights")
    p.add_argument("--model-id", default=None, help="explicit HF model id / checkpoint dir for Mode B")
    p.add_argument("--width", type=float, default=300.0, help="physical width in mm (20-2000)")
    p.add_argument("--depth", type=float, default=8.0, help="relief depth in mm (0.5-100)")
    p.add_argument("--base", type=float, default=6.0, help="base thickness in mm (>=0.5)")
    p.add_argument("--smoothing", type=float, default=1.0, help="gaussian sigma in px (0-4)")
    p.add_argument("--gamma", type=float, default=1.0, help="contrast curve exponent")
    p.add_argument("--invert", action="store_true", help="raised <-> recessed")
    p.add_argument("--remove-bg", default="auto", choices=["auto", "on", "off"], help="background masking")
    p.add_argument("--fill-holes", action="store_true", help="fill holes inside the mask (Mode A)")
    p.add_argument("--profile", default="dome", choices=["dome", "pillow", "bevel", "flat_round"], help="Mode A profile")
    p.add_argument("--transition-width", default="auto", help="Mode A bevel width in mm, or 'auto'")
    p.add_argument("--detail", default="standard", choices=["draft", "standard", "high"], help="mesh triangle budget")
    p.add_argument("--tool", type=float, default=1.0, help="tool diameter in mm (warnings + resolution)")
    p.add_argument("--size", type=int, default=0, help="inference long side (multiple of 14); 0/omit = auto")
    p.add_argument("--tta", action="store_true", help="flip test-time augmentation (Mode B)")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--no-model", action="store_true", help="skip Mode B even if requested (classic modes only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        img = load_image(args.input)
    except ImageLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    params = ReliefParams(
        width_mm=args.width, relief_depth_mm=args.depth, base_thickness_mm=args.base,
        smoothing=args.smoothing, gamma=args.gamma, invert=args.invert,
        remove_bg=args.remove_bg, fill_holes=args.fill_holes, mode=args.mode,
        profile=args.profile, transition_width_mm=args.transition_width,
        detail=args.detail, tool_diameter_mm=args.tool,
        inference_size=(args.size if args.size and args.size > 0 else "auto"),
        tta=args.tta, model=args.model,
    )

    model = None
    if params.mode in ("auto", "B") and not args.no_model:
        model_id = args.model_id or (
            args.model if args.model != "zero-shot" else "depth-anything/Depth-Anything-V2-Small-hf"
        )
        print(f"loading model: {model_id} ...")
        model = ReliefModel(model_id, device=args.device)

    try:
        result = generate_relief(img.rgb, img.alpha, params, model=model)
    except (ValueError, MeshError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    exported = result.save(args.out_dir)
    print(f"\nmode {result.info['mode_used']} | S = {result.info['S']} | mesh {result.validation['n_triangles']} tris | watertight: {result.validation['is_watertight']}")
    for w in result.warnings:
        print(f"  warning: {w}")
    for k, p in exported.items():
        print(f"  {k:16s} {Path(p).name}")
    print(json.dumps(result.meta, indent=2)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
