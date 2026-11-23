"""M7 — ReliefForge prediction app (Gradio). Served in Docker for the course.

Upload a 2D decorative design -> relief heightmap + watertight STL/OBJ/3MF + TIFF,
with a 3D preview (OBJ: Gradio's Model3D has no .3mf) and a downloadable ZIP.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
import numpy as np

from reliefkit import preview as rk_preview
from reliefkit.io import load_image
from reliefkit.model import ReliefModel
from reliefkit.pipeline import ReliefParams, generate_relief
from reliefkit.preprocess import analyze_image, suggest_mode

ZERO_SHOT_ID = "depth-anything/Depth-Anything-V2-Small-hf"
FINETUNED_DIR = os.environ.get("RELIEFFORGE_FINETUNED", str(ROOT / "models" / "finetuned"))
FINETUNED_FALLBACK_HF = os.environ.get("RELIEFFORGE_FINETUNED_HF", "")  # e.g. <user>/reliefforge-da2-small

_model_cache: dict[str, ReliefModel] = {}


def get_model(which: str):
    key = which
    if key in _model_cache:
        return _model_cache[key]
    if which == "fine-tuned":
        if Path(FINETUNED_DIR).exists():
            model = ReliefModel(FINETUNED_DIR)
        elif FINETUNED_FALLBACK_HF:
            model = ReliefModel(FINETUNED_FALLBACK_HF)
        else:
            model = ReliefModel(ZERO_SHOT_ID)
            model.info_note = "fine-tuned weights not found: running the zero-shot model"
    else:
        model = ReliefModel(ZERO_SHOT_ID)
    _model_cache[key] = model
    return model


def generate(
    image,
    mode: str,
    model_choice: str,
    width_mm: float,
    relief_depth_mm: float,
    base_thickness_mm: float,
    smoothing: float,
    gamma: float,
    invert: bool,
    remove_bg: str,
    fill_holes: bool,
    profile: str,
    transition_width_mm: float,
    detail: str,
    tool_diameter_mm: float,
    size_choice: str,
):
    if image is None:
        raise gr.Error("upload an image first")
    out_dir = Path(tempfile.mkdtemp(prefix="reliefforge_"))
    img = load_image(image)
    tw = "auto" if transition_width_mm <= 0 else float(transition_width_mm)
    size = "auto" if size_choice == "auto" else int(size_choice)
    params = ReliefParams(
        width_mm=width_mm, relief_depth_mm=relief_depth_mm, base_thickness_mm=base_thickness_mm,
        smoothing=smoothing, gamma=gamma, invert=invert, remove_bg=remove_bg, fill_holes=fill_holes,
        mode=mode, profile=profile, transition_width_mm=tw, detail=detail,
        tool_diameter_mm=tool_diameter_mm, inference_size=size, model=model_choice,
    )
    analysis = analyze_image(img.rgb)
    model = get_model(model_choice) if params.mode in ("auto", "B") else None
    try:
        result = generate_relief(img.rgb, img.alpha, params, model=model, analysis=analysis)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"generation failed: {exc}")
    exported = result.save(out_dir)
    strip = rk_preview.preview_figure(img.rgb, result.h01, result.mask)
    report = [
        f"**mode used:** {result.info['mode_used']} (suggested: {result.info['mode_suggested']})",
        f"**inference size:** {result.info['S']} px | pitch {result.meta['pixel_pitch_mm']:.3f} mm/px",
        f"**mesh:** {result.validation['n_triangles']:,} triangles, watertight: {result.validation['is_watertight']}, dims {result.meta['width_mm']:.0f} x {result.meta['height_mm']:.0f} x {result.meta['z_max_mm']:.1f} mm",
    ]
    if result.warnings:
        report.append("**warnings:** " + "; ".join(result.warnings))
    note = getattr(model, "info_note", None)
    if note:
        report.append(f"⚠ {note}")
    return strip, str(exported["obj"]), str(exported["zip"]), "\n\n".join(report)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ReliefForge — 2D design → CNC relief") as app:
        gr.Markdown("# ReliefForge — 2D decorative design → CAM-importable 2.5D relief\nUpload a design (flat ornament, line art, or a photo/render of a carved panel), set the physical size, and download a watertight STL/OBJ/3MF + 16-bit heightmap. Units: millimetres.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="filepath", label="Design image (jpg/png/webp)")
                mode = gr.Radio(["auto", "A", "B", "C"], value="auto", label="Mode (A design→relief, B AI model, C gray→height)")
                model_choice = gr.Radio(["fine-tuned", "zero-shot"], value="fine-tuned", label="Mode B weights")
                with gr.Row():
                    width_mm = gr.Slider(20, 2000, 300, step=5, label="Physical width (mm)")
                    relief_depth_mm = gr.Slider(0.5, 100, 8, step=0.5, label="Relief depth (mm)")
                    base_thickness_mm = gr.Slider(0.5, 100, 6, step=0.5, label="Base thickness (mm)")
                with gr.Row():
                    smoothing = gr.Slider(0, 4, 1.0, step=0.1, label="Smoothing σ (px)")
                    gamma = gr.Slider(0.3, 3, 1.0, step=0.05, label="Contrast γ")
                    tool_diameter_mm = gr.Slider(0.1, 20, 1.0, step=0.1, label="Tool diameter (mm)")
                with gr.Row():
                    invert = gr.Checkbox(False, label="Invert (raised ↔ recessed)")
                    fill_holes = gr.Checkbox(False, label="Fill holes in the mask")
                    remove_bg = gr.Radio(["auto", "on", "off"], value="auto", label="Background masking")
                with gr.Accordion("Advanced (Mode A / resolution)", open=False):
                    profile = gr.Radio(["dome", "pillow", "bevel", "flat_round"], value="dome", label="Mode A profile")
                    transition_width_mm = gr.Slider(0, 30, 0, step=0.5, label="Transition width (mm) — 0 = auto (3% of the long side)")
                    detail = gr.Radio(["draft", "standard", "high"], value="standard", label="Mesh triangle budget")
                    size_choice = gr.Radio(["auto", "518", "756", "1036"], value="auto", label="Inference size (px)")
                go = gr.Button("Generate relief", variant="primary")
            with gr.Column():
                strip = gr.Image(label="input | shaded relief | heightmap | mask")
                viewer = gr.Model3D(label="3D preview (OBJ, mm)")
                report = gr.Markdown()
                zipped = gr.File(label="Download ZIP (STL/OBJ/3MF + 16-bit TIFF + sidecar)")
        go.click(
            generate,
            inputs=[image, mode, model_choice, width_mm, relief_depth_mm, base_thickness_mm, smoothing, gamma, invert, remove_bg, fill_holes, profile, transition_width_mm, detail, tool_diameter_mm, size_choice],
            outputs=[strip, viewer, zipped, report],
        )
        gr.Markdown("Height from a single image is *relative and plausible*, not measured geometry. No undercuts. STL/OBJ carry no units — import as **mm**. Review the result before cutting.")
    return app


if __name__ == "__main__":
    build_app().queue(max_size=4).launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
