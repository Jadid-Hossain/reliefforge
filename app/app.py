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
from PIL import Image
from reliefkit import preview as rk_preview
from reliefkit.io import load_image
from reliefkit.model import ReliefModel
from reliefkit.pipeline import ReliefParams, generate_relief
from reliefkit.preprocess import analyze_image
from reliefkit.studio_relight import preprocess_relief_image

ZERO_SHOT_ID = "depth-anything/Depth-Anything-V2-Small-hf"
FINETUNED_DIR = os.environ.get("RELIEFFORGE_FINETUNED", str(ROOT / "models" / "finetuned"))
FINETUNED_FALLBACK_HF = os.environ.get("RELIEFFORGE_FINETUNED_HF", "")
_model_cache: dict[str, ReliefModel] = {}
_wood_tex_cache: Image.Image | None = None

EMPTY_REPORT_MD = (
    "<div class='rf-empty-report'>"
    "Upload an image on the left and click <strong>Generate 3D Relief</strong> to inspect your model and download files."
    "</div>"
)

MATERIAL_PRESETS: dict[str, dict] = {
    "clay": {
        "rgb": (214, 204, 192),
        "metallic": 0.04,
        "roughness": 0.70,
        "emissive": [0.03, 0.028, 0.025],
        "wood": False,
    },
    "gold": {
        "rgb": (232, 184, 72),
        "metallic": 0.62,
        "roughness": 0.24,
        "emissive": [0.06, 0.042, 0.012],
        "wood": False,
    },
    "bronze": {
        "rgb": (184, 126, 68),
        "metallic": 0.56,
        "roughness": 0.30,
        "emissive": [0.045, 0.028, 0.012],
        "wood": False,
    },
    "wood": {
        "rgb": (152, 102, 58),
        "metallic": 0.06,
        "roughness": 0.42,
        "emissive": [0.028, 0.018, 0.008],
        "wood": True,
    },
    "silver": {
        "rgb": (214, 222, 232),
        "metallic": 0.66,
        "roughness": 0.22,
        "emissive": [0.04, 0.044, 0.05],
        "wood": False,
    },
}

CUSTOM_CSS = """
<style>
.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
}
#rf-header-row {
    align-items: center !important;
    padding: 14px 20px;
    margin-bottom: 12px;
    border-radius: 14px;
    border: 1px solid var(--border-color-primary);
    background: linear-gradient(135deg, rgba(234, 88, 12, 0.10), rgba(15, 23, 42, 0.25));
}
.rf-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.rf-logo-box {
    width: 44px;
    height: 44px;
    border-radius: 11px;
    background: linear-gradient(135deg, #f97316, #c2410c);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 4px 14px rgba(249, 115, 22, 0.28);
}
.rf-title {
    margin: 0 !important;
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    line-height: 1.15;
}
.rf-subtitle {
    margin: 3px 0 0 0 !important;
    font-size: 0.92rem;
    color: var(--body-text-color-subdued);
}
.rf-mode-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 2px 0 12px 0;
}
@media (max-width: 960px) {
    .rf-mode-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 560px) {
    .rf-mode-grid {
        grid-template-columns: 1fr;
    }
}
.rf-mode-card {
    padding: 11px 13px;
    border-radius: 10px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.rf-mode-head {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 4px;
    font-weight: 650;
    font-size: 0.88rem;
}
.rf-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 999px;
    background: rgba(249, 115, 22, 0.16);
    color: #fb923c;
    border: 1px solid rgba(249, 115, 22, 0.35);
}
.rf-mode-card p {
    margin: 0;
    font-size: 0.81rem;
    line-height: 1.4;
    color: var(--body-text-color-subdued);
}
.rf-section-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #fb923c;
    margin: 6px 0 2px 2px;
}
.rf-guide-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    padding: 4px 2px;
}
@media (max-width: 820px) {
    .rf-guide-grid {
        grid-template-columns: 1fr;
    }
}
.rf-guide-box {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}
.rf-guide-box h4 {
    margin: 0 0 8px 0;
    font-size: 0.9rem;
    color: #fb923c;
}
.rf-guide-box ul {
    margin: 0;
    padding-left: 18px;
    font-size: 0.84rem;
    line-height: 1.55;
}
.rf-guide-box li {
    margin-bottom: 4px;
}
.rf-empty-report {
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px dashed var(--border-color-primary);
    color: var(--body-text-color-subdued);
    font-size: 0.88rem;
}
.rf-footer {
    margin-top: 14px;
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
    font-size: 0.82rem;
    color: var(--body-text-color-subdued);
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 8px;
}
/* Keep sliders clean without horizontal scrollbars */
.rf-control-group .block {
    overflow-x: hidden !important;
}
#rf-reset-top {
    min-height: 42px;
    font-weight: 600;
}
#relief-3d-viewer:fullscreen {
    background: #141417 !important;
    width: 100vw !important;
    height: 100vh !important;
    padding: 12px !important;
}
#relief-3d-viewer:fullscreen canvas {
    height: 92vh !important;
}
</style>
"""

HEADER_HTML = """
<div class="rf-brand">
    <div class="rf-logo-box" aria-hidden="true">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L3 7l9 5 9-5-9-5z"></path>
            <path d="M3 17l9 5 9-5"></path>
            <path d="M3 12l9 5 9-5"></path>
        </svg>
    </div>
    <div>
        <h1 class="rf-title">3dRelief</h1>
        <p class="rf-subtitle">Turn artwork, sketches, and wood carving photos into clean, millimetre-accurate 3D models ready for CNC cutting.</p>
    </div>
</div>
"""

MODE_GUIDE_HTML = """
<div class="rf-mode-grid">
    <div class="rf-mode-card">
        <div class="rf-mode-head"><span class="rf-badge">Auto</span> Smart Detection</div>
        <p>Inspects your uploaded image and automatically selects Mode A, B, or C for you.</p>
    </div>
    <div class="rf-mode-card">
        <div class="rf-mode-head"><span class="rf-badge">Mode A</span> Flat Artwork</div>
        <p>Best for black-and-white line art, logos, stencils, and flat vector ornaments without shading.</p>
    </div>
    <div class="rf-mode-card">
        <div class="rf-mode-head"><span class="rf-badge">Mode B</span> Photo &amp; Carving</div>
        <p>Best for real wood carving photos, furniture panels, sculptures, and shaded 3D renders.</p>
    </div>
    <div class="rf-mode-card">
        <div class="rf-mode-head"><span class="rf-badge">Mode C</span> Grayscale Heightmap</div>
        <p>Best when you already have a grayscale depth map or alpha brush (bright = raised, dark = recessed).</p>
    </div>
</div>
"""

TUTORIAL_HTML = """
<div class="rf-guide-grid">
    <div class="rf-guide-box">
        <h4>1. Quick Start &amp; Photo Cleanup</h4>
        <ul>
            <li><strong>Clean &amp; Relight Raw Photo:</strong> Removes wood grain, stains, watermarks, and harsh side shadows before building the 3D surface. Turn <strong>ON</strong> for real workshop photos; leave <strong>OFF</strong> for flat art or clean studio renders.</li>
            <li><strong>Conversion Mode:</strong> Leave on <strong>Auto</strong> if unsure, or manually pick Mode A (flat shapes), Mode B (real photos), or Mode C (grayscale heightmaps).</li>
            <li><strong>Mode B AI Model:</strong> <em>Fine-tuned</em> is trained specifically for carved panels; <em>Zero-shot</em> uses the general base model for comparison.</li>
        </ul>
    </div>
    <div class="rf-guide-box">
        <h4>2. Physical Dimensions (mm)</h4>
        <ul>
            <li><strong>Physical Width:</strong> Final width of your 3D piece in millimetres. Height adjusts automatically to preserve proportions.</li>
            <li><strong>Carving Depth:</strong> How tall the raised carving rises above the background plane (e.g., 6&ndash;15 mm for furniture panels).</li>
            <li><strong>Base Slab:</strong> Solid backing slab added beneath the carving so the exported 3D model is watertight and machinable.</li>
        </ul>
    </div>
    <div class="rf-guide-box">
        <h4>3. Surface &amp; Carving Controls</h4>
        <ul>
            <li><strong>Smoothing (&sigma;):</strong> Softens noisy pixels and rough edges. Keep low (<strong>0.5&ndash;0.8</strong>) for crisp carving details, or higher (<strong>1.2&ndash;2.0</strong>) to calm noisy photos.</li>
            <li><strong>Depth Contrast (&gamma;):</strong> Adjusts depth curve. Values above <strong>1.0</strong> make mid-level carvings pop deeper; values below <strong>1.0</strong> soften steep transitions.</li>
            <li><strong>CNC Bit Diameter:</strong> Simulates your carving bit size (mm) to warn if details are narrower than your cutter.</li>
            <li><strong>Invert Relief:</strong> Flips raised ridges into recessed pockets (engraving / mold mode).</li>
            <li><strong>Fill Mask Holes:</strong> Closes small unwanted gaps inside the foreground design mask.</li>
            <li><strong>Flatten Background:</strong> Keeps the outer background completely flat at zero height (<em>Auto</em> decides automatically).</li>
        </ul>
    </div>
    <div class="rf-guide-box">
        <h4>4. 3D Preview Materials &amp; Advanced Options</h4>
        <ul>
            <li><strong>3D Material Finish:</strong> Switch instantly between <em>Clay</em>, <em>Gold</em>, <em>Antique Bronze</em>, <em>Carved Wood</em>, and <em>Silver</em> without re-running the AI. Metallic gold/bronze highlights make subtle carving depth easy to inspect.</li>
            <li><strong>Lighting Style:</strong> <em>Studio</em> gives balanced sheen, <em>High Gloss</em> maximizes bright edge reflections, and <em>Raking Light</em> adds angled shadow &amp; cavity shading so every groove stands out clearly.</li>
            <li><strong>Mode A Edge Shape &amp; Mesh Quality:</strong> Choose cross-section shape for flat art (Dome/Pillow/Bevel/Flat Round) and adjust triangle density (Draft/Standard/High).</li>
        </ul>
    </div>
</div>
"""

FOOTER_HTML = """
<div class="rf-footer">
    <span><strong>3dRelief</strong> &mdash; Single-image 3D relief generator. All dimensions are scaled in <strong>millimetres (mm)</strong>.</span>
    <span>Tip: When opening STL or OBJ files in ArtCAM, Aspire, or Fusion 360, confirm import units are set to <strong>mm</strong> and check toolpaths before cutting.</span>
</div>
"""


def _get_wood_texture(size: int = 512) -> Image.Image:
    global _wood_tex_cache
    if _wood_tex_cache is not None:
        return _wood_tex_cache
    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    warp = (
        0.036 * np.sin(2.0 * np.pi * x * 2.0 + 0.6)
        + 0.018 * np.sin(2.0 * np.pi * x * 5.0 + 2.2 * y)
        + 0.010 * np.cos(2.0 * np.pi * (x * 3.0 - y * 2.0))
    )
    rings = np.sin(2.0 * np.pi * (y * 13.0 + warp * 13.0))
    fine = np.sin(2.0 * np.pi * (y * 58.0 + warp * 20.0)) * 0.25
    rng = np.random.default_rng(42)
    fiber = rng.normal(0.0, 0.045, (size, size)).astype(np.float32)
    fiber = 0.5 * fiber + 0.25 * np.roll(fiber, 1, axis=1) + 0.25 * np.roll(fiber, -1, axis=1)
    t = np.clip(0.5 + 0.34 * rings + 0.14 * fine + fiber, 0.0, 1.0)
    dark = np.array([86, 48, 24], dtype=np.float32)
    light = np.array([182, 126, 74], dtype=np.float32)
    rgb = dark * (1.0 - t[..., None]) + light * t[..., None]
    _wood_tex_cache = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return _wood_tex_cache


def _export_styled_glb(obj_path: str, material_key: str = "clay", lighting_key: str = "studio") -> str:
    """Create a styled GLB preview with PBR reflectivity, wood texture, or raking relief lighting."""
    preset = MATERIAL_PRESETS.get(material_key, MATERIAL_PRESETS["clay"])
    out_dir = Path(obj_path).parent
    glb_path = out_dir / f"preview_{material_key}_{lighting_key}.glb"
    if glb_path.exists():
        return str(glb_path)

    try:
        import trimesh
        from trimesh.visual import ColorVisuals, TextureVisuals
        from trimesh.visual.material import PBRMaterial

        mesh = trimesh.load(obj_path, force="mesh")
        verts = np.asarray(mesh.vertices, dtype=np.float32)

        if lighting_key == "raking":
            # Crisp 35-deg directional studio raking light + Z-depth cavity darkening
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
            n_len = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.maximum(n_len, 1e-6)

            key_dir = np.array([0.58, 0.44, 0.68], dtype=np.float32)
            key_dir /= np.linalg.norm(key_dir)
            fill_dir = np.array([-0.46, -0.24, 0.85], dtype=np.float32)
            fill_dir /= np.linalg.norm(fill_dir)
            view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            diff_key = np.clip(normals @ key_dir, 0.0, 1.0)
            diff_fill = np.clip(normals @ fill_dir, 0.0, 1.0)
            half_vec = key_dir + view_dir
            half_vec /= np.linalg.norm(half_vec)
            spec_pow = 34.0 if preset["metallic"] > 0.3 else 18.0
            spec_gain = 120.0 if preset["metallic"] > 0.3 else 55.0
            spec = np.power(np.clip(normals @ half_vec, 0.0, 1.0), spec_pow)

            z = verts[:, 2]
            z_span = max(float(z.max() - z.min()), 1e-5)
            z01 = (z - z.min()) / z_span
            cavity = 0.66 + 0.42 * np.power(z01, 0.72)

            shade = (0.24 + 0.74 * diff_key + 0.22 * diff_fill) * cavity

            if preset["wood"]:
                tex_arr = np.asarray(_get_wood_texture(512), dtype=np.float32)
                h, w = tex_arr.shape[:2]
                min_xy = verts[:, :2].min(axis=0)
                span_xy = np.maximum(verts[:, :2].max(axis=0) - min_xy, 1e-6)
                u = np.clip(((verts[:, 0] - min_xy[0]) / span_xy[0]) * (w - 1), 0, w - 1).astype(np.int32)
                v = np.clip(((verts[:, 1] - min_xy[1]) / span_xy[1]) * (h - 1), 0, h - 1).astype(np.int32)
                base = tex_arr[v, u]
            else:
                base = np.full((len(verts), 3), preset["rgb"], dtype=np.float32)

            rgb = np.clip(base * shade[:, None] + spec[:, None] * spec_gain, 0, 255).astype(np.uint8)
            rgba = np.hstack([rgb, np.full((len(verts), 1), 255, dtype=np.uint8)])
            mesh.visual = ColorVisuals(mesh=mesh, vertex_colors=rgba)
        else:
            # Real glTF 2.0 PBR material with metallic sheen & optional procedural wood texture
            min_xy = verts[:, :2].min(axis=0)
            span_xy = np.maximum(verts[:, :2].max(axis=0) - min_xy, 1e-6)
            uv = (verts[:, :2] - min_xy) / span_xy

            metallic = float(preset["metallic"])
            roughness = float(preset["roughness"])
            emissive = list(preset["emissive"])

            if lighting_key == "gloss":
                roughness = max(0.10, roughness * 0.55)
                metallic = min(0.82, metallic + 0.14 if not preset["wood"] else 0.14)
                emissive = [min(0.12, c * 1.35) for c in emissive]

            if preset["wood"]:
                base_tex = _get_wood_texture(512)
            else:
                base_tex = Image.new("RGB", (16, 16), preset["rgb"])

            mat = PBRMaterial(
                name=f"{material_key}_{lighting_key}",
                baseColorTexture=base_tex,
                metallicFactor=metallic,
                roughnessFactor=roughness,
                emissiveFactor=emissive,
            )
            mesh.visual = TextureVisuals(uv=uv, material=mat)

        mesh.export(str(glb_path))
        return str(glb_path)
    except Exception:
        traceback.print_exc()
        return obj_path


def restyle_3d_preview(obj_path: str | None, material_key: str, lighting_key: str):
    """Instantly restyle the current 3D preview without re-running inference."""
    if not obj_path or not Path(obj_path).exists():
        return None
    return _export_styled_glb(obj_path, material_key, lighting_key)


def get_model(which: str) -> ReliefModel:
    if which in _model_cache:
        return _model_cache[which]
    if which == "fine-tuned":
        if Path(FINETUNED_DIR).exists():
            model = ReliefModel(FINETUNED_DIR)
        elif FINETUNED_FALLBACK_HF:
            model = ReliefModel(FINETUNED_FALLBACK_HF)
        else:
            model = ReliefModel(ZERO_SHOT_ID)
            model.info_note = "Fine-tuned weights not found locally; running base zero-shot model."
    else:
        model = ReliefModel(ZERO_SHOT_ID)
    _model_cache[which] = model
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
    preprocess_image: bool = False,
    material_key: str = "clay",
    lighting_key: str = "studio",
):
    if image is None:
        raise gr.Error("Please upload an image first before generating a 3D relief.")

    out_dir = Path(tempfile.mkdtemp(prefix="3drelief_"))
    img = load_image(image)
    rgb_in, alpha_in, preprocess_note = img.rgb, img.alpha, None

    if preprocess_image:
        rgb_in, ok, status = preprocess_relief_image(rgb_in)
        preprocess_note = f"**Photo cleanup:** {status}"
        if ok:
            alpha_in = None
            if mode == "auto":
                mode = "B"

    tw = "auto" if transition_width_mm <= 0 else float(transition_width_mm)
    size = "auto" if size_choice == "auto" else int(size_choice)

    params = ReliefParams(
        width_mm=width_mm,
        relief_depth_mm=relief_depth_mm,
        base_thickness_mm=base_thickness_mm,
        smoothing=smoothing,
        gamma=gamma,
        invert=invert,
        remove_bg=remove_bg,
        fill_holes=fill_holes,
        mode=mode,
        profile=profile,
        transition_width_mm=tw,
        detail=detail,
        tool_diameter_mm=tool_diameter_mm,
        inference_size=size,
        model=model_choice,
    )

    analysis = analyze_image(rgb_in)
    model = get_model(model_choice) if params.mode in ("auto", "B") else None

    try:
        result = generate_relief(rgb_in, alpha_in, params, model=model, analysis=analysis)
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(f"3D generation failed: {exc}")

    exported = result.save(out_dir)
    strip = rk_preview.preview_figure(rgb_in, result.h01, result.mask)

    watertight_text = "Yes (watertight)" if result.validation["is_watertight"] else "Check mesh"
    report_lines = [
        "### Model Summary",
        f"- **Mode used:** Mode {result.info['mode_used']} *(auto-suggested: {result.info['mode_suggested']})*",
        f"- **Physical size:** {result.meta['width_mm']:.0f} &times; {result.meta['height_mm']:.0f} &times; {result.meta['z_max_mm']:.1f} mm",
        f"- **3D mesh:** {result.validation['n_triangles']:,} triangles &middot; Solid: {watertight_text}",
        f"- **Resolution:** {result.info['S']} px &middot; {result.meta['pixel_pitch_mm']:.3f} mm/px",
    ]
    if preprocess_note:
        report_lines.append(f"- {preprocess_note}")
    if result.warnings:
        report_lines.append("- **Notes:** " + "; ".join(result.warnings))
    if getattr(model, "info_note", None):
        report_lines.append(f"- **Model note:** {model.info_note}")

    obj_path = str(exported["obj"])
    preview_3d = _export_styled_glb(obj_path, material_key, lighting_key)

    return strip, preview_3d, str(exported["zip"]), "\n".join(report_lines), obj_path


def reset_workspace():
    """Stop active processing (via cancels) and reset all inputs and outputs to defaults."""
    return (
        None,            # image
        "auto",          # mode
        "fine-tuned",    # model_choice
        300,             # width_mm
        8,               # relief_depth_mm
        6,               # base_thickness_mm
        1.0,             # smoothing
        1.0,             # gamma
        False,           # invert
        "auto",          # remove_bg
        False,           # fill_holes
        "dome",          # profile
        0,               # transition_width_mm
        "standard",      # detail
        1.0,             # tool_diameter_mm
        "auto",          # size_choice
        False,           # preprocess_image
        "clay",          # material_preset
        "studio",        # lighting_preset
        None,            # strip
        None,            # viewer
        None,            # zipped
        EMPTY_REPORT_MD, # report
        None,            # mesh_obj_state
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="3dRelief — Image to 3D CNC Relief") as app:
        gr.HTML(CUSTOM_CSS)
        mesh_obj_state = gr.State(value=None)

        # Header Bar with Global Refresh / Reset button
        with gr.Row(elem_id="rf-header-row"):
            with gr.Column(scale=5, min_width=280):
                gr.HTML(HEADER_HTML)
            with gr.Column(scale=1, min_width=180):
                reset_top_btn = gr.Button(
                    "Reset / New Project",
                    variant="secondary",
                    elem_id="rf-reset-top",
                )

        # Mode Selection Visual Guide
        gr.HTML(MODE_GUIDE_HTML)

        # Collapsible Tutorial & Settings Guide Dropdown
        with gr.Accordion("Tutorial & Settings Guide — Click to see what every option does", open=False):
            gr.HTML(TUTORIAL_HTML)

        # Main Two-Column Studio Workspace
        with gr.Row(equal_height=False):
            # Left Column: Input & Controls
            with gr.Column(scale=5, min_width=400, elem_classes=["rf-control-group"]):
                gr.HTML("<div class='rf-section-label'>1. Input Image</div>")
                image = gr.Image(type="filepath", label="Upload Design or Photo (JPG, PNG, WebP)", height=250)
                preprocess_image = gr.Checkbox(
                    False,
                    label="Clean & relight raw photo (recommended for wood photos)",
                    info="Removes wood grain, stains, shadows, and workshop clutter before generating 3D depth.",
                )

                gr.HTML("<div class='rf-section-label'>2. Conversion Mode</div>")
                mode = gr.Radio(
                    choices=[
                        ("Auto Detect", "auto"),
                        ("Mode A — Flat Art", "A"),
                        ("Mode B — Photo / Carving", "B"),
                        ("Mode C — Heightmap", "C"),
                    ],
                    value="auto",
                    label="Processing Mode",
                )
                model_choice = gr.Radio(
                    choices=[
                        ("Fine-tuned (Carving Specialist)", "fine-tuned"),
                        ("Zero-shot (Base Model)", "zero-shot"),
                    ],
                    value="fine-tuned",
                    label="Mode B AI Model",
                    info="Used when Mode B is active.",
                )

                gr.HTML("<div class='rf-section-label'>3. Physical Size &amp; Depth (mm)</div>")
                with gr.Group():
                    width_mm = gr.Slider(
                        20, 2000, 300, step=5,
                        label="Physical Width (mm)",
                        info="Final width of the 3D piece (height scales automatically).",
                    )
                    relief_depth_mm = gr.Slider(
                        0.5, 100, 8, step=0.5,
                        label="Carving Depth (mm)",
                        info="How tall the raised carving stands above the base plane.",
                    )
                    base_thickness_mm = gr.Slider(
                        0.5, 100, 6, step=0.5,
                        label="Base Slab Thickness (mm)",
                        info="Solid backing thickness added beneath the relief.",
                    )

                gr.HTML("<div class='rf-section-label'>4. Surface &amp; Detail Tuning</div>")
                with gr.Group():
                    smoothing = gr.Slider(
                        0, 4, 1.0, step=0.1,
                        label="Smoothing (px)",
                        info="Lower (0.5-0.8) keeps grooves crisp; higher softens noise.",
                    )
                    gamma = gr.Slider(
                        0.3, 3, 1.0, step=0.05,
                        label="Depth Contrast",
                        info="Values above 1.0 make mid-level carvings pop deeper.",
                    )
                    tool_diameter_mm = gr.Slider(
                        0.1, 20, 1.0, step=0.1,
                        label="CNC Bit Diameter (mm)",
                        info="Used to check if fine grooves are narrower than your cutter.",
                    )

                with gr.Group():
                    with gr.Row():
                        invert = gr.Checkbox(
                            False,
                            label="Invert Relief",
                            info="Swap raised and recessed",
                        )
                        fill_holes = gr.Checkbox(
                            False,
                            label="Fill Mask Holes",
                            info="Close small inner gaps",
                        )
                    remove_bg = gr.Radio(
                        choices=[("Auto", "auto"), ("On", "on"), ("Off", "off")],
                        value="auto",
                        label="Flatten Background",
                        info="Keep the outer background completely flat at zero height.",
                    )

                with gr.Accordion("Advanced Options (Edge Shape & Mesh Quality)", open=False):
                    profile = gr.Radio(
                        choices=[
                            ("Dome", "dome"),
                            ("Pillow", "pillow"),
                            ("Bevel", "bevel"),
                            ("Flat Round", "flat_round"),
                        ],
                        value="dome",
                        label="Mode A Edge Shape",
                        info="Cross-section profile for flat artwork.",
                    )
                    transition_width_mm = gr.Slider(
                        0, 30, 0, step=0.5,
                        label="Edge Transition Width (mm)",
                        info="0 = automatic (3% of longest side).",
                    )
                    detail = gr.Radio(
                        choices=[("Draft", "draft"), ("Standard", "standard"), ("High", "high")],
                        value="standard",
                        label="Mesh Quality",
                        info="Higher quality creates a denser, smoother 3D mesh.",
                    )
                    size_choice = gr.Radio(
                        choices=[("Auto", "auto"), ("518 px", "518"), ("756 px", "756"), ("1036 px", "1036")],
                        value="auto",
                        label="Processing Resolution",
                        info="Higher resolution preserves finer details at slightly slower speed.",
                    )

                with gr.Row():
                    go = gr.Button("Generate 3D Relief", variant="primary", scale=3)
                    reset_bottom_btn = gr.Button("Stop & Reset", variant="secondary", scale=1)

            # Right Column: 3D Preview, Live Material/Lighting, Breakdown & Download
            with gr.Column(scale=7, min_width=440):
                gr.HTML("<div class='rf-section-label'>Interactive 3D Preview</div>")
                viewer = gr.Model3D(
                    label="3D Model Preview (drag to rotate, scroll to zoom)",
                    elem_id="relief-3d-viewer",
                    height=470,
                )

                with gr.Group():
                    material_preset = gr.Radio(
                        choices=[
                            ("Matte Clay", "clay"),
                            ("Polished Gold", "gold"),
                            ("Antique Bronze", "bronze"),
                            ("Carved Wood", "wood"),
                            ("Brushed Silver", "silver"),
                        ],
                        value="clay",
                        label="3D Preview Material (switches instantly)",
                    )
                    with gr.Row():
                        lighting_preset = gr.Radio(
                            choices=[
                                ("Studio Balanced", "studio"),
                                ("High Gloss", "gloss"),
                                ("Raking Light (Deep Detail)", "raking"),
                            ],
                            value="studio",
                            label="Lighting & Surface Sheen",
                            scale=3,
                        )
                        fullscreen_btn = gr.Button("Fullscreen 3D (Esc to exit)", size="sm", scale=1)

                gr.HTML("<div class='rf-section-label'>Processing Breakdown</div>")
                strip = gr.Image(label="Input  |  Shaded 3D Relief  |  Heightmap  |  Mask")

                with gr.Row():
                    with gr.Column(scale=3):
                        report = gr.Markdown(EMPTY_REPORT_MD)
                    with gr.Column(scale=2):
                        zipped = gr.File(label="Download 3D Package (STL, OBJ, 3MF + 16-bit Heightmap)")

        gr.HTML(FOOTER_HTML)

        fullscreen_btn.click(
            None,
            None,
            None,
            js="()=>{const el=document.getElementById('relief-3d-viewer');if(!el)return;if(!document.fullscreenElement){el.requestFullscreen().catch(()=>{});}else{document.exitFullscreen().catch(()=>{});}}",
        )

        all_inputs = [
            image, mode, model_choice, width_mm, relief_depth_mm,
            base_thickness_mm, smoothing, gamma, invert, remove_bg,
            fill_holes, profile, transition_width_mm, detail,
            tool_diameter_mm, size_choice, preprocess_image,
            material_preset, lighting_preset,
        ]
        all_outputs = [strip, viewer, zipped, report, mesh_obj_state]
        reset_targets = all_inputs + all_outputs

        gen_event = go.click(
            generate,
            inputs=all_inputs,
            outputs=all_outputs,
        )

        # Live material & lighting switching without re-running inference
        material_preset.change(
            restyle_3d_preview,
            inputs=[mesh_obj_state, material_preset, lighting_preset],
            outputs=[viewer],
            queue=False,
        )
        lighting_preset.change(
            restyle_3d_preview,
            inputs=[mesh_obj_state, material_preset, lighting_preset],
            outputs=[viewer],
            queue=False,
        )

        reset_top_btn.click(
            reset_workspace,
            inputs=None,
            outputs=reset_targets,
            cancels=[gen_event],
            queue=False,
        )
        reset_bottom_btn.click(
            reset_workspace,
            inputs=None,
            outputs=reset_targets,
            cancels=[gen_event],
            queue=False,
        )

    return app


if __name__ == "__main__":
    build_app().queue(max_size=4).launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
    )
