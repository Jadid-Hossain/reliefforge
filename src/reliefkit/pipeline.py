"""The one entry point the app and the CLI use: image -> validated relief files."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from . import mesh as rk_mesh
from . import modes as rk_modes
from . import postprocess as rk_post
from . import preprocess as rk_pre
from .export import build_meta, export_all
from .preprocess import resize_arr, resize_mask

APP_VERSION = "reliefforge-0.1.0"
SLOPE_WARN_DEG = 70.0
PITCH_TOOL_FACTOR = 3.0


@dataclass
class ReliefParams:
    width_mm: float = 300.0
    relief_depth_mm: float = 8.0
    base_thickness_mm: float = 6.0
    smoothing: float = 1.0
    gamma: float = 1.0
    invert: bool = False
    remove_bg: str = "auto"  # auto | on | off | rembg
    fill_holes: bool = False
    mode: str = "auto"  # auto | A | B | C
    profile: str = "dome"  # Mode A profile
    edge: float = 0.5
    transition_width_mm: float | str = "auto"
    r_px: float | None = None
    detail: str = "standard"
    tool_diameter_mm: float = 1.0
    inference_size: int | str = "auto"
    tta: bool = False
    invert_luminance: bool = False  # Mode C only
    model: str = "fine-tuned"  # fine-tuned | zero-shot

    def validate(self) -> None:
        if not (20 <= self.width_mm <= 2000):
            raise ValueError("width_mm must be in [20, 2000]")
        if not (0.5 <= self.relief_depth_mm <= 100):
            raise ValueError("relief_depth_mm must be in [0.5, 100]")
        if not (0.5 <= self.base_thickness_mm <= 100):
            raise ValueError(f"base_thickness_mm must be >= 0.5 mm for a solid (got {self.base_thickness_mm})")
        if not (0 <= self.smoothing <= 4):
            raise ValueError("smoothing must be in [0, 4]")
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.remove_bg not in ("auto", "on", "off", "rembg"):
            raise ValueError("remove_bg must be auto|on|off|rembg")
        if self.mode not in ("auto", "A", "B", "C"):
            raise ValueError("mode must be auto|A|B|C")
        if self.profile not in rk_modes.PROFILES:
            raise ValueError(f"profile must be one of {rk_modes.PROFILES}")
        if self.detail not in rk_mesh.TRIANGLE_BUDGETS:
            raise ValueError(f"detail must be one of {sorted(rk_mesh.TRIANGLE_BUDGETS)}")
        if not (0.1 <= self.tool_diameter_mm <= 20):
            raise ValueError("tool_diameter_mm must be in [0.1, 20]")
        if self.transition_width_mm not in (None, "auto") and float(self.transition_width_mm) <= 0:
            raise ValueError("transition_width_mm must be positive")


@dataclass
class ReliefResult:
    h01: np.ndarray
    mask: np.ndarray | None
    mesh: object
    mesh_info: dict
    validation: dict
    warnings: list[str]
    info: dict
    meta: dict
    exported: dict[str, object] = field(default_factory=dict)

    def save(self, out_dir) -> dict:
        self.exported = export_all(
            self.h01, self.mesh, self.mesh_info, self.validation, out_dir, self.meta,
            make_zip=True, skip_3mf=False,
        )
        return self.exported


def max_slope_deg(h01: np.ndarray, depth_mm: float, pitch_mm: float) -> float:
    gy, gx = np.gradient(np.asarray(h01, np.float32))
    return float(np.degrees(np.arctan(np.hypot(gx, gy).max() * depth_mm / max(pitch_mm, 1e-9))))


def generate_relief(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    params: ReliefParams,
    model=None,
    *,
    analysis: dict | None = None,
) -> ReliefResult:
    """Run the full pipeline on an RGB uint8 image. `model` is a ReliefModel (Mode B)."""
    params.validate()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb must be H x W x 3 uint8")
    info: dict = {"params": {k: v for k, v in asdict(params).items()}, "warnings": []}
    warnings: list[str] = info["warnings"]
    H_src, W_src = rgb.shape[:2]

    # 1) diagnostics + mask (on the source grid)
    a = analysis if analysis is not None else rk_pre.analyze_image(rgb)
    mode_suggested, _ = rk_pre.suggest_mode(rgb)
    info["mode_suggested"] = mode_suggested
    mode = mode_suggested if params.mode == "auto" else params.mode

    mask, mask_info = rk_pre.get_mask(rgb, alpha, params.remove_bg, analysis=a)
    info["mask"] = mask_info

    # 2) working resolution
    height_mm = params.width_mm * H_src / W_src
    s_info = rk_pre.choose_inference_size(
        max(H_src, W_src), params.width_mm, height_mm, params.tool_diameter_mm,
        device=(getattr(model, "device", "cpu") if model is not None else "cpu"),
        override=params.inference_size, mode=mode,
    )
    S = s_info.S
    info["S"] = S
    info["size_info"] = {"target_px": round(s_info.target_px, 1), "capped_by_source": s_info.capped_by_source, "capped_by_device": s_info.capped_by_device}
    if s_info.capped_by_source:
        warnings.append(f"inference size capped by the source resolution (S = {S})")
    if s_info.capped_by_device:
        warnings.append("running on CPU: S reduced to 518 for speed")

    # Mode B works on an S-sized grid (the model letterboxes internally); the classic
    # modes keep the source resolution capped at 1036 px.
    if mode == "B":
        work_rgb = resize_arr(rgb, int(round(W_src * S / max(H_src, W_src))), int(round(H_src * S / max(H_src, W_src))))
    else:
        f = min(1.0, 1036 / max(H_src, W_src))
        work_rgb = rgb if f == 1.0 else resize_arr(rgb, int(round(W_src * f)), int(round(H_src * f)))
    Hh, Wh = work_rgb.shape[:2]
    work_mask = None if mask is None else resize_mask(mask, Wh, Hh)

    # 3) height
    pixel_pitch_mm = params.width_mm / Wh
    if mode == "A":
        h_raw, mode_info = rk_modes.design_to_height(
            work_rgb, work_mask, profile=params.profile, edge=params.edge, r_px=params.r_px,
            transition_width_mm=params.transition_width_mm, pixel_pitch_mm=pixel_pitch_mm,
            fill=params.fill_holes,
        )
    elif mode == "C":
        h_raw, mode_info = rk_modes.gray_to_height(work_rgb, work_mask, gamma=params.gamma, invert_luminance=params.invert_luminance)
    elif mode == "B":
        if model is None:
            raise ValueError("mode B needs a ReliefModel instance")
        from .tile_fusion import boost_relief
        dev = str(getattr(model, "device", "cpu"))
        long_side = 1400 if dev == "cpu" else 2048
        tf_out = boost_relief(
            rgb,
            predict_fn=lambda im: model.predict(im, S=518, tta=False),
            global_predict_fn=lambda im: model.predict(im, S=S, tta=params.tta),
            long_side=long_side,
            tile=518,
            overlap=0.45,
            flip_tta=(dev != "cpu"),
            mask=mask,
        )
        h_raw = tf_out["height"]
        Hh, Wh = h_raw.shape[:2]
        work_mask = None if mask is None else resize_mask(mask, Wh, Hh)
        mode_info = {
            "model": getattr(model, "model_id", "?"),
            "S": S,
            "tta": params.tta,
            "tile_fusion": True,
            "n_tiles": tf_out["n_tiles"],
            "fine_shape": [Hh, Wh],
        }
    else:
        raise ValueError(f"unknown mode {mode}")
    info["mode_used"] = mode
    info["mode_info"] = mode_info

    # 4) post-processing (fixed order)
    h01, post_info = rk_post.finalize_height(
        h_raw, work_mask, invert=params.invert, gamma=params.gamma, smoothing=params.smoothing,
    )
    info["postprocess"] = {k: v for k, v in post_info.items() if k != "warnings"}
    warnings.extend(post_info.get("warnings", []))

    # 5) mesh + validation
    mesh, mesh_info = rk_mesh.build_relief_mesh(
        h01, params.width_mm, params.relief_depth_mm, params.base_thickness_mm, detail=params.detail,
    )
    validation = rk_mesh.validate_mesh(mesh, params.width_mm, params.width_mm * Hh / Wh, mesh_info["expected_z_max_mm"])
    if not validation["is_valid"]:
        raise rk_mesh.MeshError(f"mesh validation failed: {validation}")

    # 6) manufacturing warnings
    pitch = params.width_mm / Wh
    if params.tool_diameter_mm and pitch > params.tool_diameter_mm / PITCH_TOOL_FACTOR:
        warnings.append(f"pixel pitch {pitch:.3f} mm > tool/3 = {params.tool_diameter_mm / PITCH_TOOL_FACTOR:.3f} mm: fine details may be lost")
    slope = max_slope_deg(h01, params.relief_depth_mm, pitch)
    info["max_slope_deg"] = round(slope, 1)
    if slope > SLOPE_WARN_DEG:
        warnings.append(f"maximum slope {slope:.0f} deg: check the tool and the finishing pass")
    if mesh_info["downsampled"]:
        warnings.append(f"mesh downsampled to the triangle budget (mesh pitch {mesh_info['mesh_pitch_x_mm']:.3f} mm; the TIFF keeps full resolution)")

    meta = build_meta(
        width_mm=params.width_mm, height_mm=params.width_mm * Hh / Wh,
        relief_depth_mm=params.relief_depth_mm, base_mm=params.base_thickness_mm,
        h_shape=h01.shape, mesh_info=mesh_info, validation=validation, mode=mode,
        model_version=(getattr(model, "model_id", "none") if mode == "B" else f"classic-{mode}"),
        invert=params.invert, infer_long_side=S, warnings=warnings,
        extra={"app_version": APP_VERSION, "mask_source": (mask_info or {}).get("source")},
    )
    return ReliefResult(h01=h01, mask=work_mask, mesh=mesh, mesh_info=mesh_info, validation=validation, warnings=warnings, info=info, meta=meta)
