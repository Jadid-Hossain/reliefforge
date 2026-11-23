"""ReliefForge — 2D decorative design -> CAM-importable 2.5D CNC relief."""
from .io import load_image, ImageData, ImageLoadError
from .preprocess import LetterboxMeta, letterbox, unletterbox, suggest_mode, get_mask
from .mesh import build_relief_mesh, validate_mesh, MeshError
from .modes import design_to_height, gray_to_height
from .postprocess import finalize_height
from .model import ReliefModel
from .pipeline import ReliefParams, ReliefResult, generate_relief, APP_VERSION

__version__ = APP_VERSION
__all__ = [
    "load_image", "ImageData", "ImageLoadError",
    "LetterboxMeta", "letterbox", "unletterbox", "suggest_mode", "get_mask",
    "build_relief_mesh", "validate_mesh", "MeshError",
    "design_to_height", "gray_to_height", "finalize_height",
    "ReliefModel", "ReliefParams", "ReliefResult", "generate_relief", "APP_VERSION",
]
