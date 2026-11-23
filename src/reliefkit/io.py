"""M1 — image input/output and the height <-> uint16 conversion used everywhere."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class ImageLoadError(ValueError):
    """Raised when a file cannot be decoded as a usable design image."""


@dataclass
class ImageData:
    rgb: np.ndarray  # uint8, H x W x 3
    alpha: np.ndarray | None = None  # uint8, H x W (255 = opaque)
    source: str = ""
    exif_applied: bool = False

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])


def _to_8bit(arr: np.ndarray) -> np.ndarray:
    """Scale a grey array of any bit depth to uint8."""
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float64)
    lo, hi = float(a.min()), float(a.max())
    if a.dtype.kind == "f" and hi <= 1.5:  # float image stored in [0, 1]
        return np.clip(a * 255.0, 0, 255).round().astype(np.uint8)
    if hi <= lo:
        return np.zeros(a.shape, np.uint8) if hi <= 0 else np.full(a.shape, 255, np.uint8)
    if hi <= 255.0:
        return np.clip(a, 0, 255).astype(np.uint8)
    top = 65535.0 if hi > 4095 else 255.0
    return np.clip(a / top * 255.0, 0, 255).round().astype(np.uint8)


def _grey_to_rgb(g: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.stack([g, g, g], axis=-1))


def load_image(path: str | Path, max_long_side: int | None = None) -> ImageData:
    """Load JPG/PNG/WEBP/BMP/TIFF of any bit depth as RGB uint8 (+ alpha when present).

    EXIF rotation is applied. When `max_long_side` is given the image is area-downscaled
    once, here, so that nothing downstream ever works on a 50-megapixel photo.
    """
    path = Path(path)
    if not path.is_file():
        raise ImageLoadError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(f"unsupported image type '{path.suffix}' (use {', '.join(SUPPORTED_EXTENSIONS)})")
    try:
        with Image.open(path) as im:
            im.load()
            exif = im.getexif() if hasattr(im, "getexif") else {}
            rotated = exif.get(0x0112, 1) not in (None, 1)
            im = ImageOps.exif_transpose(im)
            mode = im.mode
            alpha = None
            if mode in ("RGBA", "LA", "PA") or (mode == "P" and "transparency" in im.info):
                arr = np.asarray(im.convert("RGBA"))
                rgb, alpha = arr[..., :3].copy(), arr[..., 3].copy()
            elif mode == "RGB":
                rgb = np.asarray(im)[..., :3].copy()
            elif mode == "L":
                rgb = _grey_to_rgb(np.asarray(im))
            elif mode in ("I;16", "I;16B", "I;16L", "I", "F"):
                rgb = _grey_to_rgb(_to_8bit(np.asarray(im)))
            else:  # palette without transparency, CMYK, YCbCr, LAB, ... -> let Pillow convert
                rgb = np.asarray(im.convert("RGB"))[..., :3].copy()
    except ImageLoadError:
        raise
    except Exception as exc:  # corrupt / truncated file
        raise ImageLoadError(f"could not decode image {path.name}: {exc}") from exc

    if max_long_side is not None and max(rgb.shape[:2]) > max_long_side:
        import cv2

        scale = max_long_side / max(rgb.shape[:2])
        size = (max(1, round(rgb.shape[1] * scale)), max(1, round(rgb.shape[0] * scale)))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = None if alpha is None else cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    return ImageData(rgb=np.ascontiguousarray(rgb), alpha=alpha, source=str(path), exif_applied=rotated)


# ------------------------------------------------------------------ height <-> uint16
def height_to_uint16(h01: np.ndarray) -> np.ndarray:
    """h in [0, 1] -> uint16 (0 = base plane, 65535 = highest point)."""
    h = np.asarray(h01, dtype=np.float32)
    if not np.isfinite(h).all():
        raise ValueError("height map contains NaN or inf")
    return np.clip(np.round(h * 65535.0), 0, 65535).astype(np.uint16)


def uint16_to_height(u16: np.ndarray) -> np.ndarray:
    return np.asarray(u16, dtype=np.float32) / 65535.0


def save_height_tiff(h01: np.ndarray, path: str | Path) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), height_to_uint16(h01), [cv2.IMWRITE_TIFF_COMPRESSION, 1]):
        raise OSError(f"could not write {path}")
    return path


def save_height_png16(h01: np.ndarray, path: str | Path) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), height_to_uint16(h01), [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise OSError(f"could not write {path}")
    return path


def save_rgb(rgb: np.ndarray, path: str | Path) -> Path:
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.ascontiguousarray(rgb)[..., ::-1]):
        raise OSError(f"could not write {path}")
    return path


def load_height(path: str | Path) -> np.ndarray:
    """Read a 16-bit height map written by save_height_* (or by the dataset generator)."""
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ImageLoadError(f"could not read height map {path}")
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return uint16_to_height(arr)


def luminance(rgb01: np.ndarray) -> np.ndarray:
    """Rec. 601 luma of an RGB image with values in [0, 1] -> float32 [0, 1]."""
    r, g, b = (rgb01[..., i].astype(np.float32) for i in range(3))
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0.0, 1.0)
