from __future__ import annotations
import base64, io, json, os, urllib.error, urllib.request, uuid
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parents[2]

UNIVERSAL_CLAY_PROMPT = (
    "Transform the carved relief in image 0 into a razor-sharp, high-definition orthographic studio render "
    "made of crisp, smooth neutral-gray (#808080) sculpting clay on a completely clean, uniform solid "
    "neutral-gray (#808080) background. Strictly preserve the exact position, count, shape, and sharp carved "
    "edges of every groove, weave cell, petal, leaf, and ridge from image 0 without blurring or distorting "
    "any detail. Completely remove wood grain, color stains, watermarks, text, stray marks, workshop "
    "background clutter, and harsh side shadows, leaving crisp high-contrast relief edges and clean flat margins."
)

def _load_cf_credentials() -> tuple[str, str]:
    acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    tok = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
    env_file = ROOT / ".env"
    if (not acc or not tok) and env_file.exists():
        for line in env_file.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("CLOUDFLARE_ACCOUNT_ID=") and not acc:
                acc = line.split("=", 1)[1].strip().strip('"').strip("'")
            elif line.startswith("CLOUDFLARE_API_TOKEN=") and not tok:
                tok = line.split("=", 1)[1].strip().strip('"').strip("'")
    return acc, tok

def _build_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"----ReliefForgeBoundary{uuid.uuid4().hex}"
    body = io.BytesIO()
    for name, val in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.write(f"{val}\r\n".encode("utf-8"))
    for name, (filename, data, content_type) in files.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"

def _sharpen_clay_render(pil_img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    # Convert to pure neutral grayscale RGB to eliminate stray color artifacts (e.g. yellow specks)
    gray = ImageOps.grayscale(pil_img)
    if gray.size != target_size:
        gray = gray.resize(target_size, Image.Resampling.LANCZOS)
    # Restore crisp groove edges and local relief contrast
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.8, percent=150, threshold=2))
    sharp = ImageEnhance.Contrast(sharp).enhance(1.12)
    return sharp.convert("RGB")

def preprocess_relief_image(
    rgb: np.ndarray,
    timeout: int = 120,
) -> tuple[np.ndarray, bool, str]:
    """Backend high-res image normalization via Cloudflare Workers AI FLUX.2 before Mode B."""
    acc, tok = _load_cf_credentials()
    if not acc or not tok:
        print("[Preprocess API] Cloudflare credentials missing.", flush=True)
        return rgb, False, "fallback to local pipeline"

    orig_h, orig_w = rgb.shape[:2]
    scale = min(1.0, 1024.0 / max(orig_w, orig_h))
    target_w = max(256, min(1024, int(round((orig_w * scale) / 64.0)) * 64))
    target_h = max(256, min(1024, int(round((orig_h * scale) / 64.0)) * 64))

    pil_in = Image.fromarray(rgb.astype(np.uint8)).resize((target_w, target_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil_in.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    candidates = [
        ("@cf/black-forest-labs/flux-2-dev", 24),
        ("@cf/black-forest-labs/flux-2-klein-4b", 8),
    ]

    last_err = "fallback to local pipeline"
    for model_id, steps in candidates:
        url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model_id}"
        fields = {
            "prompt": UNIVERSAL_CLAY_PROMPT,
            "steps": str(steps),
            "width": str(target_w),
            "height": str(target_h),
        }
        body, content_type = _build_multipart(fields, {"input_image_0": ("input.png", png_bytes, "image/png")})
        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {tok}",
                "Content-Type": content_type,
                "User-Agent": "ReliefForge/1.0",
            },
            method="POST",
        )
        print(f"[Preprocess API] Calling {model_id} ({target_w}x{target_h}, {steps} steps) ...", flush=True)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_ct = resp.headers.get("Content-Type", "")
                raw = resp.read()

            if resp_ct.startswith("image/"):
                img_bytes = raw
            else:
                data = json.loads(raw.decode("utf-8"))
                b64_out = (data.get("result") or {}).get("image", "")
                if not b64_out:
                    print(f"[Preprocess API] Unexpected JSON response: {str(data)[:220]}", flush=True)
                    continue
                img_bytes = base64.b64decode(b64_out)

            pil_raw = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            pil_out = _sharpen_clay_render(pil_raw, (orig_w, orig_h))
            print(f"[Preprocess API] Success via {model_id}!", flush=True)
            return np.asarray(pil_out, dtype=np.uint8), True, "applied"

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")[:240]
            print(f"[Preprocess API HTTP {e.code}] {err_body}", flush=True)
            last_err = f"fallback to local pipeline (HTTP {e.code})"
        except Exception as e:
            print(f"[Preprocess API Error] {e}", flush=True)
            last_err = "fallback to local pipeline"

    return rgb, False, last_err
