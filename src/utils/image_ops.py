from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 64


def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_pil(data: str) -> Image.Image:
    if "," in data:
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def resize_rgb(img: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return img.convert("RGB").resize((size, size), Image.BICUBIC)


def center_bbox(size: int = IMAGE_SIZE) -> Tuple[int, int, int, int]:
    # README says central 21x21 region for 64x64 inference data.
    box = size // 3
    left = (size - box) // 2
    top = (size - box) // 2
    return left, top, box, box


def clamp_bbox(x: int, y: int, w: int, h: int, size: int = IMAGE_SIZE) -> Tuple[int, int, int, int]:
    w = max(1, min(int(w), size))
    h = max(1, min(int(h), size))
    x = max(0, min(int(x), size - w))
    y = max(0, min(int(y), size - h))
    return x, y, w, h


def make_mask_np(bbox: Tuple[int, int, int, int], size: int = IMAGE_SIZE) -> np.ndarray:
    x, y, w, h = clamp_bbox(*bbox, size=size)
    mask = np.zeros((size, size), dtype=np.float32)
    mask[y : y + h, x : x + w] = 1.0
    return mask


def preview_mask(img: Image.Image, bbox: Tuple[int, int, int, int], size: int = IMAGE_SIZE) -> Image.Image:
    img = resize_rgb(img, size)
    out = img.copy()
    x, y, w, h = clamp_bbox(*bbox, size=size)
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x, y, x + w - 1, y + h - 1], fill=(0, 0, 0, 180), outline=(255, 255, 255, 255), width=1)
    return Image.alpha_composite(out.convert("RGBA"), overlay).convert("RGB")
