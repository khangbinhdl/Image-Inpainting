from __future__ import annotations

import os
from typing import Tuple

import gradio as gr
import requests
from PIL import Image

from src.utils.image_ops import center_bbox, pil_to_base64_png, base64_to_pil, preview_mask, resize_rgb

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


DISPLAY_SIZE = 512
CENTER_MASK_MODE = "Center masked image"
CUSTOM_MASK_MODE = "Clean image + custom rectangle mask"


def to_display_size(img: Image.Image | None) -> Image.Image | None:
    if img is None:
        return None
    return img.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)


def bbox_from_mode(mode: str, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    return center_bbox() if mode == CENTER_MASK_MODE else (x, y, w, h)


def update_preview(image, mode, x, y, w, h):
    if image is None:
        return None
    img = resize_rgb(Image.fromarray(image) if not isinstance(image, Image.Image) else image)
    bbox = bbox_from_mode(mode, x, y, w, h)
    return to_display_size(preview_mask(img, bbox))


def infer(image, mode, x, y, w, h, model_type, scheduler_type, steps, device, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Please upload an image first.")
    img = resize_rgb(Image.fromarray(image) if not isinstance(image, Image.Image) else image)
    bbox = bbox_from_mode(mode, x, y, w, h)
    payload = {
        "image_base64": pil_to_base64_png(img),
        "model_type": model_type,
        "mask_mode": "center" if mode == CENTER_MASK_MODE else "custom",
        "x": int(bbox[0]),
        "y": int(bbox[1]),
        "w": int(bbox[2]),
        "h": int(bbox[3]),
        "scheduler_type": scheduler_type,
        "num_steps": int(steps),
        "device": device,
    }
    progress(0.05, desc="Sending image to API")
    resp = requests.post(f"{API_URL}/inpaint", json=payload, timeout=300)
    if resp.status_code != 200:
        raise gr.Error(resp.text)
    data = resp.json()
    progress(1.0, desc=f"Completed {data.get('steps_done', 1)} timestep(s)")
    return (
        to_display_size(base64_to_pil(data["preview_base64"])),
        to_display_size(base64_to_pil(data["image_base64"])),
        f"Done: {data.get('steps_done', 1)} timestep(s)",
    )


def model_ui_visibility(model_type):
    return (
        gr.update(visible=model_type == "diffusion"),
        gr.update(visible=model_type in ["diffusion", "flowmatching"]),
    )


def mask_ui_visibility(mode):
    visible = mode == CUSTOM_MASK_MODE
    return [gr.update(visible=visible) for _ in range(4)]


with gr.Blocks(title="Image Inpainting") as demo:
    gr.Markdown("# Image Inpainting - VAE / Diffusion / Flow Matching")

    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Upload image", type="pil", height=320)
            mode = gr.Radio([CENTER_MASK_MODE, CUSTOM_MASK_MODE], value=CENTER_MASK_MODE, label="Mask mode")
            with gr.Row():
                x = gr.Slider(0, 63, value=24, step=1, label="Mask x", visible=False)
                y = gr.Slider(0, 63, value=24, step=1, label="Mask y", visible=False)
            with gr.Row():
                w = gr.Slider(1, 64, value=21, step=1, label="Mask width", visible=False)
                h = gr.Slider(1, 64, value=21, step=1, label="Mask height", visible=False)

            model_type = gr.Radio(["vae", "diffusion", "flowmatching"], value="vae", label="Model")
            scheduler_type = gr.Radio(["ddim", "dpm-solver++", "unipc"], value="ddim", label="Scheduler type", visible=False)
            steps = gr.Slider(2, 300, value=100, step=1, label="Inference timesteps", visible=False)
            device = gr.Radio(["cpu", "cuda", "mps"], value="cpu", label="Device")
            run = gr.Button("Run inference", variant="primary")
        with gr.Column():
            preview = gr.Image(label="Mask preview", type="pil", height=280)
            output = gr.Image(label="Inpainted output", type="pil", height=280)
            status = gr.Textbox(label="Status")

    for comp in [image, mode, x, y, w, h]:
        comp.change(update_preview, inputs=[image, mode, x, y, w, h], outputs=preview)
    mode.change(mask_ui_visibility, inputs=mode, outputs=[x, y, w, h])
    model_type.change(model_ui_visibility, inputs=model_type, outputs=[scheduler_type, steps])
    run.click(infer, inputs=[image, mode, x, y, w, h, model_type, scheduler_type, steps, device], outputs=[preview, output, status])


if __name__ == "__main__":
    demo.launch(server_name=os.getenv("UI_HOST", "localhost"), server_port=int(os.getenv("UI_PORT", "8501")))
