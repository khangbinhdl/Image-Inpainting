from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

from src.ml.models import ConditionalUNetVAE, IMAGE_SIZE, VAE_CHANNELS, build_unet_7ch
from src.utils.image_ops import make_mask_np, resize_rgb

MODEL_PATHS = {
    "vae": "saved_models/vae_best.pth",
    "diffusion": "saved_models/diffusion_best.pth",
    "flowmatching": "saved_models/flow_best.pth",
}


def resolve_device(requested: str) -> torch.device:
    requested = (requested or "cpu").lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device("mps")
    return torch.device("cpu")


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    arr = np.asarray(resize_rgb(img), dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.float32, non_blocking=False)


def mask_to_tensor(mask_np: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.detach().float().cpu().squeeze(0).clamp(-1, 1)
    arr = ((t + 1.0) * 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def make_condition(image_t: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    return image_t * (1.0 - mask_t) + (-1.0) * mask_t


@dataclass
class InpaintResult:
    image: Image.Image
    preview: Image.Image
    steps_done: int
    model_type: str
    scheduler_type: Optional[str] = None


class ModelCache:
    def __init__(self, model_dir: str = "saved_models"):
        self.model_dir = Path(model_dir)
        self._model = None
        self._key = None

    def clear(self):
        self._model = None
        self._key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self, model_type: str, device: torch.device):
        model_type = model_type.lower()
        key = (model_type, str(device))
        if self._key == key and self._model is not None:
            return self._model
        self.clear()

        if model_type == "vae":
            model = ConditionalUNetVAE(image_size=IMAGE_SIZE, z_dim=256, channels=VAE_CHANNELS, layers_per_block=2)
            ckpt = self.model_dir / "vae_best.pth"
        elif model_type == "diffusion":
            model = build_unet_7ch()
            ckpt = self.model_dir / "diffusion_best.pth"
        elif model_type in {"flowmatching", "flow_matching", "flow"}:
            model = build_unet_7ch()
            ckpt = self.model_dir / "flow_best.pth"
            model_type = "flowmatching"
        else:
            raise ValueError(f"Unsupported model_type={model_type}")

        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}. Put weights in saved_models/.")
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        self._model = model
        self._key = key
        return model


CACHE = ModelCache(os.getenv("MODEL_DIR", "saved_models"))


def scheduler_factory(name: str):
    name = (name or "ddim").lower()
    common = dict(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2", prediction_type="sample")
    if name == "ddim":
        return DDIMScheduler(**common)
    if name in {"dpm", "dpm-solver++", "dpmsolver"}:
        return DPMSolverMultistepScheduler(**common, algorithm_type="dpmsolver++", solver_order=2)
    if name == "unipc":
        return UniPCMultistepScheduler(**common)
    raise ValueError("scheduler_type must be one of: ddim, dpm-solver++, unipc")


@torch.inference_mode()
def run_vae(model, cond_image, mask, progress_cb: Optional[Callable[[int, int], None]] = None):
    final_image, _ = model.sample(cond_image, mask)
    if progress_cb:
        progress_cb(1, 1)
    return final_image.clamp(-1, 1), 1


@torch.inference_mode()
def run_diffusion(model, cond_image, mask, scheduler_type: str, steps: int, device: torch.device, progress_cb=None):
    steps = int(max(1, min(steps, 1000)))
    visible = cond_image * (1 - mask)
    scheduler = scheduler_factory(scheduler_type)
    scheduler.set_timesteps(steps, device=device)
    x = visible + torch.randn_like(cond_image) * mask
    total = len(scheduler.timesteps)
    for idx, t in enumerate(scheduler.timesteps):
        x = visible + x * mask
        t_batch = torch.full((x.size(0),), int(t.item()) if torch.is_tensor(t) else int(t), device=device, dtype=torch.long)
        pred_x0 = model(torch.cat([x, cond_image, mask], dim=1), t_batch).sample
        x = scheduler.step(model_output=pred_x0, timestep=t, sample=x).prev_sample
        x = visible + x * mask
        if progress_cb:
            progress_cb(idx + 1, total)
    return x.clamp(-1, 1), total


@torch.inference_mode()
def run_flowmatching(model, cond_image, mask, steps: int, device: torch.device, progress_cb=None):
    steps = int(max(2, min(steps, 1000)))
    visible = cond_image * (1 - mask)
    y = visible + torch.randn_like(cond_image) * mask
    t_steps = torch.linspace(0.0, 1.0, steps, device=device)
    total = steps - 1
    for i in range(total):
        t = t_steps[i]
        dt = t_steps[i + 1] - t
        y = visible + y * mask
        t_model = torch.full((y.size(0),), float(t.item() * 999.0), device=device, dtype=torch.float32)
        v = model(torch.cat([y, cond_image, mask], dim=1), t_model).sample
        y = visible + (y + v * dt * mask) * mask
        if progress_cb:
            progress_cb(i + 1, total)
    return y.clamp(-1, 1), total


def inpaint(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    model_type: str = "vae",
    scheduler_type: str = "ddim",
    num_steps: int = 100,
    device_name: str = "cpu",
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> InpaintResult:
    device = resolve_device(device_name)
    model_type = model_type.lower().replace(" ", "")
    model = CACHE.get(model_type, device)

    image = resize_rgb(image)
    mask_np = make_mask_np(bbox)
    image_t = pil_to_tensor(image, device)
    mask_t = mask_to_tensor(mask_np, device)
    cond = make_condition(image_t, mask_t)

    try:
        if model_type == "vae":
            out, done = run_vae(model, cond, mask_t, progress_cb)
            sched = None
        elif model_type == "diffusion":
            out, done = run_diffusion(model, cond, mask_t, scheduler_type, num_steps, device, progress_cb)
            sched = scheduler_type
        else:
            out, done = run_flowmatching(model, cond, mask_t, num_steps, device, progress_cb)
            sched = None
        return InpaintResult(image=tensor_to_pil(out), preview=tensor_to_pil(cond), steps_done=done, model_type=model_type, scheduler_type=sched)
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
