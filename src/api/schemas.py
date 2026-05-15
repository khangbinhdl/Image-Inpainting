from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class InpaintRequest(BaseModel):
    image_base64: str
    model_type: Literal["vae", "diffusion", "flowmatching"] = "vae"
    mask_mode: Literal["center", "custom"] = "center"
    x: int = Field(24, ge=0, le=63)
    y: int = Field(24, ge=0, le=63)
    w: int = Field(21, ge=1, le=64)
    h: int = Field(21, ge=1, le=64)
    scheduler_type: Literal["ddim", "dpm-solver++", "unipc"] = "ddim"
    num_steps: int = Field(100, ge=1, le=1000)
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class InpaintResponse(BaseModel):
    image_base64: str
    preview_base64: str
    model_type: str
    scheduler_type: Optional[str] = None
    steps_done: int
