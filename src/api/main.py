from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import InpaintRequest, InpaintResponse
from src.ml.inference import inpaint
from src.utils.image_ops import base64_to_pil, center_bbox, pil_to_base64_png, preview_mask

app = FastAPI(title="Image Inpainting Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inpaint", response_model=InpaintResponse)
def inpaint_endpoint(req: InpaintRequest):
    try:
        image = base64_to_pil(req.image_base64)
        bbox = center_bbox() if req.mask_mode == "center" else (req.x, req.y, req.w, req.h)
        result = inpaint(
            image=image,
            bbox=bbox,
            model_type=req.model_type,
            scheduler_type=req.scheduler_type,
            num_steps=req.num_steps,
            device_name=req.device,
        )
        return InpaintResponse(
            image_base64=pil_to_base64_png(result.image),
            preview_base64=pil_to_base64_png(preview_mask(image, bbox)),
            model_type=result.model_type,
            scheduler_type=result.scheduler_type,
            steps_done=result.steps_done,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
