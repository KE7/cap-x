import base64
import io
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import (
    Owlv2ForObjectDetection,
    Owlv2Processor,
    OwlViTForObjectDetection,
    OwlViTProcessor,
)

from capx.serving.drain_batcher import DrainBatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROC: Any | None = None
_MODEL: Any | None = None
_DEVICE = "cuda"
_THRESHOLD = 0.05
_BATCHER: DrainBatcher[Any, Any] | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    global _BATCHER
    _BATCHER = DrainBatcher(_do_detect_batch, name="owlvit")
    await _BATCHER.start()
    try:
        yield
    finally:
        await _BATCHER.close()
        _BATCHER = None


app = FastAPI(lifespan=_lifespan)


def _encode_image(image: np.ndarray | Image.Image) -> str:
    pil_image = (
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).convert("RGB")
        if isinstance(image, np.ndarray)
        else image
    )
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _decode_image(image_base64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")


class DetectRequest(BaseModel):
    image_base64: str
    texts: list[list[str]] | None = None
    threshold: float | None = None


class Detection(BaseModel):
    label: str
    score: float
    box: list[float]


class DetectResponse(BaseModel):
    detections: list[Detection]


@dataclass
class _DetectWork:
    image: Image.Image
    texts: list[str]
    threshold: float


def _do_detect_batch(works: list[_DetectWork]) -> list[DetectResponse]:
    """Perform one OWL-ViT forward pass and post-process each HTTP request."""
    assert _MODEL is not None and _PROC is not None
    images = [np.asarray(work.image, dtype=np.uint8) for work in works]
    inputs = _PROC(text=[work.texts for work in works], images=images, return_tensors="pt")
    with torch.no_grad():
        outputs = _MODEL(**{key: value.to(_MODEL.device) for key, value in inputs.items()})
    target_sizes = torch.tensor([image.shape[:2] for image in images], device=_MODEL.device)
    # Post-process at the most permissive request threshold, then apply each
    # request's threshold independently so the response contract stays exact.
    results = _PROC.post_process_grounded_object_detection(
        outputs, threshold=0.0, target_sizes=target_sizes
    )
    responses: list[DetectResponse] = []
    for work, result in zip(works, results, strict=True):
        detections = [
            Detection(
                label=work.texts[int(label)],
                score=float(score.item()),
                box=[round(value, 2) for value in box.detach().cpu().numpy().tolist()],
            )
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"], strict=True
            )
            if float(score.item()) >= work.threshold
        ]
        responses.append(DetectResponse(detections=detections))
    return responses


@app.post("/detect", response_model=DetectResponse)
async def detect_endpoint(req: DetectRequest):
    if _MODEL is None or _PROC is None or _BATCHER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        texts = req.texts if req.texts is not None else [["an object"]]
        if len(texts) != 1:
            raise HTTPException(
                status_code=400, detail="texts must contain one label list per image request"
            )
        return await _BATCHER.submit(
            "detect",
            _DetectWork(
                _decode_image(req.image_base64),
                texts[0],
                req.threshold if req.threshold is not None else _THRESHOLD,
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error during detection: %s", exc)
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}") from exc


def main(
    model_name: str = "google/owlv2-large-patch14-ensemble",
    device: str = "cuda",
    port: int = 8117,
    host: str = "127.0.0.1",
    threshold: float = 0.05,
):
    global _PROC, _MODEL, _DEVICE, _THRESHOLD
    _DEVICE, _THRESHOLD = device, threshold
    logger.info("Loading OWL-ViT model: %s on %s...", model_name, device)
    if "owlv2" in model_name.lower():
        _PROC, _MODEL = (
            Owlv2Processor.from_pretrained(model_name),
            Owlv2ForObjectDetection.from_pretrained(model_name),
        )
    else:
        _PROC, _MODEL = (
            OwlViTProcessor.from_pretrained(model_name),
            OwlViTForObjectDetection.from_pretrained(model_name),
        )
    _MODEL = _MODEL.to(device)
    _MODEL.eval()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
