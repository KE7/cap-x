import asyncio
import base64
import io
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
import tyro
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

from capx.serving.drain_batcher import DrainBatcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROCESSOR: Any | None = None
_MODEL: Any | None = None
_DEVICE = "cuda"
_BATCHER: DrainBatcher[Any, Any] | None = None


@asynccontextmanager
async def _lifespan(_: FastAPI):
    """One worker owns SAM3's mutable processor state for the whole service."""
    global _BATCHER
    _BATCHER = DrainBatcher(_run_batch, name="sam3")
    await _BATCHER.start()
    try:
        yield
    finally:
        await _BATCHER.close()
        _BATCHER = None


app = FastAPI(lifespan=_lifespan)


def _to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
        if hasattr(tensor, "dtype") and tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    return np.asarray(tensor)


def decode_image(base64_str: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(base64_str))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc


def encode_mask(mask: np.ndarray) -> str:
    return base64.b64encode(mask.astype(np.uint8).tobytes()).decode("utf-8")


def encode_array(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("utf-8")


class SegmentRequest(BaseModel):
    image_base64: str
    text_prompt: str


class PointPromptRequest(BaseModel):
    image_base64: str
    point_coords: list[float]


class PointPromptResponse(BaseModel):
    scores: list[float]
    masks_base64: str
    masks_shape: list[int]
    masks_dtype: str


class MaskData(BaseModel):
    mask_base64: str
    shape: list[int]
    box: list[float]
    score: float
    label: str


class SegmentResponse(BaseModel):
    results: list[MaskData]


@dataclass
class _TextWork:
    image: Image.Image
    text_prompt: str


@dataclass
class _PointWork:
    image: Image.Image
    point_coords: tuple[float, float]


def _autocast_context():
    device_type = "cuda" if "cuda" in _DEVICE else "mps" if "mps" in _DEVICE else "cpu"
    return torch.autocast(device_type, dtype=torch.bfloat16)


def _segment_responses_from_batch(
    outputs: dict[str, Any], works: list[_TextWork]
) -> list[SegmentResponse]:
    """Apply per-image thresholds, scaling, and serialization after one forward."""
    masks = outputs.get("pred_masks")
    boxes = outputs.get("pred_boxes")
    logits = outputs.get("pred_logits")
    presence = outputs.get("presence_logit_dec")
    if masks is None or boxes is None or logits is None or presence is None:
        return [SegmentResponse(results=[]) for _ in works]

    scores = (logits.sigmoid() * presence.sigmoid().unsqueeze(1)).squeeze(-1)
    responses: list[SegmentResponse] = []
    for index, work in enumerate(works):
        height, width = work.image.height, work.image.width
        keep = scores[index] > _PROCESSOR.confidence_threshold
        item_masks = (
            torch.nn.functional.interpolate(
                masks[index][keep].unsqueeze(1),
                (height, width),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .sigmoid()
        )
        item_boxes = boxes[index][keep]
        item_boxes = torch.stack(
            (
                item_boxes[:, 0] - item_boxes[:, 2] / 2,
                item_boxes[:, 1] - item_boxes[:, 3] / 2,
                item_boxes[:, 0] + item_boxes[:, 2] / 2,
                item_boxes[:, 1] + item_boxes[:, 3] / 2,
            ),
            dim=-1,
        ) * item_boxes.new_tensor([width, height, width, height])
        results = [
            MaskData(
                mask_base64=encode_mask(mask > 0.5),
                shape=mask.shape,
                box=box.tolist(),
                score=float(score),
                label=work.text_prompt,
            )
            for mask, box, score in zip(
                _to_numpy(item_masks),
                _to_numpy(item_boxes),
                _to_numpy(scores[index][keep]),
                strict=True,
            )
        ]
        results.sort(key=lambda item: item.score, reverse=True)
        responses.append(SegmentResponse(results=results))
    return responses


def _do_segment_batch(works: list[_TextWork]) -> list[SegmentResponse]:
    """Run every currently queued text request through one SAM3 forward pass."""
    assert _PROCESSOR is not None and _MODEL is not None
    batch_size = len(works)
    with _autocast_context():
        state = _PROCESSOR.set_image_batch([work.image for work in works])
        state["backbone_out"].update(
            _MODEL.backbone.forward_text([work.text_prompt for work in works], device=_DEVICE)
        )
        find_input = replace(
            _PROCESSOR.find_stage,
            img_ids=torch.arange(batch_size, device=_DEVICE, dtype=torch.long),
            text_ids=torch.arange(batch_size, device=_DEVICE, dtype=torch.long),
        )
        outputs = _MODEL.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_input,
            find_target=None,
            geometric_prompt=_MODEL._get_dummy_prompt(num_prompts=batch_size),
        )
    return _segment_responses_from_batch(outputs, works)


def _point_response(masks: Any, scores: Any) -> PointPromptResponse:
    masks_np, scores_np = np.asarray(masks), np.asarray(scores)
    if masks_np.size == 0 or scores_np.size == 0:
        return PointPromptResponse(
            scores=[], masks_base64="", masks_shape=(0, 0, 0), masks_dtype="float32"
        )
    order = np.argsort(scores_np)[::-1]
    masks_np, scores_np = masks_np[order], scores_np[order]
    return PointPromptResponse(
        scores=scores_np.astype(float).tolist(),
        masks_base64=encode_array(masks_np),
        masks_shape=tuple(masks_np.shape),
        masks_dtype=str(masks_np.dtype),
    )


def _do_segment_point_batch(works: list[_PointWork]) -> list[PointPromptResponse]:
    """Batch image embeddings and use SAM3's prompt-mode batch API."""
    assert _PROCESSOR is not None and _MODEL is not None
    with _autocast_context():
        state = _PROCESSOR.set_image_batch([work.image for work in works])
        masks, scores, _ = _MODEL.predict_inst_batch(
            state,
            point_coords_batch=[np.array([work.point_coords], dtype=np.float32) for work in works],
            point_labels_batch=[np.array([1], dtype=np.int64) for _ in works],
            multimask_output=True,
        )
    return [_point_response(mask, score) for mask, score in zip(masks, scores, strict=True)]


def _run_batch(kind: str, works: list[Any]) -> list[Any]:
    if kind == "text":
        return _do_segment_batch(works)
    if kind == "point":
        return _do_segment_point_batch(works)
    raise ValueError(f"Unknown SAM3 request kind: {kind}")


@app.post("/segment", response_model=SegmentResponse)
async def segment(req: SegmentRequest):
    if _PROCESSOR is None or _BATCHER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        return await _BATCHER.submit(
            "text", _TextWork(decode_image(req.image_base64), req.text_prompt)
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


@app.post("/segment_point", response_model=PointPromptResponse)
async def segment_point(req: PointPromptRequest):
    if _PROCESSOR is None or _MODEL is None or _BATCHER is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    if getattr(_MODEL, "inst_interactive_predictor", None) is None:
        raise HTTPException(
            status_code=503, detail="Instance interactivity not enabled on SAM3 model"
        )
    try:
        return await _BATCHER.submit(
            "point", _PointWork(decode_image(req.image_base64), tuple(req.point_coords))
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Point prompt inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc


def main(device: str = "cuda", port: int = 8114, host: str = "127.0.0.1"):
    global _MODEL, _PROCESSOR, _DEVICE
    _DEVICE = device
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if "cuda" in device:
            torch.cuda.set_device(int(device.split(":")[-1]) if ":" in device else 0)
    logger.info("Loading SAM3 model...")
    _MODEL = build_sam3_image_model(enable_inst_interactivity=True)
    if hasattr(_MODEL, "to") and device:
        _MODEL = _MODEL.to(device)
    _PROCESSOR = Sam3Processor(_MODEL, device=device, confidence_threshold=0.0)
    logger.info("SAM3 model loaded on %s. Starting server...", device)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
