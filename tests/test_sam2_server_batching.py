import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from capx.serving import launch_sam2_server as server


def _image_base64(value: int) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.full((3, 4, 3), value, dtype=np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class _Inputs(dict):
    def to(self, _: str):
        return self


class _BoxProcessor:
    def __init__(self) -> None:
        self.input_boxes = None

    def __call__(self, *, images, input_boxes, return_tensors):
        self.input_boxes = input_boxes
        return _Inputs(original_sizes=torch.tensor([[3, 4], [3, 4]]), reshaped_input_sizes=None)

    def post_process_masks(self, *_):
        return [
            torch.tensor([[[[1.0, -1.0], [1.0, -1.0]]]]),
            torch.tensor([[[[-1.0, 1.0], [-1.0, 1.0]]]]),
        ]


class _PointProcessor:
    def __init__(self) -> None:
        self.input_points = None

    def __call__(self, *, images, input_points, return_tensors):
        self.input_points = input_points
        return _Inputs(original_sizes=torch.tensor([[3, 4], [3, 4]]))

    def post_process_masks(self, *_):
        return [
            torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]]),
            torch.tensor([[[[4.0]], [[5.0]], [[6.0]]]]),
        ]


class _Model:
    device = "cpu"

    def __call__(self, **_):
        return SimpleNamespace(
            pred_masks=torch.empty(2),
            iou_scores=torch.tensor([[[0.2, 0.9, 0.1]], [[0.8, 0.3, 0.7]]]),
        )


def test_box_prompt_batch_uses_one_model_call_and_one_box_per_image(monkeypatch) -> None:
    processor = _BoxProcessor()
    monkeypatch.setattr(server, "_PROCESSOR", processor)
    monkeypatch.setattr(server, "_MODEL", _Model())
    requests = [
        server.SegmentRequest(image_base64=_image_base64(1), box=[1, 2, 3, 4]),
        server.SegmentRequest(image_base64=_image_base64(2), box=[5, 6, 7, 8]),
    ]

    responses = server._do_box_prompt_batch(requests)

    assert processor.input_boxes == [[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]]
    assert len(responses) == 2
    assert responses[0].masks[0].score == pytest.approx(0.2)


def test_point_prompt_batch_restores_each_request_order(monkeypatch) -> None:
    processor = _PointProcessor()
    monkeypatch.setattr(server, "_PROCESSOR", processor)
    monkeypatch.setattr(server, "_MODEL", _Model())
    monkeypatch.setattr(server, "_DEVICE", "cpu")
    requests = [
        server.PointPromptRequest(image_base64=_image_base64(1), point_coords=(1, 2)),
        server.PointPromptRequest(image_base64=_image_base64(2), point_coords=(3, 4)),
    ]

    responses = server._do_point_prompt_batch(requests)

    assert processor.input_points == [[[[1.0, 2.0]]], [[[3.0, 4.0]]]]
    assert np.allclose(
        [response.scores for response in responses], [[0.9, 0.2, 0.1], [0.8, 0.7, 0.3]]
    )
