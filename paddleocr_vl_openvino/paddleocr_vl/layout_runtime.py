"""Shared OpenVINO layout-model preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .device_policy import validate_layout_device


def prepare_layout_model_for_device(core: Any, layout_xml: str | Path, layout_device: str) -> Any:
    import openvino as ov

    layout_device = validate_layout_device(layout_device)
    model = core.read_model(str(layout_xml))
    prep = ov.preprocess.PrePostProcessor(model)
    prep.input("image").tensor().set_layout(ov.Layout("NCHW"))
    prep.input("image").preprocess().scale([255, 255, 255])
    if layout_device.startswith("NPU"):
        prep.input("im_shape").model().set_layout(ov.Layout("N..."))
        prep.input("scale_factor").model().set_layout(ov.Layout("N..."))
        prep.input("image").model().set_layout(ov.Layout("NCHW"))
    model = prep.build()
    if layout_device.startswith("NPU"):
        ov.set_batch(model, 1)
    return model
