"""Device placement policy for the production runtime."""

from __future__ import annotations


def normalize_device_name(device: str) -> str:
    return str(device or "").upper().strip()


def is_npu_device(device: str) -> bool:
    normalized = normalize_device_name(device)
    return normalized == "NPU" or normalized.startswith("NPU.")


def is_supported_vlm_device(device: str) -> bool:
    normalized = normalize_device_name(device)
    return normalized == "CPU" or normalized == "GPU" or normalized.startswith("GPU.")


def is_supported_layout_device(device: str) -> bool:
    normalized = normalize_device_name(device)
    return (
        normalized == "CPU"
        or normalized == "GPU"
        or normalized.startswith("GPU.")
        or normalized == "NPU"
        or normalized.startswith("NPU.")
    )


def validate_vlm_device(device: str) -> str:
    normalized = normalize_device_name(device)
    if is_supported_vlm_device(normalized):
        return normalized
    raise ValueError(
        "PaddleOCR-VL is restricted to explicit GPU/CPU in this project. "
        "Use vlm_device='GPU' for production, and reserve layout_device='NPU' "
        "for layout detection."
    )


def validate_layout_device(device: str) -> str:
    normalized = normalize_device_name(device)
    if is_supported_layout_device(normalized):
        return normalized
    raise ValueError(
        "Layout detection is restricted to explicit NPU/GPU/CPU devices. "
        "Use layout_device='NPU' for production; AUTO/MULTI/HETERO are not "
        "part of the supported placement."
    )
