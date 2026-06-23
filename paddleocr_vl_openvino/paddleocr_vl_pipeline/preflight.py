"""Preflight checks for production OpenVINO device placement."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from ..paddleocr_vl.device_policy import (
        normalize_device_name,
        validate_layout_device,
        validate_vlm_device,
    )
    from ..paddleocr_vl.layout_runtime import prepare_layout_model_for_device
except ImportError:  # pragma: no cover - supports direct module execution paths.
    from paddleocr_vl.device_policy import (
        normalize_device_name,
        validate_layout_device,
        validate_vlm_device,
    )
    from paddleocr_vl.layout_runtime import prepare_layout_model_for_device


def resolve_layout_xml(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_file():
        if path.suffix.lower() != ".xml":
            raise ValueError(f"layout model path is not an .xml file: {path}")
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"layout model path does not exist: {path}")

    preferred = path / "DocLayoutV3.xml"
    if preferred.is_file():
        return preferred
    xml_files = sorted(path.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"no .xml layout model found in: {path}")
    return xml_files[0]


def ov_device_available(requested_device: str, available_devices: Iterable[str]) -> bool:
    requested = normalize_device_name(requested_device)
    available = [normalize_device_name(device) for device in available_devices]
    if requested in available:
        return True
    if requested == "GPU":
        return any(device == "GPU" or device.startswith("GPU.") for device in available)
    if requested == "NPU":
        return any(device == "NPU" or device.startswith("NPU.") for device in available)
    return False


def compile_layout_model(layout_xml: Path, layout_device: str, cache_dir: Optional[str]) -> float:
    import openvino as ov

    layout_device = validate_layout_device(layout_device)
    core = ov.Core()
    model = prepare_layout_model_for_device(core, layout_xml, layout_device)

    config: Dict[str, str] = {}
    if cache_dir:
        config["CACHE_DIR"] = str(Path(cache_dir).expanduser())

    start = time.perf_counter()
    if config:
        compiled = core.compile_model(model, layout_device, config=config)
    else:
        compiled = core.compile_model(model, layout_device)
    compiled.create_infer_request()
    return time.perf_counter() - start


def run_layout_npu_preflight(
    *,
    layout_model: str,
    layout_device: str,
    vlm_device: str,
    compile_layout: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import openvino as ov

    layout_device = validate_layout_device(layout_device)
    vlm_device = validate_vlm_device(vlm_device)
    core = ov.Core()
    available_devices = list(core.available_devices)
    layout_xml = resolve_layout_xml(layout_model)
    bin_path = layout_xml.with_suffix(".bin")

    checks: Dict[str, Any] = {
        "ok": True,
        "available_devices": available_devices,
        "layout_device": layout_device,
        "vlm_device": vlm_device,
        "layout_model": str(layout_xml),
        "layout_bin_exists": bin_path.is_file(),
        "layout_device_available": ov_device_available(layout_device, available_devices),
        "vlm_device_available": ov_device_available(vlm_device, available_devices),
        "vlm_policy": "explicit GPU/CPU only",
        "layout_policy": "NPU by default",
    }

    errors: List[str] = []
    if not checks["layout_bin_exists"]:
        errors.append(f"layout .bin file not found: {bin_path}")
    if not checks["layout_device_available"]:
        errors.append(
            f"layout device {layout_device!r} is not available; "
            f"OpenVINO reports {available_devices!r}"
        )
    if not checks["vlm_device_available"]:
        errors.append(
            f"VLM device {vlm_device!r} is not available; "
            f"OpenVINO reports {available_devices!r}"
        )

    if compile_layout and not errors:
        try:
            checks["layout_compile_seconds"] = compile_layout_model(
                layout_xml,
                layout_device,
                cache_dir,
            )
        except Exception as exc:
            errors.append(f"layout compile failed on {layout_device}: {exc}")

    checks["errors"] = errors
    checks["ok"] = not errors
    return checks
