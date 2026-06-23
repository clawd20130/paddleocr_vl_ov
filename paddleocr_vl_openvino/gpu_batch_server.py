from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from paddleocr_vl_openvino.paddleocr_vl.device_policy import (
    validate_layout_device,
    validate_vlm_device,
)
from paddleocr_vl_openvino.paddleocr_vl_pipeline import (
    PaddleOCRVL,
    PaddleOCRVLBatchLane,
)
from paddleocr_vl_openvino.paddleocr_vl_pipeline.preflight import run_layout_npu_preflight


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return float(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str) -> List[str]:
    value = os.environ.get(name)
    if value in (None, ""):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


class OCRRequest(BaseModel):
    image_path: Optional[str] = None
    image_base64: Optional[str] = None
    include_text: bool = False
    max_new_tokens: Optional[int] = None
    prompt_label: str = "ocr"
    layout_threshold: float = 0.3
    layout_shape_mode: str = "auto"
    early_stop_ratio: Optional[float] = None
    vlm_min_pixels: Optional[int] = None
    vlm_max_pixels: Optional[int] = None


class OCRResponse(BaseModel):
    model: str
    seconds: float
    blocks: int
    chars: int
    text_hash: str
    text: Optional[str] = None
    timings: Dict[str, Any] = Field(default_factory=dict)


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    early_stop_ratio: Optional[float] = None
    vlm_min_pixels: Optional[int] = None
    vlm_max_pixels: Optional[int] = None
    max_images_per_flush: Optional[int] = None
    use_layout_detection: Optional[bool] = None


class OpenAIBatchChatRequest(BaseModel):
    requests: List[OpenAIChatRequest] = Field(default_factory=list)
    max_images_per_flush: Optional[int] = None


class ServerSettings(BaseModel):
    layout_model_path: str = "/home/kevinzhow/models/PP-DoclayoutV3-ov"
    vlm_model_path: str = "/home/kevinzhow/models/PaddleOCR-VL-1.6-ov-llm-int8"
    layout_device: str = "NPU"
    vlm_device: str = "GPU"
    max_images_per_flush: int = 1
    flush_timeout_seconds: float = 0.05
    vlm_batch_size: int = 16
    early_stop_ratio: float = 0.75
    max_new_tokens: int = 1024
    vlm_min_pixels: int = 112896
    vlm_max_pixels: int = 376320
    llm_int8_compress: bool = False
    llm_int8_quant: bool = False
    result_timeout_seconds: float = 1200.0
    warmup_image: Optional[str] = "/home/kevinzhow/glmocr-bench-images/page.png"
    warmup_images: List[str] = Field(default_factory=list)
    warmup_enabled: bool = False
    preflight_compile_layout: bool = True

    @field_validator("layout_device")
    @classmethod
    def _validate_layout_device(cls, value: str) -> str:
        return validate_layout_device(value)

    @field_validator("vlm_device")
    @classmethod
    def _validate_vlm_device(cls, value: str) -> str:
        return validate_vlm_device(value)


def settings_from_env() -> ServerSettings:
    return ServerSettings(
        layout_model_path=os.environ.get(
            "PADDLEOCRVL_LAYOUT_MODEL_PATH",
            "/home/kevinzhow/models/PP-DoclayoutV3-ov",
        ),
        vlm_model_path=os.environ.get(
            "PADDLEOCRVL_VLM_MODEL_PATH",
            "/home/kevinzhow/models/PaddleOCR-VL-1.6-ov-llm-int8",
        ),
        layout_device=os.environ.get("PADDLEOCRVL_LAYOUT_DEVICE", "NPU"),
        vlm_device=os.environ.get("PADDLEOCRVL_VLM_DEVICE", "GPU"),
        max_images_per_flush=_env_int("PADDLEOCRVL_MAX_IMAGES_PER_FLUSH", 1),
        flush_timeout_seconds=_env_float("PADDLEOCRVL_FLUSH_TIMEOUT_SECONDS", 0.05),
        vlm_batch_size=_env_int("PADDLEOCRVL_VLM_BATCH_SIZE", 16),
        early_stop_ratio=_env_float("PADDLEOCRVL_EARLY_STOP_RATIO", 0.75),
        max_new_tokens=_env_int("PADDLEOCRVL_MAX_NEW_TOKENS", 1024),
        vlm_min_pixels=_env_int("PADDLEOCRVL_VLM_MIN_PIXELS", 112896),
        vlm_max_pixels=_env_int("PADDLEOCRVL_VLM_MAX_PIXELS", 376320),
        llm_int8_compress=_env_bool("PADDLEOCRVL_LLM_INT8_COMPRESS", False),
        llm_int8_quant=_env_bool("PADDLEOCRVL_LLM_INT8_QUANT", False),
        result_timeout_seconds=_env_float("PADDLEOCRVL_RESULT_TIMEOUT_SECONDS", 1200.0),
        warmup_image=os.environ.get(
            "PADDLEOCRVL_WARMUP_IMAGE",
            "/home/kevinzhow/glmocr-bench-images/page.png",
        ),
        warmup_images=_env_list("PADDLEOCRVL_WARMUP_IMAGES"),
        warmup_enabled=_env_bool("PADDLEOCRVL_WARMUP_ENABLED", False),
        preflight_compile_layout=_env_bool("PADDLEOCRVL_PREFLIGHT_COMPILE_LAYOUT", True),
    )


def _block_content(block: Any) -> str:
    content = getattr(block, "content", None)
    if content is None and isinstance(block, dict):
        content = block.get("content") or block.get("result")
    return str(content or "")


def _result_text(result: Any) -> str:
    parsing = result.get("parsing_res_list", []) if isinstance(result, dict) else []
    return "\n".join(_block_content(block) for block in parsing)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _native_blocks(result: Any) -> List[Dict[str, Any]]:
    parsing = result.get("parsing_res_list", []) if isinstance(result, dict) else []
    blocks: List[Dict[str, Any]] = []
    for index, block in enumerate(parsing):
        if isinstance(block, dict):
            label = block.get("label") or block.get("block_label") or "text"
            content = block.get("content") or block.get("block_content") or ""
            bbox = block.get("bbox") or block.get("bbox_2d") or block.get("block_bbox")
            polygon = block.get("polygon_points") or block.get("block_polygon_points")
            group_id = block.get("group_id")
        else:
            label = getattr(block, "label", "text")
            content = getattr(block, "content", "")
            bbox = getattr(block, "bbox", None)
            polygon = getattr(block, "polygon_points", None)
            group_id = getattr(block, "group_id", None)

        native_block: Dict[str, Any] = {
            "index": index,
            "label": str(label or "text"),
            "content": str(content or ""),
            "bbox_2d": _jsonable(bbox),
        }
        if polygon is not None:
            native_block["polygon"] = _jsonable(polygon)
        if group_id is not None:
            native_block["group_id"] = _jsonable(group_id)
        blocks.append(native_block)
    return blocks


def _native_page_size(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    payload: Dict[str, Any] = {}
    if result.get("width") is not None:
        payload["page_width"] = _jsonable(result.get("width"))
    if result.get("height") is not None:
        payload["page_height"] = _jsonable(result.get("height"))
    return payload


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _decode_image_base64(payload: str) -> np.ndarray:
    if "," in payload and payload.lstrip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    raw = base64.b64decode(payload)
    array = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode image_base64")
    return image


def _request_input(payload: OCRRequest) -> Any:
    if payload.image_path:
        return payload.image_path
    if payload.image_base64:
        return _decode_image_base64(payload.image_base64)
    raise ValueError("image_path or image_base64 is required")


def _extract_text_prompt(messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
            continue
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _extract_openai_image_url(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, dict):
            content = [content]
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                value = image_url.get("url")
            else:
                value = image_url
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError("OpenAI request did not include an image_url")


def _openai_image_input(image_url: str) -> Any:
    if image_url.startswith("data:image"):
        return _decode_image_base64(image_url)
    if image_url.startswith("file://"):
        return image_url[7:]
    if os.path.exists(image_url):
        return image_url
    return _decode_image_base64(image_url)


def _prompt_label_from_text(prompt: str) -> str:
    normalized = prompt.strip().lower()
    if "seal" in normalized:
        return "seal"
    if "chart" in normalized:
        return "chart"
    return "ocr"


def create_app(settings: Optional[ServerSettings] = None) -> FastAPI:
    resolved_settings = settings or settings_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        preflight = run_layout_npu_preflight(
            layout_model=resolved_settings.layout_model_path,
            layout_device=resolved_settings.layout_device,
            vlm_device=resolved_settings.vlm_device,
            compile_layout=resolved_settings.preflight_compile_layout,
            cache_dir=os.environ.get("PADDLEOCRVL_OV_CACHE_DIR"),
        )
        if not preflight.get("ok"):
            raise RuntimeError(f"device preflight failed: {preflight.get('errors')}")

        init_start = time.perf_counter()
        pipeline = PaddleOCRVL(
            layout_model_path=resolved_settings.layout_model_path,
            vlm_model_path=resolved_settings.vlm_model_path,
            vlm_device=resolved_settings.vlm_device,
            layout_device=resolved_settings.layout_device,
            use_chart_recognition=False,
            use_seal_recognition=False,
            llm_int4_compress=False,
            vision_int8_quant=False,
            llm_int8_compress=resolved_settings.llm_int8_compress,
            llm_int8_quant=resolved_settings.llm_int8_quant,
            vlm_min_pixels=resolved_settings.vlm_min_pixels,
            vlm_max_pixels=resolved_settings.vlm_max_pixels,
        )
        init_seconds = time.perf_counter() - init_start

        warmup_seconds = None
        warmup_results: List[Dict[str, Any]] = []
        if resolved_settings.warmup_enabled:
            warmup_images = (
                resolved_settings.warmup_images
                if resolved_settings.warmup_images
                else ([resolved_settings.warmup_image] if resolved_settings.warmup_image else [])
            )
            warmup_total_start = time.perf_counter()
            for warmup_image in warmup_images:
                warmup_start = time.perf_counter()
                list(
                    pipeline.predict_batch_lane(
                        [warmup_image],
                        max_images_per_flush=1,
                        layout_threshold=0.3,
                        layout_shape_mode="auto",
                        use_chart_recognition=False,
                        use_seal_recognition=False,
                        vlm_batch_size=resolved_settings.vlm_batch_size,
                        early_stop_ratio=resolved_settings.early_stop_ratio,
                        max_new_tokens=resolved_settings.max_new_tokens,
                        vlm_min_pixels=resolved_settings.vlm_min_pixels,
                        vlm_max_pixels=resolved_settings.vlm_max_pixels,
                    )
                )
                warmup_results.append(
                    {
                        "image": warmup_image,
                        "seconds": time.perf_counter() - warmup_start,
                    }
                )
            warmup_seconds = time.perf_counter() - warmup_total_start

        lane = PaddleOCRVLBatchLane(
            pipeline,
            max_images_per_flush=resolved_settings.max_images_per_flush,
            flush_timeout_seconds=resolved_settings.flush_timeout_seconds,
            vlm_batch_size=resolved_settings.vlm_batch_size,
            default_predict_kwargs={
                "layout_threshold": 0.3,
                "layout_shape_mode": "auto",
                "use_chart_recognition": False,
                "use_seal_recognition": False,
                "early_stop_ratio": resolved_settings.early_stop_ratio,
                "max_new_tokens": resolved_settings.max_new_tokens,
                "vlm_min_pixels": resolved_settings.vlm_min_pixels,
                "vlm_max_pixels": resolved_settings.vlm_max_pixels,
            },
        )

        app.state.pipeline = pipeline
        app.state.batch_lane = lane
        app.state.settings = resolved_settings
        app.state.preflight = preflight
        app.state.init_seconds = init_seconds
        app.state.warmup_seconds = warmup_seconds
        app.state.warmup_results = warmup_results
        try:
            yield
        finally:
            lane.close()

    app = FastAPI(title="PaddleOCR-VL OpenVINO GPU Batch Lane", lifespan=lifespan)

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {
            "ok": True,
            "settings": app.state.settings.model_dump(),
            "preflight": app.state.preflight,
            "device_policy": {
                "layout": "explicit NPU/GPU/CPU only; NPU by default",
                "vlm": "explicit GPU/CPU only",
            },
            "init_seconds": app.state.init_seconds,
            "warmup_seconds": app.state.warmup_seconds,
            "warmup_results": app.state.warmup_results,
        }

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": "paddleocrvl-gpu-batch",
                    "object": "model",
                    "owned_by": "paddleocr-vl-openvino",
                }
            ],
        }

    @app.post("/v1/ocr", response_model=OCRResponse)
    async def ocr(payload: OCRRequest) -> OCRResponse:
        try:
            image_input = _request_input(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        settings = app.state.settings
        predict_kwargs: Dict[str, Any] = {
            "layout_threshold": payload.layout_threshold,
            "layout_shape_mode": payload.layout_shape_mode,
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "prompt_label": payload.prompt_label,
            "early_stop_ratio": (
                settings.early_stop_ratio
                if payload.early_stop_ratio is None
                else payload.early_stop_ratio
            ),
            "max_new_tokens": (
                settings.max_new_tokens
                if payload.max_new_tokens is None
                else payload.max_new_tokens
            ),
            "vlm_min_pixels": (
                settings.vlm_min_pixels
                if payload.vlm_min_pixels is None
                else payload.vlm_min_pixels
            ),
            "vlm_max_pixels": (
                settings.vlm_max_pixels
                if payload.vlm_max_pixels is None
                else payload.vlm_max_pixels
            ),
        }

        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                app.state.batch_lane.predict,
                image_input,
                settings.result_timeout_seconds,
                **predict_kwargs,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        seconds = time.perf_counter() - start

        text = _result_text(result)
        parsing = result.get("parsing_res_list", []) if isinstance(result, dict) else []
        return OCRResponse(
            model="PaddleOCR-VL-OpenVINO-GPU-BatchLane",
            seconds=seconds,
            blocks=len(parsing),
            chars=len(text),
            text_hash=_short_hash(text),
            text=text if payload.include_text else None,
            timings={
                "queue_plus_inference_seconds": seconds,
                "max_images_per_flush": settings.max_images_per_flush,
                "flush_timeout_seconds": settings.flush_timeout_seconds,
                "vlm_batch_size": settings.vlm_batch_size,
            },
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: OpenAIChatRequest) -> Dict[str, Any]:
        try:
            image_url = _extract_openai_image_url(payload.messages)
        except Exception:
            prompt = _extract_text_prompt(payload.messages)
            return {
                "id": f"paddleocrvl-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.model or "PaddleOCR-VL-OpenVINO-GPU-BatchLane",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "ok" if prompt else "",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "completion_tokens": 0,
                    "prompt_tokens": 0,
                    "total_tokens": 0,
                },
            }

        try:
            image_input = _openai_image_input(image_url)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        settings = app.state.settings
        prompt = _extract_text_prompt(payload.messages)
        predict_kwargs: Dict[str, Any] = {
            "layout_threshold": 0.3,
            "layout_shape_mode": "auto",
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "prompt_label": _prompt_label_from_text(prompt),
            "early_stop_ratio": (
                settings.early_stop_ratio
                if payload.early_stop_ratio is None
                else payload.early_stop_ratio
            ),
            "max_new_tokens": payload.max_tokens or settings.max_new_tokens,
            "vlm_min_pixels": (
                settings.vlm_min_pixels
                if payload.vlm_min_pixels is None
                else payload.vlm_min_pixels
            ),
            "vlm_max_pixels": (
                settings.vlm_max_pixels
                if payload.vlm_max_pixels is None
                else payload.vlm_max_pixels
            ),
        }
        if payload.max_images_per_flush is not None:
            predict_kwargs["max_images_per_flush"] = payload.max_images_per_flush
        if payload.use_layout_detection is not None:
            predict_kwargs["use_layout_detection"] = payload.use_layout_detection

        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                app.state.batch_lane.predict,
                image_input,
                settings.result_timeout_seconds,
                **predict_kwargs,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        seconds = time.perf_counter() - start

        text = _result_text(result)
        parsing = result.get("parsing_res_list", []) if isinstance(result, dict) else []
        native_blocks = _native_blocks(result)
        native_page_size = _native_page_size(result)
        return {
            "id": f"paddleocrvl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.model or "PaddleOCR-VL-OpenVINO-GPU-BatchLane",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "total_tokens": 0,
            },
            "paddleocrvl": {
                "seconds": seconds,
                "blocks": len(parsing),
                "chars": len(text),
                "text_hash": _short_hash(text),
                "native_blocks": native_blocks,
                **native_page_size,
                "max_images_per_flush": (
                    payload.max_images_per_flush or settings.max_images_per_flush
                ),
                "flush_timeout_seconds": settings.flush_timeout_seconds,
                "vlm_batch_size": settings.vlm_batch_size,
                "vlm_min_pixels": predict_kwargs["vlm_min_pixels"],
                "vlm_max_pixels": predict_kwargs["vlm_max_pixels"],
                "use_layout_detection": predict_kwargs.get("use_layout_detection", settings.model_dump().get("use_layout_detection")),
            },
        }

    @app.post("/v1/chat/completions/batch")
    async def chat_completions_batch(payload: OpenAIBatchChatRequest) -> Dict[str, Any]:
        if not payload.requests:
            raise HTTPException(status_code=400, detail="requests must not be empty")

        image_inputs = []
        prompt_labels = []
        max_tokens_values = []
        early_stop_ratio_values = []
        vlm_min_pixels_values = []
        vlm_max_pixels_values = []
        use_layout_detection_values = []
        for request in payload.requests:
            try:
                image_url = _extract_openai_image_url(request.messages)
            except Exception as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"each batch request must include one image_url: {exc}",
                ) from exc
            try:
                image_inputs.append(_openai_image_input(image_url))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            prompt = _extract_text_prompt(request.messages)
            prompt_labels.append(_prompt_label_from_text(prompt))
            max_tokens_values.append(request.max_tokens)
            early_stop_ratio_values.append(request.early_stop_ratio)
            vlm_min_pixels_values.append(request.vlm_min_pixels)
            vlm_max_pixels_values.append(request.vlm_max_pixels)
            use_layout_detection_values.append(request.use_layout_detection)

        unique_prompt_labels = set(prompt_labels)
        if len(unique_prompt_labels) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible prompt label",
            )

        requested_max_tokens = [value for value in max_tokens_values if value is not None]
        if requested_max_tokens and len(set(requested_max_tokens)) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible max_tokens value",
            )
        requested_early_stop_ratio = [
            value for value in early_stop_ratio_values if value is not None
        ]
        if requested_early_stop_ratio and len(set(requested_early_stop_ratio)) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible early_stop_ratio value",
            )
        requested_vlm_min_pixels = [
            value for value in vlm_min_pixels_values if value is not None
        ]
        if requested_vlm_min_pixels and len(set(requested_vlm_min_pixels)) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible vlm_min_pixels value",
            )
        requested_vlm_max_pixels = [
            value for value in vlm_max_pixels_values if value is not None
        ]
        if requested_vlm_max_pixels and len(set(requested_vlm_max_pixels)) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible vlm_max_pixels value",
            )

        requested_use_layout_detection = [
            value for value in use_layout_detection_values if value is not None
        ]
        if requested_use_layout_detection and len(set(requested_use_layout_detection)) != 1:
            raise HTTPException(
                status_code=400,
                detail="batch requests must use one compatible use_layout_detection value",
            )

        settings = app.state.settings
        max_images_per_flush = (
            payload.max_images_per_flush or settings.max_images_per_flush
        )
        predict_kwargs: Dict[str, Any] = {
            "layout_threshold": 0.3,
            "layout_shape_mode": "auto",
            "use_chart_recognition": False,
            "use_seal_recognition": False,
            "prompt_label": prompt_labels[0],
            "early_stop_ratio": (
                requested_early_stop_ratio[0]
                if requested_early_stop_ratio
                else settings.early_stop_ratio
            ),
            "max_new_tokens": requested_max_tokens[0] if requested_max_tokens else settings.max_new_tokens,
            "vlm_min_pixels": requested_vlm_min_pixels[0] if requested_vlm_min_pixels else settings.vlm_min_pixels,
            "vlm_max_pixels": requested_vlm_max_pixels[0] if requested_vlm_max_pixels else settings.vlm_max_pixels,
            "max_images_per_flush": max_images_per_flush,
        }
        if requested_use_layout_detection:
            predict_kwargs["use_layout_detection"] = requested_use_layout_detection[0]

        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(
                app.state.batch_lane.predict_many,
                image_inputs,
                settings.result_timeout_seconds,
                **predict_kwargs,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        seconds = time.perf_counter() - start

        response_results = []
        for index, result in enumerate(results):
            text = _result_text(result)
            parsing = result.get("parsing_res_list", []) if isinstance(result, dict) else []
            native_blocks = _native_blocks(result)
            native_page_size = _native_page_size(result)
            response_results.append(
                {
                    "id": f"paddleocrvl-{int(time.time() * 1000)}-{index}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": (
                        payload.requests[index].model
                        or "PaddleOCR-VL-OpenVINO-GPU-BatchLane"
                    ),
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "completion_tokens": 0,
                        "prompt_tokens": 0,
                        "total_tokens": 0,
                    },
                    "paddleocrvl": {
                        "seconds": seconds,
                        "blocks": len(parsing),
                        "chars": len(text),
                        "text_hash": _short_hash(text),
                        "native_blocks": native_blocks,
                        **native_page_size,
                    },
                }
            )

        return {
            "object": "batch.chat.completion",
            "created": int(time.time()),
            "results": response_results,
            "paddleocrvl": {
                "seconds": seconds,
                "items": len(response_results),
                "max_images_per_flush": max_images_per_flush,
                "flush_timeout_seconds": settings.flush_timeout_seconds,
                "vlm_batch_size": settings.vlm_batch_size,
                "early_stop_ratio": predict_kwargs["early_stop_ratio"],
                "vlm_min_pixels": predict_kwargs["vlm_min_pixels"],
                "vlm_max_pixels": predict_kwargs["vlm_max_pixels"],
                "use_layout_detection": predict_kwargs.get("use_layout_detection"),
            },
        }

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8021")))
    args = parser.parse_args()
    uvicorn.run("paddleocr_vl_openvino.gpu_batch_server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
