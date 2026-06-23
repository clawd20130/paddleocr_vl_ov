#!/usr/bin/env python3
"""Safe preflight checks for the production layout-NPU / VLM-GPU placement."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from paddleocr_vl_openvino.paddleocr_vl_pipeline.preflight import (
    run_layout_npu_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout-model",
        default=os.environ.get(
            "PADDLEOCRVL_LAYOUT_MODEL_PATH",
            "/home/kevinzhow/models/PP-DoclayoutV3-ov",
        ),
        help="Layout model .xml path or directory.",
    )
    parser.add_argument(
        "--layout-device",
        default=os.environ.get("PADDLEOCRVL_LAYOUT_DEVICE", "NPU"),
        help="Layout device; explicit NPU/GPU/CPU only, default NPU.",
    )
    parser.add_argument(
        "--vlm-device",
        default=os.environ.get("PADDLEOCRVL_VLM_DEVICE", "GPU"),
        help="VLM device; only CPU/GPU/GPU.x are accepted.",
    )
    parser.add_argument("--cache-dir", default=os.environ.get("PADDLEOCRVL_OV_CACHE_DIR"))
    parser.add_argument(
        "--compile-layout",
        action="store_true",
        help="Also compile the layout model on the requested device. This does not run VLM inference.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args()

    try:
        result = run_layout_npu_preflight(
            layout_model=args.layout_model,
            layout_device=args.layout_device,
            vlm_device=args.vlm_device,
            compile_layout=args.compile_layout,
            cache_dir=args.cache_dir,
        )
    except Exception as exc:
        result = {"ok": False, "errors": [str(exc)]}

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print("ok:", result.get("ok"))
        for key in ("available_devices", "layout_device", "vlm_device", "layout_model"):
            if key in result:
                print(f"{key}: {result[key]}")
        for error in result.get("errors", []):
            print(f"error: {error}", file=sys.stderr)
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
