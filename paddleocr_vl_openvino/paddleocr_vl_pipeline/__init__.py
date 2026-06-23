"""
PaddleOCR-VL Pipeline 模块
"""

from .batch_lane import PaddleOCRVLBatchLane
from .ov_paddleocr_vl_pipeline import PaddleOCRVL, PaddleOCRVLResult, PaddleOCRVLBlock

__all__ = [
    'PaddleOCRVL',
    'PaddleOCRVLBatchLane',
    'PaddleOCRVLResult',
    'PaddleOCRVLBlock',
]
