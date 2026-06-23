import numpy as np
import torch

from paddleocr_vl_openvino.paddleocr_vl.ov_paddleocr_vl import (
    LOC_TOKEN_ID_END,
    OVPaddleOCRVLForCausalLM,
    _openvino_config_from_env,
)


def _torch_select(logits_row, generated_ids, penalty, block_label):
    logits = torch.from_numpy(logits_row.copy()).unsqueeze(0)
    logits = OVPaddleOCRVLForCausalLM._apply_repetition_penalty(
        logits,
        generated_ids,
        penalty,
    )
    logits = OVPaddleOCRVLForCausalLM._suppress_loc_tokens(logits, block_label)
    return int(logits.squeeze(0).argmax().item())


def test_numpy_token_selection_matches_torch_filters_for_text_blocks():
    logits = np.zeros(LOC_TOKEN_ID_END + 5, dtype=np.float32)
    logits[5] = 6.0
    logits[7] = 10.0
    logits[LOC_TOKEN_ID_END] = 99.0

    actual = OVPaddleOCRVLForCausalLM._select_next_token_np(
        logits.copy(),
        generated_ids=[7],
        penalty=2.0,
        block_label="text",
    )

    assert actual == _torch_select(logits, [7], 2.0, "text")
    assert actual == 5


def test_numpy_token_selection_preserves_loc_tokens_for_spotting_blocks():
    logits = np.zeros(LOC_TOKEN_ID_END + 5, dtype=np.float32)
    logits[5] = 6.0
    logits[LOC_TOKEN_ID_END] = 99.0

    actual = OVPaddleOCRVLForCausalLM._select_next_token_np(
        logits.copy(),
        generated_ids=[],
        penalty=1.0,
        block_label="spotting",
    )

    assert actual == _torch_select(logits, [], 1.0, "spotting")
    assert actual == LOC_TOKEN_ID_END


def test_openvino_cache_dir_is_env_only(monkeypatch):
    monkeypatch.delenv("PADDLEOCRVL_OV_CACHE_DIR", raising=False)
    assert "CACHE_DIR" not in _openvino_config_from_env()

    monkeypatch.setenv("PADDLEOCRVL_OV_CACHE_DIR", "/tmp/ov-cache")
    assert _openvino_config_from_env()["CACHE_DIR"] == "/tmp/ov-cache"
