import numpy as np
import torch

from paddleocr_vl_openvino.paddleocr_vl.ov_paddleocr_vl import (
    LOC_TOKEN_ID_END,
    OVPaddleOCRVLForCausalLM,
    _bucketed_length,
    _env_int,
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


def test_env_int_falls_back_for_unset_or_invalid_values(monkeypatch):
    monkeypatch.delenv("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", raising=False)
    assert _env_int("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", 0) == 0

    monkeypatch.setenv("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", "64")
    assert _env_int("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", 0) == 64

    monkeypatch.setenv("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", "bad")
    assert _env_int("PADDLEOCRVL_OV_DECODE_MASK_BUCKET", 32) == 32


def test_bucketed_length_rounds_up_without_exceeding_max():
    assert _bucketed_length(326, 0, 400) == 326
    assert _bucketed_length(326, 64, 400) == 384
    assert _bucketed_length(385, 64, 400) == 400


def test_beam_idx_matches_openvino_i32_input():
    generated = OVPaddleOCRVLForCausalLM._beam_idx_array(3)
    assert generated.dtype == np.int32
    assert generated.tolist() == [0, 1, 2]

    reordered = OVPaddleOCRVLForCausalLM._beam_idx_array(torch.tensor([2, 0, 1]))
    assert reordered.dtype == np.int32
    assert reordered.tolist() == [2, 0, 1]
