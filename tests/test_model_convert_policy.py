from pathlib import Path


def test_model_convert_vlm_export_rejects_npu_devices_statically():
    source = Path("model_convert/ov_paddleocr_vl.py").read_text()
    assert "def _validate_vlm_export_device" in source
    assert "Do not generate NPU variants of the VLM graphs" in source
    assert source.count("_validate_vlm_export_device(device)") >= 5
