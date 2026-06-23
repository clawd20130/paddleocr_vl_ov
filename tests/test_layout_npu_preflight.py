import tempfile
from pathlib import Path

from paddleocr_vl_openvino.paddleocr_vl_pipeline.preflight import (
    ov_device_available,
    resolve_layout_xml,
)


def test_ov_device_available_matches_device_families():
    assert ov_device_available("NPU", ["CPU", "NPU"])
    assert ov_device_available("npu", ["NPU.0"])
    assert ov_device_available("GPU", ["GPU.0"])
    assert ov_device_available("GPU.0", ["GPU.0"])
    assert ov_device_available("CPU", ["CPU", "NPU"])
    assert not ov_device_available("NPU", ["CPU", "GPU"])
    assert not ov_device_available("GPU.1", ["GPU.0"])
    assert not ov_device_available("AUTO", ["CPU", "GPU"])


def test_resolve_layout_xml_prefers_doclayoutv3():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        other = tmp_path / "other.xml"
        preferred = tmp_path / "DocLayoutV3.xml"
        other.write_text("<xml />")
        preferred.write_text("<xml />")
        assert resolve_layout_xml(str(tmp_path)) == preferred


def test_resolve_layout_xml_rejects_non_xml_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = Path(tmp_dir) / "model.onnx"
        model.write_text("")
        try:
            resolve_layout_xml(str(model))
        except ValueError:
            pass
        else:
            raise AssertionError("expected non-xml layout model path to be rejected")
