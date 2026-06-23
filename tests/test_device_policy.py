from paddleocr_vl_openvino.paddleocr_vl.device_policy import (
    is_supported_layout_device,
    is_supported_vlm_device,
    validate_layout_device,
    validate_vlm_device,
)


def test_vlm_device_policy_accepts_explicit_cpu_and_gpu():
    expected = {
        "CPU": "CPU",
        "cpu": "CPU",
        "GPU": "GPU",
        "gpu": "GPU",
        "GPU.0": "GPU.0",
        "gpu.1": "GPU.1",
    }
    for device, normalized in expected.items():
        assert is_supported_vlm_device(device)
        assert validate_vlm_device(device) == normalized


def test_vlm_device_policy_rejects_automatic_or_npu_routes():
    for device in [
        "NPU",
        "AUTO",
        "AUTO:GPU,CPU",
        "AUTO:NPU,GPU",
        "MULTI:GPU,CPU",
        "MULTI:NPU,GPU",
        "HETERO:GPU,CPU",
        "HETERO:NPU,GPU",
        "",
    ]:
        assert not is_supported_vlm_device(device)
        try:
            validate_vlm_device(device)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {device!r} to be rejected")


def test_layout_device_policy_accepts_explicit_cpu_gpu_and_npu():
    expected = {
        "CPU": "CPU",
        "gpu": "GPU",
        "GPU.0": "GPU.0",
        "npu": "NPU",
        "NPU.0": "NPU.0",
    }
    for device, normalized in expected.items():
        assert is_supported_layout_device(device)
        assert validate_layout_device(device) == normalized


def test_layout_device_policy_rejects_automatic_routes():
    for device in [
        "AUTO",
        "AUTO:GPU,CPU",
        "AUTO:NPU,GPU",
        "MULTI:GPU,CPU",
        "HETERO:GPU,CPU",
        "",
    ]:
        assert not is_supported_layout_device(device)
        try:
            validate_layout_device(device)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {device!r} to be rejected")
