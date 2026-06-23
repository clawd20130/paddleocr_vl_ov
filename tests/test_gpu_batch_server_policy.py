import os

from paddleocr_vl_openvino.gpu_batch_server import (
    ServerSettings,
    create_app,
    settings_from_env,
)


_ENV_KEYS = [
    "PADDLEOCRVL_LAYOUT_DEVICE",
    "PADDLEOCRVL_VLM_DEVICE",
    "PADDLEOCRVL_MAX_IMAGES_PER_FLUSH",
    "PADDLEOCRVL_WARMUP_ENABLED",
    "PADDLEOCRVL_PREFLIGHT_COMPILE_LAYOUT",
]


def _with_env(updates, callback):
    previous = {key: os.environ.get(key) for key in _ENV_KEYS}
    try:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        for key, value in updates.items():
            os.environ[key] = value
        return callback()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_server_settings_defaults_use_bounded_startup_policy():
    def check():
        settings = settings_from_env()
        assert settings.layout_device == "NPU"
        assert settings.vlm_device == "GPU"
        assert settings.max_images_per_flush == 1
        assert settings.preflight_compile_layout is True
        assert settings.warmup_enabled is False

    _with_env({}, check)


def test_server_settings_rejects_unsupported_env_devices():
    def check_vlm():
        try:
            settings_from_env()
        except ValueError:
            return
        raise AssertionError("expected VLM NPU device to be rejected")

    _with_env({"PADDLEOCRVL_VLM_DEVICE": "NPU"}, check_vlm)

    def check_layout():
        try:
            settings_from_env()
        except ValueError:
            return
        raise AssertionError("expected layout AUTO device to be rejected")

    _with_env({"PADDLEOCRVL_LAYOUT_DEVICE": "AUTO"}, check_layout)


def test_server_settings_validates_direct_device_values():
    settings = ServerSettings(layout_device="npu", vlm_device="gpu.0")
    assert settings.layout_device == "NPU"
    assert settings.vlm_device == "GPU.0"

    for kwargs in [
        {"vlm_device": "NPU"},
        {"vlm_device": "AUTO"},
        {"layout_device": "AUTO"},
        {"layout_device": "MULTI:NPU,GPU"},
    ]:
        try:
            ServerSettings(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {kwargs!r} to be rejected")


def test_create_app_construction_does_not_load_models():
    app = create_app(
        ServerSettings(
            layout_model_path="/missing/layout",
            vlm_model_path="/missing/vlm",
            warmup_enabled=False,
            preflight_compile_layout=True,
        )
    )
    assert app.title == "PaddleOCR-VL OpenVINO GPU Batch Lane"
    assert "pipeline" not in app.state._state
