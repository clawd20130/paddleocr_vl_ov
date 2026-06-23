from paddleocr_vl_openvino.paddleocr_vl_pipeline.batch_lane import PaddleOCRVLBatchLane
import time


class FakePipeline:
    def __init__(self):
        self.calls = []

    def predict_batch_lane(self, inputs, **kwargs):
        self.calls.append((tuple(inputs), dict(kwargs)))
        for item in inputs:
            yield f"result:{item}"


def test_batch_lane_groups_compatible_requests():
    pipeline = FakePipeline()
    lane = PaddleOCRVLBatchLane(
        pipeline,
        max_images_per_flush=2,
        flush_timeout_seconds=0.5,
        vlm_batch_size=128,
    )
    try:
        first = lane.submit("a")
        second = lane.submit("b")

        assert first.result(timeout=2) == "result:a"
        assert second.result(timeout=2) == "result:b"
    finally:
        lane.close()

    assert pipeline.calls == [
        (
            ("a", "b"),
            {
                "max_images_per_flush": 2,
                "vlm_batch_size": 128,
            },
        )
    ]


def test_batch_lane_keeps_incompatible_kwargs_separate():
    pipeline = FakePipeline()
    lane = PaddleOCRVLBatchLane(
        pipeline,
        max_images_per_flush=2,
        flush_timeout_seconds=0.01,
        vlm_batch_size=128,
    )
    try:
        first = lane.submit("a", prompt_label="ocr")
        second = lane.submit("b", prompt_label="chart")

        assert first.result(timeout=2) == "result:a"
        assert second.result(timeout=2) == "result:b"
    finally:
        lane.close()

    assert pipeline.calls == [
        (
            ("a",),
            {
                "max_images_per_flush": 1,
                "prompt_label": "ocr",
                "vlm_batch_size": 128,
            },
        ),
        (
            ("b",),
            {
                "max_images_per_flush": 1,
                "prompt_label": "chart",
                "vlm_batch_size": 128,
            },
        ),
    ]


def test_batch_lane_uses_request_max_images_per_flush_for_single_requests():
    pipeline = FakePipeline()
    lane = PaddleOCRVLBatchLane(
        pipeline,
        max_images_per_flush=2,
        flush_timeout_seconds=0.5,
        vlm_batch_size=128,
    )
    try:
        first = lane.submit("a", max_images_per_flush=3)
        second = lane.submit("b", max_images_per_flush=3)
        third = lane.submit("c", max_images_per_flush=3)

        assert first.result(timeout=2) == "result:a"
        assert second.result(timeout=2) == "result:b"
        assert third.result(timeout=2) == "result:c"
    finally:
        lane.close()

    assert pipeline.calls == [
        (
            ("a", "b", "c"),
            {
                "max_images_per_flush": 3,
                "vlm_batch_size": 128,
            },
        )
    ]


def test_batch_lane_waits_for_request_flush_limit_not_lane_default():
    pipeline = FakePipeline()
    lane = PaddleOCRVLBatchLane(
        pipeline,
        max_images_per_flush=2,
        flush_timeout_seconds=0.2,
        vlm_batch_size=128,
    )
    try:
        first = lane.submit("a", max_images_per_flush=3)
        time.sleep(0.03)
        second = lane.submit("b", max_images_per_flush=3)
        time.sleep(0.03)

        assert not first.done()
        assert not second.done()

        third = lane.submit("c", max_images_per_flush=3)
        assert first.result(timeout=2) == "result:a"
        assert second.result(timeout=2) == "result:b"
        assert third.result(timeout=2) == "result:c"
    finally:
        lane.close()

    assert pipeline.calls == [
        (
            ("a", "b", "c"),
            {
                "max_images_per_flush": 3,
                "vlm_batch_size": 128,
            },
        )
    ]


def test_batch_lane_predict_many_runs_as_one_group():
    pipeline = FakePipeline()
    lane = PaddleOCRVLBatchLane(
        pipeline,
        max_images_per_flush=2,
        flush_timeout_seconds=0.01,
        vlm_batch_size=128,
    )
    try:
        result = lane.predict_many(
            ["a", "b", "c"],
            timeout=2,
            max_images_per_flush=3,
        )
    finally:
        lane.close()

    assert result == ["result:a", "result:b", "result:c"]
    assert pipeline.calls == [
        (
            ("a", "b", "c"),
            {
                "max_images_per_flush": 3,
                "vlm_batch_size": 128,
            },
        )
    ]
