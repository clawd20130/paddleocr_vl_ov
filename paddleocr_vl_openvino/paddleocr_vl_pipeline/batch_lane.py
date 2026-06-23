from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _freeze_for_key(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze_for_key(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_for_key(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_for_key(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


@dataclass
class _BatchLaneRequest:
    image_input: Any
    kwargs: Dict[str, Any]
    future: Future
    enqueued_at: float
    max_images_per_flush: Optional[int] = None
    returns_many: bool = False
    many_max_images_per_flush: Optional[int] = None

    @property
    def key(self) -> Tuple[Tuple[str, Any], ...]:
        if self.returns_many:
            return (("__batch_lane_many__", True),)
        items = sorted((key, _freeze_for_key(value)) for key, value in self.kwargs.items())
        items.append(("__max_images_per_flush__", self.max_images_per_flush))
        return tuple(items)


class PaddleOCRVLBatchLane:
    """Small request queue for GPU-first PaddleOCR-VL batching.

    The wrapped ``PaddleOCRVL`` instance is still used from one worker thread.
    The queue only moves the batch boundary up to independent image requests and
    then calls ``predict_batch_lane()`` with a small compatible request window.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        max_images_per_flush: int = 1,
        flush_timeout_seconds: float = 0.05,
        vlm_batch_size: int = 128,
        default_predict_kwargs: Optional[Dict[str, Any]] = None,
        worker_name: str = "paddleocrvl-gpu-batch-lane",
    ) -> None:
        if max_images_per_flush <= 0:
            raise ValueError("max_images_per_flush must be >= 1")
        if flush_timeout_seconds < 0:
            raise ValueError("flush_timeout_seconds must be >= 0")
        if vlm_batch_size <= 0:
            raise ValueError("vlm_batch_size must be >= 1")

        self.pipeline = pipeline
        self.max_images_per_flush = max_images_per_flush
        self.flush_timeout_seconds = flush_timeout_seconds
        self.default_predict_kwargs = dict(default_predict_kwargs or {})
        self.default_predict_kwargs.setdefault("vlm_batch_size", vlm_batch_size)

        self._condition = threading.Condition()
        self._pending: List[_BatchLaneRequest] = []
        self._closing = False
        self._worker = threading.Thread(target=self._run, name=worker_name, daemon=True)
        self._worker.start()

    def submit(self, image_input: Any, **predict_kwargs: Any) -> Future:
        kwargs = dict(self.default_predict_kwargs)
        kwargs.update(predict_kwargs)
        max_images_per_flush = kwargs.pop("max_images_per_flush", None)
        if max_images_per_flush is not None:
            max_images_per_flush = max(1, int(max_images_per_flush))
        future: Future = Future()
        request = _BatchLaneRequest(
            image_input=image_input,
            kwargs=kwargs,
            future=future,
            enqueued_at=time.monotonic(),
            max_images_per_flush=max_images_per_flush,
        )
        with self._condition:
            if self._closing:
                raise RuntimeError("PaddleOCRVLBatchLane is closed")
            self._pending.append(request)
            self._condition.notify()
        return future

    def predict(self, image_input: Any, timeout: Optional[float] = None, **predict_kwargs: Any) -> Any:
        return self.submit(image_input, **predict_kwargs).result(timeout=timeout)

    def submit_many(self, image_inputs: Iterable[Any], **predict_kwargs: Any) -> Future:
        inputs = list(image_inputs)
        if not inputs:
            future: Future = Future()
            future.set_result([])
            return future
        kwargs = dict(self.default_predict_kwargs)
        kwargs.update(predict_kwargs)
        many_max_images_per_flush = kwargs.pop("max_images_per_flush", None)
        future: Future = Future()
        request = _BatchLaneRequest(
            image_input=inputs,
            kwargs=kwargs,
            future=future,
            enqueued_at=time.monotonic(),
            returns_many=True,
            many_max_images_per_flush=many_max_images_per_flush,
        )
        with self._condition:
            if self._closing:
                raise RuntimeError("PaddleOCRVLBatchLane is closed")
            self._pending.append(request)
            self._condition.notify()
        return future

    def predict_many(self, image_inputs: Iterable[Any], timeout: Optional[float] = None, **predict_kwargs: Any) -> List[Any]:
        return self.submit_many(image_inputs, **predict_kwargs).result(timeout=timeout)

    def map(self, image_inputs: Iterable[Any], timeout: Optional[float] = None, **predict_kwargs: Any) -> List[Any]:
        futures = [self.submit(image_input, **predict_kwargs) for image_input in image_inputs]
        return [future.result(timeout=timeout) for future in futures]

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        if wait and self._worker.is_alive():
            self._worker.join()

    def __enter__(self) -> "PaddleOCRVLBatchLane":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if not self._pending and self._closing:
                    return
                batch = self._take_batch_locked()

            self._flush(batch)

    def _take_batch_locked(self) -> List[_BatchLaneRequest]:
        first = self._pending[0]
        if first.returns_many:
            del self._pending[0]
            return [first]

        max_images_per_flush = first.max_images_per_flush or self.max_images_per_flush
        deadline = first.enqueued_at + self.flush_timeout_seconds

        while not self._closing:
            if self._compatible_count_locked(first.key, max_images_per_flush) >= max_images_per_flush:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._condition.wait(remaining)

        batch = [first]
        del self._pending[0]

        index = 0
        while index < len(self._pending) and len(batch) < max_images_per_flush:
            candidate = self._pending[index]
            if candidate.key == first.key:
                batch.append(candidate)
                del self._pending[index]
            else:
                index += 1

        return batch

    def _compatible_count_locked(self, key: Tuple[Tuple[str, Any], ...], limit: int) -> int:
        count = 0
        for request in self._pending:
            if request.key == key:
                count += 1
                if count >= limit:
                    return count
        return count

    def _flush(self, batch: List[_BatchLaneRequest]) -> None:
        live_batch = [request for request in batch if not request.future.cancelled()]
        if not live_batch:
            return

        try:
            kwargs = dict(live_batch[0].kwargs)
            if len(live_batch) == 1 and live_batch[0].returns_many:
                image_inputs = list(live_batch[0].image_input)
                many_max_images_per_flush = (
                    live_batch[0].many_max_images_per_flush or len(image_inputs)
                )
                results = list(
                    self.pipeline.predict_batch_lane(
                        image_inputs,
                        max_images_per_flush=many_max_images_per_flush,
                        **kwargs,
                    )
                )
                if len(results) != len(image_inputs):
                    raise RuntimeError(
                        "predict_batch_lane returned "
                        f"{len(results)} results for {len(image_inputs)} images"
                    )
                live_batch[0].future.set_result(results)
                return

            results = list(
                self.pipeline.predict_batch_lane(
                    [request.image_input for request in live_batch],
                    max_images_per_flush=len(live_batch),
                    **kwargs,
                )
            )
            if len(results) != len(live_batch):
                raise RuntimeError(
                    "predict_batch_lane returned "
                    f"{len(results)} results for {len(live_batch)} requests"
                )
        except BaseException as exc:
            for request in live_batch:
                if not request.future.cancelled():
                    request.future.set_exception(exc)
            return

        for request, result in zip(live_batch, results):
            if not request.future.cancelled():
                request.future.set_result(result)
