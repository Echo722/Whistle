"""Behavioral tests for CUDA kernel probing, CPU fallback, and Qwen3 OOM batch halving."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from audiosearch.inference import runtime
from audiosearch.inference.qwen3_encoder import Qwen3TextEncoder, Qwen3TextEncoderConfig
from audiosearch.inference.spaces import EmbeddingSpace, ModelUnavailableError


class _FakeBrokenKernelCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def empty_cache() -> None:
        return None


def _make_broken_kernel_torch():
    def _ones(*_args, **_kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    return SimpleNamespace(
        float32="float32",
        cuda=_FakeBrokenKernelCuda(),
        ones=_ones,
    )


def test_plan_device_falls_back_to_cpu_when_cuda_kernel_fails(monkeypatch):
    runtime.reset_cuda_probe_cache()
    try:
        monkeypatch.setattr(runtime, "require_torch", _make_broken_kernel_torch)
        plan = runtime.plan_device("auto")
        assert plan.device == "cpu"
        assert "no kernel image" in plan.reason

        with pytest.raises(ModelUnavailableError, match="no kernel image"):
            runtime.plan_device("cuda")
    finally:
        runtime.reset_cuda_probe_cache()


def test_qwen3_encode_raw_halves_batch_on_oom():
    class _FakeOOMError(RuntimeError):
        pass

    class _FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def __getitem__(self, item) -> _FakeTensor:
            return _FakeTensor(self.arr[item])

        def to(self, _device):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self.arr

    class _FakeBatch(dict):
        def to(self, _device):
            return self

    class _FakeTokenizer:
        def __call__(self, texts, **_kwargs):
            n = len(texts)
            mask = np.ones((n, 2), dtype=np.int64)
            return _FakeBatch(
                {
                    "input_ids": _FakeTensor(np.arange(n * 2, dtype=np.int64).reshape(n, 2)),
                    "attention_mask": _FakeTensor(mask),
                }
            )

    class _FakeModel:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def parameters(self):
            yield SimpleNamespace(device="cuda:0")

        def __call__(self, **encoded):
            n = encoded["attention_mask"].arr.shape[0]
            self.batch_sizes.append(n)
            if n > 2:
                raise _FakeOOMError("CUDA out of memory")
            hidden = np.zeros((n, 2, 4), dtype=np.float32)
            for i in range(n):
                hidden[i, -1, :] = np.array([1.0, 0.5, 0.25, 0.0], dtype=np.float32) + i
            return SimpleNamespace(last_hidden_state=_FakeTensor(hidden))

    empty_cache_calls = 0

    class _FakeCuda:
        OutOfMemoryError = _FakeOOMError

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            nonlocal empty_cache_calls
            empty_cache_calls += 1

    fake_torch = SimpleNamespace(
        cuda=_FakeCuda(),
        inference_mode=nullcontext,
    )

    enc = Qwen3TextEncoder(Qwen3TextEncoderConfig(mrl_dim=4, batch_size=4))
    enc._torch = fake_torch
    enc._tokenizer = _FakeTokenizer()
    fake_model = _FakeModel()
    enc._model = fake_model
    enc.space = EmbeddingSpace(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="test",
        tower="text",
        preprocessing_version="qwen3-lasttoken-mrl4",
        dimension=4,
        normalized=True,
    )

    monkeypatch_last = None
    try:
        from audiosearch.inference import qwen3_encoder

        monkeypatch_last = qwen3_encoder.last_token_pool
        qwen3_encoder.last_token_pool = lambda hs, _mask: _FakeTensor(hs.arr[:, -1, :])

        vecs = enc.encode_documents(["a", "b", "c", "d"])
        assert vecs.shape == (4, 4)
        assert fake_model.batch_sizes == [4, 2, 2]
        assert empty_cache_calls == 1
    finally:
        if monkeypatch_last is not None:
            qwen3_encoder.last_token_pool = monkeypatch_last
