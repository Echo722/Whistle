"""Qwen3-Embedding text encoder with MRL truncation and BF16 support."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .clap2_encoder import snapshot_directory
from .runtime import plan_device, require_torch, require_transformers
from .spaces import EmbeddingSpace, ModelFingerprint, ModelUnavailableError, l2_normalize

DEFAULT_QWEN3_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QWEN3_REVISION = "latest"
DEFAULT_QWEN3_MRL_DIM = 512
QWEN3_QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Official last-token pooling for causal/left-padded Qwen3 embedding models."""
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    batch_idx = last_hidden_states.new_empty((batch_size,), dtype=sequence_lengths.dtype)
    batch_idx.copy_(attention_mask.new_tensor(range(batch_size)))
    return last_hidden_states[batch_idx, sequence_lengths]


@dataclass
class Qwen3TextEncoderConfig:
    model_id: str = DEFAULT_QWEN3_MODEL
    revision: str = DEFAULT_QWEN3_REVISION
    device: str = "auto"
    precision: str = "auto"
    mrl_dim: int | None = DEFAULT_QWEN3_MRL_DIM
    max_tokens: int = 256
    batch_size: int = 64
    query_prefix: str = QWEN3_QUERY_PREFIX
    cache_root: Path | None = None


class Qwen3TextEncoder:
    """Offline-first Qwen3-Embedding encoder returning normalized MRL vectors."""

    def __init__(self, config: Qwen3TextEncoderConfig | None = None) -> None:
        self.config = config or Qwen3TextEncoderConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._precision: str = "fp32"
        self.space: EmbeddingSpace | None = None
        self.fingerprint: ModelFingerprint | None = None
        self.snapshot: Path | None = None

    def _resolve_precision(self, torch: Any, device_str: str) -> str:
        pref = self.config.precision
        if pref not in {"auto", "bf16", "fp32"}:
            raise ValueError(f"unsupported precision {pref!r}; expected auto|bf16|fp32")
        if pref == "fp32" or not device_str.startswith("cuda"):
            return "fp32"
        bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        return "bf16" if bf16_ok else "fp32"

    def _autocast_ctx(self):
        if self._precision == "bf16" and self._torch is not None:
            return self._torch.amp.autocast("cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    def load(self) -> EmbeddingSpace:
        torch = require_torch()
        transformers = require_transformers()
        snapshot = snapshot_directory(
            self.config.model_id, self.config.revision, self.config.cache_root
        )
        device = plan_device(self.config.device)
        resolved_precision = self._resolve_precision(torch, device.device)
        self._torch = torch
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(snapshot),
                padding_side="left",
                local_files_only=True,
            )
            model = transformers.AutoModel.from_pretrained(
                str(snapshot),
                local_files_only=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Qwen3 text model {self.config.model_id} could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        model.eval()
        model.to(device.device)
        self._tokenizer = tokenizer
        self._model = model
        self._precision = resolved_precision
        self.snapshot = snapshot

        hidden_dim = int(model.config.hidden_size)
        out_dim = int(self.config.mrl_dim) if self.config.mrl_dim else hidden_dim
        if out_dim <= 0 or out_dim > hidden_dim:
            raise ValueError(f"invalid MRL dimension {out_dim} for hidden_size {hidden_dim}")

        resolved_revision = snapshot.name
        self.space = EmbeddingSpace(
            model_id=self.config.model_id,
            revision=resolved_revision,
            tower="text",
            preprocessing_version=f"qwen3-lasttoken-mrl{out_dim}",
            dimension=out_dim,
            normalized=True,
        )
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.fingerprint = ModelFingerprint(
            model_id=self.config.model_id,
            revision=resolved_revision,
            library="transformers",
            library_version=str(getattr(transformers, "__version__", "unknown")),
            device=device.device,
            dtype="bfloat16" if resolved_precision == "bf16" else "float32",
            weights_bytes=params_bytes,
        )
        return self.space

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def _require_loaded(self) -> None:
        if self._model is None or self._tokenizer is None or self.space is None:
            raise ModelUnavailableError("Qwen3 text encoder is not loaded; call load() first")

    def _encode_raw(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        self._require_loaded()
        assert self.space is not None
        if not texts:
            return np.zeros((0, self.space.dimension), dtype=np.float32)

        torch = self._torch
        device = next(self._model.parameters()).device
        size = batch_size or self.config.batch_size
        out_dim = self.space.dimension
        outputs: list[np.ndarray] = []

        def _is_oom(exc: BaseException) -> bool:
            cuda_mod = getattr(torch, "cuda", None)
            oom_cls = getattr(cuda_mod, "OutOfMemoryError", None) if cuda_mod is not None else None
            if oom_cls is not None and isinstance(exc, oom_cls):
                return True
            return "out of memory" in str(exc).lower()

        def _encode_chunk(sub_chunk: list[str]) -> list[np.ndarray]:
            try:
                encoded = self._tokenizer(
                    sub_chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_tokens,
                    return_tensors="pt",
                ).to(device)
                with torch.inference_mode(), self._autocast_ctx():
                    model_out = self._model(**encoded)
                    pooled = last_token_pool(model_out.last_hidden_state, encoded["attention_mask"])
                    pooled = pooled[:, :out_dim]
                return [pooled.float().cpu().numpy()]
            except Exception as exc:  # noqa: BLE001 - halve batch and retry on OOM
                if _is_oom(exc):
                    cuda_mod = getattr(torch, "cuda", None)
                    if cuda_mod is not None and cuda_mod.is_available():
                        cuda_mod.empty_cache()
                    if len(sub_chunk) > 1:
                        mid = len(sub_chunk) // 2
                        return _encode_chunk(sub_chunk[:mid]) + _encode_chunk(sub_chunk[mid:])
                raise

        for start in range(0, len(texts), size):
            chunk = [str(t) if t else " " for t in texts[start : start + size]]
            outputs.extend(_encode_chunk(chunk))

        return l2_normalize(np.concatenate(outputs, axis=0))

    def encode_documents(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> np.ndarray:
        """Encode indexed asset metadata texts without query instruction."""
        return self._encode_raw(texts, batch_size=batch_size)

    def encode_queries(
        self, texts: Sequence[str], *, batch_size: int | None = None
    ) -> np.ndarray:
        """Encode search queries with Qwen3's official retrieval instruction prefix."""
        prefix = self.config.query_prefix
        formatted = [f"{prefix}{t}" if prefix else t for t in texts]
        return self._encode_raw(formatted, batch_size=batch_size)


__all__ = [
    "DEFAULT_QWEN3_MODEL",
    "DEFAULT_QWEN3_MRL_DIM",
    "DEFAULT_QWEN3_REVISION",
    "QWEN3_QUERY_PREFIX",
    "Qwen3TextEncoder",
    "Qwen3TextEncoderConfig",
    "last_token_pool",
]
