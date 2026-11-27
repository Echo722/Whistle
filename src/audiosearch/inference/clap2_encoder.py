"""CLAP2 audio-text encoder, loaded the way its authors publish it: OpenCLIP.

CLAP2 (``laion/clap2-*``) is not a ``transformers`` CLAP checkpoint. Its audio
tower is a NaFlex vision transformer over log-mel patches: the number of tokens
follows the clip length instead of being padded to a fixed 10 s crop, the front
end resamples to 32 kHz and cuts 128-bin log-mel into 16x16 patches, and the text
tower is a RoPE transformer with a GPT-2 style BPE tokenizer. Swapping the model
name inside ``ClapModel`` therefore cannot work; the official entry point is
``open_clip.create_model_from_pretrained`` plus the model's own preprocess
factory, which is what this module wraps.

Two things are deliberately project-side, and both are recorded in the space id:

* the window layout (10 s / 5 s hop, anchored at 0) — the only choice CLAP2
  leaves open, because it accepts any length;
* nothing else. Resampling, mel geometry, patch size, tokenizer and pooling all
  come from the checkpoint's own ``open_clip_config.json``.

Text and audio towers come from one checkpoint and write into one
``EmbeddingSpace``; comparing a CLAP2 text query with a CLAP2 audio vector is
legal, comparing it with a ``clap-htsat-unfused`` vector is not (different ids).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .runtime import plan_device, require_torch
from .spaces import EmbeddingSpace, ModelFingerprint, ModelUnavailableError, l2_normalize

DEFAULT_CLAP2_MODEL = "laion/clap2-large-audioxl-s1B"
SMALL_CLAP2_MODEL = "laion/clap2-small-audioxl-s1B"
# Pinned default revision or "latest" to pick the newest cached snapshot.
DEFAULT_CLAP2_REVISION = "latest"
# CLAP2's own front end: mono, 32 kHz, 128-bin log-mel, 16x16 patches.
CLAP2_TARGET_SAMPLE_RATE = 32_000
CLAP2_WINDOW_SECONDS = 10.0
CLAP2_PREPROCESSING_VERSION = "clap2-openclip-v1/mono32k/logmel128-patch16/10s"


def encode_in_length_buckets(
    lengths: Sequence[int],
    batch_size: int,
    run_batch: Callable[[list[int]], np.ndarray],
    is_oom: Callable[[BaseException], bool],
) -> np.ndarray:
    """Run ``run_batch`` over length-sorted buckets and restore input order.

    ``run_batch`` receives original indices and returns a ``(k, dim)`` array. A
    failure for which ``is_oom`` is true splits the batch in half and retries;
    a single-item batch (or any other error) is re-raised.
    """
    n = len(lengths)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)
    size = max(1, int(batch_size))
    order = sorted(range(n), key=lambda i: (lengths[i], i))

    def run(idxs: list[int]) -> list[tuple[list[int], np.ndarray]]:
        try:
            return [(idxs, np.asarray(run_batch(idxs)))]
        except Exception as exc:  # noqa: BLE001 - only OOM is recoverable
            if len(idxs) > 1 and is_oom(exc):
                mid = len(idxs) // 2
                return run(idxs[:mid]) + run(idxs[mid:])
            raise

    out: np.ndarray | None = None
    for start in range(0, n, size):
        for idxs, rows in run(order[start : start + size]):
            if out is None:
                out = np.empty((n, rows.shape[1]), dtype=rows.dtype)
            out[idxs] = rows
    assert out is not None
    return out


def snapshot_directory(model_id: str, revision: str, cache_root: Path | None) -> Path:
    """Locate the HF snapshot of ``model_id`` inside a local cache.

    ``cache_root`` is an ``HF_HOME``-style directory (the one that contains
    ``hub/``); the default resolves via ``HF_HOME``. If ``revision`` is empty or
    ``"latest"``, pick the newest snapshot directory on disk; otherwise require
    the exact pinned revision folder. Never triggers a network fetch.
    """
    if cache_root is None:
        cache_root = Path(_default_hf_home())
    repo = "models--" + model_id.replace("/", "--")
    snapshots = cache_root / "hub" / repo / "snapshots"
    if revision in ("", "latest"):
        if snapshots.is_dir():
            candidates = sorted(
                (p for p in snapshots.iterdir() if p.is_dir()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0]
        raise ModelUnavailableError(
            f"No cached snapshot found for {model_id} in {snapshots}; download it locally first"
        )
    candidate = snapshots / revision
    if candidate.is_dir():
        return candidate
    raise ModelUnavailableError(
        f"CLAP2 snapshot {model_id}@{revision[:12]} is not in {snapshots}; "
        "download it with huggingface_hub.snapshot_download(repo, revision=...) first"
    )


def _default_hf_home() -> str:
    import os

    return os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")


@dataclass
class Clap2EncoderConfig:
    model_id: str = DEFAULT_CLAP2_MODEL
    revision: str = DEFAULT_CLAP2_REVISION
    device: str = "auto"
    precision: str = "auto"
    batch_size: int = 8
    # Directory holding ``hub/models--<org>--<name>/snapshots/<sha>``.
    cache_root: Path | None = None


class Clap2Encoder:
    """Paired CLAP2 towers behind the same surface as ``ClapEncoder``.

    The checkpoint is always read from a pinned local snapshot: an experiment that
    silently fetched whatever ``main`` pointed at would not be reproducible, so a
    missing snapshot is a typed error instead of an implicit download.
    """

    def __init__(self, config: Clap2EncoderConfig | None = None):
        self.config = config or Clap2EncoderConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._transform: Any = None
        self._torch: Any = None
        self._open_clip: Any = None
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
        if pref == "bf16" and not bf16_ok:
            return "fp32"
        return "bf16" if bf16_ok else "fp32"

    def _autocast_ctx(self):
        if self._precision == "bf16" and self._torch is not None:
            return self._torch.amp.autocast("cuda", dtype=self._torch.bfloat16)
        return nullcontext()

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> EmbeddingSpace:
        torch = require_torch()
        try:
            import open_clip
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ModelUnavailableError(
                "open_clip_torch is not installed; CLAP2 needs the official OpenCLIP loader"
            ) from exc

        snapshot = snapshot_directory(
            self.config.model_id, self.config.revision, self.config.cache_root
        )
        device = plan_device(self.config.device)
        resolved_precision = self._resolve_precision(torch, device.device)
        self._torch = torch
        self._open_clip = open_clip
        model_name = f"local-dir:{snapshot}"
        try:
            model, preprocess = open_clip.create_model_from_pretrained(
                model_name, device=device.device
            )
            tokenizer = open_clip.get_tokenizer(model_name)
        except Exception as exc:  # noqa: BLE001 - library raises many types
            raise ModelUnavailableError(
                f"CLAP2 checkpoint {self.config.model_id} could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._transform = preprocess()
        self._precision = resolved_precision
        self.snapshot = snapshot
        resolved_revision = snapshot.name
        self.space = clap2_space(
            model_id=self.config.model_id,
            revision=resolved_revision,
            dimension=int(model.embed_dim),
        )
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.fingerprint = ModelFingerprint(
            model_id=self.config.model_id,
            revision=resolved_revision,
            library="open_clip",
            library_version=str(getattr(open_clip, "__version__", "unknown")),
            device=device.device,
            dtype="bfloat16" if resolved_precision == "bf16" else "float32",
            weights_bytes=params_bytes,
        )
        return self.space

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._transform = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    @property
    def device(self) -> str:
        return str(next(self._model.parameters()).device) if self._model is not None else "unloaded"

    def _require_loaded(self) -> None:
        if self._model is None or self._tokenizer is None or self._transform is None:
            raise ModelUnavailableError("CLAP2 encoder is not loaded; call load() first")

    # -- audio tower --------------------------------------------------------
    def _model_device(self):
        return next(self._model.parameters()).device

    def _patchify(self, samples: np.ndarray, sample_rate_hz: int) -> dict[str, Any]:
        """Run the checkpoint's own front end; it resamples and takes mono itself."""
        waveform = self._torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32))
        return self._transform((waveform[None, :], int(sample_rate_hz)))

    def _stack(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad a batch to the longest clip; ``patch_valid`` marks the padding.

        CLAP2's own contract: padded patches are masked, so batched embeddings
        equal the ones obtained clip by clip. This function is the only batching
        code on this path, which is why the equivalence test exercises it.
        """
        torch = self._torch
        return {
            key: torch.nn.utils.rnn.pad_sequence([item[key] for item in items], batch_first=True)
            for key in items[0]
        }

    def prepare_audio_item(self, samples: np.ndarray, sample_rate_hz: int) -> dict[str, Any]:
        """CPU front end (resample + log-mel + patchify) for one mono window.

        Stateless, so it can run on worker threads while the GPU encodes the
        previous batch.
        """
        self._require_loaded()
        return self._patchify(samples, int(sample_rate_hz))

    def encode_prepared(
        self,
        prepared: Sequence[dict[str, Any]],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode already-patchified windows; returns rows in input order.

        Windows are bucketed by token count so a short window is not padded to
        the longest one in its batch, and a CUDA OOM halves the batch and retries.
        """
        self._require_loaded()
        if not prepared:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        size = max(1, int(batch_size or self.config.batch_size))
        torch = self._torch
        device = self._model_device()
        lengths = [int(item["patches"].shape[0]) for item in prepared]

        def run_batch(idxs: list[int]) -> np.ndarray:
            batch = self._stack([prepared[i] for i in idxs])
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.inference_mode(), self._autocast_ctx():
                embeddings = self._model.encode_audio(batch, normalize=True)
            return embeddings.float().cpu().numpy()

        def is_oom(exc: BaseException) -> bool:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                return True
            return False

        return l2_normalize(encode_in_length_buckets(lengths, size, run_batch, is_oom))

    def encode_audio_items(
        self,
        items: Sequence[tuple[np.ndarray, int]],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode mono ``(waveform, sample_rate_hz)`` items in mixed-rate batches."""
        self._require_loaded()
        if not items:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        prepared = [self.prepare_audio_item(row, int(sr)) for row, sr in items]
        return self.encode_prepared(prepared, batch_size=batch_size)

    def encode_audio_mono(
        self,
        samples: np.ndarray | list[np.ndarray],
        sample_rate_hz: int,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode mono windows given at their native sample rate.

        Windows may differ in length; CLAP2 keeps the native token count, so a
        short final window is not zero-padded into a fixed 10 s crop.
        """
        self._require_loaded()
        if isinstance(samples, np.ndarray):
            rows = [samples] if samples.ndim == 1 else list(samples)
        else:
            rows = list(samples)
        return self.encode_audio_items(
            [(row, int(sample_rate_hz)) for row in rows],
            batch_size=batch_size,
        )

    # -- text tower ---------------------------------------------------------
    def encode_texts(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """CLAP2 text tower → the same space as ``encode_audio_mono``."""
        self._require_loaded()
        if not texts:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        torch = self._torch
        device = self._model_device()
        size = batch_size or self.config.batch_size
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), size):
            tokens = self._tokenizer(texts[start : start + size]).to(device)
            with torch.inference_mode(), self._autocast_ctx():
                embeddings = self._model.encode_text(tokens, normalize=True)
            outputs.append(embeddings.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs, axis=0))


def clap2_space(*, model_id: str, revision: str, dimension: int) -> EmbeddingSpace:
    """Identity of a CLAP2 space; text and audio share it by construction."""
    return EmbeddingSpace(
        model_id=model_id,
        revision=revision,
        tower="audio",
        preprocessing_version=CLAP2_PREPROCESSING_VERSION,
        dimension=dimension,
        normalized=True,
    )


__all__ = [
    "CLAP2_PREPROCESSING_VERSION",
    "CLAP2_TARGET_SAMPLE_RATE",
    "CLAP2_WINDOW_SECONDS",
    "DEFAULT_CLAP2_MODEL",
    "DEFAULT_CLAP2_REVISION",
    "SMALL_CLAP2_MODEL",
    "Clap2Encoder",
    "Clap2EncoderConfig",
    "clap2_space",
    "encode_in_length_buckets",
    "snapshot_directory",
]
