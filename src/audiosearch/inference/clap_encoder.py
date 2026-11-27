"""Paired CLAP text/audio encoder (``laion/clap-htsat-unfused``).

Both towers come from one checkpoint and one revision, and they write into one
``EmbeddingSpace`` — that pairing is the only reason a text query may be compared
against an audio vector at all. Preprocessing follows the checkpoint's own
feature extractor configuration (48 kHz mono, 10 s window, ``repeatpad``), with
the only project-side choice being how a long source is cut into windows.

The resampler is part of the identity, not an implementation detail: v1 of this
module used a filter whose anti-aliasing corner sat at the wrong frequency, so
its vectors were built from aliased input. ``clap-fe-v2`` names the fixed filter
so a v1 artifact can never be mistaken for, reused as, or mixed with a v2 one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from audiosearch.audioio.resample import RESAMPLE_VERSION, resample

from .runtime import plan_device, require_torch, require_transformers
from .spaces import EmbeddingSpace, ModelFingerprint, ModelUnavailableError, l2_normalize

DEFAULT_CLAP_MODEL = "laion/clap-htsat-unfused"
DEFAULT_CLAP_REVISION = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
# Feature-extractor settings copied from the checkpoint for the space id.
TARGET_SAMPLE_RATE = 48_000
WINDOW_SECONDS = 10.0
# Projection width of laion/clap-htsat-unfused; a constant so a model-free
# capability check can compare dimensions without loading the checkpoint.
EXPECTED_DIMENSION = 512
# v1 = the buggy resampler (input-Nyquist corner, per-block history reset).
# v2 = polyphase-v2 resampler; the exact implementation tag is embedded so a
# future filter change cannot silently reuse v2 rows.
PREPROCESSING_SCHEME = "clap-fe-v2"
PREPROCESSING_VERSION = f"{PREPROCESSING_SCHEME}/mono48k/repeatpad/10s/{RESAMPLE_VERSION}"
# Artifacts written before the fix. Kept for diagnostics and downgrade checks only.
LEGACY_PREPROCESSING_VERSIONS: tuple[str, ...] = ("clap-fe-v1/mono48k/repeatpad/10s",)


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """One analysis window on the original file, in original-file seconds."""

    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def plan_windows(
    duration_seconds: float,
    *,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float | None = None,
    max_windows: int | None = None,
) -> list[AudioWindow]:
    """Cover a file with overlapping windows, anchored at 0 so onsets are kept.

    Files shorter than one window are a single window covering the whole file.
    Every returned window is on the original file and never extends past its
    duration, so a hit maps back without a coordinate shift.
    """
    if duration_seconds <= 0:
        raise ValueError("duration must be positive")
    if window_seconds <= 0:
        raise ValueError("window seconds must be positive")
    hop = hop_seconds if hop_seconds is not None else window_seconds / 2
    if hop <= 0:
        raise ValueError("hop seconds must be positive")
    if duration_seconds <= window_seconds:
        return [AudioWindow(0.0, duration_seconds)]
    windows: list[AudioWindow] = []
    start = 0.0
    while True:
        end = min(start + window_seconds, duration_seconds)
        windows.append(AudioWindow(start, end))
        if end >= duration_seconds:
            break
        start += hop
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def coverage(windows: list[AudioWindow], duration_seconds: float) -> float:
    """Fraction of the file touched by at least one window."""
    if duration_seconds <= 0 or not windows:
        return 0.0
    merged: list[list[float]] = []
    for window in sorted(windows, key=lambda w: w.start_seconds):
        if merged and window.start_seconds <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], window.end_seconds)
        else:
            merged.append([window.start_seconds, window.end_seconds])
    covered = sum(end - start for start, end in merged)
    return min(1.0, covered / duration_seconds)


@dataclass
class ClapEncoderConfig:
    model_id: str = DEFAULT_CLAP_MODEL
    revision: str = DEFAULT_CLAP_REVISION
    device: str = "auto"
    batch_size: int = 8
    # Log-mel frames per chunk; the feature extractor pads/truncates to this.
    max_audio_frames: int = 1000


class ClapEncoder:
    """Paired CLAP encoder with one shared space for both towers."""

    def __init__(self, config: ClapEncoderConfig | None = None, *, local_only: bool = True):
        self.config = config or ClapEncoderConfig()
        self.local_only = local_only
        self._model: Any = None
        self._tokenizer: Any = None
        self._feature_extractor: Any = None
        self._torch: Any = None
        self.space: EmbeddingSpace | None = None
        self.fingerprint: ModelFingerprint | None = None

    # -- lifecycle ----------------------------------------------------------
    def load(self) -> EmbeddingSpace:
        torch = require_torch()
        transformers = require_transformers()
        device = plan_device(self.config.device)
        self._torch = torch
        try:
            model = transformers.ClapModel.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.local_only,
            )
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.local_only,
            )
            extractor = transformers.ClapFeatureExtractor.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                local_files_only=self.local_only,
            )
        except OSError as exc:
            raise ModelUnavailableError(
                f"CLAP model {self.config.model_id}@{self.config.revision} "
                f"is not available locally: {type(exc).__name__}"
            ) from exc
        model.eval()
        model.to(device.device)
        self._model = model
        self._tokenizer = tokenizer
        self._feature_extractor = extractor
        # One space for both towers: that is what makes text↔audio comparison legal.
        self.space = EmbeddingSpace(
            model_id=self.config.model_id,
            revision=self.config.revision,
            tower="audio",
            preprocessing_version=PREPROCESSING_VERSION,
            dimension=int(model.config.projection_dim),
            normalized=True,
        )
        params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        self.fingerprint = ModelFingerprint(
            model_id=self.config.model_id,
            revision=self.config.revision,
            library="transformers",
            library_version=transformers.__version__,
            device=device.device,
            dtype=device.dtype,
            weights_bytes=params_bytes,
        )
        return self.space

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        self._feature_extractor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else "unloaded"

    def _require_loaded(self) -> None:
        if self._model is None or self._tokenizer is None or self._feature_extractor is None:
            raise ModelUnavailableError("CLAP encoder is not loaded; call load() first")

    # -- audio tower --------------------------------------------------------
    def _features(self, batch: list[np.ndarray]):
        encoded = self._feature_extractor(
            raw_speech=batch,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
            truncation=False,
            padding="repeatpad",
        )
        return encoded["input_features"].to(self._model.device), encoded.get("is_longer")

    def encode_audio_mono(
        self,
        samples: np.ndarray | list[np.ndarray],
        sample_rate_hz: int,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode mono float32 windows; each element is one window.

        Windows may differ in length (the last window of a file is usually
        shorter than the others). They are resampled to 48 kHz here and then
        handed to the checkpoint's own feature extractor with ``repeatpad``, so
        no window is zero-padded by this project.
        """
        self._require_loaded()
        if isinstance(samples, np.ndarray):
            data = [samples] if samples.ndim == 1 else list(samples)
        else:
            data = list(samples)
        prepared = [
            resample(row.astype(np.float32, copy=False), sample_rate_hz, TARGET_SAMPLE_RATE)
            if sample_rate_hz != TARGET_SAMPLE_RATE
            else row.astype(np.float32, copy=False)
            for row in data
        ]
        return self._encode_prepared(prepared, batch_size=batch_size)

    def _encode_prepared(
        self, prepared: list[np.ndarray], *, batch_size: int | None = None
    ) -> np.ndarray:
        self._require_loaded()
        torch = self._torch
        size = batch_size or self.config.batch_size
        outputs: list[np.ndarray] = []
        for start in range(0, len(prepared), size):
            features, is_longer = self._features(prepared[start : start + size])
            with torch.inference_mode():
                embeddings = self._model.get_audio_features(
                    input_features=features,
                    is_longer=is_longer.to(self._model.device) if is_longer is not None else None,
                )
            outputs.append(embeddings.float().cpu().numpy())
        if not outputs:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        return l2_normalize(np.concatenate(outputs, axis=0))

    # -- text tower ---------------------------------------------------------
    def encode_texts(self, texts: list[str], *, batch_size: int | None = None) -> np.ndarray:
        """CLAP text tower → the same space as ``encode_audio_mono``."""
        self._require_loaded()
        if not texts:
            return np.zeros((0, self.space.dimension), dtype=np.float32)  # type: ignore[union-attr]
        torch = self._torch
        size = batch_size or self.config.batch_size
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), size):
            tokens = self._tokenizer(
                texts[start : start + size],
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            tokens = {key: value.to(self._model.device) for key, value in tokens.items()}
            with torch.inference_mode():
                embeddings = self._model.get_text_features(**tokens)
            outputs.append(embeddings.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs, axis=0))


__all__ = [
    "DEFAULT_CLAP_MODEL",
    "DEFAULT_CLAP_REVISION",
    "EXPECTED_DIMENSION",
    "LEGACY_PREPROCESSING_VERSIONS",
    "PREPROCESSING_SCHEME",
    "PREPROCESSING_VERSION",
    "TARGET_SAMPLE_RATE",
    "WINDOW_SECONDS",
    "AudioWindow",
    "ClapEncoder",
    "ClapEncoderConfig",
    "coverage",
    "plan_windows",
]
