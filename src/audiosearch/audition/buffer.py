"""Decoding one fetched file into a buffer at the output sample rate.

Why the output rate and not the file's: the audio callback must never convert.
Doing the conversion up front means the buffer's frame index *is* the playback
position, which is what makes the playhead trustworthy instead of estimated.

The whole file is decoded at once. Audition is one file at a time and the
library is mostly short (43% under two seconds), so a cap is a cheaper guard
than a streaming decoder: a file past the cap is refused with a typed error
rather than silently truncated.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from audiosearch.audioio.decode import open_audio_file, open_pedalboard_audio
from audiosearch.audioio.resample import resample

DEFAULT_MAX_SECONDS = 120.0


class AuditionError(RuntimeError):
    """Anything that stops bytes from becoming a playable buffer."""


class UnsupportedFormatError(AuditionError):
    """The bytes are not audio libsndfile can read."""


class TooLongError(AuditionError):
    """The file is longer than audition will hold in memory."""


@dataclass(frozen=True, slots=True)
class PlaybackBuffer:
    """Samples ready to hand to the output stream, plus what produced them."""

    samples: np.ndarray  # (frames, channels) float32 at ``sample_rate_hz``
    sample_rate_hz: int
    duration_seconds: float
    source_sample_rate_hz: int
    resampled: bool

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])

    @property
    def channel_count(self) -> int:
        return int(self.samples.shape[1])


def _decode_soundfile(
    handle: sf.SoundFile,
    *,
    target_rate: int,
    max_seconds: float,
) -> PlaybackBuffer:
    with handle:
        source_rate = int(handle.samplerate)
        frames = int(handle.frames)
        if source_rate <= 0 or frames <= 0:
            raise UnsupportedFormatError("音频没有可播放的采样")
        duration = frames / source_rate
        if duration > max_seconds:
            raise TooLongError(f"{duration:.1f} 秒超过试听上限 {max_seconds:.0f} 秒")
        try:
            samples = handle.read(dtype="float32", always_2d=True)
        except Exception as exc:
            raise UnsupportedFormatError(f"读取音频失败：{exc}") from exc

    resampled = source_rate != target_rate
    if resampled:
        samples = resample(samples, source_rate, target_rate)
    return PlaybackBuffer(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate_hz=target_rate,
        duration_seconds=duration,
        source_sample_rate_hz=source_rate,
        resampled=resampled,
    )


def decode_for_playback(
    data: bytes,
    *,
    target_rate: int,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> PlaybackBuffer:
    """Decode ``data`` to float32 at ``target_rate``.

    Raises ``UnsupportedFormatError`` for undecodable bytes and ``TooLongError``
    when the header claims more than ``max_seconds``. Both are checked from the
    header before any samples are read, so a huge file costs a header parse.
    """
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    try:
        handle = sf.SoundFile(io.BytesIO(data))
    except Exception as exc:  # libsndfile raises LibsndfileError, not OSError
        raise UnsupportedFormatError(f"无法识别音频格式：{exc}") from exc
    return _decode_soundfile(handle, target_rate=target_rate, max_seconds=max_seconds)


def decode_file_for_playback(
    path: str | Path,
    *,
    target_rate: int,
    max_seconds: float = DEFAULT_MAX_SECONDS,
) -> PlaybackBuffer:
    """Decode local audio file at ``path`` to float32 at ``target_rate``."""
    if target_rate <= 0:
        raise ValueError("target_rate must be positive")
    try:
        handle = open_audio_file(path)
    except Exception as exc:
        raise UnsupportedFormatError(f"无法识别音频格式：{exc}") from exc
    try:
        return _decode_soundfile(handle, target_rate=target_rate, max_seconds=max_seconds)
    except UnsupportedFormatError as primary_exc:
        try:
            pb_handle = open_pedalboard_audio(path)
        except Exception:
            raise primary_exc from None
        return _decode_soundfile(pb_handle, target_rate=target_rate, max_seconds=max_seconds)


__all__ = [
    "AuditionError",
    "PlaybackBuffer",
    "TooLongError",
    "UnsupportedFormatError",
    "decode_file_for_playback",
    "decode_for_playback",
]
