"""Bounded, streaming WAVE decoding into float32.

Nothing here loads a whole file: callers get frame blocks of a fixed size and
decide when to stop. Time is always original-file seconds; the sample index →
time conversion uses the file's own sample rate, so a decoded window keeps the
same start/end on the source file.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .errors import AssetOfflineError, DecodeError, SourceRangeError
from .wav import WavInfo, parse_wav

# 65_536 stereo frames is 512 KiB of float32: small enough to interleave with
# other work, large enough that per-block overhead disappears.
DEFAULT_BLOCK_FRAMES = 65_536
_MAX_BLOCK_BYTES = 8 * 1024 * 1024


def _decode_ints(raw: bytes, sample_format: str, channels: int) -> np.ndarray:
    """Convert interleaved integer PCM bytes to float32 in [-1, 1]."""
    if sample_format == "pcm_u8":
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_format == "pcm_s16":
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_format == "pcm_s24":
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        # Sign-extend from 24 bits.
        values = (values ^ 0x800000) - 0x800000
        data = values.astype(np.float32) / 8388608.0
    elif sample_format == "pcm_s32":
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif sample_format == "float32":
        data = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif sample_format == "float64":
        data = np.frombuffer(raw, dtype="<f8").astype(np.float64).astype(np.float32)
    else:  # pragma: no cover - rejected while parsing fmt
        raise DecodeError(f"unhandled sample format: {sample_format}")
    return np.ascontiguousarray(data.reshape(-1, channels))


@dataclass(frozen=True, slots=True)
class DecodedWindow:
    """A block of original-file audio, already mapped back to source seconds."""

    start_seconds: float
    end_seconds: float
    # Shape (frames, channels); channel order is the original file order.
    samples: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.samples.shape[0])


class WavReader:
    """Streaming reader over one WAVE file."""

    def __init__(self, path: Path, stream: BinaryIO, info: WavInfo) -> None:
        self.path = path
        self._stream = stream
        self.info = info

    def __enter__(self) -> WavReader:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._stream.close()

    # -- geometry -----------------------------------------------------------
    @property
    def sample_rate_hz(self) -> int:
        return self.info.sample_rate_hz

    @property
    def channel_count(self) -> int:
        return self.info.channel_count

    def seconds_to_frame(self, seconds: float) -> int:
        return int(round(seconds * self.info.sample_rate_hz))

    def frame_to_seconds(self, frame: int) -> float:
        return frame / self.info.sample_rate_hz

    def frame_bounds(self, start_seconds: float, end_seconds: float) -> tuple[int, int]:
        """Map an original-file range to whole frames, clamped to the payload."""
        if start_seconds < 0 or end_seconds <= start_seconds:
            raise SourceRangeError("source range must be non-empty and non-negative")
        first = self.seconds_to_frame(start_seconds)
        last = self.seconds_to_frame(end_seconds)
        first = max(0, min(first, self.info.frames))
        last = max(0, min(last, self.info.frames))
        if last <= first:
            raise SourceRangeError("source range maps to zero frames in this file")
        return first, last

    # -- reading ------------------------------------------------------------
    def read_frames(self, start_frame: int, count: int) -> np.ndarray:
        """Read ``count`` interleaved frames starting at ``start_frame``."""
        if count <= 0:
            return np.zeros((0, self.info.channel_count), dtype=np.float32)
        block_align = self.info.block_align
        available = max(0, min(count, self.info.frames - start_frame))
        if available <= 0:
            return np.zeros((0, self.info.channel_count), dtype=np.float32)
        byte_count = available * block_align
        self._stream.seek(self.info.data_offset + start_frame * block_align)
        raw = self._stream.read(byte_count)
        if len(raw) != byte_count:
            raise DecodeError(
                f"short read: wanted {byte_count} bytes, got {len(raw)} "
                f"(file may have been truncated)"
            )
        return _decode_ints(raw, self.info.sample_format, self.info.channel_count)

    def iter_frames(
        self,
        start_frame: int,
        end_frame: int,
        *,
        block_frames: int = DEFAULT_BLOCK_FRAMES,
    ) -> Iterator[np.ndarray]:
        """Yield blocks of at most ``block_frames`` frames between the bounds."""
        block_frames = max(1, min(block_frames, _MAX_BLOCK_BYTES // self.info.block_align))
        position = start_frame
        while position < end_frame:
            take = min(block_frames, end_frame - position)
            block = self.read_frames(position, take)
            if block.shape[0] == 0:
                break
            yield block
            position += block.shape[0]

    def read_range(
        self,
        start_seconds: float,
        end_seconds: float,
        *,
        channels: list[int] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Decode a range fully into memory; the caller owns the size decision."""
        first, last = self.frame_bounds(start_seconds, end_seconds)
        blocks = list(self.iter_frames(first, last))
        samples = (
            np.concatenate(blocks, axis=0)
            if blocks
            else np.zeros((0, self.info.channel_count), dtype=np.float32)
        )
        if channels is None:
            return samples, first
        for channel in channels:
            if not 0 <= channel < self.info.channel_count:
                raise SourceRangeError(f"source channel {channel} is out of range")
        return samples[:, list(channels)], first

    def read_range_mono(
        self,
        start_seconds: float,
        end_seconds: float,
        *,
        block_frames: int = DEFAULT_BLOCK_FRAMES,
    ) -> np.ndarray:
        """Decode a range as mono float32 (mean of channels), block by block."""
        first, last = self.frame_bounds(start_seconds, end_seconds)
        parts = [block.mean(axis=1, dtype=np.float32) for block in self.iter_frames(first, last)]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)


def open_wav(path: Path) -> WavReader:
    """Open a WAVE file for streaming reads; raises typed audio errors."""
    if not path.is_file():
        raise AssetOfflineError(f"audio file is missing: {path.name}")
    try:
        stream = path.open("rb")
    except OSError as exc:
        raise AssetOfflineError(f"audio file cannot be opened: {type(exc).__name__}") from exc
    try:
        info = parse_wav(stream, file_size=path.stat().st_size)
    except Exception:
        stream.close()
        raise
    return WavReader(path, stream, info)


def probe(path: Path) -> WavInfo:
    """Header-only probe; does not decode audio and never leaves the file open."""
    with open_wav(path) as wav:
        return wav.info


_SAMPLE_FORMAT_TO_SUBTYPE: dict[str, str] = {
    "pcm_u8": "PCM_U8",
    "pcm_s16": "PCM_16",
    "pcm_s24": "PCM_24",
    "pcm_s32": "PCM_32",
    "float32": "FLOAT",
    "float64": "DOUBLE",
}


class _SoundFileWavFallback:
    """Read-only ``soundfile.SoundFile`` facade backed by pure-Python ``WavReader``."""

    def __init__(self, reader: WavReader) -> None:
        self._reader = reader
        self._pos = 0

    @property
    def samplerate(self) -> int:
        return int(self._reader.info.sample_rate_hz)

    @property
    def frames(self) -> int:
        return int(self._reader.info.frames)

    @property
    def channels(self) -> int:
        return int(self._reader.info.channel_count)

    @property
    def subtype(self) -> str:
        return _SAMPLE_FORMAT_TO_SUBTYPE.get(self._reader.info.sample_format, "PCM_16")

    def __len__(self) -> int:
        return self.frames

    def seekable(self) -> bool:
        return True

    def seek(self, frames: int) -> int:
        self._pos = max(0, min(int(frames), self.frames))
        return self._pos

    def read(
        self,
        frames: int = -1,
        dtype: str = "float32",
        always_2d: bool = True,
    ) -> np.ndarray:
        remaining = max(0, self.frames - self._pos)
        count = remaining if frames < 0 else min(max(0, int(frames)), remaining)
        if count <= 0:
            arr = np.zeros((0, self.channels), dtype=np.float32)
        else:
            arr = self._reader.read_frames(self._pos, count)
            self._pos += int(arr.shape[0])
        if dtype != "float32":
            arr = arr.astype(dtype, copy=False)
        if not always_2d and self.channels == 1:
            return arr[:, 0]
        return arr

    def close(self) -> None:
        self._reader.close()

    def __enter__(self) -> _SoundFileWavFallback:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class _SoundFilePedalboardFallback:
    """Read-only ``soundfile.SoundFile`` facade backed by ``pedalboard.io.AudioFile``."""

    def __init__(self, audio_file: object) -> None:
        self._af = audio_file
        self._pos = 0
        self._samplerate = int(audio_file.samplerate)
        self._frames = int(audio_file.frames)
        self._channels = int(audio_file.num_channels)
        self._subtype = str(getattr(audio_file, "file_dtype", "FLOAT") or "FLOAT")

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def subtype(self) -> str:
        return self._subtype

    def __len__(self) -> int:
        return self._frames

    def seekable(self) -> bool:
        return True

    def seek(self, frames: int) -> int:
        self._pos = max(0, min(int(frames), self._frames))
        self._af.seek(self._pos)
        return self._pos

    def read(
        self,
        frames: int = -1,
        dtype: str = "float32",
        always_2d: bool = True,
    ) -> np.ndarray:
        remaining = max(0, self._frames - self._pos)
        count = remaining if frames < 0 else min(max(0, int(frames)), remaining)
        if count <= 0:
            arr = np.zeros((0, self._channels), dtype=dtype)
        else:
            raw = self._af.read(count)
            arr = np.ascontiguousarray(raw.T, dtype=dtype)
            self._pos += int(arr.shape[0])
        if not always_2d and self._channels == 1:
            return arr[:, 0]
        return arr

    def close(self) -> None:
        self._af.close()

    def __enter__(self) -> _SoundFilePedalboardFallback:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


def open_pedalboard_audio(path: str | Path) -> _SoundFilePedalboardFallback:
    """Open ``path`` via ``pedalboard.io.AudioFile`` wrapped as a SoundFile-like reader."""
    from pedalboard.io import AudioFile

    return _SoundFilePedalboardFallback(AudioFile(str(Path(path))))


def open_audio_file(path: str | Path):
    """Open ``path`` via soundfile, falling back to pure-Python WAV or pedalboard."""
    import soundfile as sf

    file_path = Path(path)
    try:
        return sf.SoundFile(str(file_path), "r")
    except Exception as sf_exc:
        if file_path.suffix.lower() == ".wav":
            try:
                return _SoundFileWavFallback(open_wav(file_path))
            except Exception:
                pass
        try:
            return open_pedalboard_audio(file_path)
        except Exception:
            pass
        raise sf_exc


__all__ = [
    "DEFAULT_BLOCK_FRAMES",
    "DecodedWindow",
    "WavReader",
    "open_audio_file",
    "open_pedalboard_audio",
    "open_wav",
    "probe",
]
