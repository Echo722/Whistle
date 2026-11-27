"""Synthetic WAVE fixtures written on the fly; the real library is never touched."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _fmt_chunk(
    *,
    format_tag: int,
    channels: int,
    rate: int,
    bits: int,
    extensible_subtag: int | None = None,
) -> bytes:
    block_align = channels * bits // 8
    body = struct.pack("<HHIIHH", format_tag, channels, rate, rate * block_align, block_align, bits)
    if format_tag == 0xFFFE:
        # cbSize(2) | wValidBitsPerSample(2) | dwChannelMask(4) | GUID(16) = 24 bytes
        sub = struct.pack("<I", extensible_subtag or 1)
        body += (
            struct.pack("<H", 22)
            + struct.pack("<H", bits)
            + struct.pack("<I", 0)
            + sub
            + b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"
        )
    return b"fmt " + struct.pack("<I", len(body)) + body


def _chunk(chunk_id: bytes, body: bytes) -> bytes:
    padding = b"\x00" if len(body) % 2 else b""
    return chunk_id + struct.pack("<I", len(body)) + body + padding


def encode_int24(samples: np.ndarray) -> bytes:
    """Interleave-free int24 encoder for test signals in [-1, 1)."""
    flat = np.asarray(samples, dtype=np.float32).reshape(-1)
    ints = np.clip(np.round(flat * 8388607.0), -8388608, 8388607).astype(np.int32)
    raw = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :3]
    return raw.tobytes()


def write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    rate: int = 48000,
    bit_depth: int = 16,
    extra_chunks: dict[bytes, bytes] | None = None,
) -> Path:
    """Write interleaved float samples as PCM WAVE; ``samples`` shape (frames, channels)."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = samples.shape[1]
    if bit_depth == 16:
        body = (np.clip(samples, -1.0, 32767 / 32768) * 32768.0).astype("<i2").tobytes()
        tag, bits = 1, 16
    elif bit_depth == 24:
        body = encode_int24(samples)
        tag, bits = 1, 24
    elif bit_depth == 32:
        # Float WAVE is allowed to exceed [-1, 1]; callers must clamp for display.
        body = samples.astype("<f4").tobytes()
        tag, bits = 3, 32
    elif bit_depth == 8:
        body = (np.clip(samples, -1.0, 1.0) * 127 + 128).astype(np.uint8).tobytes()
        tag, bits = 1, 8
    else:
        raise ValueError(f"unsupported test bit depth: {bit_depth}")
    chunks = [_fmt_chunk(format_tag=tag, channels=channels, rate=rate, bits=bits)]
    for chunk_id, chunk_body in (extra_chunks or {}).items():
        chunks.append(_chunk(chunk_id, chunk_body))
    chunks.append(_chunk(b"data", body))
    payload = b"WAVE" + b"".join(chunks)
    path.write_bytes(b"RIFF" + struct.pack("<I", len(payload)) + payload)
    return path


def write_extensible_wav(path: Path, samples: np.ndarray, *, rate: int = 48000) -> Path:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    channels = samples.shape[1]
    fmt = _fmt_chunk(format_tag=0xFFFE, channels=channels, rate=rate, bits=32, extensible_subtag=3)
    data = _chunk(b"data", samples.astype("<f4").tobytes())
    payload = b"WAVE" + fmt + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(payload)) + payload)
    return path


def tone(frames: int, rate: int, frequency: float, *, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(frames, dtype=np.float64) / rate
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
