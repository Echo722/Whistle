"""Synthetic WAVE files carrying metadata chunks, written on the fly.

Everything is built byte by byte so a test can put exactly one chunk in exactly one
state (odd length, missing pad byte, declared size past EOF, RF64 header). The real
library is never touched and no fixture is committed.
"""

from __future__ import annotations

import struct
from pathlib import Path

PCM = 1


def chunk(chunk_id: bytes, body: bytes, *, pad: bool = True) -> bytes:
    """One RIFF chunk, word aligned unless ``pad`` is disabled on purpose."""
    padding = b"\x00" if pad and len(body) % 2 else b""
    return chunk_id + struct.pack("<I", len(body)) + body + padding


def fmt_chunk(*, channels: int = 1, rate: int = 48000, bits: int = 16) -> bytes:
    block_align = channels * bits // 8
    body = struct.pack("<HHIIHH", PCM, channels, rate, rate * block_align, block_align, bits)
    return chunk(b"fmt ", body)


def data_chunk(*, frames: int = 480, channels: int = 1, bits: int = 16) -> bytes:
    return chunk(b"data", b"\x00" * (frames * channels * bits // 8))


def wave_file(
    *,
    before_data: tuple[bytes, ...] = (),
    after_data: tuple[bytes, ...] = (),
    frames: int = 480,
    channels: int = 1,
    riff_id: bytes = b"RIFF",
) -> bytes:
    """A complete PCM WAVE file with optional extra chunks on either side of ``data``."""
    body = (
        b"WAVE"
        + fmt_chunk(channels=channels)
        + b"".join(before_data)
        + data_chunk(frames=frames, channels=channels)
        + b"".join(after_data)
    )
    return riff_id + struct.pack("<I", len(body)) + body


def write_file(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def bext_body(
    *,
    description: str = "",
    originator: str = "",
    originator_reference: str = "",
    date: bytes = b"",
    time_value: bytes = b"",
    time_reference: int = 0,
    version: int = 0,
    umid: bytes = b"",
    coding_history: bytes = b"",
    fixed_bytes: int = 602,
) -> bytes:
    """Build a ``bext`` payload with the documented BWF field offsets."""
    body = (
        _fixed(description, 256)
        + _fixed(originator, 32)
        + _fixed(originator_reference, 32)
        + _fixed_raw(date, 10)
        + _fixed_raw(time_value, 8)
        + struct.pack("<Q", time_reference)
        + struct.pack("<H", version)
        + _fixed_raw(umid, 64)
        # Fixed fields through the UMID occupy 412 bytes; the reserved block fills the
        # rest of the documented 602-byte header.
        + b"\x00" * max(fixed_bytes - 412, 0)
    )
    return body + coding_history


def bext_chunk(**kwargs) -> bytes:
    return chunk(b"bext", bext_body(**kwargs))


def info_body(fields: list[tuple[bytes, bytes]], *, list_type: bytes = b"INFO") -> bytes:
    inner = b"".join(chunk(tag, value) for tag, value in fields)
    return list_type + inner


def info_chunk(fields: list[tuple[bytes, bytes]], *, list_type: bytes = b"INFO") -> bytes:
    return chunk(b"LIST", info_body(fields, list_type=list_type))


def ixml_chunk(text: str | bytes) -> bytes:
    payload = text.encode("utf-8") if isinstance(text, str) else text
    return chunk(b"iXML", payload)


def _fixed(text: str, width: int) -> bytes:
    return _fixed_raw(text.encode("utf-8"), width)


def _fixed_raw(raw: bytes, width: int) -> bytes:
    return raw[:width].ljust(width, b"\x00")
