"""Minimal RIFF/RF64 WAVE reader with no third-party dependency.

The reader only looks at container structure: it never loads audio data. That
keeps header probing cheap for large libraries and lets the caller decide how
many frames to pull into memory.

Chunk parsing follows the RIFF layout rather than assuming tags sit at the
front: ``LIST``/``bext``/``iXML``/``ID3`` chunks may appear before or after the
``data`` chunk (see the project plan, section 6.1).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from .errors import DecodeError, UnsupportedFormatError

PCM = 0x0001
IEEE_FLOAT = 0x0003
EXTENSIBLE = 0xFFFE

# 12-byte tail shared by every KSDATAFORMAT_SUBTYPE_* GUID (after the 4-byte code).
_KSDATAFORMAT_TAIL: Final = b"\x00\x00\x10\x00\x80\x00\x00\xaa\x00\x38\x9b\x71"

_RIFF_HEADERS: Final = {b"RIFF": False, b"RIFX": False, b"RF64": True, b"BW64": True}


@dataclass(frozen=True, slots=True)
class WavInfo:
    """What the container says about the audio payload."""

    sample_rate_hz: int
    channel_count: int
    frames: int
    bit_depth: int
    # One of pcm_u8 / pcm_s16 / pcm_s24 / pcm_s32 / float32 / float64.
    sample_format: str
    block_align: int
    data_offset: int
    data_bytes: int

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate_hz


def _sample_format(format_tag: int, bits: int) -> str:
    if format_tag == PCM:
        try:
            return {8: "pcm_u8", 16: "pcm_s16", 24: "pcm_s24", 32: "pcm_s32"}[bits]
        except KeyError:
            raise UnsupportedFormatError(f"unsupported PCM bit depth: {bits}") from None
    if format_tag == IEEE_FLOAT:
        try:
            return {32: "float32", 64: "float64"}[bits]
        except KeyError:
            raise UnsupportedFormatError(f"unsupported float bit depth: {bits}") from None
    raise UnsupportedFormatError(f"unsupported WAVE format tag: 0x{format_tag:04x}")


def _parse_fmt(body: bytes) -> tuple[str, int, int, int, int]:
    """Return (sample_format, sample_rate, channels, bit_depth, block_align)."""
    if len(body) < 16:
        raise DecodeError("fmt chunk is shorter than 16 bytes")
    format_tag, channels, rate, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", body[:16])
    if format_tag == EXTENSIBLE:
        if len(body) < 40:
            raise DecodeError("extensible fmt chunk is truncated")
        sub_tag = struct.unpack("<I", body[24:28])[0]
        if body[28:40] != _KSDATAFORMAT_TAIL:
            raise UnsupportedFormatError("extensible WAVE uses an unknown subformat GUID")
        format_tag = sub_tag
    if channels <= 0:
        raise UnsupportedFormatError("WAVE declares zero channels")
    if rate <= 0:
        raise UnsupportedFormatError("WAVE declares an invalid sample rate")
    if block_align <= 0 or block_align != channels * bits // 8:
        # A wrong block_align would silently shear channels apart.
        raise UnsupportedFormatError("WAVE block alignment is inconsistent with the format")
    return _sample_format(format_tag, bits), rate, channels, bits, block_align


def parse_wav(stream: BinaryIO, *, file_size: int | None = None) -> WavInfo:
    """Scan RIFF chunks of an already-open binary stream and return the payload info."""
    stream.seek(0)
    header = stream.read(12)
    if len(header) < 12:
        raise UnsupportedFormatError("file is shorter than a RIFF header")
    riff, _riff_size, wave = header[:4], struct.unpack("<I", header[4:8])[0], header[8:12]
    if riff not in _RIFF_HEADERS:
        raise UnsupportedFormatError(f"not a RIFF/RF64 container: {riff!r}")
    if riff == b"RIFX":
        raise UnsupportedFormatError("big-endian RIFX WAVE is not supported")
    if wave != b"WAVE":
        raise UnsupportedFormatError("RIFF container is not WAVE")
    is_rf64 = _RIFF_HEADERS[riff]

    if file_size is None:
        current = stream.tell()
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(current)

    fmt: tuple[str, int, int, int, int] | None = None
    data_offset: int | None = None
    data_bytes: int | None = None
    rf64_data_bytes: int | None = None
    position = 12
    while position + 8 <= file_size:
        stream.seek(position)
        chunk_header = stream.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id = chunk_header[:4]
        chunk_size = struct.unpack("<I", chunk_header[4:8])[0]
        body_start = position + 8
        if chunk_id == b"ds64" and is_rf64:
            body = stream.read(28)
            if len(body) >= 16:
                # ds64: riffSize(8), dataSize(8), sampleCount(8), tableLength(4)
                rf64_data_bytes = struct.unpack("<Q", body[8:16])[0]
        elif chunk_id == b"fmt ":
            if fmt is None:
                fmt = _parse_fmt(stream.read(chunk_size))
        elif chunk_id == b"data":
            if data_offset is None:
                data_offset = body_start
                size = chunk_size
                declared = True
                if is_rf64 and size == 0xFFFFFFFF:
                    if rf64_data_bytes is None:
                        raise DecodeError("RF64 data chunk without a ds64 size")
                    size = rf64_data_bytes
                elif size == 0 or size == 0xFFFFFFFF:
                    # Streaming writers leave a zero/placeholder size.
                    size = file_size - body_start
                    declared = False
                if declared and size > file_size - body_start:
                    # Keep the declared size: reading past the truncation must
                    # surface as a decode error, not silently shorter audio.
                    data_bytes = size
                else:
                    data_bytes = min(size, file_size - body_start)
        if fmt is not None and data_offset is not None:
            break
        # Chunks are word aligned; odd sizes carry one pad byte.
        position = body_start + chunk_size + (chunk_size & 1)

    if fmt is None:
        raise UnsupportedFormatError("WAVE has no fmt chunk")
    if data_offset is None or data_bytes is None:
        raise UnsupportedFormatError("WAVE has no data chunk")
    sample_format, rate, channels, bits, block_align = fmt
    return WavInfo(
        sample_rate_hz=rate,
        channel_count=channels,
        frames=data_bytes // block_align,
        bit_depth=bits,
        sample_format=sample_format,
        block_align=block_align,
        data_offset=data_offset,
        data_bytes=data_bytes,
    )


def read_wav_info(path: Path) -> WavInfo:
    with path.open("rb") as stream:
        return parse_wav(stream)
