"""Cover art for one audio file, ported from MediaEye.

Priority is the one MediaEye arrived at and the one that matches a packaged
library's actual layout:

1. a folder image *named* as a cover (``Artwork.png``, ``folder.jpg`` …)
2. a picture embedded in the file (ID3v2 in MP3 and in WAV's ``id3 `` chunk,
   FLAC ``PICTURE``)
3. any other image nearby

The directory walk is the valuable part: same folder, its asset-named subfolders,
then three levels up with decaying weights, scanning only asset-named subfolders
at each ancestor. Without that last restriction, previewing a file in ``PackA``
picks up ``PackB``'s cover from the shared parent.

Resolution is cached per *directory*: a pack's several hundred sounds share one
cover, and a scrolling list must not rescan for each row.
"""

from __future__ import annotations

import base64
import os
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# Stems that read as "this file is the cover".
COVER_KEYWORDS = ("folder", "cover", "front", "album", "artwork", "art", "jacket", "poster")
# Folder names that say "art and documents live here" in a packaged library.
ASSET_FOLDER_NAMES = {
    "artwork",
    "art",
    "documents",
    "document",
    "docs",
    "doc",
    "images",
    "pictures",
    "pics",
    "covers",
    "scans",
}
# Ancestor weights: closer is better, and an asset folder beats loose files.
ANCESTOR_WEIGHTS = ((0.60, 0.85), (0.40, 0.70), (0.25, 0.55))

_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

_MAX_ID3_CHUNK_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CoverImage:
    data: bytes
    media_type: str
    source: str  # folder-named | embedded | folder-nearby


def _score_name(stem: str, audio_stem: str) -> int:
    """How much this filename claims to be the cover (MediaEye's scale)."""
    if stem in COVER_KEYWORDS:
        return 4
    if any(keyword in stem for keyword in COVER_KEYWORDS):
        return 3
    if audio_stem and stem == audio_stem:
        return 2
    return 1


def _image_files(directory: Path) -> list[Path]:
    try:
        return [
            entry
            for entry in directory.iterdir()
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        ]
    except OSError:
        return []


def _subdirectories(directory: Path) -> list[Path]:
    try:
        return [entry for entry in directory.iterdir() if entry.is_dir()]
    except OSError:
        return []


def find_folder_image(audio_path: Path) -> tuple[Path | None, bool]:
    """Best nearby image and whether its name claims to be the cover.

    Returns ``(path, named)``; ``named`` is what decides whether this beats an
    embedded picture or only serves as the fallback.
    """
    audio_dir = audio_path.parent
    audio_stem = audio_path.stem.lower()
    best_path: Path | None = None
    best_score = 0.0
    best_named = False

    def scan(directory: Path, weight: float) -> None:
        nonlocal best_path, best_score, best_named
        for image in _image_files(directory):
            raw = _score_name(image.stem.lower(), audio_stem)
            score = raw * weight
            if score > best_score:
                best_score = score
                best_path = image
                best_named = raw >= 3

    scan(audio_dir, 1.0)
    for sub in _subdirectories(audio_dir):
        scan(sub, 0.9 if sub.name.lower() in ASSET_FOLDER_NAMES else 0.7)

    current = audio_dir
    for dir_weight, asset_weight in ANCESTOR_WEIGHTS:
        parent = current.parent
        if parent == current:
            break
        scan(parent, dir_weight)
        # Only asset-named folders up here: a loose image in a sibling pack is
        # another pack's cover, not this one's.
        for sub in _subdirectories(parent):
            if sub.name.lower() in ASSET_FOLDER_NAMES:
                scan(sub, asset_weight)
        current = parent

    return best_path, best_named


# ---------------------------------------------------------------- embedded art
def _de_unsync(buf: bytes) -> bytes:
    return buf.replace(b"\xff\x00", b"\xff")


def _sniff_image_mime(data: bytes, declared: str = "") -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    clean = declared.strip().lower()
    if clean in {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/bmp", "image/gif"}:
        return "image/jpeg" if clean == "image/jpg" else clean
    return clean or None


def _picture_from_id3(data: bytes) -> tuple[bytes, str] | None:
    """First APIC/PIC frame of an ID3v2 tag at the start of ``data``."""
    if len(data) < 10 or data[:3] != b"ID3":
        return None
    major = data[3]
    if major < 2 or major > 4:
        return None
    flags = data[5]
    size = (data[6] & 0x7F) << 21 | (data[7] & 0x7F) << 14
    size |= (data[8] & 0x7F) << 7 | (data[9] & 0x7F)
    body = data[10 : 10 + size]
    if major == 3 and (flags & 0x80):
        body = _de_unsync(body)
    position = 0
    if flags & 0x40:  # extended header
        if len(body) >= 4:
            ext = (
                int.from_bytes(body[:4], "big")
                if major <= 3
                else (body[0] & 0x7F) << 21 | (body[1] & 0x7F) << 14
                | (body[2] & 0x7F) << 7 | (body[3] & 0x7F)
            )
            position = ext + (4 if major <= 3 else 0)
    frame_header = 6 if major == 2 else 10
    id_length = 3 if major == 2 else 4
    while position + frame_header <= len(body):
        frame_id = body[position : position + id_length]
        if not frame_id.strip(b"\x00"):
            break
        if major == 2:
            frame_size = int.from_bytes(body[position + 3 : position + 6], "big")
        elif major == 3:
            frame_size = int.from_bytes(body[position + 4 : position + 8], "big")
        else:
            raw = body[position + 4 : position + 8]
            frame_size = (raw[0] & 0x7F) << 21 | (raw[1] & 0x7F) << 14
            frame_size |= (raw[2] & 0x7F) << 7 | (raw[3] & 0x7F)
        if frame_size <= 0 or position + frame_header + frame_size > len(body):
            break
        frame = body[position + frame_header : position + frame_header + frame_size]
        if major == 4:
            fflags = body[position + 9]
            if fflags & 0x02:
                frame = _de_unsync(frame)
            if (fflags & 0x01) and len(frame) >= 4:
                frame = frame[4:]
        if frame_id in {b"APIC", b"PIC"}:
            return _parse_apic(frame, major)
        position += frame_header + frame_size
    return None


def _parse_apic(frame: bytes, major: int) -> tuple[bytes, str] | None:
    if len(frame) < 4:
        return None
    encoding = frame[0]
    declared_mime = ""
    if major == 2:
        fmt = frame[1:4]
        declared_mime = {b"JPG": "image/jpeg", b"PNG": "image/png"}.get(fmt, "")
        cursor = 4
    else:
        end = frame.find(b"\x00", 1)
        if end < 0:
            return None
        declared_mime = frame[1:end].decode("latin-1").lower()
        cursor = end + 1
    if cursor >= len(frame):
        return None
    cursor += 1  # picture type
    if encoding in (1, 2):  # UTF-16 description, double-null
        while cursor + 1 < len(frame) and not (frame[cursor] == 0 and frame[cursor + 1] == 0):
            cursor += 2
        cursor += 2
    else:
        end = frame.find(b"\x00", cursor)
        if end < 0:
            return None
        cursor = end + 1
    if cursor >= len(frame):
        return None
    pic_bytes = frame[cursor:]
    mime = _sniff_image_mime(pic_bytes, declared_mime)
    if mime is None:
        return None
    return pic_bytes, mime


def _parse_flac_picture_block(block: bytes) -> tuple[bytes, str] | None:
    try:
        offset = 4  # picture type
        mime_len = int.from_bytes(block[offset : offset + 4], "big")
        offset += 4
        mime = block[offset : offset + mime_len].decode("latin-1").lower()
        offset += mime_len
        desc_len = int.from_bytes(block[offset : offset + 4], "big")
        offset += 4 + desc_len
        offset += 16  # width, height, depth, colours
        data_len = int.from_bytes(block[offset : offset + 4], "big")
        offset += 4
        if offset + data_len > len(block):
            return None
        return block[offset : offset + data_len], mime
    except (IndexError, ValueError):
        return None


def _picture_from_flac(data: bytes) -> tuple[bytes, str] | None:
    if len(data) < 4 or data[:4] != b"fLaC":
        return None
    position = 4
    while position + 4 <= len(data):
        header = data[position]
        block_type = header & 0x7F
        length = int.from_bytes(data[position + 1 : position + 4], "big")
        start = position + 4
        end = start + length
        if end > len(data):
            return None
        if block_type == 6:  # PICTURE
            return _parse_flac_picture_block(data[start:end])
        if header & 0x80:  # last block
            return None
        position = end
    return None


def _wav_chunk(data: bytes, wanted: bytes) -> bytes | None:
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RF64") or data[8:12] != b"WAVE":
        return None
    wanted_low = wanted.lower()
    position = 12
    while position + 8 <= len(data):
        chunk_id = data[position : position + 4]
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        body = data[position + 8 : position + 8 + size]
        if chunk_id == wanted or chunk_id.lower() == wanted_low:
            return body
        position += 8 + size + (size & 1)
    return None


def _read_wav_id3_chunk(handle: BinaryIO) -> bytes | None:
    """Stream-walk RIFF/RF64 WAVE chunks by seeking past PCM payloads for ``id3 ``/``ID3 ``."""
    handle.seek(0)
    header = handle.read(12)
    if len(header) < 12 or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
        return None
    is_rf64 = header[:4] == b"RF64"
    ds64_data_size: int | None = None
    while True:
        chunk_hdr = handle.read(8)
        if len(chunk_hdr) < 8:
            return None
        chunk_id = chunk_hdr[:4]
        raw_size = int.from_bytes(chunk_hdr[4:8], "little")
        if is_rf64 and chunk_id == b"ds64" and 24 <= raw_size <= _MAX_ID3_CHUNK_BYTES:
            ds64 = handle.read(raw_size)
            if len(ds64) < raw_size:
                return None
            if len(ds64) >= 16:
                ds64_data_size = int.from_bytes(ds64[8:16], "little")
            if raw_size & 1:
                handle.seek(1, os.SEEK_CUR)
            continue

        actual_size = raw_size
        if is_rf64 and chunk_id == b"data" and raw_size == 0xFFFFFFFF and ds64_data_size is not None:
            actual_size = ds64_data_size

        if chunk_id.lower() == b"id3 ":
            if actual_size <= 0 or actual_size > _MAX_ID3_CHUNK_BYTES:
                return None
            return handle.read(actual_size)

        handle.seek(actual_size + (actual_size & 1), os.SEEK_CUR)


def _read_aiff_id3_chunk(handle: BinaryIO) -> bytes | None:
    """Stream-walk FORM AIFF/AIFC chunks for an ``ID3 ``/``id3 `` chunk."""
    handle.seek(0)
    header = handle.read(12)
    if len(header) < 12 or header[:4] != b"FORM" or header[8:12] not in (b"AIFF", b"AIFC"):
        return None
    while True:
        chunk_hdr = handle.read(8)
        if len(chunk_hdr) < 8:
            return None
        chunk_id = chunk_hdr[:4]
        size = int.from_bytes(chunk_hdr[4:8], "big")
        if chunk_id.lower() == b"id3 ":
            if size <= 0 or size > _MAX_ID3_CHUNK_BYTES:
                return None
            return handle.read(size)
        handle.seek(size + (size & 1), os.SEEK_CUR)


def _read_id3_header_and_body(handle: BinaryIO) -> bytes | None:
    handle.seek(0)
    head = handle.read(10)
    if len(head) < 10 or head[:3] != b"ID3":
        return None
    size = (head[6] & 0x7F) << 21 | (head[7] & 0x7F) << 14 | (head[8] & 0x7F) << 7 | (head[9] & 0x7F)
    if size <= 0 or size > _MAX_ID3_CHUNK_BYTES:
        return None
    return head + handle.read(size)


def _picture_via_mutagen(audio_path: Path) -> tuple[bytes, str] | None:
    """Best-effort cover extraction via mutagen for M4A, OGG/Opus, or flagged ID3 APIC frames."""
    try:
        import mutagen

        mf = mutagen.File(str(audio_path))
        if mf is None or not getattr(mf, "tags", None):
            return None
        tags = mf.tags
        # ID3 (MP3 / WAV / AIFF)
        if hasattr(tags, "getall"):
            apics = tags.getall("APIC")
            if apics:
                frame = apics[0]
                data = bytes(getattr(frame, "data", b"") or b"")
                mime = str(getattr(frame, "mime", "") or "image/jpeg").lower()
                if data:
                    return data, mime
        # MP4 / M4A covr atom
        if "covr" in tags and tags["covr"]:
            raw = bytes(tags["covr"][0])
            if raw:
                mime = "image/png" if raw.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
                return raw, mime
        # Vorbis / Opus metadata_block_picture
        if "metadata_block_picture" in tags and tags["metadata_block_picture"]:
            b64_val = str(tags["metadata_block_picture"][0])
            block = base64.b64decode(b64_val)
            return _parse_flac_picture_block(block)
    except Exception:
        return None
    return None


def embedded_picture(audio_path: Path) -> tuple[bytes, str] | None:
    """The file's own artwork, if it carries any."""
    try:
        with audio_path.open("rb") as handle:
            head = handle.read(12)
            if len(head) < 4:
                return None
            if head[:4] == b"fLaC":
                handle.seek(0)
                return _picture_from_flac(handle.read())
            if head[:3] == b"ID3":
                tag = _read_id3_header_and_body(handle)
                if tag:
                    pic = _picture_from_id3(tag)
                    if pic is not None:
                        return pic
                    return _picture_via_mutagen(audio_path)
                return None
            if head[:4] in (b"RIFF", b"RF64") and len(head) >= 12 and head[8:12] == b"WAVE":
                tag = _read_wav_id3_chunk(handle)
                if tag:
                    pic = _picture_from_id3(tag)
                    if pic is not None:
                        return pic
                    return _picture_via_mutagen(audio_path)
                return None
            if head[:4] == b"FORM" and len(head) >= 12 and head[8:12] in (b"AIFF", b"AIFC"):
                tag = _read_aiff_id3_chunk(handle)
                if tag:
                    pic = _picture_from_id3(tag)
                    if pic is not None:
                        return pic
                    return _picture_via_mutagen(audio_path)
                return None
    except OSError:
        return None
    if audio_path.suffix.lower() in {".m4a", ".mp4", ".ogg", ".opus"}:
        return _picture_via_mutagen(audio_path)
    return None


class CoverResolver:
    """Resolve covers, caching the directory scan per folder."""

    def __init__(self, *, cache_limit: int = 512) -> None:
        self._cache: OrderedDict[Path, tuple[Path | None, bool]] = OrderedDict()
        self._limit = cache_limit

    def _folder_image(self, directory: Path, audio_path: Path) -> tuple[Path | None, bool]:
        hit = self._cache.get(directory)
        if hit is not None:
            self._cache.move_to_end(directory)
            return hit
        result = find_folder_image(audio_path)
        self._cache[directory] = result
        while len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return result

    def resolve(self, audio_path: Path) -> CoverImage | None:
        folder_path, named = self._folder_image(audio_path.parent, audio_path)
        if folder_path is not None and named:
            return self._image_file(folder_path, "folder-named")
        embedded = embedded_picture(audio_path)
        if embedded is not None:
            data, media_type = embedded
            return CoverImage(data=data, media_type=media_type, source="embedded")
        if folder_path is not None:
            return self._image_file(folder_path, "folder-nearby")
        return None

    @staticmethod
    def _image_file(path: Path, source: str) -> CoverImage | None:
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return CoverImage(
            data=data,
            media_type=_MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream"),
            source=source,
        )


__all__ = [
    "CoverImage",
    "CoverResolver",
    "embedded_picture",
    "find_folder_image",
]
