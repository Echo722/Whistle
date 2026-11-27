"""Quadruple-safe explicit metadata writer & batch undo for WAV, MP3, FLAC, OGG, OPUS, and AIFF.

Safety guarantees:
1. Audio data never modified: pre-write & post-write SHA-256 over audio frames/chunks must match.
2. Unknown chunks & non-INFO LISTs (e.g., ``LIST/adtl``) are preserved byte-for-byte in WAV.
3. Same-directory temporary file + flush/fsync + verification + atomic ``os.replace``.
4. Pre-write backup into ``.local/metadata-backups/<batch_id>/`` + ``write_batches`` audit + undo.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import struct
import time
import uuid
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from audiosearch.catalog.embedded import (
    BEXT_FIXED_BYTES,
    EmbeddedStatus,
    read_embedded_metadata,
)
from audiosearch.library.metadata_fields import (
    EDITABLE_FIELD_KEYS,
    FIELD_SPEC_MAP,
    extract_canonical_from_raw,
)
from audiosearch.library.scanner import _read_non_wav_tags
from audiosearch.library.store import FileRecord, LibraryStore

SUPPORTED_WRITE_EXTS = frozenset({"wav", "mp3", "flac", "ogg", "opus", "aif", "aiff"})
_COPY_BUF_BYTES = 1024 * 1024
_RIFF_MAX_PAYLOAD = 0xFFFFFFF0

_USER_TAG_TO_FIELD: tuple[tuple[str, str], ...] = (
    ("DESCRIPTION", "description"),
    ("KEYWORDS", "keywords"),
    ("CATID", "cat_id"),
    ("CATEGORY", "category"),
    ("SUBCATEGORY", "subcategory"),
    ("FXNAME", "fx_name"),
    ("LIBRARY", "library"),
    ("DESIGNER", "designer"),
    ("NOTES", "notes"),
)
_FIELD_TO_USER_TAG: dict[str, str] = {field: tag for tag, field in _USER_TAG_TO_FIELD}

_VORBIS_ALL_MANAGED_KEYS: tuple[str, ...] = (
    "DESCRIPTION",
    "COMMENT",
    "KEYWORDS",
    "CATID",
    "CATEGORY",
    "SUBCATEGORY",
    "FXNAME",
    "TITLE",
    "LIBRARY",
    "ALBUM",
    "DESIGNER",
    "ARTIST",
    "GENRE",
    "NOTES",
)

_ID3_MANAGED_TXXX: tuple[str, ...] = (
    "KEYWORDS",
    "CATID",
    "CATEGORY",
    "SUBCATEGORY",
    "FXNAME",
    "LIBRARY",
    "DESIGNER",
    "NOTES",
    "DESCRIPTION",
)


class AudioIntegrityError(RuntimeError):
    """Raised when post-write audio bitstream hash differs from pre-write hash."""


@dataclass(frozen=True, slots=True)
class FieldDiff:
    field: str
    label: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class FileWritePreview:
    file_id: int
    path: Path
    filename: str
    ext: str
    diffs: tuple[FieldDiff, ...]
    target_values: dict[str, str]
    audio_sha256: str


@dataclass(frozen=True, slots=True)
class SkippedWriteItem:
    file_id: int
    filename: str
    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class WriteBackPreview:
    items: tuple[FileWritePreview, ...]
    skipped: tuple[SkippedWriteItem, ...]


@dataclass(frozen=True, slots=True)
class WriteFileOutcome:
    file_id: int
    filename: str
    path: str
    success: bool
    message: str
    audio_sha256_before: str = ""
    audio_sha256_after: str = ""


@dataclass(frozen=True, slots=True)
class WriteBatchResult:
    batch_id: str
    status: str
    backup_dir: str
    succeeded: tuple[WriteFileOutcome, ...]
    failed: tuple[WriteFileOutcome, ...]
    skipped: tuple[SkippedWriteItem, ...]


@dataclass(frozen=True, slots=True)
class UndoBatchResult:
    batch_id: str
    status: str
    restored: tuple[WriteFileOutcome, ...]
    failed: tuple[WriteFileOutcome, ...]


# --------------------------------------------------------------------------- hashing
def _stream_hash(stream: BinaryIO, start: int, length: int) -> str:
    h = hashlib.sha256()
    stream.seek(start)
    remaining = length
    while remaining > 0:
        chunk = stream.read(min(_COPY_BUF_BYTES, remaining))
        if not chunk:
            raise ValueError("音频流意外截断，无法完成完整性校验")
        h.update(chunk)
        remaining -= len(chunk)
    return h.hexdigest()


def hash_wav_data_chunk(path: Path) -> str:
    """Compute SHA-256 of the exact RIFF/WAVE ``data`` chunk payload."""
    size_total = path.stat().st_size
    with path.open("rb") as fh:
        hdr = fh.read(12)
        if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise ValueError(f"仅支持标准 RIFF/WAVE 文件写回与校验：{path.name}")
        pos = 12
        while pos + 8 <= size_total:
            fh.seek(pos)
            ch = fh.read(8)
            if len(ch) < 8:
                break
            cid = ch[:4]
            csize = struct.unpack("<I", ch[4:8])[0]
            body_start = pos + 8
            if cid == b"data":
                if csize == 0xFFFFFFFF or body_start + csize > size_total:
                    raise ValueError("WAV data chunk 长度无效或被截断")
                return _stream_hash(fh, body_start, csize)
            pos = body_start + csize + (csize & 1)
    raise ValueError(f"未找到 WAV data chunk：{path.name}")


def hash_aiff_ssnd_chunk(path: Path) -> str:
    """Compute SHA-256 of the exact FORM/AIFF ``SSND`` chunk payload."""
    size_total = path.stat().st_size
    with path.open("rb") as fh:
        hdr = fh.read(12)
        if len(hdr) < 12 or hdr[:4] != b"FORM" or hdr[8:12] not in (b"AIFF", b"AIFC"):
            raise ValueError(f"仅支持标准 FORM/AIFF 文件：{path.name}")
        pos = 12
        while pos + 8 <= size_total:
            fh.seek(pos)
            ch = fh.read(8)
            if len(ch) < 8:
                break
            cid = ch[:4]
            csize = struct.unpack(">I", ch[4:8])[0]
            body_start = pos + 8
            if cid == b"SSND":
                if body_start + csize > size_total:
                    raise ValueError("AIFF SSND chunk 长度无效或被截断")
                return _stream_hash(fh, body_start, csize)
            pos = body_start + csize + (csize & 1)
    raise ValueError(f"未找到 AIFF SSND chunk：{path.name}")


def hash_flac_audio_frames(path: Path) -> str:
    """Compute SHA-256 of all FLAC audio frames after the final metadata block."""
    size_total = path.stat().st_size
    with path.open("rb") as fh:
        if fh.read(4) != b"fLaC":
            raise ValueError(f"不是有效的 FLAC 文件：{path.name}")
        pos = 4
        while pos + 4 <= size_total:
            fh.seek(pos)
            bh = fh.read(4)
            if len(bh) < 4:
                break
            is_last = bool(bh[0] & 0x80)
            block_len = int.from_bytes(bh[1:4], "big")
            pos += 4 + block_len
            if pos > size_total:
                raise ValueError("FLAC metadata block 超出文件末尾")
            if is_last:
                return _stream_hash(fh, pos, size_total - pos)
    raise ValueError(f"FLAC 元数据块结构不完整：{path.name}")


def hash_mp3_audio_frames(path: Path) -> str:
    """Compute SHA-256 of MPEG audio bitstream, ignoring ID3v2/ID3v1/APEv2 tags."""
    size_total = path.stat().st_size
    with path.open("rb") as fh:
        start = 0
        while start + 10 <= size_total:
            fh.seek(start)
            hdr = fh.read(10)
            if len(hdr) < 10 or hdr[:3] != b"ID3":
                break
            flags = hdr[5]
            raw_sz = hdr[6:10]
            if any(b & 0x80 for b in raw_sz):
                break
            tag_sz = (raw_sz[0] << 21) | (raw_sz[1] << 14) | (raw_sz[2] << 7) | raw_sz[3]
            step = 10 + tag_sz + (10 if (flags & 0x10) else 0)
            if start + step > size_total:
                break
            start += step

        end = size_total
        # Strip trailing ID3v1 (128 bytes) and/or APEv2 footer/tag if present.
        for _ in range(2):
            if end - start >= 128:
                fh.seek(end - 128)
                if fh.read(3) == b"TAG":
                    end -= 128
                    continue
            if end - start >= 32:
                fh.seek(end - 32)
                ape = fh.read(32)
                if len(ape) == 32 and ape[:8] == b"APETAGEX":
                    ape_size = struct.unpack("<I", ape[12:16])[0]
                    ape_flags = struct.unpack("<I", ape[20:24])[0]
                    has_header = bool(ape_flags & (1 << 31))
                    total_ape = ape_size + (32 if has_header else 0)
                    if total_ape <= end - start:
                        end -= total_ape
                        continue
            break

        if end <= start:
            raise ValueError(f"未找到有效 MP3 音频帧：{path.name}")
        return _stream_hash(fh, start, end - start)


def hash_ogg_audio_packets(path: Path) -> str:
    """Compute SHA-256 over Ogg stream packets excluding the Vorbis/Opus comment header (packet #1)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        data = fh.read()
    pos = 0
    packet_idx = 0
    packet_buf = bytearray()
    n = len(data)
    found_page = False
    while pos + 27 <= n:
        if data[pos : pos + 4] != b"OggS":
            raise ValueError(f"不是有效的 Ogg 流：{path.name}")
        found_page = True
        n_segs = data[pos + 26]
        seg_table_start = pos + 27
        seg_table_end = seg_table_start + n_segs
        if seg_table_end > n:
            raise ValueError("Ogg 页段表超出文件末尾")
        seg_table = data[seg_table_start:seg_table_end]
        payload_start = seg_table_end
        payload_len = sum(seg_table)
        payload_end = payload_start + payload_len
        if payload_end > n:
            raise ValueError("Ogg 页载荷超出文件末尾")

        offset = payload_start
        for seg_len in seg_table:
            packet_buf.extend(data[offset : offset + seg_len])
            offset += seg_len
            if seg_len < 255:
                # Packet complete
                if packet_idx != 1:
                    h.update(struct.pack("<I", len(packet_buf)))
                    h.update(packet_buf)
                packet_idx += 1
                packet_buf.clear()
        pos = payload_end
    if not found_page:
        raise ValueError(f"未找到 Ogg 页面：{path.name}")
    if packet_buf:
        if packet_idx != 1:
            h.update(struct.pack("<I", len(packet_buf)))
            h.update(packet_buf)
    return h.hexdigest()


def compute_audio_hash(path: Path, ext: str) -> str:
    ext_low = ext.lower().lstrip(".")
    if ext_low == "wav":
        return hash_wav_data_chunk(path)
    if ext_low == "flac":
        return hash_flac_audio_frames(path)
    if ext_low == "mp3":
        return hash_mp3_audio_frames(path)
    if ext_low in ("ogg", "opus"):
        return hash_ogg_audio_packets(path)
    if ext_low in ("aif", "aiff"):
        return hash_aiff_ssnd_chunk(path)
    raise ValueError(f"不支持写回的文件格式：.{ext_low}")


# --------------------------------------------------------------------------- WAV chunk helpers
def _utf8_prefix(text: str, max_bytes: int) -> bytes:
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_bytes:
        return raw
    cut = raw[:max_bytes]
    while cut:
        try:
            cut.decode("utf-8")
            return cut
        except UnicodeDecodeError:
            cut = cut[:-1]
    return b""


def _pad_chunk_slot(chunk_id: bytes, body: bytes) -> bytes:
    pad = b"\x00" if (len(body) & 1) else b""
    return chunk_id + struct.pack("<I", len(body)) + body + pad


def _walk_riff_chunks(path: Path) -> list[tuple[bytes, int, int]]:
    file_size = path.stat().st_size
    with path.open("rb") as fh:
        hdr = fh.read(12)
        if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
            raise ValueError(f"仅支持标准 RIFF/WAVE 文件写回：{path.name}")
        chunks: list[tuple[bytes, int, int]] = []
        pos = 12
        while pos + 8 <= file_size:
            fh.seek(pos)
            ch = fh.read(8)
            if len(ch) < 8:
                break
            cid = ch[:4]
            csize = struct.unpack("<I", ch[4:8])[0]
            body_start = pos + 8
            if body_start + csize > file_size:
                raise ValueError(f"WAV chunk {_safe_cid(cid)} 长度超出文件末尾")
            chunks.append((cid, body_start, csize))
            pos = body_start + csize + (csize & 1)
    return chunks


def _safe_cid(cid: bytes) -> str:
    return cid.decode("ascii", errors="replace")


def _build_bext_body(existing_body: bytes | None, canonical: dict[str, str]) -> bytes:
    if existing_body is not None and len(existing_body) >= BEXT_FIXED_BYTES:
        fixed = bytearray(existing_body[:BEXT_FIXED_BYTES])
        coding_history = existing_body[BEXT_FIXED_BYTES:]
    else:
        fixed = bytearray(BEXT_FIXED_BYTES)
        struct.pack_into("<H", fixed, 346, 1)  # BWF v1
        coding_history = b""

    desc = canonical.get("description", "").strip() or canonical.get("fx_name", "").strip()
    orig = canonical.get("designer", "").strip() or canonical.get("library", "").strip()
    desc_bytes = _utf8_prefix(desc, 256)
    orig_bytes = _utf8_prefix(orig, 32)
    fixed[0:256] = desc_bytes.ljust(256, b"\x00")
    fixed[256:288] = orig_bytes.ljust(32, b"\x00")
    if coding_history and not coding_history.endswith(b"\x00"):
        coding_history += b"\x00"
    return bytes(fixed) + coding_history


def _parse_existing_list_info(body: bytes | None) -> dict[bytes, bytes]:
    out: dict[bytes, bytes] = {}
    if not body or len(body) < 4 or body[:4] != b"INFO":
        return out
    pos = 4
    while pos + 8 <= len(body):
        tag = body[pos : pos + 4]
        sz = struct.unpack("<I", body[pos + 4 : pos + 8])[0]
        start = pos + 8
        end = min(len(body), start + sz)
        out[tag] = body[start:end]
        pos = start + sz + (sz & 1)
    return out


def _build_list_info_body(existing_body: bytes | None, canonical: dict[str, str]) -> bytes | None:
    subtags = _parse_existing_list_info(existing_body)
    mapping: tuple[tuple[bytes, str], ...] = (
        (b"INAM", canonical.get("fx_name", "").strip()),
        (b"IART", canonical.get("designer", "").strip()),
        (b"ICMT", canonical.get("description", "").strip()),
        (b"IGNR", canonical.get("category", "").strip()),
        (b"ISBJ", canonical.get("subcategory", "").strip()),
        (b"IKEY", canonical.get("keywords", "").strip()),
        (b"ISRC", canonical.get("library", "").strip()),
    )
    for tag, val in mapping:
        if val:
            subtags[tag] = val.encode("utf-8") + b"\x00"
        else:
            subtags.pop(tag, None)

    if not subtags:
        return None
    parts = [b"INFO"]
    for tag, val_bytes in subtags.items():
        parts.append(_pad_chunk_slot(tag, val_bytes))
    return b"".join(parts)


def _find_child(parent: ElementTree.Element, name_upper: str) -> ElementTree.Element | None:
    for child in parent:
        if child.tag.rsplit("}", 1)[-1].upper() == name_upper:
            return child
    return None


def _set_or_remove_xml_child(parent: ElementTree.Element, tag_name: str, value: str) -> None:
    existing = _find_child(parent, tag_name.upper())
    cleaned = value.strip()
    if cleaned:
        if existing is None:
            existing = ElementTree.SubElement(parent, tag_name)
        existing.text = cleaned
    elif existing is not None:
        parent.remove(existing)


def _build_ixml_body(existing_body: bytes | None, canonical: dict[str, str]) -> bytes | None:
    root: ElementTree.Element | None = None
    if existing_body:
        raw = existing_body[3:] if existing_body[:3] == b"\xef\xbb\xbf" else existing_body
        try:
            text = raw.decode("utf-8")
            parsed = ElementTree.fromstring(text)
            if parsed.tag.rsplit("}", 1)[-1].upper() == "BWFXML":
                root = parsed
        except Exception:
            root = None

    if root is None:
        root = ElementTree.Element("BWFXML")
        ver = ElementTree.SubElement(root, "IXML_VERSION")
        ver.text = "1.50"

    _set_or_remove_xml_child(root, "PROJECT", canonical.get("library", ""))
    _set_or_remove_xml_child(root, "NOTE", canonical.get("description", ""))

    user_el = _find_child(root, "USER")
    if user_el is None:
        user_el = ElementTree.Element("USER")
        root.append(user_el)

    for xml_tag, field_key in _USER_TAG_TO_FIELD:
        _set_or_remove_xml_child(user_el, xml_tag, canonical.get(field_key, ""))

    if len(user_el) == 0:
        root.remove(user_el)

    has_any_content = any(
        child.tag.rsplit("}", 1)[-1].upper() != "IXML_VERSION" for child in root
    )
    if not has_any_content:
        return None
    xml_bytes = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml_bytes


def _write_wav_chunks_atomic(
    path: Path,
    *,
    new_bext: bytes | None,
    new_info: bytes | None,
    new_ixml: bytes | None,
    drop_absent_managed: bool = False,
) -> None:
    """Rebuild RIFF/WAVE file atomically preserving all unknown chunks and exact ``data`` stream."""
    pre_hash = hash_wav_data_chunk(path)
    chunks = _walk_riff_chunks(path)
    tmp_path = path.with_name(f".audiosearch_tmp_{uuid.uuid4().hex[:10]}_{path.name}")

    # Compute chunk plan before streaming so RIFF total size is exact upfront.
    out_items: list[tuple[str, bytes, int, int, bytes | None]] = []
    # item: (mode, chunk_id, src_offset, src_size, replacement_body)
    placed_bext = new_bext is None and not drop_absent_managed
    placed_info = new_info is None and not drop_absent_managed
    placed_ixml = new_ixml is None and not drop_absent_managed

    with path.open("rb") as src:
        for cid, offset, size in chunks:
            if cid == b"bext":
                if not placed_bext and new_bext is not None:
                    out_items.append(("mem", b"bext", 0, 0, new_bext))
                placed_bext = True
                continue
            if cid == b"iXML":
                if not placed_ixml and new_ixml is not None:
                    out_items.append(("mem", b"iXML", 0, 0, new_ixml))
                placed_ixml = True
                continue
            if cid == b"LIST":
                src.seek(offset)
                list_type = src.read(4) if size >= 4 else b""
                if list_type == b"INFO":
                    if not placed_info and new_info is not None:
                        out_items.append(("mem", b"LIST", 0, 0, new_info))
                    placed_info = True
                    continue
            if cid == b"data":
                # Ensure any newly added metadata chunks are emitted before ``data`` if absent.
                if not placed_bext and new_bext is not None:
                    out_items.append(("mem", b"bext", 0, 0, new_bext))
                placed_bext = True
                if not placed_info and new_info is not None:
                    out_items.append(("mem", b"LIST", 0, 0, new_info))
                placed_info = True
                if not placed_ixml and new_ixml is not None:
                    out_items.append(("mem", b"iXML", 0, 0, new_ixml))
                placed_ixml = True
            out_items.append(("copy", cid, offset, size, None))

        if not placed_bext and new_bext is not None:
            out_items.append(("mem", b"bext", 0, 0, new_bext))
        if not placed_info and new_info is not None:
            out_items.append(("mem", b"LIST", 0, 0, new_info))
        if not placed_ixml and new_ixml is not None:
            out_items.append(("mem", b"iXML", 0, 0, new_ixml))

        riff_payload = 4  # b"WAVE"
        for mode, _cid, _off, csize, body in out_items:
            sz = len(body) if mode == "mem" and body is not None else csize
            riff_payload += 8 + sz + (sz & 1)
        if riff_payload > _RIFF_MAX_PAYLOAD:
            raise ValueError("写入后 RIFF 体积超过 4GB 上限（需 RF64），已中止写回")

        try:
            with tmp_path.open("wb") as dst:
                dst.write(b"RIFF")
                dst.write(struct.pack("<I", riff_payload))
                dst.write(b"WAVE")
                for mode, cid, offset, csize, body in out_items:
                    if mode == "mem":
                        assert body is not None
                        dst.write(_pad_chunk_slot(cid, body))
                    else:
                        dst.write(cid)
                        dst.write(struct.pack("<I", csize))
                        src.seek(offset)
                        rem = csize
                        while rem > 0:
                            buf = src.read(min(_COPY_BUF_BYTES, rem))
                            if not buf:
                                raise ValueError("复制原始 chunk 时文件意外结束")
                            dst.write(buf)
                            rem -= len(buf)
                        if csize & 1:
                            dst.write(b"\x00")
                dst.flush()
                os.fsync(dst.fileno())
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    try:
        post_hash = hash_wav_data_chunk(tmp_path)
        if post_hash != pre_hash:
            raise AudioIntegrityError(
                f"WAV 音频数据块写前后哈希不一致（{pre_hash[:12]} != {post_hash[:12]}），已中止并回滚"
            )
        verify_meta = read_embedded_metadata(tmp_path)
        if verify_meta.status in (EmbeddedStatus.MALFORMED, EmbeddedStatus.FAILED):
            raise ValueError(f"写回后 WAV 元数据自检未通过：{'; '.join(verify_meta.notes)}")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_wav_managed_chunks(path: Path) -> tuple[bytes | None, bytes | None, bytes | None]:
    chunks = _walk_riff_chunks(path)
    bext_body: bytes | None = None
    info_body: bytes | None = None
    ixml_body: bytes | None = None
    with path.open("rb") as fh:
        for cid, off, sz in chunks:
            if cid == b"bext" and bext_body is None:
                fh.seek(off)
                bext_body = fh.read(sz)
            elif cid == b"iXML" and ixml_body is None:
                fh.seek(off)
                ixml_body = fh.read(sz)
            elif cid == b"LIST" and info_body is None and sz >= 4:
                fh.seek(off)
                lt = fh.read(4)
                if lt == b"INFO":
                    fh.seek(off)
                    info_body = fh.read(sz)
    return bext_body, info_body, ixml_body


# --------------------------------------------------------------------------- mutagen writers
def _serialize_mutagen_backup(path: Path, ext: str) -> dict[str, Any]:
    import mutagen
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis

    ext_low = ext.lower().lstrip(".")
    if ext_low == "mp3":
        try:
            id3 = ID3(str(path))
        except ID3NoHeaderError:
            return {"kind": "id3", "had_tags": False, "frames": []}
        frames_b64: list[dict[str, Any]] = []
        for frame in id3.values():
            fid = str(getattr(frame, "FrameID", ""))
            if fid == "TXXX":
                frames_b64.append(
                    {"fid": "TXXX", "desc": str(frame.desc), "text": [str(x) for x in frame.text], "encoding": int(frame.encoding)}
                )
            elif fid == "COMM":
                frames_b64.append(
                    {
                        "fid": "COMM",
                        "lang": str(frame.lang),
                        "desc": str(frame.desc),
                        "text": [str(x) for x in frame.text],
                        "encoding": int(frame.encoding),
                    }
                )
            elif fid in ("TIT2", "TPE1", "TALB", "TCON"):
                frames_b64.append(
                    {"fid": fid, "text": [str(x) for x in frame.text], "encoding": int(frame.encoding)}
                )
        return {"kind": "id3", "had_tags": True, "frames": frames_b64}

    if ext_low in ("aif", "aiff"):
        aiff = AIFF(str(path))
        if not aiff.tags:
            return {"kind": "id3", "had_tags": False, "frames": []}
        frames_b64 = []
        for frame in aiff.tags.values():
            fid = str(getattr(frame, "FrameID", ""))
            if fid == "TXXX":
                frames_b64.append(
                    {"fid": "TXXX", "desc": str(frame.desc), "text": [str(x) for x in frame.text], "encoding": int(frame.encoding)}
                )
            elif fid == "COMM":
                frames_b64.append(
                    {
                        "fid": "COMM",
                        "lang": str(frame.lang),
                        "desc": str(frame.desc),
                        "text": [str(x) for x in frame.text],
                        "encoding": int(frame.encoding),
                    }
                )
            elif fid in ("TIT2", "TPE1", "TALB", "TCON"):
                frames_b64.append(
                    {"fid": fid, "text": [str(x) for x in frame.text], "encoding": int(frame.encoding)}
                )
        return {"kind": "id3", "had_tags": True, "frames": frames_b64}

    cls_map = {"flac": FLAC, "ogg": OggVorbis, "opus": OggOpus}
    loader = cls_map.get(ext_low)
    audio = loader(str(path)) if loader else mutagen.File(str(path))
    if audio is None or not getattr(audio, "tags", None):
        return {"kind": "vorbis", "had_tags": False, "items": []}
    items: list[tuple[str, list[str]]] = []
    for k in audio.tags.keys():
        val = audio.tags.get(k)
        if isinstance(val, (list, tuple)):
            items.append((str(k), [str(x) for x in val]))
        elif val is not None:
            items.append((str(k), [str(val)]))
    return {"kind": "vorbis", "had_tags": True, "items": items}


def _apply_id3_canonical(tags: Any, canonical: dict[str, str]) -> None:
    from mutagen.id3 import COMM, TALB, TCON, TIT2, TPE1, TXXX

    # Remove managed TXXX tags
    for desc_key in _ID3_MANAGED_TXXX:
        tags.delall(f"TXXX:{desc_key}")
    tags.delall("COMM")
    tags.delall("TIT2")
    tags.delall("TPE1")
    tags.delall("TALB")
    tags.delall("TCON")

    desc = canonical.get("description", "").strip()
    if desc:
        tags.add(COMM(encoding=3, lang="eng", desc="", text=[desc]))
        tags.add(TXXX(encoding=3, desc="DESCRIPTION", text=[desc]))

    fx_name = canonical.get("fx_name", "").strip()
    if fx_name:
        tags.add(TIT2(encoding=3, text=[fx_name]))
        tags.add(TXXX(encoding=3, desc="FXNAME", text=[fx_name]))

    designer = canonical.get("designer", "").strip()
    if designer:
        tags.add(TPE1(encoding=3, text=[designer]))
        tags.add(TXXX(encoding=3, desc="DESIGNER", text=[designer]))

    library = canonical.get("library", "").strip()
    if library:
        tags.add(TALB(encoding=3, text=[library]))
        tags.add(TXXX(encoding=3, desc="LIBRARY", text=[library]))

    category = canonical.get("category", "").strip()
    if category:
        tags.add(TCON(encoding=3, text=[category]))
        tags.add(TXXX(encoding=3, desc="CATEGORY", text=[category]))

    for field_key, txxx_desc in (
        ("keywords", "KEYWORDS"),
        ("cat_id", "CATID"),
        ("subcategory", "SUBCATEGORY"),
        ("notes", "NOTES"),
    ):
        val = canonical.get(field_key, "").strip()
        if val:
            tags.add(TXXX(encoding=3, desc=txxx_desc, text=[val]))


def _restore_id3_from_backup(tags: Any, backup_data: dict[str, Any]) -> None:
    from mutagen.id3 import COMM, TALB, TCON, TIT2, TPE1, TXXX

    for desc_key in _ID3_MANAGED_TXXX:
        tags.delall(f"TXXX:{desc_key}")
    tags.delall("COMM")
    tags.delall("TIT2")
    tags.delall("TPE1")
    tags.delall("TALB")
    tags.delall("TCON")

    cls_map = {"TIT2": TIT2, "TPE1": TPE1, "TALB": TALB, "TCON": TCON}
    for fr in backup_data.get("frames", []):
        fid = fr["fid"]
        if fid == "TXXX":
            tags.add(TXXX(encoding=int(fr.get("encoding", 3)), desc=fr["desc"], text=fr["text"]))
        elif fid == "COMM":
            tags.add(
                COMM(
                    encoding=int(fr.get("encoding", 3)),
                    lang=fr.get("lang", "eng"),
                    desc=fr.get("desc", ""),
                    text=fr["text"],
                )
            )
        elif fid in cls_map:
            tags.add(cls_map[fid](encoding=int(fr.get("encoding", 3)), text=fr["text"]))


def _apply_vorbis_canonical(tags: Any, canonical: dict[str, str]) -> None:
    for k in _VORBIS_ALL_MANAGED_KEYS:
        if k in tags:
            del tags[k]

    pairs = (
        ("DESCRIPTION", canonical.get("description", "").strip()),
        ("COMMENT", canonical.get("description", "").strip()),
        ("KEYWORDS", canonical.get("keywords", "").strip()),
        ("CATID", canonical.get("cat_id", "").strip()),
        ("CATEGORY", canonical.get("category", "").strip()),
        ("GENRE", canonical.get("category", "").strip()),
        ("SUBCATEGORY", canonical.get("subcategory", "").strip()),
        ("FXNAME", canonical.get("fx_name", "").strip()),
        ("TITLE", canonical.get("fx_name", "").strip()),
        ("LIBRARY", canonical.get("library", "").strip()),
        ("ALBUM", canonical.get("library", "").strip()),
        ("DESIGNER", canonical.get("designer", "").strip()),
        ("ARTIST", canonical.get("designer", "").strip()),
        ("NOTES", canonical.get("notes", "").strip()),
    )
    for key, val in pairs:
        if val:
            tags[key] = [val]


def _write_mutagen_atomic(
    path: Path,
    ext: str,
    *,
    canonical: dict[str, str] | None = None,
    restore_backup: dict[str, Any] | None = None,
) -> None:
    from mutagen.aiff import AIFF
    from mutagen.flac import FLAC
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis

    ext_low = ext.lower().lstrip(".")
    pre_hash = compute_audio_hash(path, ext_low)
    tmp_path = path.with_name(f".audiosearch_tmp_{uuid.uuid4().hex[:10]}_{path.name}")
    shutil.copy2(path, tmp_path)

    try:
        if ext_low == "mp3":
            try:
                id3 = ID3(str(tmp_path))
            except ID3NoHeaderError:
                id3 = ID3()
            if restore_backup is not None:
                if not restore_backup.get("had_tags", False):
                    id3.delete(str(tmp_path))
                else:
                    _restore_id3_from_backup(id3, restore_backup)
                    id3.save(str(tmp_path), v2_version=3)
            else:
                assert canonical is not None
                _apply_id3_canonical(id3, canonical)
                id3.save(str(tmp_path), v2_version=3)
        elif ext_low in ("aif", "aiff"):
            aiff = AIFF(str(tmp_path))
            if restore_backup is not None:
                if not restore_backup.get("had_tags", False):
                    aiff.delete()
                else:
                    if aiff.tags is None:
                        aiff.add_tags()
                    _restore_id3_from_backup(aiff.tags, restore_backup)
                    aiff.save()
            else:
                assert canonical is not None
                if aiff.tags is None:
                    aiff.add_tags()
                _apply_id3_canonical(aiff.tags, canonical)
                aiff.save()
        else:
            cls_map = {"flac": FLAC, "ogg": OggVorbis, "opus": OggOpus}
            audio = cls_map[ext_low](str(tmp_path))
            if audio.tags is None:
                audio.add_tags()
            assert audio.tags is not None
            if restore_backup is not None:
                if not restore_backup.get("had_tags", False):
                    audio.delete()
                else:
                    audio.tags.clear()
                    for k, vals in restore_backup.get("items", []):
                        audio.tags[k] = list(vals)
                    audio.save()
            else:
                assert canonical is not None
                _apply_vorbis_canonical(audio.tags, canonical)
                audio.save()

        post_hash = compute_audio_hash(tmp_path, ext_low)
        if post_hash != pre_hash:
            raise AudioIntegrityError(
                f"{ext_low.upper()} 音频帧写前后哈希不一致（{pre_hash[:12]} != {post_hash[:12]}），已中止并回滚"
            )
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- sync helpers
def _read_disk_metadata_snapshot(
    path: Path, ext: str
) -> tuple[str, str, list[tuple[str, str]]]:
    """Read freshly written metadata from ``path`` for DB resync."""
    ext_low = ext.lower().lstrip(".")
    bext_dict: dict[str, str] = {}
    ixml_dict: dict[str, str] = {}
    emb_fields: list[tuple[str, str]] = []
    if ext_low == "wav":
        meta = read_embedded_metadata(path)
        for item in meta.values:
            emb_fields.append((item.source, item.value))
            if item.family == "bext":
                bext_dict[item.source] = item.value
            elif item.family == "ixml":
                ixml_dict[item.source] = item.value
    else:
        _d, _p, emb_fields = _read_non_wav_tags(path)
    return (
        json.dumps(bext_dict, ensure_ascii=False),
        json.dumps(ixml_dict, ensure_ascii=False),
        emb_fields,
    )


# --------------------------------------------------------------------------- public API
def _is_writable_file(path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    try:
        mode = path.stat().st_mode
        if not (mode & stat.S_IWRITE):
            return False
    except OSError:
        return False
    return os.access(path, os.W_OK | os.R_OK)


def build_write_back_preview(
    store: LibraryStore,
    file_ids: list[int] | None = None,
) -> WriteBackPreview:
    """Compute a Dry-run diff preview for all dirty files (or explicit ``file_ids``)."""
    target_ids = (
        list(dict.fromkeys(int(x) for x in file_ids))
        if file_ids is not None
        else store.dirty_file_ids()
    )
    if not target_ids:
        return WriteBackPreview(items=(), skipped=())

    recs = store.get_files_by_ids(target_ids)
    items: list[FileWritePreview] = []
    skipped: list[SkippedWriteItem] = []

    for fid in target_ids:
        rec = recs.get(fid)
        if rec is None:
            continue
        raw_rows = store.get_metadata_field_rows(fid)
        dirty_rows = [(f, v) for f, v, _src, d in raw_rows if d]
        if not dirty_rows:
            skipped.append(
                SkippedWriteItem(fid, rec.filename, str(rec.full_path), "无待写回修改")
            )
            continue

        ext_low = rec.ext.lower().lstrip(".")
        if ext_low not in SUPPORTED_WRITE_EXTS:
            skipped.append(
                SkippedWriteItem(fid, rec.filename, str(rec.full_path), f"暂不支持写回格式 .{ext_low}")
            )
            continue

        if not rec.is_available or not rec.full_path.exists():
            skipped.append(
                SkippedWriteItem(fid, rec.filename, str(rec.full_path), "文件离线或不存在")
            )
            continue

        if not _is_writable_file(rec.full_path):
            skipped.append(
                SkippedWriteItem(fid, rec.filename, str(rec.full_path), "文件为只读或无写入权限")
            )
            continue

        clean_rows = [(f, v, src, 0) for f, v, src, d in raw_rows if not d]
        embedded_before = extract_canonical_from_raw(rec, clean_rows)
        after_canon = extract_canonical_from_raw(rec, raw_rows)
        target_values = {k: after_canon[k][0] for k in EDITABLE_FIELD_KEYS}

        diffs: list[FieldDiff] = []
        for field_key, after_val in dirty_rows:
            if field_key not in FIELD_SPEC_MAP:
                continue
            spec = FIELD_SPEC_MAP[field_key]
            before_val = embedded_before.get(field_key, ("", "embedded", False))[0]
            if before_val != after_val or after_val:
                diffs.append(
                    FieldDiff(
                        field=field_key,
                        label=spec.label_zh,
                        before=before_val,
                        after=after_val,
                    )
                )

        try:
            audio_sha = compute_audio_hash(rec.full_path, ext_low)
        except Exception as exc:
            skipped.append(
                SkippedWriteItem(fid, rec.filename, str(rec.full_path), f"音频校验准备失败：{exc}")
            )
            continue

        items.append(
            FileWritePreview(
                file_id=fid,
                path=rec.full_path,
                filename=rec.filename,
                ext=ext_low,
                diffs=tuple(diffs),
                target_values=target_values,
                audio_sha256=audio_sha,
            )
        )

    return WriteBackPreview(items=tuple(items), skipped=tuple(skipped))


def execute_write_back(
    store: LibraryStore,
    file_ids: list[int] | None = None,
    *,
    backup_root: Path | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> WriteBatchResult:
    """Back up existing metadata, write atomically with pre/post audio SHA-256 checks, and log batch."""
    preview = build_write_back_preview(store, file_ids)
    b_root = backup_root or (store.db_path.parent / "metadata-backups")
    batch_id = time.strftime("batch_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    batch_dir = b_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    if not preview.items:
        return WriteBatchResult(
            batch_id=batch_id,
            status="empty",
            backup_dir=str(batch_dir),
            succeeded=(),
            failed=(),
            skipped=preview.skipped,
        )

    store.create_write_batch(
        batch_id,
        file_count=len(preview.items),
        backup_dir=str(batch_dir),
        status="pending",
    )

    manifest_entries: list[dict[str, Any]] = []
    succeeded: list[WriteFileOutcome] = []
    failed: list[WriteFileOutcome] = []
    total = len(preview.items)

    for idx, item in enumerate(preview.items, start=1):
        if on_progress is not None:
            on_progress(idx, total, item.filename)

        rec = store.get_file(item.file_id)
        if rec is None:
            failed.append(
                WriteFileOutcome(
                    item.file_id,
                    item.filename,
                    str(item.path),
                    False,
                    "库记录已不存在",
                )
            )
            continue

        raw_rows = store.get_metadata_field_rows(item.file_id)
        before_canon = {
            k: v
            for k, (v, _src, d) in extract_canonical_from_raw(
                rec, [(f, val, s, 0) for f, val, s, d in raw_rows if not d]
            ).items()
            if v
        }
        after_canon_for_undo = dict(item.target_values)

        try:
            if item.ext == "wav":
                old_bext, old_info, old_ixml = _read_wav_managed_chunks(item.path)
                backup_payload = {
                    "format": "wav",
                    "bext_b64": None if old_bext is None else base64.b64encode(old_bext).decode("ascii"),
                    "info_b64": None if old_info is None else base64.b64encode(old_info).decode("ascii"),
                    "ixml_b64": None if old_ixml is None else base64.b64encode(old_ixml).decode("ascii"),
                }
            else:
                backup_payload = {
                    "format": "mutagen",
                    "tags": _serialize_mutagen_backup(item.path, item.ext),
                }

            backup_file = batch_dir / f"{item.file_id}.json"
            backup_record = {
                "file_id": item.file_id,
                "path": str(item.path),
                "ext": item.ext,
                "audio_sha256": item.audio_sha256,
                "before_canonical": before_canon,
                "after_canonical": after_canon_for_undo,
                "payload": backup_payload,
            }
            backup_file.write_text(json.dumps(backup_record, ensure_ascii=False), encoding="utf-8")

            if item.ext == "wav":
                new_bext = _build_bext_body(old_bext, item.target_values)
                new_info = _build_list_info_body(old_info, item.target_values)
                new_ixml = _build_ixml_body(old_ixml, item.target_values)
                _write_wav_chunks_atomic(
                    item.path,
                    new_bext=new_bext,
                    new_info=new_info,
                    new_ixml=new_ixml,
                    drop_absent_managed=True,
                )
            else:
                _write_mutagen_atomic(item.path, item.ext, canonical=item.target_values)

            post_hash = compute_audio_hash(item.path, item.ext)
            st = item.path.stat()
            bext_json, ixml_json, emb_fields = _read_disk_metadata_snapshot(item.path, item.ext)
            store.sync_file_after_metadata_write(
                item.file_id,
                size_bytes=int(st.st_size),
                mtime_ns=int(st.st_mtime_ns),
                bext_json=bext_json,
                ixml_json=ixml_json,
                embedded_fields=emb_fields,
                canonical_values=item.target_values,
                fallback_package=rec.package,
                keep_dirty=False,
            )
            manifest_entries.append(
                {
                    "file_id": item.file_id,
                    "filename": item.filename,
                    "path": str(item.path),
                    "backup_file": backup_file.name,
                }
            )
            succeeded.append(
                WriteFileOutcome(
                    file_id=item.file_id,
                    filename=item.filename,
                    path=str(item.path),
                    success=True,
                    message="写回完成且音频哈希校验通过",
                    audio_sha256_before=item.audio_sha256,
                    audio_sha256_after=post_hash,
                )
            )
        except Exception as exc:
            failed.append(
                WriteFileOutcome(
                    file_id=item.file_id,
                    filename=item.filename,
                    path=str(item.path),
                    success=False,
                    message=str(exc),
                    audio_sha256_before=item.audio_sha256,
                )
            )

    manifest_path = batch_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "batch_id": batch_id,
                "created_at": time.time(),
                "entries": manifest_entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    final_status = "committed" if succeeded else "failed"
    store.update_write_batch_status(batch_id, final_status, file_count=len(succeeded))
    return WriteBatchResult(
        batch_id=batch_id,
        status=final_status,
        backup_dir=str(batch_dir),
        succeeded=tuple(succeeded),
        failed=tuple(failed),
        skipped=preview.skipped,
    )


def undo_write_batch(
    store: LibraryStore,
    batch_id: str,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> UndoBatchResult:
    """Restore pre-write file metadata from backup with pre/post audio SHA-256 verification."""
    batch = store.get_write_batch(batch_id)
    if batch is None:
        raise ValueError(f"未找到写回批次：{batch_id}")
    if batch.status == "reverted":
        raise ValueError("该批次已撤销")

    batch_dir = Path(batch.backup_dir)
    manifest_path = batch_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"备份清单不存在：{manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(manifest.get("entries", []))
    restored: list[WriteFileOutcome] = []
    failed: list[WriteFileOutcome] = []
    total = len(entries)

    for idx, entry in enumerate(entries, start=1):
        fid = int(entry["file_id"])
        filename = str(entry["filename"])
        path = Path(str(entry["path"]))
        if on_progress is not None:
            on_progress(idx, total, filename)

        backup_path = batch_dir / str(entry["backup_file"])
        if not backup_path.exists():
            failed.append(WriteFileOutcome(fid, filename, str(path), False, "缺少备份数据"))
            continue

        backup_rec = json.loads(backup_path.read_text(encoding="utf-8"))
        ext = str(backup_rec["ext"]).lower().lstrip(".")
        expected_sha = str(backup_rec["audio_sha256"])

        if not _is_writable_file(path):
            failed.append(WriteFileOutcome(fid, filename, str(path), False, "文件不存在或无写入权限"))
            continue

        try:
            cur_sha = compute_audio_hash(path, ext)
            if cur_sha != expected_sha:
                raise AudioIntegrityError(
                    "文件音频内容已与备份记录不一致，为防覆盖外部修改已拒绝撤销"
                )

            payload = backup_rec["payload"]
            if payload["format"] == "wav":
                old_bext = base64.b64decode(payload["bext_b64"]) if payload.get("bext_b64") else None
                old_info = base64.b64decode(payload["info_b64"]) if payload.get("info_b64") else None
                old_ixml = base64.b64decode(payload["ixml_b64"]) if payload.get("ixml_b64") else None
                _write_wav_chunks_atomic(
                    path,
                    new_bext=old_bext,
                    new_info=old_info,
                    new_ixml=old_ixml,
                    drop_absent_managed=True,
                )
            else:
                _write_mutagen_atomic(path, ext, restore_backup=payload["tags"])

            post_sha = compute_audio_hash(path, ext)
            if post_sha != expected_sha:
                raise AudioIntegrityError("撤销后音频哈希校验失败")

            st = path.stat()
            bext_json, ixml_json, emb_fields = _read_disk_metadata_snapshot(path, ext)
            rec_now = store.get_file(fid)
            if rec_now is not None:
                after_vals = {
                    k: str(v)
                    for k, v in (backup_rec.get("after_canonical") or {}).items()
                    if str(v).strip()
                }
                store.sync_file_after_metadata_write(
                    fid,
                    size_bytes=int(st.st_size),
                    mtime_ns=int(st.st_mtime_ns),
                    bext_json=bext_json,
                    ixml_json=ixml_json,
                    embedded_fields=emb_fields,
                    canonical_values=after_vals,
                    fallback_package=rec_now.package,
                    keep_dirty=bool(after_vals),
                )
            restored.append(
                WriteFileOutcome(
                    fid,
                    filename,
                    str(path),
                    True,
                    "已撤销写回并恢复原文件元数据",
                    audio_sha256_before=expected_sha,
                    audio_sha256_after=post_sha,
                )
            )
        except Exception as exc:
            failed.append(WriteFileOutcome(fid, filename, str(path), False, str(exc)))

    final_status = "reverted" if not failed else ("reverted" if restored else batch.status)
    if final_status == "reverted":
        store.update_write_batch_status(batch_id, "reverted")
    return UndoBatchResult(
        batch_id=batch_id,
        status=final_status,
        restored=tuple(restored),
        failed=tuple(failed),
    )


__all__ = [
    "AudioIntegrityError",
    "FieldDiff",
    "FileWritePreview",
    "SUPPORTED_WRITE_EXTS",
    "SkippedWriteItem",
    "UndoBatchResult",
    "WriteBackPreview",
    "WriteBatchResult",
    "WriteFileOutcome",
    "build_write_back_preview",
    "compute_audio_hash",
    "execute_write_back",
    "hash_aiff_ssnd_chunk",
    "hash_flac_audio_frames",
    "hash_mp3_audio_frames",
    "hash_ogg_audio_packets",
    "hash_wav_data_chunk",
    "undo_write_batch",
]
