"""Read the text metadata a WAVE file already carries, without decoding audio.

Scope of this round: RIFF/WAVE only, and three chunk families:

* ``bext`` — the EBU Broadcast Wave (BWF v0/v1/v2) description block.
* ``LIST/INFO`` — the RIFF INFO tag list (``INAM``, ``ICMT``, ...).
* ``iXML`` — the iXML production-metadata document.

Everything here is bounded and explicit. Only a metadata chunk body is read, and only
below a size limit; ``data`` is skipped with a seek, never read. An odd-sized chunk is
followed by one pad byte. A chunk whose declared size runs past the end of the file is
reported as truncated instead of being glued to the chunk that follows, and a container
this build does not understand (RF64/BW64/RIFX) is reported as
``unsupported_container`` instead of being parsed with the wrong layout.

iXML is parsed with ``xml.etree`` after two cheap guards: a document declaring a DTD or
an entity is rejected outright, and the body is size-limited before it reaches the
parser. ``ElementTree`` never resolves external entities, so no iXML field can cause a
file, URL or entity expansion to be read.

Nothing here invents a value. A field is either present in a chunk and cleanly
decodable, or it is absent. Present-but-uninterpretable data (a BWF origination date
that is not a date, a truncated chunk, a broken XML document) becomes a note; the
caller stores those notes so a human can see why a field is missing.

Field mapping (see ``INFO_FIELDS`` and ``IXML_FIELDS`` for the tag tables):

===========================  ==============================  ====================
chunk / field                catalog field                   searchable
===========================  ==============================  ====================
bext Description             ``description``                 yes (description)
bext Originator              ``bext_originator``             no
bext OriginatorReference     ``bext_originator_reference``   no
bext OriginationDate/Time    ``bext_origination_*``          no (must be a date)
bext TimeReference/Version   ``bext_time_reference``/``_version``  no
bext UMID                   ``bext_umid``                   no
bext CodingHistory           ``bext_coding_history``         no
LIST/INFO ICMT/ISBJ/IGNR     ``keyword``                     yes (keywords)
LIST/INFO INAM/IART/...      ``info_*``                      no
iXML NOTE                    ``keyword``                     yes (keywords)
iXML PROJECT/SCENE/TAKE/...  ``ixml_*``                      no
iXML TRACK/NAME              ``ixml_track_name``             no
===========================  ==============================  ====================

Deliberately not imported: the BWF loudness block (v1+) is a numeric measurement, so it
cannot help retrieval, and a writer that sets version 1 without measuring leaves zeros
that would be published as a real 0.00 dBFS reading. ``LIST/adtl`` labels are timeline
markers about a region of the file, not catalog metadata. Project, scene, take, tape and
track names are production identifiers and are never promoted to a description.
"""

from __future__ import annotations

import os
import re
import struct
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, replace
from datetime import date, time
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

# Bump when the field mapping or the decoding rules change: stored reads carrying an
# older version are re-read, because a new mapping can only be applied by looking at
# the file again.
EMBEDDED_META_VERSION = "embedded-v1"

# What a parsed value is allowed to do once it reaches the catalog.
ROLE_DESCRIPTION = "description"
ROLE_KEYWORD = "keyword"
ROLE_VALUE = "value"

# Source families. A read authorises deleting the old values of a family only when that
# family was actually covered: a broken iXML chunk says nothing about the old iXML
# keywords, so it must not be able to erase them. Values keep their per-tag source
# (``embedded:info:ICMT``); the family is the first segment after ``embedded:``.
FAMILY_BEXT = "bext"
FAMILY_INFO = "info"
FAMILY_IXML = "ixml"
FAMILIES = (FAMILY_BEXT, FAMILY_INFO, FAMILY_IXML)

# Hard bounds. A metadata chunk in a real library is a few hundred bytes to a few
# hundred kilobytes; anything near these numbers is either a broken header or a file
# this reader should not try to swallow.
MAX_METADATA_CHUNK_BYTES = 4 * 1024 * 1024
MAX_IXML_CHARS = 1_000_000
MAX_FIELD_CHARS = 1024
MAX_CHUNKS = 4096

# Fixed part of a BWF ``bext`` chunk (ITU-R BS.1352 / EBU Tech 3285).
BEXT_FIXED_BYTES = 602

METADATA_CHUNKS = frozenset({b"bext", b"LIST", b"iXML"})

RIFF_CONTAINERS = {b"RIFF": "RIFF", b"RF64": "RF64", b"BW64": "BW64", b"RIFX": "RIFX"}

# RIFF INFO tags this reader maps. Everything else is left alone and named in a note:
# importing a tag whose meaning is unknown would put unexplained text into search.
INFO_FIELDS: dict[bytes, tuple[str, str]] = {
    b"INAM": ("info_title", ROLE_VALUE),
    b"IART": ("info_artist", ROLE_VALUE),
    b"ICMT": ("info_comment", ROLE_KEYWORD),
    b"ISBJ": ("info_subject", ROLE_KEYWORD),
    b"IGNR": ("info_genre", ROLE_KEYWORD),
    b"IKEY": ("info_keywords", ROLE_KEYWORD),
    b"ICRD": ("info_creation_date", ROLE_VALUE),
    b"ISFT": ("info_software", ROLE_VALUE),
    b"ICOP": ("info_copyright", ROLE_VALUE),
    b"ISRC": ("info_source", ROLE_VALUE),
    b"IENG": ("info_engineer", ROLE_VALUE),
    b"ITCH": ("info_technician", ROLE_VALUE),
    b"ISRF": ("info_source_form", ROLE_VALUE),
}

# Direct children of ``<BWFXML>`` this reader maps. Project/scene/take/tape/track names
# are production identifiers, not descriptions: they are stored as their own fields and
# never promoted to ``description``.
IXML_FIELDS: dict[str, tuple[str, str]] = {
    "IXML_VERSION": ("ixml_version", ROLE_VALUE),
    "PROJECT": ("ixml_project", ROLE_VALUE),
    "SCENE": ("ixml_scene", ROLE_VALUE),
    "TAKE": ("ixml_take", ROLE_VALUE),
    "TAPE": ("ixml_tape", ROLE_VALUE),
    "NOTE": ("keyword", ROLE_KEYWORD),
}

# Soundminer / AudioSearchTool ``<BWFXML><USER>...</USER></BWFXML>`` child elements.
IXML_USER_FIELDS: dict[str, tuple[str, str]] = {
    "DESCRIPTION": ("ixml_user_description", ROLE_DESCRIPTION),
    "KEYWORDS": ("ixml_user_keywords", ROLE_KEYWORD),
    "CATID": ("ixml_user_cat_id", ROLE_VALUE),
    "CATEGORY": ("ixml_user_category", ROLE_KEYWORD),
    "SUBCATEGORY": ("ixml_user_subcategory", ROLE_KEYWORD),
    "FXNAME": ("ixml_user_fx_name", ROLE_VALUE),
    "LIBRARY": ("ixml_user_library", ROLE_VALUE),
    "DESIGNER": ("ixml_user_designer", ROLE_VALUE),
    "NOTES": ("ixml_user_notes", ROLE_VALUE),
}

_ISO_DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_ISO_TIME = re.compile(r"^[0-9]{2}:[0-9]{2}:[0-9]{2}$")


class EmbeddedStatus(StrEnum):
    """What one read of one file produced.

    ``OK`` and ``PARTIAL`` both carry usable values and differ in whether the whole
    chunk table was covered: a partial read saw some metadata and then hit a problem, so
    only the families it did cover may be replaced. ``NO_METADATA`` means the whole file
    was walked and no supported field was present, which *is* evidence that nothing is
    there. The other three carry no trustworthy coverage at all.
    """

    OK = "ok"
    PARTIAL = "partial"
    NO_METADATA = "no_metadata"
    UNSUPPORTED_CONTAINER = "unsupported_container"
    MALFORMED = "malformed"
    FAILED = "failed"


class Coverage(StrEnum):
    """How much of a file one read actually covered.

    This is the distinction the storage layer needs and that a single status cannot
    express: "the iXML chunk is broken" and "there is no iXML chunk" lead to the same
    old-value question but not to the same answer.
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    NONE = "none"


@dataclass(frozen=True)
class EmbeddedValue:
    """One field exactly as it was found, with the chunk it came from."""

    field: str
    value: str
    source: str
    role: str

    @property
    def family(self) -> str:
        """``embedded:info:ICMT`` -> ``info``."""
        parts = self.source.split(":")
        return parts[1] if len(parts) > 1 else parts[0]

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "value": self.value,
            "source": self.source,
            "family": self.family,
            "role": self.role,
        }


def family_of_source(source: str) -> str:
    """The source family of a stored ``metadata_values.source`` value."""
    parts = source.split(":")
    return parts[1] if len(parts) > 1 else parts[0]


@dataclass(frozen=True)
class EmbeddedMetadata:
    """The result of reading one file: usable values, notes, and an explicit status."""

    status: EmbeddedStatus
    values: tuple[EmbeddedValue, ...] = ()
    notes: tuple[str, ...] = ()
    chunks: tuple[str, ...] = ()
    container: str | None = None
    parser_version: str = EMBEDDED_META_VERSION
    # Per family (``bext``/``info``/``ixml``): was it fully walked, partly read, or not
    # reached at all? A family absent from this map was never seen in the file.
    covered: tuple[tuple[str, Coverage], ...] = ()

    @property
    def coverage(self) -> dict[str, Coverage]:
        return dict(self.covered)

    def family_coverage(self, family: str) -> Coverage:
        return self.coverage.get(family, Coverage.NONE)

    def replaced_families(self) -> frozenset[str]:
        """Families whose old values this read is allowed to remove wholesale.

        ``COMPLETE`` is the only answer that authorises a delete: it means the whole
        chunk table was walked and that family's chunks were read without a problem, so
        whatever is missing really is absent from the file. ``PARTIAL`` says the family
        was seen but its content could not be trusted (a broken XML document, a
        truncated or oversized chunk); keeping the previous values is the honest result.
        """
        return frozenset(
            family for family, coverage in self.covered if coverage is Coverage.COMPLETE
        )

    def families_with_values(self) -> frozenset[str]:
        return frozenset(value.family for value in self.values)

    @property
    def description(self) -> str | None:
        for value in self.values:
            if value.role == ROLE_DESCRIPTION:
                return value.value
        return None

    @property
    def field_count(self) -> int:
        return len(self.values)

    def as_dict(self) -> dict:
        return {
            "status": str(self.status),
            "parser_version": self.parser_version,
            "container": self.container,
            "chunks": list(self.chunks),
            "covered": {family: str(coverage) for family, coverage in self.covered},
            "replaced_families": sorted(self.replaced_families()),
            "field_count": self.field_count,
            "notes": list(self.notes),
            "values": [value.as_dict() for value in self.values],
        }


def read_embedded_metadata(path: Path | str) -> EmbeddedMetadata:
    """Read one file; an unreadable file is a ``failed`` result, not an exception."""
    return read_embedded_verified(path).metadata


@dataclass(frozen=True)
class EmbeddedRead:
    """One read together with the evidence that it really described the expected bytes.

    The importer must not record metadata it cannot attribute to the revision it stored,
    so the file is opened first and the *handle* is measured, not just the path: the
    bytes that get parsed are then the bytes that were measured. ``stable`` says the
    handle still had the same size/mtime after parsing, and ``matches_expected`` says
    those attributes were the ones the manifest named.
    """

    metadata: EmbeddedMetadata
    expected: tuple[int, int] | None = None
    observed: tuple[int, int] | None = None
    stable: bool = False
    matches_expected: bool = False

    @property
    def trustworthy(self) -> bool:
        """True only when the parsed bytes provably were the expected revision."""
        return self.stable and self.matches_expected and self.observed is not None


def read_embedded_verified(
    path: Path | str, *, expect: tuple[int, int] | None = None
) -> EmbeddedRead:
    """Open, measure the handle, parse, measure again.

    The handle is measured with ``os.fstat`` so the parse cannot silently run against a
    different file than the one that was checked: opening and checking the same handle is
    the only way to be sure, and a second path-level ``stat`` would prove nothing about
    the bytes that were already read.
    """
    target = Path(path)
    try:
        with target.open("rb") as stream:
            stat = os.fstat(stream.fileno())
            observed = (int(stat.st_size), int(stat.st_mtime_ns))
            metadata = parse_embedded_stream(stream, file_size=observed[0])
            after = os.fstat(stream.fileno())
    except OSError as exc:
        return EmbeddedRead(
            metadata=EmbeddedMetadata(
                status=EmbeddedStatus.FAILED,
                notes=(f"无法读取文件属性或内容（{type(exc).__name__}）",),
            ),
            expected=expect,
        )
    stable = (int(after.st_size), int(after.st_mtime_ns)) == observed
    if not stable:
        # The bytes moved under the reader: nothing parsed here can be attributed to a
        # single revision, so the result is reported as a failure rather than used.
        metadata = replace(
            metadata,
            status=EmbeddedStatus.FAILED,
            notes=(*metadata.notes, "读取期间文件属性发生变化，本次结果不可用"),
            covered=(),
        )
    return EmbeddedRead(
        metadata=metadata,
        expected=expect,
        observed=observed,
        stable=stable,
        matches_expected=expect is None or observed == expect,
    )


def parse_embedded_stream(stream: BinaryIO, *, file_size: int) -> EmbeddedMetadata:
    """Walk the RIFF chunk table of an open stream and parse the metadata chunks.

    Coverage is tracked per source family because that is what decides whether the
    storage layer may delete the previous values of that family. A broken ``iXML`` chunk
    is not evidence that the old iXML keywords are gone; a *complete* walk that finds no
    ``iXML`` chunk is.
    """
    header = stream.read(12)
    if len(header) < 12:
        return EmbeddedMetadata(
            status=EmbeddedStatus.MALFORMED, notes=("文件短于 12 字节的 RIFF 头",)
        )
    riff = header[:4]
    container = RIFF_CONTAINERS.get(riff)
    if container is None:
        return EmbeddedMetadata(
            status=EmbeddedStatus.UNSUPPORTED_CONTAINER,
            notes=(f"不是 RIFF 容器（{_chunk_name(riff)}）",),
        )
    if container != "RIFF":
        # RF64/BW64 relocate the data size into a ``ds64`` chunk and RIFX is
        # big-endian; reading either with the RIFF layout would misplace every
        # following chunk, so this build declines rather than guesses.
        return EmbeddedMetadata(
            status=EmbeddedStatus.UNSUPPORTED_CONTAINER,
            container=container,
            notes=(f"{container} 容器本轮不解析（需要 ds64 或大端布局）",),
        )
    if header[8:12] != b"WAVE":
        return EmbeddedMetadata(
            status=EmbeddedStatus.UNSUPPORTED_CONTAINER,
            container=container,
            notes=("RIFF 容器不是 WAVE",),
        )

    chunks: list[str] = []
    values: list[EmbeddedValue] = []
    notes: list[str] = []
    problems: list[str] = []
    family_seen: set[str] = set()
    family_problem: set[str] = set()
    position = 12
    count = 0
    complete_scan = True
    while position + 8 <= file_size:
        count += 1
        if count > MAX_CHUNKS:
            notes.append(f"chunk 数量超过上限 {MAX_CHUNKS}，停止解析")
            problems.append("chunk 数量超限")
            complete_scan = False
            break
        stream.seek(position)
        chunk_header = stream.read(8)
        if len(chunk_header) < 8:
            notes.append(f"位置 {position} 的 chunk 头被截断")
            problems.append("chunk 头截断")
            complete_scan = False
            break
        chunk_id = chunk_header[:4]
        size = struct.unpack("<I", chunk_header[4:8])[0]
        body_start = position + 8
        available = max(file_size - body_start, 0)
        name = _chunk_name(chunk_id)
        chunks.append(name)

        if chunk_id == b"data":
            if size in (0, 0xFFFFFFFF):
                # A streaming writer left a placeholder size: the end of the audio is
                # not known, so nothing after it can be located either. Anything the file
                # holds beyond this point is unscanned, which is *not* evidence of
                # absence.
                notes.append("data: 长度字段是占位值，其后内容未扫描")
                complete_scan = False
                break
            position = body_start + size + (size & 1)
            if position > file_size:
                notes.append("data: 声明长度超出文件末尾，其后内容未扫描")
                complete_scan = False
                break
            continue

        if chunk_id in METADATA_CHUNKS:
            family = _family_of_chunk(chunk_id)
            family_seen.add(family)
            if size > MAX_METADATA_CHUNK_BYTES:
                notes.append(f"{name}: {size} 字节超过上限 {MAX_METADATA_CHUNK_BYTES}，未读取")
                problems.append(f"{name} 超过大小上限")
                family_problem.add(family)
            else:
                body = stream.read(min(size, available))
                if len(body) < size:
                    notes.append(f"{name}: chunk 声明 {size} 字节，实际只读到 {len(body)} 字节")
                    problems.append(f"{name} 被截断")
                    family_problem.add(family)
                parsed, problem = _parse_metadata_chunk(chunk_id, body, notes)
                values.extend(parsed)
                if problem:
                    problems.append(problem)
                    family_problem.add(family)

        if size > available:
            # The declared size runs past EOF. What exists was already read; there is no
            # reliable next chunk position, so the walk ends here with a note.
            notes.append(f"{name}: 声明 {size} 字节但文件只剩 {available} 字节，解析到此为止")
            problems.append(f"{name} 长度超出文件末尾")
            complete_scan = False
            break
        position = body_start + size + (size & 1)

    covered = _coverage(family_seen, family_problem, complete_scan=complete_scan)
    if complete_scan and not family_seen and not problems:
        notes.append("没有 bext / LIST-INFO / iXML chunk")

    if complete_scan and not problems:
        status = EmbeddedStatus.OK if values else EmbeddedStatus.NO_METADATA
    elif values:
        # Something was usable but the file was not fully covered (a broken or truncated
        # chunk, or a scan that stopped early). Per-family coverage records exactly which
        # of those values may replace stored ones.
        status = EmbeddedStatus.PARTIAL
    elif not complete_scan:
        # The walk stopped before the end and produced nothing usable. This is "the file
        # could not be scanned", not "the metadata is gone": every family stays unscanned.
        status = EmbeddedStatus.PARTIAL
    else:
        # Fully walked, nothing usable: a structural failure.
        status = EmbeddedStatus.MALFORMED
    return EmbeddedMetadata(
        status=status,
        values=tuple(values),
        notes=tuple(notes),
        chunks=tuple(chunks),
        container=container,
        covered=covered,
    )


def _family_of_chunk(chunk_id: bytes) -> str:
    if chunk_id == b"bext":
        return FAMILY_BEXT
    if chunk_id == b"LIST":
        return FAMILY_INFO
    return FAMILY_IXML


def _coverage(
    family_seen: set[str], family_problem: set[str], *, complete_scan: bool
) -> tuple[tuple[str, Coverage], ...]:
    """Per-family coverage, including families the file does not contain.

    A family that was never seen is still ``COMPLETE`` when the whole chunk table was
    walked: that absence is exactly what authorises removing stale values of that family.
    """
    covered: list[tuple[str, Coverage]] = []
    for family in FAMILIES:
        if family in family_problem:
            covered.append((family, Coverage.PARTIAL))
        elif complete_scan:
            covered.append((family, Coverage.COMPLETE))
        elif family in family_seen:
            covered.append((family, Coverage.PARTIAL))
        else:
            covered.append((family, Coverage.NONE))
    return tuple(covered)


def _parse_metadata_chunk(
    chunk_id: bytes, body: bytes, notes: list[str]
) -> tuple[list[EmbeddedValue], str | None]:
    if chunk_id == b"bext":
        return _parse_bext(body, notes), None
    if chunk_id == b"LIST":
        return _parse_list(body, notes)
    return _parse_ixml(body, notes)


# --------------------------------------------------------------------------- bext
def _parse_bext(body: bytes, notes: list[str]) -> list[EmbeddedValue]:
    if len(body) < BEXT_FIXED_BYTES:
        notes.append(f"bext: 只有 {len(body)} 字节，短于规范固定长度 {BEXT_FIXED_BYTES}")
    values: list[EmbeddedValue] = []

    description = _field_text(body[0:256], notes, "bext.Description")
    if description:
        values.append(EmbeddedValue("description", description, "embedded:bext", ROLE_DESCRIPTION))

    for name, start, length, label in (
        ("bext_originator", 256, 32, "bext.Originator"),
        ("bext_originator_reference", 288, 32, "bext.OriginatorReference"),
        (
            "bext_coding_history",
            BEXT_FIXED_BYTES,
            len(body) - BEXT_FIXED_BYTES,
            "bext.CodingHistory",
        ),
    ):
        text = _field_text(_slice(body, start, length), notes, label)
        if text:
            values.append(EmbeddedValue(name, text, "embedded:bext", ROLE_VALUE))

    text_date = _field_text(_slice(body, 320, 10), notes, "bext.OriginationDate", ascii_only=True)
    if text_date:
        # The field is defined as YYYY-MM-DD; a value that is not a real date is a
        # broken writer, not a date this catalog may show.
        if _ISO_DATE.match(text_date) and _is_date(text_date):
            values.append(
                EmbeddedValue("bext_origination_date", text_date, "embedded:bext", ROLE_VALUE)
            )
        else:
            notes.append(f"bext.OriginationDate 不是可用的 YYYY-MM-DD，已跳过：{text_date!r}")

    text_time = _field_text(_slice(body, 330, 8), notes, "bext.OriginationTime", ascii_only=True)
    if text_time:
        if _ISO_TIME.match(text_time) and _is_time(text_time):
            values.append(
                EmbeddedValue("bext_origination_time", text_time, "embedded:bext", ROLE_VALUE)
            )
        else:
            notes.append(f"bext.OriginationTime 不是可用的 HH:MM:SS，已跳过：{text_time!r}")

    if len(body) >= 346:
        time_reference = struct.unpack("<Q", body[338:346])[0]
        # Zero means "no timecode was written", which is the default in every writer
        # that does not timestamp; importing it would add a field that says nothing.
        if time_reference:
            values.append(
                EmbeddedValue(
                    "bext_time_reference", str(time_reference), "embedded:bext", ROLE_VALUE
                )
            )
    version = struct.unpack("<H", body[346:348])[0] if len(body) >= 348 else None
    if version:
        values.append(EmbeddedValue("bext_version", str(version), "embedded:bext", ROLE_VALUE))

    umid = _slice(body, 348, 64)
    if umid.strip(b"\x00"):
        values.append(EmbeddedValue("bext_umid", umid.hex(), "embedded:bext", ROLE_VALUE))

    # The BWF loudness block (v1+) is deliberately not imported: it is a numeric
    # measurement, not text, so it cannot help retrieval, and writers that set
    # version 1 without measuring leave a block of zeros that would be published as a
    # real 0.00 dBFS reading. Not reading it is the honest default.
    return values


# ----------------------------------------------------------------------- LIST/INFO
def _parse_list(body: bytes, notes: list[str]) -> tuple[list[EmbeddedValue], str | None]:
    if len(body) < 4:
        notes.append("LIST: 不足 4 字节，无法判断列表类型")
        return [], "LIST 头截断"
    list_type = body[:4]
    if list_type != b"INFO":
        # ``adtl`` labels/comments are timeline markers about a region of the file, not
        # catalog metadata; ``INFO`` is the only list this round maps.
        notes.append(f"LIST/{_chunk_name(list_type)}: 不是 INFO 列表，未导入")
        return [], None

    values: list[EmbeddedValue] = []
    unmapped: list[str] = []
    problem: str | None = None
    position = 4
    while position + 8 <= len(body):
        tag = body[position : position + 4]
        size = struct.unpack("<I", body[position + 4 : position + 8])[0]
        start = position + 8
        end = start + size
        if end > len(body):
            notes.append(f"LIST/INFO {_chunk_name(tag)}: 声明 {size} 字节超出列表，按截断读取")
            problem = "LIST/INFO 子块超出列表长度"
            end = len(body)
        raw = body[start:end]
        position = start + size + (size & 1)
        mapped = INFO_FIELDS.get(tag)
        if mapped is None:
            unmapped.append(_chunk_name(tag))
            continue
        text = _field_text(raw, notes, f"LIST/INFO.{_chunk_name(tag)}")
        if not text:
            continue
        field_name, role = mapped
        values.append(EmbeddedValue(field_name, text, f"embedded:info:{_chunk_name(tag)}", role))
    if unmapped:
        notes.append("LIST/INFO 含未映射字段：" + ", ".join(sorted(set(unmapped))))
    return values, problem


# ---------------------------------------------------------------------------- iXML
def _parse_ixml(body: bytes, notes: list[str]) -> tuple[list[EmbeddedValue], str | None]:
    if body[:3] == b"\xef\xbb\xbf":
        body = body[3:]
    head = body[:8192].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in head:
        # iXML has no DTD. Refusing outright is both cheaper and safer than relying on
        # the parser's entity handling, and keeps "no external resource is ever read"
        # true by construction.
        notes.append("iXML: 含 DOCTYPE/ENTITY 声明，已拒绝解析")
        return [], "iXML 含 DTD 或实体声明"
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        notes.append("iXML: 不是合法 UTF-8，未解析")
        return [], "iXML 编码不是 UTF-8"
    if len(text) > MAX_IXML_CHARS:
        notes.append(f"iXML: 超过 {MAX_IXML_CHARS} 字符上限，未解析")
        return [], "iXML 超过字符上限"
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        notes.append(f"iXML: XML 解析失败（{exc}）")
        return [], "iXML 不是合法 XML"
    if _local_name(root.tag).upper() != "BWFXML":
        notes.append(f"iXML: 根元素是 {_local_name(root.tag)!r}，不是 BWFXML")
        return [], "iXML 根元素不是 BWFXML"

    values: list[EmbeddedValue] = []
    for child in root:
        tag = _local_name(child.tag).upper()
        if tag == "USER":
            for user_child in child:
                utag = _local_name(user_child.tag).upper()
                umapped = IXML_USER_FIELDS.get(utag)
                if umapped is None:
                    continue
                uval = " ".join((user_child.text or "").split())
                if not uval:
                    continue
                ufield, urole = umapped
                values.append(
                    EmbeddedValue(
                        ufield,
                        uval[:MAX_FIELD_CHARS],
                        f"embedded:ixml:USER:{utag}",
                        urole,
                    )
                )
            continue
        mapped = IXML_FIELDS.get(tag)
        if mapped is None:
            continue
        value = " ".join((child.text or "").split())
        if not value:
            continue
        field_name, role = mapped
        values.append(
            EmbeddedValue(field_name, value[:MAX_FIELD_CHARS], f"embedded:ixml:{tag}", role)
        )

    # Track names are production identifiers: stored, never promoted to a description.
    for element in root.iter():
        if _local_name(element.tag).upper() != "TRACK":
            continue
        for child in element:
            if _local_name(child.tag).upper() != "NAME":
                continue
            name = " ".join((child.text or "").split())
            if name:
                values.append(
                    EmbeddedValue(
                        "ixml_track_name",
                        name[:MAX_FIELD_CHARS],
                        "embedded:ixml:TRACK",
                        ROLE_VALUE,
                    )
                )
    return values, None


# ------------------------------------------------------------------------- helpers
def _slice(body: bytes, start: int, length: int) -> bytes:
    if length <= 0 or start >= len(body):
        return b""
    return body[start : start + length]


def _field_text(
    raw: bytes, notes: list[str], label: str, *, ascii_only: bool = False
) -> str | None:
    """Decode one fixed-width text field; ``None`` when it is empty.

    Fields are NUL padded, and some writers pad at the front as well, so NULs and
    surrounding whitespace are removed before the value is cut at its first interior
    NUL. A value that is not valid UTF-8 is decoded as cp1252 and reported: guessing a
    legacy code page would silently produce text nobody can verify. Fields that the
    format defines as ASCII (the BWF date/time pair) are decoded strictly, so binary
    content is reported instead of being turned into odd characters.
    """
    stripped = raw.strip(b"\x00 \t\r\n")
    if not stripped:
        return None
    head = stripped.split(b"\x00", 1)[0]
    try:
        text = head.decode("ascii" if ascii_only else "utf-8")
    except UnicodeDecodeError:
        if ascii_only:
            notes.append(f"{label}: 内容不是 ASCII，疑似二进制数据，已跳过")
            return None
        text = head.decode("cp1252", "replace")
        notes.append(f"{label}: 不是合法 UTF-8，按 cp1252 解码")
    printable = "".join(char if char.isprintable() else " " for char in text)
    cleaned = " ".join(printable.split())
    if not cleaned:
        return None
    if len(cleaned) > MAX_FIELD_CHARS:
        notes.append(f"{label}: 超过 {MAX_FIELD_CHARS} 字符，已截断")
        return cleaned[:MAX_FIELD_CHARS]
    return cleaned


def _chunk_name(chunk_id: bytes) -> str:
    if chunk_id and all(32 <= byte < 127 for byte in chunk_id):
        return chunk_id.decode("ascii")
    return chunk_id.hex()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _is_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_time(value: str) -> bool:
    try:
        time.fromisoformat(value)
    except ValueError:
        return False
    return True
