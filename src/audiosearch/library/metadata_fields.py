"""Canonical R5 editable metadata schema, mapping helpers, and batch transforms."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MetadataFieldSpec:
    key: str
    label_zh: str
    label_en: str
    multiline: bool = False


EDITABLE_FIELDS: tuple[MetadataFieldSpec, ...] = (
    MetadataFieldSpec("description", "描述", "Description", multiline=True),
    MetadataFieldSpec("keywords", "关键词", "Keywords", multiline=False),
    MetadataFieldSpec("cat_id", "UCS 编号", "CatID", multiline=False),
    MetadataFieldSpec("category", "主分类", "Category", multiline=False),
    MetadataFieldSpec("subcategory", "子分类", "SubCategory", multiline=False),
    MetadataFieldSpec("fx_name", "音效名称", "FXName", multiline=False),
    MetadataFieldSpec("library", "素材库 / 包", "Library", multiline=False),
    MetadataFieldSpec("designer", "设计师", "Designer", multiline=False),
    MetadataFieldSpec("notes", "元数据备注", "Notes", multiline=True),
)

EDITABLE_FIELD_KEYS: tuple[str, ...] = tuple(spec.key for spec in EDITABLE_FIELDS)
FIELD_SPEC_MAP: dict[str, MetadataFieldSpec] = {spec.key: spec for spec in EDITABLE_FIELDS}

BATCH_OPS: tuple[tuple[str, str], ...] = (
    ("set", "设为固定值"),
    ("prepend", "前置添加"),
    ("append", "末尾追加"),
    ("replace", "查找替换"),
    ("regex_replace", "正则替换"),
    ("clear", "清空字段"),
)


def apply_field_transform(
    current: str,
    op: str,
    value: str = "",
    *,
    find: str = "",
    replace_with: str = "",
    case_sensitive: bool = False,
) -> str:
    """Apply a single deterministic batch metadata transform."""
    base = current or ""
    if op == "set":
        return value
    if op == "clear":
        return ""
    if op == "prepend":
        return f"{value}{base}" if value else base
    if op == "append":
        return f"{base}{value}" if value else base
    if op == "replace":
        if not find:
            return base
        if case_sensitive:
            return base.replace(find, replace_with)
        pattern = re.compile(re.escape(find), re.IGNORECASE)
        return pattern.sub(lambda _m: replace_with, base)
    if op == "regex_replace":
        if not find:
            return base
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(find, flags)
        except re.error as exc:
            raise ValueError(f"正则表达式无效：{exc}") from exc
        try:
            return pattern.sub(replace_with, base)
        except re.error as exc:
            raise ValueError(f"替换模板无效：{exc}") from exc
    raise ValueError(f"不支持的批量操作：{op!r}")


_RAW_FALLBACK_ORDER: dict[str, tuple[str, ...]] = {
    "description": (
        "embedded:ixml:USER:DESCRIPTION",
        "embedded:bext",
        "embedded:info:ICMT",
        "embedded:info:ISBJ",
        "embedded:tag:DESCRIPTION",
        "embedded:tag:COMMENT",
    ),
    "keywords": (
        "embedded:ixml:USER:KEYWORDS",
        "embedded:info:IKEY",
        "embedded:ixml:NOTE",
        "embedded:info:ICMT",
        "embedded:tag:KEYWORDS",
    ),
    "cat_id": (
        "embedded:ixml:USER:CATID",
        "embedded:tag:CATID",
    ),
    "category": (
        "embedded:ixml:USER:CATEGORY",
        "embedded:info:IGNR",
        "embedded:tag:CATEGORY",
        "embedded:tag:GENRE",
    ),
    "subcategory": (
        "embedded:ixml:USER:SUBCATEGORY",
        "embedded:info:ISBJ",
        "embedded:tag:SUBCATEGORY",
    ),
    "fx_name": (
        "embedded:ixml:USER:FXNAME",
        "embedded:info:INAM",
        "embedded:tag:FXNAME",
        "embedded:tag:TITLE",
    ),
    "library": (
        "embedded:ixml:USER:LIBRARY",
        "embedded:ixml:PROJECT",
        "embedded:info:ISRC",
        "embedded:tag:LIBRARY",
        "embedded:tag:ALBUM",
    ),
    "designer": (
        "embedded:ixml:USER:DESIGNER",
        "embedded:info:IART",
        "embedded:bext",
        "embedded:tag:DESIGNER",
        "embedded:tag:ARTIST",
    ),
    "notes": (
        "embedded:ixml:USER:NOTES",
        "embedded:tag:NOTES",
    ),
}


def extract_canonical_from_raw(
    file_rec: Any,
    raw_fields: list[tuple[str, str, str, int]],
) -> dict[str, tuple[str, str, bool]]:
    """Resolve canonical R5 fields as ``{key: (value, source, is_dirty)}``."""
    by_name: dict[str, tuple[str, str, int]] = {}
    for field_name, value, source, dirty in raw_fields:
        existing = by_name.get(field_name)
        if existing is None or (dirty and not existing[2]):
            by_name[field_name] = (value, source, int(dirty))

    resolved: dict[str, tuple[str, str, bool]] = {}
    for key in EDITABLE_FIELD_KEYS:
        direct = by_name.get(key)
        if direct is not None and (direct[2] or direct[1] != "embedded" or direct[0]):
            resolved[key] = (direct[0], direct[1], bool(direct[2]))
            continue
        val = ""
        src = "embedded"
        for raw_key in _RAW_FALLBACK_ORDER.get(key, ()):
            # Skip bext for designer fallback because embedded:bext stores Description.
            if key == "designer" and raw_key == "embedded:bext":
                continue
            cand = by_name.get(raw_key)
            if cand is not None and cand[0]:
                val, src, _ = cand
                break
        if not val and key == "description" and getattr(file_rec, "description", ""):
            val = str(file_rec.description)
            src = "embedded"
        elif not val and key == "library" and getattr(file_rec, "package", ""):
            val = str(file_rec.package)
            src = "derived"
        resolved[key] = (val, src, False)
    return resolved


def compute_file_summary_columns(
    canonical: dict[str, tuple[str, str, bool]],
    raw_fields: list[tuple[str, str, str, int]],
    fallback_package: str = "",
) -> tuple[str, str, str]:
    """Compute ``(description, package, embedded_text)`` synced into ``files`` & FTS5."""
    desc = canonical.get("description", ("", "", False))[0].strip()
    if not desc:
        desc = canonical.get("fx_name", ("", "", False))[0].strip()
    pkg = canonical.get("library", ("", "", False))[0].strip() or fallback_package

    tokens: list[str] = []
    seen: set[str] = set()

    def _add(text: str) -> None:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            tokens.append(cleaned)

    for key in EDITABLE_FIELD_KEYS:
        _add(canonical.get(key, ("", "", False))[0])
    for field_name, value, _source, _dirty in raw_fields:
        if field_name not in FIELD_SPEC_MAP:
            _add(value)
    return desc, pkg, " ".join(tokens)


__all__ = [
    "BATCH_OPS",
    "EDITABLE_FIELDS",
    "EDITABLE_FIELD_KEYS",
    "FIELD_SPEC_MAP",
    "InspectorSummary",
    "MetadataFieldSpec",
    "apply_field_transform",
    "compute_file_summary_columns",
    "extract_canonical_from_raw",
    "extract_inspector_summary",
]


@dataclass(frozen=True, slots=True)
class InspectorSummary:
    """Structured core metadata summary and human-friendly field list for the inspector."""

    brand: str
    package: str
    ucs_label: str
    folder_path: str
    formatted_fields: tuple[tuple[str, str, str], ...]
    path_segments: tuple[str, ...] = ()
    root_label: str = ""


_INSPECTOR_LABEL_MAP: dict[str, tuple[str, int]] = {
    "designer": ("厂牌 / 设计师", 10),
    "embedded:ixml:USER:DESIGNER": ("设计师 (DESIGNER)", 11),
    "embedded:info:IART": ("厂牌 / 作者 (IART)", 12),
    "embedded:tag:DESIGNER": ("设计师 (DESIGNER)", 13),
    "embedded:tag:ARTIST": ("作者 (ARTIST)", 14),
    "library": ("素材包 (Library)", 20),
    "embedded:ixml:USER:LIBRARY": ("素材库 (LIBRARY)", 21),
    "embedded:ixml:PROJECT": ("工程 (PROJECT)", 22),
    "embedded:tag:LIBRARY": ("素材库 (LIBRARY)", 23),
    "embedded:tag:ALBUM": ("专辑 / 库 (ALBUM)", 24),
    "fx_name": ("音效名称 (FXName)", 30),
    "embedded:ixml:USER:FXNAME": ("音效名称 (FXNAME)", 31),
    "embedded:info:INAM": ("标题 (INAM)", 32),
    "embedded:tag:FXNAME": ("音效名称 (FXNAME)", 33),
    "embedded:tag:TITLE": ("标题 (TITLE)", 34),
    "cat_id": ("UCS 编号 (CatID)", 40),
    "embedded:ixml:USER:CATID": ("UCS 编号 (CATID)", 41),
    "embedded:tag:CATID": ("UCS 编号 (CATID)", 42),
    "category": ("主分类 (Category)", 43),
    "embedded:ixml:USER:CATEGORY": ("主分类 (CATEGORY)", 44),
    "embedded:info:IGNR": ("分类 (IGNR)", 45),
    "embedded:tag:CATEGORY": ("主分类 (CATEGORY)", 46),
    "embedded:tag:GENRE": ("流派 / 分类 (GENRE)", 47),
    "subcategory": ("子分类 (SubCategory)", 48),
    "embedded:ixml:USER:SUBCATEGORY": ("子分类 (SUBCATEGORY)", 49),
    "embedded:tag:SUBCATEGORY": ("子分类 (SUBCATEGORY)", 50),
    "keywords": ("关键词 (Keywords)", 60),
    "embedded:ixml:USER:KEYWORDS": ("关键词 (KEYWORDS)", 61),
    "embedded:info:IKEY": ("关键词 (IKEY)", 62),
    "embedded:tag:KEYWORDS": ("关键词 (KEYWORDS)", 63),
    "description": ("描述 (Description)", 70),
    "embedded:ixml:USER:DESCRIPTION": ("描述 (DESCRIPTION)", 71),
    "embedded:info:ICMT": ("注释 (ICMT)", 72),
    "embedded:info:ISBJ": ("主题 (ISBJ)", 73),
    "embedded:tag:DESCRIPTION": ("描述 (DESCRIPTION)", 74),
    "embedded:tag:COMMENT": ("注释 (COMMENT)", 75),
    "notes": ("备注 (Notes)", 80),
    "embedded:ixml:USER:NOTES": ("备注 (NOTES)", 81),
    "embedded:ixml:NOTE": ("工程备注 (NOTE)", 82),
    "embedded:tag:NOTES": ("备注 (NOTES)", 83),
    "embedded:info:ISRC": ("来源 (ISRC)", 90),
    "embedded:info:ICOP": ("版权 (ICOP)", 91),
    "embedded:info:ICRD": ("创建日期 (ICRD)", 92),
    "embedded:info:IENG": ("工程师 (IENG)", 93),
    "embedded:info:ITCH": ("技术员 (ITCH)", 94),
    "embedded:info:ISRF": ("原始介质 (ISRF)", 95),
    "embedded:info:ISFT": ("写入软件 (ISFT)", 200),
    "embedded:ixml:IXML_VERSION": ("iXML 版本", 201),
    "embedded:bext": ("BWF (bext)", 210),
}

_SOURCE_PRIORITY = {"user": 3, "ai": 2, "imported_me": 1, "embedded": 0}
_HEX_RE = re.compile(r"^[0-9a-fA-F]{32,}$")
_SOFTWARE_NOISE_PREFIXES = (
    "pro tools",
    "reaper",
    "soundminer",
    "basehead",
    "nuendo",
    "cubase",
    "logic pro",
    "ableton",
    "adobe audition",
    "audition",
    "audiofinder",
    "soundly",
    "wavelab",
    "izotope",
    "avid",
    "sony",
    "steinberg",
    "audacity",
    "fl studio",
)


def _is_software_or_boilerplate(value: str, source: str) -> bool:
    """Reject DAW software names and copyright boilerplate in embedded metadata."""
    if source in ("user", "ai"):
        return False
    low = value.strip().casefold()
    if not low:
        return True
    if low.startswith(("©", "(c)")) or "all rights reserved" in low:
        return True
    return any(low == prefix or low.startswith(f"{prefix} ") for prefix in _SOFTWARE_NOISE_PREFIXES)


def _lookup_first(lookup: dict[str, tuple[str, str]], keys: tuple[str, ...]) -> str:
    for key in keys:
        entry = lookup.get(key)
        if entry is not None and entry[0].strip():
            return entry[0].strip()
    return ""


def _lookup_clean(lookup: dict[str, tuple[str, str]], keys: tuple[str, ...]) -> str:
    for key in keys:
        entry = lookup.get(key)
        if entry is None:
            continue
        val = entry[0].strip()
        if val and not _is_software_or_boilerplate(val, entry[1]):
            return val
    return ""


def _friendly_field_meta(field_name: str, value: str) -> tuple[str, int]:
    mapped = _INSPECTOR_LABEL_MAP.get(field_name)
    if mapped is not None:
        label, order = mapped
        if field_name == "embedded:bext" and _HEX_RE.match(value.strip()):
            return label, 300
        return label, order
    if field_name.startswith("embedded:bext:"):
        return f"BWF {field_name.split(':', 2)[-1]}", 150
    if field_name.startswith("embedded:info:"):
        tag = field_name.split(":", 2)[-1]
        return f"RIFF INFO ({tag})", 160
    if field_name.startswith("embedded:ixml:USER:"):
        tag = field_name.split(":", 3)[-1]
        return f"iXML ({tag})", 140
    if field_name.startswith("embedded:ixml:"):
        tag = field_name.split(":", 2)[-1]
        return f"iXML ({tag})", 170
    if field_name.startswith("embedded:tag:"):
        tag = field_name.split(":", 2)[-1]
        return f"标签 ({tag})", 165
    return field_name, 180


def extract_inspector_summary(
    rec: Any,
    fields: list[tuple[str, str, str]],
) -> InspectorSummary:
    """Extract promoted brand/package/UCS/path and readable field rows for the inspector."""
    lookup: dict[str, tuple[str, str]] = {}
    for field_name, value, source in fields:
        clean_val = str(value or "").strip()
        if not clean_val:
            continue
        src = str(source or "embedded")
        prio = _SOURCE_PRIORITY.get(src, 0)
        prev = lookup.get(field_name)
        if prev is None or prio > _SOURCE_PRIORITY.get(prev[1], 0):
            lookup[field_name] = (clean_val, src)

    package = _lookup_clean(
        lookup,
        (
            "library",
            "embedded:ixml:USER:LIBRARY",
            "embedded:tag:LIBRARY",
            "embedded:tag:ALBUM",
            "embedded:ixml:PROJECT",
        ),
    )

    brand = _lookup_clean(
        lookup,
        (
            "designer",
            "embedded:ixml:USER:DESIGNER",
            "embedded:info:IART",
            "embedded:tag:DESIGNER",
            "embedded:tag:ARTIST",
        ),
    )

    if brand and package and brand.casefold() == package.casefold():
        package = ""

    rel_path = str(getattr(rec, "rel_path", "") or "").replace("\\", "/").strip("/")
    parts = [p.strip() for p in rel_path.split("/") if p.strip()]
    path_segments = tuple(parts[:-1]) if len(parts) > 1 else ()

    from pathlib import Path

    root_path_str = str(getattr(rec, "root_path", "") or "").strip()
    root_label = (Path(root_path_str).name or root_path_str) if root_path_str else ""

    cat_id = _lookup_first(lookup, ("cat_id", "embedded:ixml:USER:CATID", "embedded:tag:CATID"))
    category = _lookup_first(
        lookup,
        (
            "category",
            "embedded:ixml:USER:CATEGORY",
            "embedded:info:IGNR",
            "embedded:tag:CATEGORY",
            "embedded:tag:GENRE",
        ),
    )
    subcategory = _lookup_first(
        lookup,
        ("subcategory", "embedded:ixml:USER:SUBCATEGORY", "embedded:tag:SUBCATEGORY"),
    )
    cat_part = " / ".join(part for part in (category, subcategory) if part)
    if cat_id and cat_part:
        ucs_label = f"{cat_id} · {cat_part}"
    else:
        ucs_label = cat_id or cat_part

    raw_full = getattr(rec, "full_path", "")
    folder_path = str(Path(raw_full).parent) if raw_full else ""

    ranked_rows: list[tuple[int, str, str, str, str]] = []
    for field_name, value, source in fields:
        clean_val = str(value or "").strip()
        if not clean_val:
            continue
        label, order = _friendly_field_meta(field_name, clean_val)
        ranked_rows.append((order, field_name, label, clean_val, str(source or "embedded")))
    ranked_rows.sort(key=lambda item: (item[0], item[1]))
    formatted_fields = tuple(
        (label, clean_val, src) for _order, _fn, label, clean_val, src in ranked_rows
    )

    return InspectorSummary(
        brand=brand,
        package=package,
        ucs_label=ucs_label,
        folder_path=folder_path,
        formatted_fields=formatted_fields,
        path_segments=path_segments,
        root_label=root_label,
    )
