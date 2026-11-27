"""R5b CSV metadata export and non-destructive preview/import helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from audiosearch.library.metadata_fields import EDITABLE_FIELD_KEYS, FIELD_SPEC_MAP
from audiosearch.library.store import FileRecord, LibraryStore

CSV_COLUMNS: tuple[str, ...] = (
    "file_id",
    "rel_path",
    "filename",
    "ext",
    "duration_sec",
    *EDITABLE_FIELD_KEYS,
    "favorite",
    "user_note",
)

IMPORTABLE_TARGETS: tuple[str, ...] = (*EDITABLE_FIELD_KEYS, "favorite", "user_note")

_HEADER_ALIASES: dict[str, str] = {
    "file_id": "file_id",
    "id": "file_id",
    "rel_path": "rel_path",
    "path": "rel_path",
    "路径": "rel_path",
    "相对路径": "rel_path",
    "filename": "filename",
    "file": "filename",
    "name": "filename",
    "名称": "filename",
    "文件名": "filename",
    "ext": "ext",
    "格式": "ext",
    "duration_sec": "duration_sec",
    "duration": "duration_sec",
    "length": "duration_sec",
    "时长": "duration_sec",
    "description": "description",
    "desc": "description",
    "comment": "description",
    "comments": "description",
    "描述": "description",
    "keywords": "keywords",
    "keyword": "keywords",
    "tags": "keywords",
    "关键词": "keywords",
    "cat_id": "cat_id",
    "catid": "cat_id",
    "ucs_catid": "cat_id",
    "ucs编号": "cat_id",
    "category": "category",
    "主分类": "category",
    "subcategory": "subcategory",
    "sub_category": "subcategory",
    "子分类": "subcategory",
    "fx_name": "fx_name",
    "fxname": "fx_name",
    "title": "fx_name",
    "音效名称": "fx_name",
    "library": "library",
    "package": "library",
    "album": "library",
    "素材库": "library",
    "素材包": "library",
    "designer": "designer",
    "creator": "designer",
    "creatorid": "designer",
    "artist": "designer",
    "设计师": "designer",
    "notes": "notes",
    "元数据备注": "notes",
    "favorite": "favorite",
    "fav": "favorite",
    "收藏": "favorite",
    "user_note": "user_note",
    "local_note": "user_note",
    "本地备注": "user_note",
}


@dataclass(frozen=True, slots=True)
class CsvExportResult:
    path: Path
    row_count: int


@dataclass(frozen=True, slots=True)
class CsvFieldChange:
    field: str
    label: str
    before: str
    after: str


@dataclass(frozen=True, slots=True)
class CsvImportRowPreview:
    row_num: int
    file_id: int
    filename: str
    changes: tuple[CsvFieldChange, ...]


@dataclass(frozen=True, slots=True)
class CsvImportSkip:
    row_num: int
    identifier: str
    reason: str


@dataclass(frozen=True, slots=True)
class CsvImportPreview:
    headers: tuple[str, ...]
    column_map: dict[str, str]
    matched: tuple[CsvImportRowPreview, ...]
    skipped: tuple[CsvImportSkip, ...]
    total_rows: int


@dataclass(frozen=True, slots=True)
class CsvImportResult:
    updated_files: int
    updated_fields: int
    skipped: tuple[CsvImportSkip, ...]


def _iter_export_records(store: LibraryStore, file_ids: list[int] | None) -> list[FileRecord]:
    if file_ids is not None:
        unique_ids = list(dict.fromkeys(int(x) for x in file_ids))
        recs = store.get_files_by_ids(unique_ids)
        return [recs[fid] for fid in unique_ids if fid in recs]
    all_recs: list[FileRecord] = []
    for batch in store.iter_files_for_indexing(online_only=False, batch_size=500):
        # Fetch full rows (favorite/note/dirty) cleanly via get_files_by_ids
        recs = store.get_files_by_ids([r.id for r in batch])
        all_recs.extend(recs[r.id] for r in batch if r.id in recs)
    return all_recs


def export_metadata_csv(
    store: LibraryStore,
    csv_path: Path | str,
    file_ids: list[int] | None = None,
) -> CsvExportResult:
    """Export file metadata to Excel-friendly UTF-8-BOM CSV."""
    target = Path(csv_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    records = _iter_export_records(store, file_ids)

    with target.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for rec in records:
            canon = store.get_canonical_metadata(rec.id)
            row = [
                str(rec.id),
                rec.rel_path,
                rec.filename,
                rec.ext,
                f"{rec.duration_sec:.3f}",
                *(canon.get(k, ("", "", False))[0] for k in EDITABLE_FIELD_KEYS),
                "1" if rec.favorite else "0",
                rec.note,
            ]
            writer.writerow(row)
    return CsvExportResult(path=target, row_count=len(records))


def detect_csv_column_map(headers: list[str]) -> dict[str, str]:
    """Map raw CSV headers to canonical field/identifier names."""
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    for h in headers:
        norm = re.sub(r"[\s/_-]+", "", h.strip().casefold())
        exact = h.strip().casefold()
        target = _HEADER_ALIASES.get(exact) or _HEADER_ALIASES.get(norm)
        if target and target not in used_targets:
            mapping[h] = target
            used_targets.add(target)
    return mapping


def read_csv_rows(csv_path: Path | str) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV trying UTF-8-BOM, UTF-8, then Windows GB18030."""
    target = Path(csv_path)
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = target.read_text(encoding=enc)
            lines = text.splitlines()
            if not lines:
                return [], []
            reader = csv.DictReader(lines)
            headers = list(reader.fieldnames or [])
            rows = [{k: str(v or "") for k, v in row.items() if k is not None} for row in reader]
            return headers, rows
        except UnicodeDecodeError as exc:
            last_err = exc
    raise ValueError(f"无法解码 CSV 文件编码：{last_err}")


def _build_lookup_tables(store: LibraryStore) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    by_rel: dict[str, list[int]] = {}
    by_name: dict[str, list[int]] = {}
    for batch in store.iter_files_for_indexing(online_only=False, batch_size=500):
        for r in batch:
            rel_key = r.rel_path.replace("\\", "/").casefold()
            by_rel.setdefault(rel_key, []).append(r.id)
            by_name.setdefault(r.filename.casefold(), []).append(r.id)
    return by_rel, by_name


def _parse_bool_cell(val: str) -> bool | None:
    low = val.strip().casefold()
    if low in ("1", "true", "yes", "y", "是", "★"):
        return True
    if low in ("0", "false", "no", "n", "否", "☆"):
        return False
    return None


def preview_metadata_csv_import(
    store: LibraryStore,
    csv_path: Path | str,
    *,
    column_map: dict[str, str] | None = None,
    overwrite_empty: bool = False,
) -> CsvImportPreview:
    """Match CSV rows to DB records and compute non-destructive field diffs."""
    headers, rows = read_csv_rows(csv_path)
    resolved_map = dict(column_map) if column_map is not None else detect_csv_column_map(headers)
    inv_map = {target: hdr for hdr, target in resolved_map.items()}

    by_rel, by_name = _build_lookup_tables(store)

    # Pre-resolve matching file_ids so we can batch fetch FileRecords
    pending_matches: list[tuple[int, int, dict[str, str]]] = []
    skipped: list[CsvImportSkip] = []

    for row_idx, row in enumerate(rows, start=2):
        fid: int | None = None
        ident = ""

        if "file_id" in inv_map:
            raw_id = row.get(inv_map["file_id"], "").strip()
            if raw_id:
                ident = raw_id
                try:
                    cand_id = int(raw_id)
                    if store.get_file(cand_id) is not None:
                        fid = cand_id
                except ValueError:
                    fid = None

        if fid is None and "rel_path" in inv_map:
            raw_rel = row.get(inv_map["rel_path"], "").strip().replace("\\", "/")
            if raw_rel:
                ident = ident or raw_rel
                cands = by_rel.get(raw_rel.casefold(), [])
                if len(cands) == 1:
                    fid = cands[0]
                elif len(cands) > 1:
                    skipped.append(CsvImportSkip(row_idx, ident, "相对路径命中多条记录，无法唯一匹配"))
                    continue

        if fid is None and "filename" in inv_map:
            raw_name = row.get(inv_map["filename"], "").strip()
            if raw_name:
                ident = ident or raw_name
                cands = by_name.get(raw_name.casefold(), [])
                if len(cands) == 1:
                    fid = cands[0]
                elif len(cands) > 1:
                    skipped.append(CsvImportSkip(row_idx, ident, "文件名在库中存在重名，需提供 file_id 或 rel_path"))
                    continue

        if fid is None:
            skipped.append(CsvImportSkip(row_idx, ident or "(空)", "未能在库中找到对应音频记录"))
            continue

        pending_matches.append((row_idx, fid, row))

    recs = store.get_files_by_ids([fid for _rn, fid, _row in pending_matches])
    matched: list[CsvImportRowPreview] = []

    for row_idx, fid, row in pending_matches:
        rec = recs.get(fid)
        if rec is None:
            skipped.append(CsvImportSkip(row_idx, str(fid), "库记录不存在"))
            continue
        canon = store.get_canonical_metadata(fid)
        changes: list[CsvFieldChange] = []

        for hdr, target in resolved_map.items():
            if target not in IMPORTABLE_TARGETS:
                continue
            new_val = row.get(hdr, "").strip()
            if not new_val and not overwrite_empty:
                continue
            if target in FIELD_SPEC_MAP:
                before_val = canon.get(target, ("", "embedded", False))[0]
                if new_val != before_val:
                    changes.append(
                        CsvFieldChange(
                            field=target,
                            label=FIELD_SPEC_MAP[target].label_zh,
                            before=before_val,
                            after=new_val,
                        )
                    )
            elif target == "favorite":
                parsed_fav = _parse_bool_cell(new_val)
                if parsed_fav is not None and parsed_fav != rec.favorite:
                    changes.append(
                        CsvFieldChange(
                            field="favorite",
                            label="收藏",
                            before="是" if rec.favorite else "否",
                            after="是" if parsed_fav else "否",
                        )
                    )
            elif target == "user_note":
                if new_val != rec.note:
                    changes.append(
                        CsvFieldChange(
                            field="user_note",
                            label="本地备注",
                            before=rec.note,
                            after=new_val,
                        )
                    )

        if changes:
            matched.append(
                CsvImportRowPreview(
                    row_num=row_idx,
                    file_id=fid,
                    filename=rec.filename,
                    changes=tuple(changes),
                )
            )

    return CsvImportPreview(
        headers=tuple(headers),
        column_map=resolved_map,
        matched=tuple(matched),
        skipped=tuple(skipped),
        total_rows=len(rows),
    )


def apply_metadata_csv_import(
    store: LibraryStore,
    csv_path: Path | str,
    *,
    column_map: dict[str, str] | None = None,
    overwrite_empty: bool = False,
) -> CsvImportResult:
    """Import CSV metadata diffs into local DB only (dirty=1, never modifies audio files)."""
    preview = preview_metadata_csv_import(
        store,
        csv_path,
        column_map=column_map,
        overwrite_empty=overwrite_empty,
    )
    updated_files = 0
    updated_fields = 0
    for item in preview.matched:
        meta_updates: dict[str, str] = {}
        fav_val: bool | None = None
        note_val: str | None = None
        for ch in item.changes:
            if ch.field in FIELD_SPEC_MAP:
                meta_updates[ch.field] = ch.after
            elif ch.field == "favorite":
                fav_val = ch.after == "是"
            elif ch.field == "user_note":
                note_val = ch.after
        if meta_updates:
            store.update_file_metadata(item.file_id, meta_updates, source="user")
        if fav_val is not None:
            store.set_favorites([item.file_id], fav_val)
        if note_val is not None:
            store.set_note(item.file_id, note_val)
        if meta_updates or fav_val is not None or note_val is not None:
            updated_files += 1
            updated_fields += len(item.changes)

    return CsvImportResult(
        updated_files=updated_files,
        updated_fields=updated_fields,
        skipped=preview.skipped,
    )


__all__ = [
    "CSV_COLUMNS",
    "CsvExportResult",
    "CsvFieldChange",
    "CsvImportPreview",
    "CsvImportResult",
    "CsvImportRowPreview",
    "CsvImportSkip",
    "IMPORTABLE_TARGETS",
    "apply_metadata_csv_import",
    "detect_csv_column_map",
    "export_metadata_csv",
    "preview_metadata_csv_import",
    "read_csv_rows",
]
