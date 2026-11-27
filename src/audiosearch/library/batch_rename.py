"""R5b template-driven batch file renaming with collision resolution & two-phase swap-safe Undo."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiosearch.library.store import FileRecord, LibraryStore

RENAME_PRESETS: tuple[tuple[str, str], ...] = (
    ("UCS 标准（含序号）", "{CatID}_{FXName}_{Designer}_{Library}_{Index:02d}"),
    ("UCS 简洁", "{CatID}_{FXName}_{Index:02d}"),
    ("分类前缀 + 原文件名", "{CatID}_{OrigName}"),
    ("素材库 + 原名 + 序号", "{Library}_{OrigName}_{Index:03d}"),
)

PLACEHOLDER_HELP: tuple[tuple[str, str], ...] = (
    ("{CatID}", "UCS 编号"),
    ("{Category}", "主分类"),
    ("{SubCategory}", "子分类"),
    ("{FXName}", "音效名称"),
    ("{Description}", "描述"),
    ("{Designer}", "设计师"),
    ("{Library}", "素材库"),
    ("{Notes}", "备注"),
    ("{OrigName}", "原文件名(不含扩展名)"),
    ("{Folder}", "直接父目录"),
    ("{Ext}", "扩展名"),
    ("{Index:02d}", "两位序号(01,02...)"),
)

_WINDOWS_ILLEGAL_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
)
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)(?::([^}]*))?\}")


@dataclass(frozen=True, slots=True)
class RenamePreviewItem:
    file_id: int
    old_path: Path
    new_path: Path
    old_filename: str
    new_filename: str
    old_rel_path: str
    new_rel_path: str
    status: str  # "ready" | "unchanged" | "conflict" | "skipped"
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RenamePreview:
    items: tuple[RenamePreviewItem, ...]
    ready_count: int
    conflict_count: int
    skipped_count: int


@dataclass(frozen=True, slots=True)
class RenameOutcome:
    file_id: int
    old_filename: str
    new_filename: str
    success: bool
    message: str


@dataclass(frozen=True, slots=True)
class RenameBatchResult:
    batch_id: str
    status: str  # "committed" | "conflict" | "empty" | "failed"
    manifest_path: str
    renamed: tuple[RenameOutcome, ...]
    failed: tuple[RenameOutcome, ...]
    skipped: tuple[RenamePreviewItem, ...]


@dataclass(frozen=True, slots=True)
class UndoRenameResult:
    batch_id: str
    status: str
    restored: tuple[RenameOutcome, ...]
    failed: tuple[RenameOutcome, ...]


def sanitize_filename_stem(raw: str, *, space_mode: str = "keep") -> str:
    """Sanitize a rendered stem for Windows filesystems while preserving readable separators."""
    text = str(raw or "")
    text = _WINDOWS_ILLEGAL_RE.sub("_", text)
    if space_mode == "underscore":
        text = re.sub(r"\s+", "_", text)
    elif space_mode == "camel":
        words = [w for w in re.split(r"\s+", text) if w]
        text = "".join(w[:1].upper() + w[1:] if len(w) > 1 else w.upper() for w in words)
    else:
        text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[_\s]*_[_\s]*", "_", text)
    text = text.strip(" ._")
    if len(text) > 180:
        text = text[:180].rstrip(" ._")
    if text.upper() in _WINDOWS_RESERVED:
        text = f"_{text}"
    return text


def render_rename_template(
    template: str,
    *,
    canonical: dict[str, str],
    rec: FileRecord,
    index: int,
    space_mode: str = "keep",
) -> str:
    """Render filename (preserving original suffix) from a placeholder template."""
    orig_p = Path(rec.filename)
    orig_stem = orig_p.stem
    ext_suffix = orig_p.suffix
    parent_rel = Path(rec.rel_path).parent
    folder_name = parent_rel.name if str(parent_rel) not in ("", ".") else ""

    values: dict[str, Any] = {
        "catid": canonical.get("cat_id", ""),
        "category": canonical.get("category", ""),
        "subcategory": canonical.get("subcategory", ""),
        "fxname": canonical.get("fx_name", "") or canonical.get("description", ""),
        "description": canonical.get("description", "") or canonical.get("fx_name", ""),
        "designer": canonical.get("designer", ""),
        "creatorid": canonical.get("designer", ""),
        "library": canonical.get("library", "") or rec.package,
        "sourceid": canonical.get("library", "") or rec.package,
        "package": rec.package,
        "notes": canonical.get("notes", ""),
        "origname": orig_stem,
        "stem": orig_stem,
        "folder": folder_name,
        "ext": ext_suffix.lstrip("."),
        "index": int(index),
    }

    def _sub(match: re.Match[str]) -> str:
        key = match.group(1).casefold()
        fmt = match.group(2)
        if key not in values:
            return match.group(0)
        val = values[key]
        if key == "index" and fmt:
            try:
                return format(int(val), fmt)
            except ValueError:
                return str(val)
        return str(val or "")

    rendered = _PLACEHOLDER_RE.sub(_sub, template.strip())
    # Clean leftover empty separator segments caused by missing fields (e.g., "_FX_" -> "FX").
    stem = sanitize_filename_stem(rendered, space_mode=space_mode)
    if not stem:
        return ""
    return f"{stem}{ext_suffix}"


def _build_rel_path(old_rel_path: str, new_filename: str) -> str:
    parent = Path(old_rel_path).parent
    if str(parent) in ("", "."):
        return new_filename
    return f"{parent.as_posix()}/{new_filename}"


def _can_write_file(path: Path) -> bool:
    return path.exists() and path.is_file() and os.access(path, os.W_OK)


def build_rename_preview(
    store: LibraryStore,
    file_ids: list[int],
    template: str,
    *,
    space_mode: str = "keep",
    collision_policy: str = "error",  # "error" | "auto_suffix"
    start_index: int = 1,
) -> RenamePreview:
    """Compute a Dry-run batch rename plan checking filesystem and in-batch collisions."""
    unique_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
    if not unique_ids:
        return RenamePreview(items=(), ready_count=0, conflict_count=0, skipped_count=0)

    recs = store.get_files_by_ids(unique_ids)
    raw_items: list[tuple[FileRecord, Path, str]] = []
    preliminary: list[RenamePreviewItem] = []

    idx_counter = int(start_index)
    for fid in unique_ids:
        rec = recs.get(fid)
        if rec is None:
            continue
        old_path = rec.full_path
        if not rec.is_available or not old_path.exists():
            preliminary.append(
                RenamePreviewItem(
                    fid, old_path, old_path, rec.filename, rec.filename, rec.rel_path, rec.rel_path,
                    "skipped", "文件离线或不存在",
                )
            )
            continue
        if not _can_write_file(old_path):
            preliminary.append(
                RenamePreviewItem(
                    fid, old_path, old_path, rec.filename, rec.filename, rec.rel_path, rec.rel_path,
                    "skipped", "文件只读或无写入权限",
                )
            )
            continue
        canon_full = store.get_canonical_metadata(fid)
        canon_map = {k: v for k, (v, _s, _d) in canon_full.items()}
        rendered_name = render_rename_template(
            template, canonical=canon_map, rec=rec, index=idx_counter, space_mode=space_mode
        )
        idx_counter += 1
        if not rendered_name:
            preliminary.append(
                RenamePreviewItem(
                    fid, old_path, old_path, rec.filename, rec.filename, rec.rel_path, rec.rel_path,
                    "conflict", "模板渲染结果为空",
                )
            )
            continue
        raw_items.append((rec, old_path, rendered_name))

    old_path_keys = {str(op).casefold() for _r, op, _n in raw_items}
    used_targets: set[str] = set()
    resolved_items: list[RenamePreviewItem] = list(preliminary)

    # First pass for unchanged vs target assignment
    for rec, old_path, candidate_name in raw_items:
        parent_dir = old_path.parent
        candidate_path = parent_dir / candidate_name
        candidate_key = str(candidate_path).casefold()

        def _is_taken(path_obj: Path, p_key: str) -> bool:
            if p_key in used_targets:
                return True
            if path_obj.exists() and p_key not in old_path_keys:
                return True
            return False

        if _is_taken(candidate_path, candidate_key):
            if collision_policy == "auto_suffix":
                stem = Path(candidate_name).stem
                suffix = Path(candidate_name).suffix
                seq = 1
                while True:
                    alt_name = f"{stem}_{seq:02d}{suffix}"
                    alt_path = parent_dir / alt_name
                    alt_key = str(alt_path).casefold()
                    if not _is_taken(alt_path, alt_key):
                        candidate_name = alt_name
                        candidate_path = alt_path
                        candidate_key = alt_key
                        break
                    seq += 1
            else:
                reason = (
                    "目标文件名与批次内其他文件重名"
                    if candidate_key in used_targets
                    else "磁盘已存在同名非本批次文件"
                )
                resolved_items.append(
                    RenamePreviewItem(
                        rec.id,
                        old_path,
                        candidate_path,
                        rec.filename,
                        candidate_name,
                        rec.rel_path,
                        _build_rel_path(rec.rel_path, candidate_name),
                        "conflict",
                        reason,
                    )
                )
                continue

        used_targets.add(candidate_key)
        if candidate_name == rec.filename:
            resolved_items.append(
                RenamePreviewItem(
                    rec.id, old_path, candidate_path, rec.filename, candidate_name,
                    rec.rel_path, rec.rel_path, "unchanged", "与原文件名一致",
                )
            )
        else:
            resolved_items.append(
                RenamePreviewItem(
                    rec.id,
                    old_path,
                    candidate_path,
                    rec.filename,
                    candidate_name,
                    rec.rel_path,
                    _build_rel_path(rec.rel_path, candidate_name),
                    "ready",
                    "",
                )
            )

    ready_count = sum(1 for x in resolved_items if x.status == "ready")
    conflict_count = sum(1 for x in resolved_items if x.status == "conflict")
    skipped_count = sum(1 for x in resolved_items if x.status in ("skipped", "unchanged"))
    return RenamePreview(
        items=tuple(resolved_items),
        ready_count=ready_count,
        conflict_count=conflict_count,
        skipped_count=skipped_count,
    )


def _two_phase_rename(pairs: list[tuple[Path, Path]]) -> None:
    """Perform cyclic-swap-safe and Windows case-only-safe batch renames with full rollback."""
    if not pairs:
        return
    token = uuid.uuid4().hex[:8]
    staged: list[tuple[Path, Path, Path]] = []  # (old_p, tmp_p, new_p)
    try:
        # Phase 1: old -> unique tmp in same folder
        for idx, (old_p, new_p) in enumerate(pairs):
            tmp_p = old_p.with_name(f".__ren_{token}_{idx}_{old_p.name}")
            os.rename(old_p, tmp_p)
            staged.append((old_p, tmp_p, new_p))
    except Exception:
        for old_p, tmp_p, _new_p in reversed(staged):
            try:
                if tmp_p.exists():
                    os.rename(tmp_p, old_p)
            except OSError:
                pass
        raise

    moved_final: list[tuple[Path, Path, Path]] = []
    try:
        # Phase 2: tmp -> final new_p
        for old_p, tmp_p, new_p in staged:
            os.rename(tmp_p, new_p)
            moved_final.append((old_p, tmp_p, new_p))
    except Exception:
        for old_p, _tmp_p, new_p in reversed(moved_final):
            try:
                if new_p.exists():
                    os.rename(new_p, old_p)
            except OSError:
                pass
        for old_p, tmp_p, _new_p in reversed(staged):
            try:
                if tmp_p.exists():
                    os.rename(tmp_p, old_p)
            except OSError:
                pass
        raise


def execute_batch_rename(
    store: LibraryStore,
    file_ids: list[int],
    template: str,
    *,
    space_mode: str = "keep",
    collision_policy: str = "error",
    start_index: int = 1,
    backup_root: Path | None = None,
) -> RenameBatchResult:
    """Execute validated batch rename, sync SQLite preserving ``file_id``, and save Undo manifest."""
    preview = build_rename_preview(
        store,
        file_ids,
        template,
        space_mode=space_mode,
        collision_policy=collision_policy,
        start_index=start_index,
    )
    b_root = backup_root or (store.db_path.parent / "metadata-backups" / "renames")
    batch_id = time.strftime("ren_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    batch_dir = b_root / batch_id
    manifest_path = batch_dir / "rename_manifest.json"
    skipped = tuple(x for x in preview.items if x.status in ("skipped", "unchanged"))

    if preview.conflict_count > 0:
        conflicts = tuple(
            RenameOutcome(x.file_id, x.old_filename, x.new_filename, False, x.reason)
            for x in preview.items
            if x.status == "conflict"
        )
        return RenameBatchResult(
            batch_id=batch_id,
            status="conflict",
            manifest_path="",
            renamed=(),
            failed=conflicts,
            skipped=skipped,
        )

    ready = [x for x in preview.items if x.status == "ready"]
    if not ready:
        return RenameBatchResult(
            batch_id=batch_id,
            status="empty",
            manifest_path="",
            renamed=(),
            failed=(),
            skipped=skipped,
        )

    batch_dir.mkdir(parents=True, exist_ok=True)
    store.create_rename_batch(
        batch_id,
        file_count=len(ready),
        template=template,
        manifest_path=str(manifest_path),
        status="pending",
    )

    try:
        _two_phase_rename([(x.old_path, x.new_path) for x in ready])
    except Exception as exc:
        store.update_rename_batch_status(batch_id, "failed", file_count=0)
        return RenameBatchResult(
            batch_id=batch_id,
            status="failed",
            manifest_path=str(manifest_path),
            renamed=(),
            failed=tuple(
                RenameOutcome(x.file_id, x.old_filename, x.new_filename, False, str(exc))
                for x in ready
            ),
            skipped=skipped,
        )

    store.apply_file_renames_db([(x.file_id, x.new_rel_path, x.new_filename) for x in ready])
    manifest_data = {
        "batch_id": batch_id,
        "created_at": time.time(),
        "template": template,
        "entries": [
            {
                "file_id": x.file_id,
                "old_path": str(x.old_path),
                "new_path": str(x.new_path),
                "old_filename": x.old_filename,
                "new_filename": x.new_filename,
                "old_rel_path": x.old_rel_path,
                "new_rel_path": x.new_rel_path,
            }
            for x in ready
        ],
    }
    manifest_path.write_text(json.dumps(manifest_data, ensure_ascii=False, indent=2), encoding="utf-8")
    store.update_rename_batch_status(batch_id, "committed", file_count=len(ready))

    return RenameBatchResult(
        batch_id=batch_id,
        status="committed",
        manifest_path=str(manifest_path),
        renamed=tuple(
            RenameOutcome(x.file_id, x.old_filename, x.new_filename, True, "改名成功")
            for x in ready
        ),
        failed=(),
        skipped=skipped,
    )


def undo_rename_batch(store: LibraryStore, batch_id: str) -> UndoRenameResult:
    """Undo a committed batch rename atomically using two-phase swap-safe restoration."""
    batch = store.get_rename_batch(batch_id)
    if batch is None:
        raise ValueError(f"未找到改名批次：{batch_id}")
    if batch.status == "reverted":
        raise ValueError("该改名批次已撤销")

    manifest_path = Path(batch.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"改名清单不存在：{manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = list(data.get("entries", []))

    new_keys = {str(Path(e["new_path"])).casefold() for e in entries}
    for e in entries:
        cur_p = Path(e["new_path"])
        old_p = Path(e["old_path"])
        if not cur_p.exists():
            raise FileNotFoundError(f"目标文件已不存在，无法撤销：{cur_p.name}")
        if old_p.exists() and str(old_p).casefold() not in new_keys:
            raise FileExistsError(f"原路径已被其他文件占用：{old_p.name}")

    _two_phase_rename([(Path(e["new_path"]), Path(e["old_path"])) for e in entries])
    store.apply_file_renames_db(
        [(int(e["file_id"]), str(e["old_rel_path"]), str(e["old_filename"])) for e in entries]
    )
    store.update_rename_batch_status(batch_id, "reverted")

    return UndoRenameResult(
        batch_id=batch_id,
        status="reverted",
        restored=tuple(
            RenameOutcome(
                int(e["file_id"]), str(e["new_filename"]), str(e["old_filename"]), True, "已恢复原文件名"
            )
            for e in entries
        ),
        failed=(),
    )


__all__ = [
    "PLACEHOLDER_HELP",
    "RENAME_PRESETS",
    "RenameBatchResult",
    "RenameOutcome",
    "RenamePreview",
    "RenamePreviewItem",
    "UndoRenameResult",
    "build_rename_preview",
    "execute_batch_rename",
    "render_rename_template",
    "sanitize_filename_stem",
    "undo_rename_batch",
]
