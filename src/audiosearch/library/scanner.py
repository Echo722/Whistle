"""Incremental background audio library scanner with pause/resume/cancel."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from audiosearch.audioio.video import VIDEO_EXTENSIONS, is_video_path, probe_video
from audiosearch.audioio.wav import read_wav_info
from audiosearch.catalog.embedded import EmbeddedMetadata, read_embedded_metadata
from audiosearch.library.schema import ensure_schema, open_connection

AUDIO_EXTENSIONS = frozenset({".wav", ".aif", ".aiff", ".flac", ".ogg", ".mp3", ".opus"})
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
BATCH_SIZE = 500
_PROGRESS_INTERVAL_SEC = 0.15

_SUBTYPE_BITS: dict[str, int] = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ALAW": 8,
    "ULAW": 8,
}


@dataclass(frozen=True, slots=True)
class ScanProgress:
    root_id: int
    scanned: int
    added: int
    updated: int
    skipped: int
    errors: int
    current_dir: str
    status: str  # scanning | paused | completed | cancelled | offline


@dataclass(frozen=True, slots=True)
class ScanReport:
    root_id: int
    scanned: int
    added: int
    updated: int
    skipped: int
    removed: int
    errors: int
    status: str


class LibraryScanner:
    """Walks a root directory and incrementally updates SQLite + FTS5."""

    def __init__(self, db_path: str | Path, *, include_videos: bool = True) -> None:
        self.db_path = Path(db_path)
        self._include_videos = bool(include_videos)
        self._pause_clear = threading.Event()
        self._pause_clear.set()  # set => running, clear => paused
        self._cancel = threading.Event()

    def pause(self) -> None:
        self._pause_clear.clear()

    def resume(self) -> None:
        self._pause_clear.set()

    def cancel(self) -> None:
        self._cancel.set()
        self._pause_clear.set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_clear.is_set()

    def scan_root(
        self,
        root_id: int,
        *,
        on_progress: Callable[[ScanProgress], None] | None = None,
    ) -> ScanReport:
        self._cancel.clear()
        self._pause_clear.set()
        conn = open_connection(self.db_path)
        ensure_schema(conn)
        try:
            row = conn.execute("SELECT * FROM roots WHERE id = ?", (root_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown root_id: {root_id}")
            root_path = Path(str(row["path"]))
            root_label = str(row["label"])

            if not root_path.exists() or not root_path.is_dir():
                conn.execute(
                    "UPDATE roots SET is_online = 0, status = 'offline' WHERE id = ?",
                    (root_id,),
                )
                conn.execute("UPDATE files SET is_available = 0 WHERE root_id = ?", (root_id,))
                conn.commit()
                if on_progress:
                    on_progress(
                        ScanProgress(root_id, 0, 0, 0, 0, 0, str(root_path), "offline")
                    )
                return ScanReport(root_id, 0, 0, 0, 0, 0, 0, "offline")

            conn.execute(
                "UPDATE roots SET is_online = 1, status = 'scanning' WHERE id = ?",
                (root_id,),
            )
            conn.commit()

            existing = _load_existing_files(conn, root_id)
            folders_by_rel = _load_existing_folders(conn, root_id)
            seen_rel_paths: set[str] = set()
            pending_rows: list[_ScannedFile] = []

            scanned = added = updated = skipped = errors = 0
            last_progress_at = 0.0
            queue: deque[tuple[Path, str, int | None]] = deque([(root_path, "", None)])

            def emit_progress(cur_dir_str: str, *, force: bool = False) -> None:
                nonlocal last_progress_at
                if on_progress is None:
                    return
                now_mono = time.monotonic()
                if not force and scanned != 1 and (now_mono - last_progress_at) < _PROGRESS_INTERVAL_SEC:
                    return
                last_progress_at = now_mono
                on_progress(
                    ScanProgress(
                        root_id,
                        scanned,
                        added,
                        updated,
                        skipped,
                        errors,
                        cur_dir_str,
                        "scanning",
                    )
                )

            while queue:
                if self._cancel.is_set():
                    break
                self._wait_if_paused(conn, root_id, on_progress, scanned, added, updated, skipped, errors, str(queue[0][0]))
                if self._cancel.is_set():
                    break

                cur_dir, cur_rel, cur_folder_id = queue.popleft()
                try:
                    entries = list(os.scandir(cur_dir))
                except OSError:
                    errors += 1
                    continue

                allowed_exts = MEDIA_EXTENSIONS if self._include_videos else AUDIO_EXTENSIONS
                subdirs: list[os.DirEntry[str]] = []
                audio_entries: list[os.DirEntry[str]] = []
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.name.startswith("._"):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            subdirs.append(entry)
                        elif entry.is_file(follow_symlinks=False):
                            ext = Path(entry.name).suffix.lower()
                            if ext in allowed_exts:
                                audio_entries.append(entry)
                    except OSError:
                        errors += 1

                subdirs.sort(key=lambda e: e.name.casefold())
                missing_subs = [
                    (sub, f"{cur_rel}/{sub.name}" if cur_rel else sub.name)
                    for sub in subdirs
                    if (f"{cur_rel}/{sub.name}" if cur_rel else sub.name) not in folders_by_rel
                ]
                if missing_subs:
                    with conn:
                        for sub, sub_rel in missing_subs:
                            folders_by_rel[sub_rel] = _insert_folder(
                                conn, root_id, cur_folder_id, sub_rel, sub.name
                            )
                for sub in subdirs:
                    sub_rel = f"{cur_rel}/{sub.name}" if cur_rel else sub.name
                    queue.append((Path(sub.path), sub_rel, folders_by_rel[sub_rel]))

                for entry in audio_entries:
                    if self._cancel.is_set():
                        break
                    self._wait_if_paused(conn, root_id, on_progress, scanned, added, updated, skipped, errors, str(cur_dir))
                    if self._cancel.is_set():
                        break

                    scanned += 1
                    rel_path = f"{cur_rel}/{entry.name}" if cur_rel else entry.name
                    seen_rel_paths.add(rel_path)

                    try:
                        st = entry.stat(follow_symlinks=False)
                    except OSError:
                        errors += 1
                        continue

                    prev = existing.get(rel_path)
                    if (
                        prev is not None
                        and prev[1] == st.st_size
                        and prev[2] == st.st_mtime_ns
                        and prev[3] == 1
                        and prev[4] == cur_folder_id
                    ):
                        skipped += 1
                    else:
                        record = _inspect_audio_file(
                            Path(entry.path),
                            root_id=root_id,
                            folder_id=cur_folder_id,
                            rel_path=rel_path,
                            root_label=root_label,
                            size_bytes=st.st_size,
                            mtime_ns=st.st_mtime_ns,
                            existing_id=None if prev is None else prev[0],
                        )
                        if record is None:
                            errors += 1
                        else:
                            if prev is None:
                                added += 1
                            else:
                                updated += 1
                            pending_rows.append(record)

                    if len(pending_rows) >= BATCH_SIZE:
                        _flush_batch(conn, pending_rows)
                        pending_rows.clear()
                        emit_progress(str(cur_dir))
                    elif scanned == 1 or (on_progress is not None and scanned % 500 == 0):
                        emit_progress(str(cur_dir), force=(scanned == 1))

                if on_progress and audio_entries:
                    emit_progress(str(cur_dir))

            if pending_rows:
                _flush_batch(conn, pending_rows)
                pending_rows.clear()

            removed = 0
            if not self._cancel.is_set():
                missing_ids = [
                    info[0] for rp, info in existing.items() if rp not in seen_rel_paths
                ]
                if missing_ids:
                    for i in range(0, len(missing_ids), 500):
                        chunk = missing_ids[i : i + 500]
                        placeholders = ",".join("?" for _ in chunk)
                        conn.execute(f"DELETE FROM files WHERE id IN ({placeholders})", chunk)
                    removed = len(missing_ids)

            if on_progress is not None and not self._cancel.is_set():
                on_progress(
                    ScanProgress(
                        root_id,
                        scanned,
                        added,
                        updated,
                        skipped,
                        errors,
                        str(root_path),
                        "finalizing",
                    )
                )

            _recompute_counts(conn, root_id)
            final_status = "cancelled" if self._cancel.is_set() else "idle"
            conn.execute(
                "UPDATE roots SET status = ?, last_scan_at = ? WHERE id = ?",
                (final_status, time.time(), root_id),
            )
            conn.commit()
            report_status = "cancelled" if self._cancel.is_set() else "completed"
            if on_progress:
                on_progress(
                    ScanProgress(
                        root_id,
                        scanned,
                        added,
                        updated,
                        skipped,
                        errors,
                        str(root_path),
                        report_status,
                    )
                )
            return ScanReport(
                root_id=root_id,
                scanned=scanned,
                added=added,
                updated=updated,
                skipped=skipped,
                removed=removed,
                errors=errors,
                status=report_status,
            )
        finally:
            conn.close()

    def _wait_if_paused(
        self,
        conn: sqlite3.Connection,
        root_id: int,
        on_progress: Callable[[ScanProgress], None] | None,
        scanned: int,
        added: int,
        updated: int,
        skipped: int,
        errors: int,
        current_dir: str,
    ) -> None:
        if self._pause_clear.is_set():
            return
        conn.execute("UPDATE roots SET status = 'paused' WHERE id = ?", (root_id,))
        conn.commit()
        if on_progress:
            on_progress(
                ScanProgress(root_id, scanned, added, updated, skipped, errors, current_dir, "paused")
            )
        self._pause_clear.wait()
        if not self._cancel.is_set():
            conn.execute("UPDATE roots SET status = 'scanning' WHERE id = ?", (root_id,))
            conn.commit()


@dataclass(frozen=True, slots=True)
class _ScannedFile:
    existing_id: int | None
    root_id: int
    folder_id: int | None
    rel_path: str
    filename: str
    ext: str
    size_bytes: int
    mtime_ns: int
    duration_sec: float
    sample_rate: int
    channels: int
    bit_depth: int
    package: str
    description: str
    embedded_text: str
    bext_json: str
    ixml_json: str
    indexed_at: float
    fields: list[tuple[str, str]]


def _load_existing_files(
    conn: sqlite3.Connection, root_id: int
) -> dict[str, tuple[int, int, int, int, int | None]]:
    rows = conn.execute(
        "SELECT id, rel_path, size_bytes, mtime_ns, is_available, folder_id FROM files WHERE root_id = ?",
        (root_id,),
    ).fetchall()
    return {
        str(r["rel_path"]): (
            int(r["id"]),
            int(r["size_bytes"]),
            int(r["mtime_ns"]),
            int(r["is_available"]),
            None if r["folder_id"] is None else int(r["folder_id"]),
        )
        for r in rows
    }


def _load_existing_folders(conn: sqlite3.Connection, root_id: int) -> dict[str, int]:
    rows = conn.execute("SELECT id, rel_path FROM folders WHERE root_id = ?", (root_id,)).fetchall()
    return {str(r["rel_path"]): int(r["id"]) for r in rows}


def _insert_folder(
    conn: sqlite3.Connection,
    root_id: int,
    parent_id: int | None,
    rel_path: str,
    name: str,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO folders(root_id, parent_id, rel_path, name, file_count)
        VALUES (?, ?, ?, ?, 0)
        ON CONFLICT(root_id, rel_path) DO UPDATE SET parent_id = excluded.parent_id
        """,
        (root_id, parent_id, rel_path, name),
    )
    if cur.lastrowid:
        return int(cur.lastrowid)
    row = conn.execute(
        "SELECT id FROM folders WHERE root_id = ? AND rel_path = ?",
        (root_id, rel_path),
    ).fetchone()
    assert row is not None
    return int(row["id"])


def _inspect_audio_file(
    path: Path,
    *,
    root_id: int,
    folder_id: int | None,
    rel_path: str,
    root_label: str,
    size_bytes: int,
    mtime_ns: int,
    existing_id: int | None,
) -> _ScannedFile | None:
    try:
        return _inspect_audio_file_impl(
            path,
            root_id=root_id,
            folder_id=folder_id,
            rel_path=rel_path,
            root_label=root_label,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            existing_id=existing_id,
        )
    except Exception:
        return None


def _inspect_audio_file_impl(
    path: Path,
    *,
    root_id: int,
    folder_id: int | None,
    rel_path: str,
    root_label: str,
    size_bytes: int,
    mtime_ns: int,
    existing_id: int | None,
) -> _ScannedFile | None:
    duration_sec = 0.0
    sample_rate = 0
    channels = 0
    bit_depth = 0
    description = ""
    embedded_text = ""
    bext_dict: dict[str, str] = {}
    ixml_dict: dict[str, str] = {}
    fields: list[tuple[str, str]] = []
    package = _derive_package(rel_path, root_label)

    if is_video_path(path):
        vinfo = probe_video(path)
        duration_sec = vinfo.duration_sec
        sample_rate = vinfo.audio_sample_rate if vinfo.has_audio else 0
        channels = vinfo.audio_channels if vinfo.has_audio else 0
        bit_depth = 0
        if vinfo.width > 0 and vinfo.height > 0:
            fields.append(("video_resolution", f"{vinfo.width}x{vinfo.height}"))
        if vinfo.fps > 0.0:
            fields.append(("video_fps", f"{vinfo.fps:.2f}"))
        if vinfo.video_codec:
            fields.append(("video_codec", vinfo.video_codec))
        fields.append(("audio_codec", vinfo.audio_codec if vinfo.has_audio else "无音轨"))
        for tag_k, tag_v in vinfo.tags.items():
            fields.append((f"video_tag_{tag_k.lower()}", tag_v))
        description = vinfo.comment or vinfo.title
        text_parts = [f[1] for f in fields if f[1]]
        embedded_text = " ".join(text_parts)
        return _ScannedFile(
            existing_id=existing_id,
            root_id=root_id,
            folder_id=folder_id,
            rel_path=rel_path,
            filename=path.name,
            ext=path.suffix.lower().lstrip("."),
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            package=package,
            description=description,
            embedded_text=embedded_text,
            bext_json=json.dumps(bext_dict, ensure_ascii=False),
            ixml_json=json.dumps(ixml_dict, ensure_ascii=False),
            indexed_at=time.time(),
            fields=fields,
        )

    try:
        import soundfile as sf

        info = sf.info(str(path))
        sample_rate = int(info.samplerate or 0)
        channels = int(info.channels or 0)
        duration_sec = float(info.duration or 0.0)
        bit_depth = _SUBTYPE_BITS.get(str(info.subtype or ""), 0)
    except Exception:
        if path.suffix.lower() == ".wav":
            try:
                wav = read_wav_info(path)
                sample_rate = wav.sample_rate_hz
                channels = wav.channel_count
                duration_sec = wav.duration_seconds
                bit_depth = wav.bit_depth
            except Exception:
                return None
        else:
            return None

    if path.suffix.lower() == ".wav":
        try:
            meta: EmbeddedMetadata = read_embedded_metadata(path)
            description = meta.description or ""
            text_parts: list[str] = []
            by_field: dict[str, str] = {}
            for item in meta.values:
                fields.append((item.source, item.value))
                text_parts.append(item.value)
                by_field.setdefault(item.field, item.value)
                if item.family == "bext":
                    bext_dict[item.source] = item.value
                elif item.family == "ixml":
                    ixml_dict[item.source] = item.value
            if not description:
                for fallback_field in (
                    "ixml_user_description",
                    "info_comment",
                    "info_subject",
                    "info_title",
                ):
                    if by_field.get(fallback_field):
                        description = by_field[fallback_field]
                        break
            embedded_text = " ".join(text_parts)
            for key in ("ixml_user_library", "ixml_project", "bext_originator", "info_source"):
                if by_field.get(key):
                    package = by_field[key].strip()
                    break
        except Exception:
            pass
    else:
        tag_desc, tag_pkg, tag_fields = _read_non_wav_tags(path)
        if tag_fields:
            fields.extend(tag_fields)
            embedded_text = " ".join(v for _, v in tag_fields if v)
        if tag_desc:
            description = tag_desc
        if tag_pkg:
            package = tag_pkg

    return _ScannedFile(
        existing_id=existing_id,
        root_id=root_id,
        folder_id=folder_id,
        rel_path=rel_path,
        filename=path.name,
        ext=path.suffix.lower().lstrip("."),
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        duration_sec=duration_sec,
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=bit_depth,
        package=package,
        description=description,
        embedded_text=embedded_text,
        bext_json=json.dumps(bext_dict, ensure_ascii=False),
        ixml_json=json.dumps(ixml_dict, ensure_ascii=False),
        indexed_at=time.time(),
        fields=fields,
    )


def _read_non_wav_tags(path: Path) -> tuple[str, str, list[tuple[str, str]]]:
    """Best-effort read of ID3 / Vorbis comment tags via ``mutagen`` for non-WAV formats."""
    try:
        import mutagen
    except ImportError:
        return "", "", []

    try:
        audio = mutagen.File(str(path), easy=False)
    except Exception:
        return "", "", []
    if audio is None or not getattr(audio, "tags", None):
        return "", "", []

    tags = audio.tags
    raw_map: dict[str, str] = {}

    if hasattr(tags, "values") and any(hasattr(v, "FrameID") for v in tags.values()):
        for frame in tags.values():
            fid = str(getattr(frame, "FrameID", "") or "").upper()
            if fid == "TXXX":
                desc = str(getattr(frame, "desc", "") or "").strip().upper()
                texts = getattr(frame, "text", []) or []
                val = " ".join(str(x).strip() for x in texts if str(x).strip())
                if desc and val:
                    raw_map[desc] = val
            elif fid == "COMM":
                texts = getattr(frame, "text", []) or []
                val = " ".join(str(x).strip() for x in texts if str(x).strip())
                if val and "COMMENT" not in raw_map:
                    raw_map["COMMENT"] = val
            elif fid in ("TIT2", "TPE1", "TALB", "TCON"):
                texts = getattr(frame, "text", []) or []
                val = " ".join(str(x).strip() for x in texts if str(x).strip())
                key_name = {"TIT2": "TITLE", "TPE1": "ARTIST", "TALB": "ALBUM", "TCON": "GENRE"}[fid]
                if val and key_name not in raw_map:
                    raw_map[key_name] = val
    elif hasattr(tags, "items"):
        for k, v in tags.items():
            key_up = str(k).strip().upper()
            if isinstance(v, (list, tuple)):
                val = " ".join(str(x).strip() for x in v if str(x).strip())
            else:
                val = str(v).strip()
            if key_up and val and key_up not in raw_map:
                raw_map[key_up] = val

    tag_order = (
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
    fields: list[tuple[str, str]] = []
    for k in tag_order:
        val = raw_map.get(k)
        if val:
            fields.append((f"embedded:tag:{k}", val))
    description = raw_map.get("DESCRIPTION") or raw_map.get("COMMENT") or raw_map.get("TITLE") or ""
    package = raw_map.get("LIBRARY") or raw_map.get("ALBUM") or ""
    return description, package, fields


def _derive_package(rel_path: str, root_label: str) -> str:
    parts = rel_path.split("/")
    return parts[0] if len(parts) > 1 else root_label


def _flush_batch(conn: sqlite3.Connection, rows: list[_ScannedFile]) -> None:
    if not rows:
        return
    with conn:
        for item in rows:
            if item.existing_id is None:
                cur = conn.execute(
                    """
                    INSERT INTO files(
                        root_id, folder_id, rel_path, filename, ext, size_bytes, mtime_ns,
                        duration_sec, sample_rate, channels, bit_depth, package, description,
                        embedded_text, bext_json, ixml_json, is_available, indexed_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                    ON CONFLICT(root_id, rel_path) DO UPDATE SET
                        folder_id = excluded.folder_id,
                        filename = excluded.filename,
                        ext = excluded.ext,
                        size_bytes = excluded.size_bytes,
                        mtime_ns = excluded.mtime_ns,
                        duration_sec = excluded.duration_sec,
                        sample_rate = excluded.sample_rate,
                        channels = excluded.channels,
                        bit_depth = excluded.bit_depth,
                        package = excluded.package,
                        description = excluded.description,
                        embedded_text = excluded.embedded_text,
                        bext_json = excluded.bext_json,
                        ixml_json = excluded.ixml_json,
                        is_available = 1,
                        indexed_at = excluded.indexed_at
                    """,
                    (
                        item.root_id,
                        item.folder_id,
                        item.rel_path,
                        item.filename,
                        item.ext,
                        item.size_bytes,
                        item.mtime_ns,
                        item.duration_sec,
                        item.sample_rate,
                        item.channels,
                        item.bit_depth,
                        item.package,
                        item.description,
                        item.embedded_text,
                        item.bext_json,
                        item.ixml_json,
                        item.indexed_at,
                    ),
                )
                file_id = int(cur.lastrowid)
                if not file_id:
                    found = conn.execute(
                        "SELECT id FROM files WHERE root_id = ? AND rel_path = ?",
                        (item.root_id, item.rel_path),
                    ).fetchone()
                    assert found is not None
                    file_id = int(found["id"])
            else:
                file_id = item.existing_id
                dirty_rows = conn.execute(
                    "SELECT field, value FROM metadata_fields WHERE file_id = ? AND dirty = 1",
                    (file_id,),
                ).fetchall()
                dirty_map = {str(r["field"]): str(r["value"]) for r in dirty_rows}
                keep_desc = dirty_map.get("description") or item.description
                keep_pkg = dirty_map.get("library") or item.package
                if dirty_map:
                    existing_row = conn.execute(
                        "SELECT embedded_text FROM files WHERE id = ?",
                        (file_id,),
                    ).fetchone()
                    keep_emb = str(existing_row["embedded_text"]) if existing_row else item.embedded_text
                else:
                    keep_emb = item.embedded_text
                conn.execute(
                    """
                    UPDATE files SET
                        folder_id = ?, filename = ?, ext = ?, size_bytes = ?, mtime_ns = ?,
                        duration_sec = ?, sample_rate = ?, channels = ?, bit_depth = ?,
                        package = ?, description = ?, embedded_text = ?, bext_json = ?,
                        ixml_json = ?, is_available = 1, indexed_at = ?
                    WHERE id = ?
                    """,
                    (
                        item.folder_id,
                        item.filename,
                        item.ext,
                        item.size_bytes,
                        item.mtime_ns,
                        item.duration_sec,
                        item.sample_rate,
                        item.channels,
                        item.bit_depth,
                        keep_pkg,
                        keep_desc,
                        keep_emb,
                        item.bext_json,
                        item.ixml_json,
                        item.indexed_at,
                        file_id,
                    ),
                )

            conn.execute(
                "DELETE FROM metadata_fields WHERE file_id = ? AND source = 'embedded' AND dirty = 0",
                (file_id,),
            )
            if item.fields:
                conn.executemany(
                    """
                    INSERT INTO metadata_fields(file_id, field, value, source, dirty, updated_at)
                    VALUES (?, ?, ?, 'embedded', 0, ?)
                    ON CONFLICT(file_id, field) DO UPDATE SET
                        value = CASE WHEN metadata_fields.dirty = 0 THEN excluded.value ELSE metadata_fields.value END,
                        source = CASE WHEN metadata_fields.dirty = 0 THEN 'embedded' ELSE metadata_fields.source END,
                        updated_at = excluded.updated_at
                    """,
                    [(file_id, f_name, f_val, item.indexed_at) for f_name, f_val in item.fields],
                )


def _recompute_counts(conn: sqlite3.Connection, root_id: int) -> None:
    total = conn.execute(
        "SELECT COUNT(*) AS c FROM files WHERE root_id = ?", (root_id,)
    ).fetchone()
    root_count = 0 if total is None else int(total["c"])
    conn.execute("UPDATE roots SET file_count = ? WHERE id = ?", (root_count, root_id))

    direct = dict(
        conn.execute(
            "SELECT folder_id, COUNT(*) FROM files WHERE root_id = ? AND folder_id IS NOT NULL GROUP BY folder_id",
            (root_id,),
        ).fetchall()
    )
    folder_rows = conn.execute(
        "SELECT id, parent_id FROM folders WHERE root_id = ?", (root_id,)
    ).fetchall()
    children: dict[int, list[int]] = {}
    for row in folder_rows:
        pid = row["parent_id"]
        if pid is not None:
            children.setdefault(int(pid), []).append(int(row["id"]))

    totals: dict[int, int] = {}

    def dfs(fid: int) -> int:
        if fid in totals:
            return totals[fid]
        val = int(direct.get(fid, 0))
        for cid in children.get(fid, ()):
            val += dfs(cid)
        totals[fid] = val
        return val

    for row in folder_rows:
        dfs(int(row["id"]))

    conn.executemany(
        "UPDATE folders SET file_count = ? WHERE id = ?",
        [(count, fid) for fid, count in totals.items()],
    )
    # Prune folders with 0 files recursively when online scan finished cleanly.
    empty_ids = [fid for fid, count in totals.items() if count == 0]
    if empty_ids:
        for i in range(0, len(empty_ids), 500):
            chunk = empty_ids[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            conn.execute(f"DELETE FROM folders WHERE id IN ({placeholders})", chunk)
