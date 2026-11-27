"""Thread-safe SQLite store for local audio library browsing and querying."""

from __future__ import annotations

import array
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiosearch.library.metadata_fields import (
    EDITABLE_FIELD_KEYS,
    FIELD_SPEC_MAP,
    apply_field_transform,
    compute_file_summary_columns,
    extract_canonical_from_raw,
)
from audiosearch.library.peaks import (
    PeakData,
    deserialize_peaks,
    peaks_to_thumbnail,
    peaks_to_thumbnail_spectral,
    serialize_peaks,
)
from audiosearch.library.schema import ensure_schema, open_connection
from audiosearch.library.search_syntax import (
    _escape_like,
    build_where_clause,
    extract_absolute_path_query,
    normalize_abs_path,
)

SORT_COLUMN_SQL: dict[str, str] = {
    "filename": "f.filename COLLATE NOCASE",
    "favorite": "COALESCE(um.favorite, 0)",
    "description": "f.description COLLATE NOCASE",
    "duration_sec": "f.duration_sec",
    "sample_rate": "f.sample_rate",
    "channels": "f.channels",
    "bit_depth": "f.bit_depth",
    "ext": "f.ext COLLATE NOCASE",
    "size_bytes": "f.size_bytes",
    "package": "f.package COLLATE NOCASE",
    "rel_path": "f.rel_path COLLATE NOCASE",
    "played_at": "COALESCE(rp.played_at, 0.0)",
}


@dataclass(frozen=True, slots=True)
class QueryFilter:
    """R4 structured filters combinable across Keyword, AI Semantic, and Views."""

    ext: str | None = None
    min_duration: float | None = None
    max_duration: float | None = None
    favorites_only: bool = False
    online_only: bool = False
    recent_only: bool = False
    collection_id: int | None = None
    include_hidden: bool = False

    def is_default(self) -> bool:
        return (
            not self.ext
            and self.min_duration is None
            and self.max_duration is None
            and not self.favorites_only
            and not self.online_only
            and not self.recent_only
            and self.collection_id is None
            and not self.include_hidden
        )

    def active_count(self) -> int:
        count = 0
        if self.ext:
            count += 1
        if self.min_duration is not None or self.max_duration is not None:
            count += 1
        if self.favorites_only:
            count += 1
        if self.online_only:
            count += 1
        return count


@dataclass(frozen=True, slots=True)
class CollectionRecord:
    id: int
    name: str
    sort_order: int
    folder_count: int
    file_count: int


@dataclass(frozen=True, slots=True)
class CollectionFolderRecord:
    id: int
    collection_id: int
    root_id: int
    folder_id: int | None
    folder_rel_path: str
    label: str
    root_path: str
    sort_order: int = 0


def _prune_overlapping_folder_mounts(
    mounts: list[tuple[int, int | None, str]],
) -> list[tuple[int, int | None, str]]:
    """Deduplicate mounts and drop child paths shadowed by an ancestor mount in the same root."""
    by_root: dict[int, dict[str, tuple[int | None, str]]] = {}
    for rid, fid, rp in mounts:
        clean_rp = str(rp or "").strip("/")
        key = clean_rp.casefold()
        root_bucket = by_root.setdefault(int(rid), {})
        if key not in root_bucket:
            root_bucket[key] = (fid, clean_rp)

    out: list[tuple[int, int | None, str]] = []
    for rid, bucket in by_root.items():
        kept: list[tuple[str, int | None, str]] = []
        for key, (fid, clean_rp) in sorted(bucket.items(), key=lambda kv: (len(kv[0]), kv[0])):
            if any(anc_key == "" or key.startswith(anc_key + "/") for anc_key, _, _ in kept):
                continue
            kept.append((key, fid, clean_rp))
            out.append((rid, fid, clean_rp))
    return out


@dataclass(frozen=True, slots=True)
class SearchHistoryRecord:
    id: int
    query: str
    mode: str
    result_count: int
    pinned: bool
    last_used_at: float


@dataclass(frozen=True, slots=True)
class RootRecord:
    id: int
    path: str
    label: str
    is_online: bool
    file_count: int
    last_scan_at: float | None
    status: str


@dataclass(frozen=True, slots=True)
class FolderRecord:
    id: int
    root_id: int
    parent_id: int | None
    rel_path: str
    name: str
    file_count: int


@dataclass(frozen=True, slots=True)
class FileRecord:
    id: int
    root_id: int
    folder_id: int | None
    root_path: str
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
    is_available: bool
    thumb_blob: bytes | None = None
    spectral_thumb_blob: bytes | None = None
    favorite: bool = False
    note: str = ""
    is_hidden: bool = False
    is_dirty: bool = False

    @property
    def full_path(self) -> Path:
        return Path(self.root_path) / Path(self.rel_path)


@dataclass(frozen=True, slots=True)
class WriteBatchRecord:
    batch_id: str
    created_at: float
    file_count: int
    backup_dir: str
    status: str


@dataclass(frozen=True, slots=True)
class RenameBatchRecord:
    batch_id: str
    created_at: float
    file_count: int
    template: str
    manifest_path: str
    status: str


class LibraryStore:
    """Thread-safe access layer over ``library.sqlite3``."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._lock = threading.Lock()
        self._conns: list[sqlite3.Connection] = []
        self._index_version = 0
        ensure_schema(self._conn())

    @property
    def index_version(self) -> int:
        with self._lock:
            return self._index_version

    def bump_index_version(self) -> int:
        with self._lock:
            self._index_version += 1
            return self._index_version

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = open_connection(self.db_path)
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        self._local = threading.local()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    # -- roots -------------------------------------------------------------
    def add_root(self, path: str | Path, label: str | None = None) -> RootRecord:
        resolved = str(Path(path).resolve())
        display = label or Path(resolved).name or resolved
        online = 1 if Path(resolved).exists() else 0
        conn = self._conn()
        existing = conn.execute("SELECT id FROM roots WHERE path = ?", (resolved,)).fetchone()
        if existing is None:
            max_row = conn.execute("SELECT COALESCE(MAX(sort_order), 0) FROM roots").fetchone()
            max_ord = 0 if max_row is None else int(max_row[0])
            next_ord = (max_ord + 1) if max_ord > 0 else 0
            conn.execute(
                """
                INSERT INTO roots(path, label, is_online, status, sort_order)
                VALUES (?, ?, ?, ?, ?)
                """,
                (resolved, display, online, "idle" if online else "offline", next_ord),
            )
        else:
            conn.execute(
                """
                UPDATE roots SET label = ?, is_online = ? WHERE id = ?
                """,
                (display, online, int(existing["id"])),
            )
        conn.commit()
        self.bump_index_version()
        row = conn.execute("SELECT * FROM roots WHERE path = ?", (resolved,)).fetchone()
        assert row is not None
        return _row_to_root(row)

    def remove_root(self, root_id: int) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM roots WHERE id = ?", (root_id,))
        conn.commit()
        self.bump_index_version()

    def list_roots(self) -> list[RootRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM roots "
                "ORDER BY CASE WHEN sort_order > 0 THEN 0 ELSE 1 END, "
                "sort_order ASC, label COLLATE NOCASE, id ASC"
            )
            .fetchall()
        )
        return [_row_to_root(r) for r in rows]

    def reorder_roots(self, ordered_root_ids: list[int]) -> None:
        current_ids = [r.id for r in self.list_roots()]
        valid = set(current_ids)
        merged: list[int] = []
        seen: set[int] = set()
        for rid in ordered_root_ids:
            rid_int = int(rid)
            if rid_int in valid and rid_int not in seen:
                seen.add(rid_int)
                merged.append(rid_int)
        for rid_int in current_ids:
            if rid_int not in seen:
                seen.add(rid_int)
                merged.append(rid_int)
        if not merged:
            return
        conn = self._conn()
        with conn:
            conn.executemany(
                "UPDATE roots SET sort_order = ? WHERE id = ?",
                [(pos, rid_int) for pos, rid_int in enumerate(merged, start=1)],
            )

    def get_root(self, root_id: int) -> RootRecord | None:
        row = self._conn().execute("SELECT * FROM roots WHERE id = ?", (root_id,)).fetchone()
        return None if row is None else _row_to_root(row)

    def refresh_roots_online(self) -> list[RootRecord]:
        conn = self._conn()
        roots = self.list_roots()
        changed = False
        for root in roots:
            online_now = Path(root.path).exists()
            if online_now != root.is_online:
                changed = True
                status = "idle" if online_now else "offline"
                conn.execute(
                    "UPDATE roots SET is_online = ?, status = ? WHERE id = ?",
                    (1 if online_now else 0, status, root.id),
                )
                conn.execute(
                    "UPDATE files SET is_available = ? WHERE root_id = ?",
                    (1 if online_now else 0, root.id),
                )
        if changed:
            conn.commit()
        return self.list_roots() if changed else roots

    def relocate_root(
        self, root_id: int, new_path: str | Path, *, label: str | None = None
    ) -> RootRecord:
        """Bind an existing root to a new path while preserving all file IDs and user metadata."""
        target = Path(new_path).resolve()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Root path does not exist or is not a directory: {target}")
        resolved = str(target)
        conn = self._conn()
        existing = conn.execute("SELECT id FROM roots WHERE path = ?", (resolved,)).fetchone()
        if existing is not None and int(existing["id"]) != int(root_id):
            raise ValueError(f"Path is already registered to another root: {resolved}")
        current = self.get_root(root_id)
        if current is None:
            raise ValueError(f"Root id {root_id} not found")
        new_label = label if label is not None else current.label
        with conn:
            conn.execute(
                "UPDATE roots SET path = ?, label = ?, is_online = 1, status = 'idle' WHERE id = ?",
                (resolved, new_label, root_id),
            )
            conn.execute(
                "UPDATE files SET is_available = 1 WHERE root_id = ?",
                (root_id,),
            )
        self.bump_index_version()
        updated = self.get_root(root_id)
        assert updated is not None
        return updated

    def mark_file_available(self, file_id: int, available: bool) -> bool:
        val = 1 if available else 0
        conn = self._conn()
        with conn:
            cur = conn.execute(
                "UPDATE files SET is_available = ? WHERE id = ? AND is_available != ?",
                (val, int(file_id), val),
            )
        if cur.rowcount > 0:
            self.bump_index_version()
            return True
        return False

    def refresh_folder_online(self, root_id: int, folder_rel_path: str) -> bool:
        rp = folder_rel_path.strip("/")
        root = self.get_root(root_id)
        if root is None:
            return False
        root_p = Path(root.path)
        root_online_now = root_p.exists()
        if not root_online_now or not rp:
            before = root.is_online
            self.refresh_roots_online()
            after_root = self.get_root(root_id)
            if after_root is None or not after_root.is_online:
                return bool(after_root is not None and before != after_root.is_online)
            conn = self._conn()
            return self._sync_root_files_availability(conn, after_root, folder_prefix="")

        conn = self._conn()
        return self._sync_root_files_availability(conn, root, folder_prefix=rp)

    def refresh_root_folders_online(self, root_id: int | None = None) -> bool:
        before_roots = {r.id: r.is_online for r in self.list_roots()}
        after_list = self.refresh_roots_online()
        after_roots = {r.id: r.is_online for r in after_list}
        changed = before_roots != after_roots

        targets = [self.get_root(root_id)] if root_id is not None else after_list
        conn = self._conn()
        for root in targets:
            if root is None or not root.is_online:
                continue
            if self._sync_root_files_availability(conn, root, folder_prefix=""):
                changed = True
        return changed

    @staticmethod
    def _listdir_names(dir_path: Path) -> set[str] | None:
        try:
            with os.scandir(dir_path) as it:
                return {entry.name.casefold() for entry in it}
        except OSError:
            return None

    def _sync_root_files_availability(
        self,
        conn: sqlite3.Connection,
        root: RootRecord,
        *,
        folder_prefix: str = "",
    ) -> bool:
        root_p = Path(root.path)
        if folder_prefix:
            prefix = f"{folder_prefix}/"
            upper = f"{folder_prefix}0"
            folder_rows = conn.execute(
                "SELECT id, rel_path FROM folders "
                "WHERE root_id = ? AND (rel_path = ? OR (rel_path >= ? AND rel_path < ?)) "
                "ORDER BY LENGTH(rel_path) ASC",
                (root.id, folder_prefix, prefix, upper),
            ).fetchall()
            file_rows = conn.execute(
                "SELECT id, folder_id, filename, is_available FROM files "
                "WHERE root_id = ? AND rel_path >= ? AND rel_path < ?",
                (root.id, prefix, upper),
            ).fetchall()
        else:
            folder_rows = conn.execute(
                "SELECT id, rel_path FROM folders WHERE root_id = ? ORDER BY LENGTH(rel_path) ASC",
                (root.id,),
            ).fetchall()
            file_rows = conn.execute(
                "SELECT id, folder_id, filename, is_available FROM files WHERE root_id = ?",
                (root.id,),
            ).fetchall()

        dir_entries: dict[int | None, set[str] | None] = {}
        if not folder_prefix:
            dir_entries[None] = self._listdir_names(root_p)

        missing_prefixes: list[str] = []
        for row in folder_rows:
            fid = int(row["id"])
            rp = str(row["rel_path"]).strip("/")
            if any(rp.startswith(mp + "/") for mp in missing_prefixes):
                dir_entries[fid] = None
            else:
                names = self._listdir_names(root_p / Path(rp))
                if names is None:
                    missing_prefixes.append(rp)
                dir_entries[fid] = names

        to_off: list[int] = []
        to_on: list[int] = []
        for fr in file_rows:
            fid = None if fr["folder_id"] is None else int(fr["folder_id"])
            entries = dir_entries.get(fid)
            exists_now = entries is not None and str(fr["filename"]).casefold() in entries
            cur_av = bool(fr["is_available"])
            if exists_now and not cur_av:
                to_on.append(int(fr["id"]))
            elif not exists_now and cur_av:
                to_off.append(int(fr["id"]))

        if not to_off and not to_on:
            return False

        with conn:
            for val, file_ids in ((0, to_off), (1, to_on)):
                for i in range(0, len(file_ids), 500):
                    chunk = file_ids[i : i + 500]
                    ph = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE files SET is_available = ? WHERE id IN ({ph})",
                        [val, *chunk],
                    )
        return True

    def relocate_folder(
        self, root_id: int, folder_id: int, new_path: str | Path
    ) -> tuple[int, int | None, str, str]:
        """Rebind a subfolder to a moved/renamed path, preserving file IDs, favorites, notes & collections."""
        target = Path(new_path).resolve()
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {target}")
        root = self.get_root(root_id)
        if root is None:
            raise ValueError(f"Root id {root_id} not found")
        folder = self.get_folder(folder_id)
        if folder is None or folder.root_id != root_id:
            raise ValueError(f"Folder id {folder_id} not found under root {root_id}")

        root_resolved = Path(root.path).resolve()
        if target == root_resolved:
            raise ValueError("不能将子目录重定位到根目录自身，请直接对根目录操作")

        old_rel = folder.rel_path.strip("/")
        old_prefix = f"{old_rel}/"
        old_upper = f"{old_rel}0"
        old_name = Path(old_rel).name
        conn = self._conn()

        try:
            new_rel_path = target.relative_to(root_resolved)
            under_same_root = True
        except ValueError:
            new_rel_path = None
            under_same_root = False

        if under_same_root and new_rel_path is not None:
            new_rel = new_rel_path.as_posix().strip("/")
            if new_rel == old_rel:
                self.refresh_root_folders_online(root_id)
                return (int(root_id), int(folder_id), old_rel, str(target))
            if new_rel.startswith(old_prefix):
                raise ValueError("不能将目录重定位到其自身子目录下")

            conflict = conn.execute(
                "SELECT id FROM folders WHERE root_id = ? AND rel_path = ?",
                (int(root_id), new_rel),
            ).fetchone()
            if conflict is not None and int(conflict["id"]) != int(folder_id):
                raise ValueError(f"目标路径下已存在索引目录：{new_rel}")

            descendants = conn.execute(
                "SELECT id, rel_path FROM folders "
                "WHERE root_id = ? AND rel_path >= ? AND rel_path < ? AND id != ?",
                (int(root_id), old_prefix, old_upper, int(folder_id)),
            ).fetchall()
            cut = len(old_prefix) + 1
            with conn:
                # Ensure intermediate ancestor folders exist under root_id.
                parts = [p for p in new_rel.split("/")[:-1] if p]
                parent_id: int | None = None
                acc: list[str] = []
                for part in parts:
                    acc.append(part)
                    anc_rel = "/".join(acc)
                    anc_row = conn.execute(
                        "SELECT id FROM folders WHERE root_id = ? AND rel_path = ?",
                        (int(root_id), anc_rel),
                    ).fetchone()
                    if anc_row is not None:
                        parent_id = int(anc_row["id"])
                    else:
                        cur = conn.execute(
                            "INSERT INTO folders(root_id, parent_id, rel_path, name, file_count) "
                            "VALUES (?, ?, ?, ?, 0)",
                            (int(root_id), parent_id, anc_rel, part),
                        )
                        parent_id = int(cur.lastrowid)

                conn.execute(
                    "UPDATE folders SET rel_path = ?, name = ?, parent_id = ? WHERE id = ?",
                    (new_rel, Path(new_rel).name, parent_id, int(folder_id)),
                )
                for desc in descendants:
                    child_rel = f"{new_rel}/{str(desc['rel_path'])[len(old_prefix):]}"
                    conn.execute(
                        "UPDATE folders SET rel_path = ? WHERE id = ?",
                        (child_rel, int(desc["id"])),
                    )
                conn.execute(
                    "UPDATE files SET rel_path = ? || SUBSTR(rel_path, ?), is_available = 1 "
                    "WHERE root_id = ? AND rel_path >= ? AND rel_path < ?",
                    (f"{new_rel}/", cut, int(root_id), old_prefix, old_upper),
                )
                conn.execute(
                    "UPDATE collection_folders SET folder_rel_path = ?, "
                    "label = CASE WHEN label = ? THEN ? ELSE label END "
                    "WHERE root_id = ? AND (folder_id = ? OR folder_rel_path = ?)",
                    (new_rel, old_name, Path(new_rel).name, int(root_id), int(folder_id), old_rel),
                )
                cf_desc = conn.execute(
                    "SELECT id, folder_rel_path FROM collection_folders "
                    "WHERE root_id = ? AND folder_rel_path >= ? AND folder_rel_path < ?",
                    (int(root_id), old_prefix, old_upper),
                ).fetchall()
                for cfd in cf_desc:
                    conn.execute(
                        "UPDATE collection_folders SET folder_rel_path = ? WHERE id = ?",
                        (f"{new_rel}/{str(cfd['folder_rel_path'])[len(old_prefix):]}", int(cfd["id"])),
                    )
                self._recompute_root_counts(conn, int(root_id))
            self.refresh_root_folders_online(root_id)
            self.bump_index_version()
            return (int(root_id), int(folder_id), new_rel, str(target))

        # Scenario B: moved outside original root -> bind/create destination root and migrate records cleanly.
        dest_root = self.add_root(target)
        dest_root_id = int(dest_root.id)
        if dest_root_id == int(root_id):
            raise ValueError("目标根目录与当前根目录相同")

        cut = len(old_prefix) + 1
        descendants = conn.execute(
            "SELECT id, parent_id, rel_path FROM folders "
            "WHERE root_id = ? AND rel_path >= ? AND rel_path < ? AND id != ?",
            (int(root_id), old_prefix, old_upper, int(folder_id)),
        ).fetchall()
        with conn:
            # Update collection_folders BEFORE deleting folder_id so ON DELETE CASCADE won't drop mounts.
            cf_exact = conn.execute(
                "SELECT id, collection_id, label FROM collection_folders "
                "WHERE root_id = ? AND (folder_id = ? OR folder_rel_path = ?)",
                (int(root_id), int(folder_id), old_rel),
            ).fetchall()
            for cfe in cf_exact:
                exists_dest = conn.execute(
                    "SELECT id FROM collection_folders "
                    "WHERE collection_id = ? AND root_id = ? AND folder_rel_path = ''",
                    (int(cfe["collection_id"]), dest_root_id),
                ).fetchone()
                if exists_dest is not None:
                    conn.execute("DELETE FROM collection_folders WHERE id = ?", (int(cfe["id"]),))
                else:
                    new_lbl = dest_root.label if str(cfe["label"]) == old_name else str(cfe["label"])
                    conn.execute(
                        "UPDATE collection_folders SET root_id = ?, folder_id = NULL, "
                        "folder_rel_path = '', label = ? WHERE id = ?",
                        (dest_root_id, new_lbl, int(cfe["id"])),
                    )

            cf_desc = conn.execute(
                "SELECT id, collection_id, folder_rel_path FROM collection_folders "
                "WHERE root_id = ? AND folder_rel_path >= ? AND folder_rel_path < ?",
                (int(root_id), old_prefix, old_upper),
            ).fetchall()
            for cfd in cf_desc:
                new_sub = str(cfd["folder_rel_path"])[len(old_prefix):]
                exists_dest = conn.execute(
                    "SELECT id FROM collection_folders "
                    "WHERE collection_id = ? AND root_id = ? AND folder_rel_path = ?",
                    (int(cfd["collection_id"]), dest_root_id, new_sub),
                ).fetchone()
                if exists_dest is not None:
                    conn.execute("DELETE FROM collection_folders WHERE id = ?", (int(cfd["id"]),))
                else:
                    conn.execute(
                        "UPDATE collection_folders SET root_id = ?, folder_rel_path = ? WHERE id = ?",
                        (dest_root_id, new_sub, int(cfd["id"])),
                    )

            for desc in descendants:
                new_sub = str(desc["rel_path"])[len(old_prefix):]
                new_pid = None if desc["parent_id"] == folder_id else desc["parent_id"]
                conn.execute(
                    "UPDATE folders SET root_id = ?, rel_path = ?, parent_id = ? WHERE id = ?",
                    (dest_root_id, new_sub, new_pid, int(desc["id"])),
                )

            conn.execute(
                "UPDATE files SET root_id = ?, folder_id = NULL, "
                "rel_path = SUBSTR(rel_path, ?), is_available = 1 "
                "WHERE root_id = ? AND folder_id = ?",
                (dest_root_id, cut, int(root_id), int(folder_id)),
            )
            conn.execute(
                "UPDATE files SET root_id = ?, rel_path = SUBSTR(rel_path, ?), is_available = 1 "
                "WHERE root_id = ? AND rel_path >= ? AND rel_path < ?",
                (dest_root_id, cut, int(root_id), old_prefix, old_upper),
            )
            conn.execute("DELETE FROM folders WHERE id = ?", (int(folder_id),))
            self._recompute_root_counts(conn, int(root_id))
            self._recompute_root_counts(conn, dest_root_id)

        self.refresh_root_folders_online(dest_root_id)
        self.bump_index_version()
        return (dest_root_id, None, "", str(target))

    @staticmethod
    def _recompute_root_counts(conn: sqlite3.Connection, root_id: int) -> None:
        folder_rows = conn.execute(
            "SELECT id, parent_id FROM folders WHERE root_id = ?", (root_id,)
        ).fetchall()
        direct_rows = conn.execute(
            "SELECT folder_id, COUNT(*) AS c FROM files WHERE root_id = ? AND folder_id IS NOT NULL GROUP BY folder_id",
            (root_id,),
        ).fetchall()
        direct = {int(r["folder_id"]): int(r["c"]) for r in direct_rows}
        children: dict[int, list[int]] = {}
        all_ids: list[int] = []
        for r in folder_rows:
            fid = int(r["id"])
            all_ids.append(fid)
            pid = r["parent_id"]
            if pid is not None:
                children.setdefault(int(pid), []).append(fid)

        memo: dict[int, int] = {}

        def _sum(fid: int) -> int:
            if fid in memo:
                return memo[fid]
            total = direct.get(fid, 0) + sum(_sum(cid) for cid in children.get(fid, ()))
            memo[fid] = total
            return total

        updates = [(_sum(fid), fid) for fid in all_ids]
        if updates:
            conn.executemany("UPDATE folders SET file_count = ? WHERE id = ?", updates)
        total_root = conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE root_id = ?", (root_id,)
        ).fetchone()
        conn.execute(
            "UPDATE roots SET file_count = ? WHERE id = ?",
            (0 if total_root is None else int(total_root["c"]), root_id),
        )

    # -- folders -----------------------------------------------------------
    def list_folders(self, root_id: int, parent_id: int | None = None) -> list[FolderRecord]:
        conn = self._conn()
        if parent_id is None:
            rows = conn.execute(
                "SELECT * FROM folders WHERE root_id = ? AND parent_id IS NULL "
                "ORDER BY name COLLATE NOCASE",
                (root_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM folders WHERE root_id = ? AND parent_id = ? "
                "ORDER BY name COLLATE NOCASE",
                (root_id, parent_id),
            ).fetchall()
        return [_row_to_folder(r) for r in rows]

    def has_child_folders(self, folder_id: int) -> bool:
        row = (
            self._conn()
            .execute("SELECT 1 FROM folders WHERE parent_id = ? LIMIT 1", (folder_id,))
            .fetchone()
        )
        return row is not None

    def folder_ids_with_children(self, folder_ids: list[int]) -> set[int]:
        ids = list(dict.fromkeys(int(fid) for fid in folder_ids))
        if not ids:
            return set()
        conn = self._conn()
        out: set[int] = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            ph = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT DISTINCT parent_id FROM folders WHERE parent_id IN ({ph})",
                chunk,
            ).fetchall()
            for r in rows:
                if r["parent_id"] is not None:
                    out.add(int(r["parent_id"]))
        return out

    def root_ids_with_folders(self) -> set[int]:
        rows = self._conn().execute("SELECT DISTINCT root_id FROM folders").fetchall()
        return {int(r["root_id"]) for r in rows}

    def list_all_folder_paths(self, root_id: int) -> dict[str, tuple[int, str]]:
        """Return mapping ``normalized_rel_path -> (folder_id, exact_rel_path)`` for one root."""
        rows = (
            self._conn()
            .execute("SELECT id, rel_path FROM folders WHERE root_id = ?", (int(root_id),))
            .fetchall()
        )
        out: dict[str, tuple[int, str]] = {}
        for r in rows:
            rel = str(r["rel_path"] or "").strip("/")
            if rel:
                out[rel.casefold()] = (int(r["id"]), rel)
        return out

    def get_folder(self, folder_id: int) -> FolderRecord | None:
        row = self._conn().execute("SELECT * FROM folders WHERE id = ?", (folder_id,)).fetchone()
        return None if row is None else _row_to_folder(row)

    # -- queries -----------------------------------------------------------
    def total_file_count(self) -> int:
        row = self._conn().execute("SELECT COALESCE(SUM(file_count), 0) AS c FROM roots").fetchone()
        if row is not None and int(row["c"]) > 0:
            return int(row["c"])
        row = self._conn().execute("SELECT COUNT(*) AS c FROM files").fetchone()
        return 0 if row is None else int(row["c"])

    def all_file_ids(self) -> set[int]:
        """Return every ``files.id`` currently in the library (online and offline)."""
        rows = self._conn().execute("SELECT id FROM files").fetchall()
        return {int(r[0]) for r in rows}

    def file_ids_for_root(self, root_id: int) -> list[int]:
        """Return all ``files.id`` rows belonging to ``root_id``."""
        rows = (
            self._conn()
            .execute("SELECT id FROM files WHERE root_id = ?", (int(root_id),))
            .fetchall()
        )
        return [int(r[0]) for r in rows]

    def has_hidden_files(self) -> bool:
        return self._conn().execute("SELECT 1 FROM hidden_files LIMIT 1").fetchone() is not None

    def _is_sole_root(self, root_id: int) -> bool:
        rows = self._conn().execute("SELECT id FROM roots LIMIT 2").fetchall()
        return len(rows) == 1 and int(rows[0]["id"]) == int(root_id)

    def _append_collection_filter(
        self,
        where_parts: list[str],
        typed_params: list[Any],
        cid: int,
    ) -> None:
        conn = self._conn()
        has_direct = (
            conn.execute(
                "SELECT 1 FROM collection_files WHERE collection_id = ? LIMIT 1",
                (cid,),
            ).fetchone()
            is not None
        )
        cf_rows = conn.execute(
            "SELECT root_id, folder_id, folder_rel_path FROM collection_folders WHERE collection_id = ?",
            (cid,),
        ).fetchall()
        pruned_mounts = _prune_overlapping_folder_mounts(
            [
                (
                    int(cf["root_id"]),
                    None if cf["folder_id"] is None else int(cf["folder_id"]),
                    str(cf["folder_rel_path"] or ""),
                )
                for cf in cf_rows
            ]
        )
        if not has_direct and not pruned_mounts:
            where_parts.append("0 = 1")
        elif len(pruned_mounts) <= 200:
            union_parts: list[str] = []
            if has_direct:
                union_parts.append("SELECT file_id FROM collection_files WHERE collection_id = ?")
                typed_params.append(cid)
            for rid, _fid, rp in pruned_mounts:
                if rp:
                    union_parts.append(
                        "SELECT id FROM files WHERE root_id = ? AND rel_path >= ? AND rel_path < ?"
                    )
                    typed_params.extend([rid, f"{rp}/", f"{rp}0"])
                else:
                    union_parts.append("SELECT id FROM files WHERE root_id = ?")
                    typed_params.append(rid)
            where_parts.append(f"f.id IN ({' UNION '.join(union_parts)})")
        else:
            coll_clauses = [
                "EXISTS (SELECT 1 FROM collection_files cf "
                "WHERE cf.collection_id = ? AND cf.file_id = f.id)"
            ]
            typed_params.append(cid)
            for rid, _fid, rp in pruned_mounts:
                if rp:
                    coll_clauses.append(
                        "(f.root_id = ? AND f.rel_path >= ? AND f.rel_path < ?)"
                    )
                    typed_params.extend([rid, f"{rp}/", f"{rp}0"])
                else:
                    coll_clauses.append("f.root_id = ?")
                    typed_params.append(rid)
            where_parts.append(f"({' OR '.join(coll_clauses)})")

    def _resolve_absolute_path_query(
        self, query: str
    ) -> tuple[bool, list[str], list[Any]] | None:
        raw_abs = extract_absolute_path_query(query)
        if raw_abs is None:
            return None
        norm_cf, norm_orig, trailing_sep = normalize_abs_path(raw_abs)
        roots = self.list_roots()
        root_entries: list[tuple[RootRecord, str, str]] = []
        for r in roots:
            r_cf, r_orig, _ = normalize_abs_path(r.path)
            if r_cf:
                root_entries.append((r, r_cf, r_orig))
        root_entries.sort(key=lambda item: len(item[1]), reverse=True)

        conn = self._conn()
        for root, root_cf, _root_orig in root_entries:
            if norm_cf == root_cf:
                return False, ["f.root_id = ?"], [root.id]
            prefix_cf = root_cf if root_cf.endswith("/") else f"{root_cf}/"
            if norm_cf.startswith(prefix_cf):
                rel_orig = norm_orig[len(prefix_cf) :].strip("/")
                if not rel_orig:
                    return False, ["f.root_id = ?"], [root.id]
                rel_cf = rel_orig.casefold()

                if not trailing_sep:
                    row = conn.execute(
                        "SELECT id FROM files WHERE root_id = ? AND rel_path = ? LIMIT 1",
                        (root.id, rel_orig),
                    ).fetchone()
                    hit_id = None if row is None else int(row["id"])
                    if hit_id is None:
                        if len(rel_orig) >= 3:
                            fts_term = '"' + rel_orig.replace('"', '""') + '"'
                            cand_rows = conn.execute(
                                "SELECT f.id, f.rel_path FROM files f "
                                "JOIN files_fts ON files_fts.rowid = f.id "
                                "WHERE f.root_id = ? AND files_fts MATCH ?",
                                (root.id, fts_term),
                            ).fetchall()
                            for cand in cand_rows:
                                if str(cand["rel_path"]).casefold() == rel_cf:
                                    hit_id = int(cand["id"])
                                    break
                        else:
                            cand = conn.execute(
                                "SELECT id FROM files WHERE root_id = ? AND rel_path = ? COLLATE NOCASE LIMIT 1",
                                (root.id, rel_orig),
                            ).fetchone()
                            if cand is not None:
                                hit_id = int(cand["id"])
                    if hit_id is not None:
                        return False, ["f.id = ?"], [hit_id]

                f_row = conn.execute(
                    "SELECT rel_path FROM folders WHERE root_id = ? AND rel_path = ? LIMIT 1",
                    (root.id, rel_orig),
                ).fetchone()
                if f_row is None:
                    f_row = conn.execute(
                        "SELECT rel_path FROM folders WHERE root_id = ? AND rel_path = ? COLLATE NOCASE LIMIT 1",
                        (root.id, rel_orig),
                    ).fetchone()
                if f_row is not None:
                    exact_rel = str(f_row["rel_path"]).strip("/")
                    return (
                        False,
                        ["f.root_id = ?", "f.rel_path >= ?", "f.rel_path < ?"],
                        [root.id, f"{exact_rel}/", f"{exact_rel}0"],
                    )

                like_pat = f"{_escape_like(rel_orig)}%"
                if len(rel_orig) >= 3:
                    fts_term = '"' + rel_orig.replace('"', '""') + '"'
                    return (
                        True,
                        [
                            "files_fts MATCH ?",
                            "f.root_id = ?",
                            "f.rel_path LIKE ? ESCAPE '\\'",
                        ],
                        [fts_term, root.id, like_pat],
                    )
                return (
                    False,
                    ["f.root_id = ?", "f.rel_path LIKE ? ESCAPE '\\'"],
                    [root.id, like_pat],
                )

        prefix_root_ids: list[int] = []
        for root, root_cf, _ in root_entries:
            if root_cf == norm_cf:
                prefix_root_ids.append(root.id)
            elif norm_cf.endswith("/") and root_cf.startswith(norm_cf):
                prefix_root_ids.append(root.id)
            elif not trailing_sep and root_cf.startswith(norm_cf):
                prefix_root_ids.append(root.id)
            elif trailing_sep and root_cf.startswith(f"{norm_cf}/"):
                prefix_root_ids.append(root.id)
        if prefix_root_ids:
            ph = ",".join("?" for _ in prefix_root_ids)
            return False, [f"f.root_id IN ({ph})"], prefix_root_ids
        return False, ["0 = 1"], []

    def _build_id_query_parts(
        self,
        *,
        query: str,
        root_id: int | None,
        folder_id: int | None,
        folder_rel_path: str | None,
        include_subfolders: bool,
        sort_column: str,
        filters: QueryFilter | None = None,
    ) -> tuple[str, list[Any]]:
        abs_plan = self._resolve_absolute_path_query(query)
        if abs_plan is not None:
            join_fts, where_parts, params = abs_plan[0], list(abs_plan[1]), list(abs_plan[2])
        else:
            join_fts, where_parts, params = build_where_clause(query, alias="f")
        from_sql = "FROM files f"
        if filters is not None and filters.online_only:
            from_sql += " JOIN roots r ON r.id = f.root_id"

        need_um_join = filters is not None and filters.favorites_only
        need_um_left = not need_um_join and sort_column == "favorite"
        if need_um_join:
            from_sql += " JOIN user_meta um ON um.file_id = f.id"
        elif need_um_left:
            from_sql += " LEFT JOIN user_meta um ON um.file_id = f.id"

        include_hidden = filters is not None and filters.include_hidden
        check_hidden = not include_hidden and self.has_hidden_files()
        if check_hidden:
            from_sql += " LEFT JOIN hidden_files hf ON hf.file_id = f.id"

        if join_fts:
            from_sql += " JOIN files_fts ON files_fts.rowid = f.id"

        if filters is not None and filters.recent_only:
            from_sql += " JOIN recent_plays rp ON rp.file_id = f.id"
        elif sort_column == "played_at":
            from_sql += " LEFT JOIN recent_plays rp ON rp.file_id = f.id"

        typed_params: list[Any] = list(params)
        if check_hidden:
            where_parts.append("hf.file_id IS NULL")

        if filters is not None:
            if filters.favorites_only:
                where_parts.append("um.favorite = 1")
            if filters.online_only:
                where_parts.append("f.is_available = 1 AND r.is_online = 1")
            if filters.ext:
                ext_norm = filters.ext.strip().lower().lstrip(".")
                if ext_norm == "__audio__":
                    where_parts.append(
                        "LOWER(f.ext) NOT IN ('mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v')"
                    )
                elif ext_norm == "__video__":
                    where_parts.append(
                        "LOWER(f.ext) IN ('mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v')"
                    )
                elif ext_norm == "aif":
                    where_parts.append("LOWER(f.ext) IN ('aif', 'aiff')")
                elif ext_norm:
                    where_parts.append("LOWER(f.ext) = ?")
                    typed_params.append(ext_norm)
            if filters.min_duration is not None:
                where_parts.append("f.duration_sec >= ?")
                typed_params.append(float(filters.min_duration))
            if filters.max_duration is not None:
                where_parts.append("f.duration_sec <= ?")
                typed_params.append(float(filters.max_duration))
            if filters.collection_id is not None:
                self._append_collection_filter(where_parts, typed_params, int(filters.collection_id))

        if abs_plan is None and root_id is not None:
            omit_sole_root = (
                folder_id is None
                and not folder_rel_path
                and include_subfolders
                and self._is_sole_root(root_id)
            )
            if not omit_sole_root:
                where_parts.append("f.root_id = ?")
                typed_params.append(root_id)
            if folder_rel_path:
                if include_subfolders:
                    prefix = f"{folder_rel_path}/"
                    upper = f"{folder_rel_path}0"
                    where_parts.append("f.rel_path >= ? AND f.rel_path < ?")
                    typed_params.extend([prefix, upper])
                elif folder_id is not None:
                    where_parts.append("f.folder_id = ?")
                    typed_params.append(folder_id)
            elif folder_id is not None:
                where_parts.append("f.folder_id = ?")
                typed_params.append(folder_id)
            elif not include_subfolders:
                where_parts.append("f.folder_id IS NULL")

        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        return from_sql + where_sql, typed_params

    def _build_query_parts(
        self,
        *,
        query: str,
        root_id: int | None,
        folder_id: int | None,
        folder_rel_path: str | None,
        include_subfolders: bool,
        filters: QueryFilter | None = None,
    ) -> tuple[str, list[Any]]:
        abs_plan = self._resolve_absolute_path_query(query)
        if abs_plan is not None:
            join_fts, where_parts, params = abs_plan[0], list(abs_plan[1]), list(abs_plan[2])
        else:
            join_fts, where_parts, params = build_where_clause(query, alias="f")
        from_sql = (
            "FROM files f JOIN roots r ON r.id = f.root_id "
            "LEFT JOIN user_meta um ON um.file_id = f.id "
            "LEFT JOIN hidden_files hf ON hf.file_id = f.id"
        )
        if join_fts:
            from_sql += " JOIN files_fts ON files_fts.rowid = f.id"
        if filters is not None and filters.recent_only:
            from_sql += " JOIN recent_plays rp ON rp.file_id = f.id"

        typed_params: list[Any] = list(params)
        if filters is None or not filters.include_hidden:
            where_parts.append("hf.file_id IS NULL")

        if filters is not None:
            if filters.favorites_only:
                where_parts.append("COALESCE(um.favorite, 0) = 1")
            if filters.online_only:
                where_parts.append("f.is_available = 1 AND r.is_online = 1")
            if filters.ext:
                ext_norm = filters.ext.strip().lower().lstrip(".")
                if ext_norm == "__audio__":
                    where_parts.append(
                        "LOWER(f.ext) NOT IN ('mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v')"
                    )
                elif ext_norm == "__video__":
                    where_parts.append(
                        "LOWER(f.ext) IN ('mp4', 'mov', 'mkv', 'webm', 'avi', 'm4v')"
                    )
                elif ext_norm == "aif":
                    where_parts.append("LOWER(f.ext) IN ('aif', 'aiff')")
                elif ext_norm:
                    where_parts.append("LOWER(f.ext) = ?")
                    typed_params.append(ext_norm)
            if filters.min_duration is not None:
                where_parts.append("f.duration_sec >= ?")
                typed_params.append(float(filters.min_duration))
            if filters.max_duration is not None:
                where_parts.append("f.duration_sec <= ?")
                typed_params.append(float(filters.max_duration))
            if filters.collection_id is not None:
                cid = int(filters.collection_id)
                conn = self._conn()
                has_direct = (
                    conn.execute(
                        "SELECT 1 FROM collection_files WHERE collection_id = ? LIMIT 1",
                        (cid,),
                    ).fetchone()
                    is not None
                )
                cf_rows = conn.execute(
                    "SELECT root_id, folder_id, folder_rel_path FROM collection_folders WHERE collection_id = ?",
                    (cid,),
                ).fetchall()
                pruned_mounts = _prune_overlapping_folder_mounts(
                    [
                        (
                            int(cf["root_id"]),
                            None if cf["folder_id"] is None else int(cf["folder_id"]),
                            str(cf["folder_rel_path"] or ""),
                        )
                        for cf in cf_rows
                    ]
                )
                if not has_direct and not pruned_mounts:
                    where_parts.append("0 = 1")
                elif len(pruned_mounts) <= 200:
                    union_parts: list[str] = []
                    if has_direct:
                        union_parts.append(
                            "SELECT file_id FROM collection_files WHERE collection_id = ?"
                        )
                        typed_params.append(cid)
                    for rid, _fid, rp in pruned_mounts:
                        if rp:
                            union_parts.append(
                                "SELECT id FROM files WHERE root_id = ? AND rel_path >= ? AND rel_path < ?"
                            )
                            typed_params.extend([rid, f"{rp}/", f"{rp}0"])
                        else:
                            union_parts.append("SELECT id FROM files WHERE root_id = ?")
                            typed_params.append(rid)
                    where_parts.append(f"f.id IN ({' UNION '.join(union_parts)})")
                else:
                    coll_clauses = [
                        "EXISTS (SELECT 1 FROM collection_files cf WHERE cf.collection_id = ? AND cf.file_id = f.id)"
                    ]
                    typed_params.append(cid)
                    for rid, _fid, rp in pruned_mounts:
                        if rp:
                            coll_clauses.append(
                                "(f.root_id = ? AND f.rel_path >= ? AND f.rel_path < ?)"
                            )
                            typed_params.extend([rid, f"{rp}/", f"{rp}0"])
                        else:
                            coll_clauses.append("f.root_id = ?")
                            typed_params.append(rid)
                    where_parts.append(f"({' OR '.join(coll_clauses)})")

        if abs_plan is None and root_id is not None:
            where_parts.append("f.root_id = ?")
            typed_params.append(root_id)
            if folder_rel_path:
                if include_subfolders:
                    # ASCII range trick: '/' (47) + 1 == '0' (48) gives an exact prefix scan.
                    prefix = f"{folder_rel_path}/"
                    upper = f"{folder_rel_path}0"
                    where_parts.append("f.rel_path >= ? AND f.rel_path < ?")
                    typed_params.extend([prefix, upper])
                elif folder_id is not None:
                    where_parts.append("f.folder_id = ?")
                    typed_params.append(folder_id)
            elif folder_id is not None:
                where_parts.append("f.folder_id = ?")
                typed_params.append(folder_id)
            elif not include_subfolders:
                where_parts.append("f.folder_id IS NULL")

        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""
        return from_sql + where_sql, typed_params

    def query_count(
        self,
        *,
        query: str = "",
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str | None = None,
        include_subfolders: bool = True,
        filters: QueryFilter | None = None,
    ) -> int:
        # Fast path: unfiltered recursive count when no hidden files exist and root is settled.
        if (
            include_subfolders
            and not query.strip()
            and (filters is None or filters.is_default())
            and self.hidden_file_count() == 0
        ):
            if folder_id is not None:
                folder = self.get_folder(folder_id)
                if folder is not None and folder.file_count > 0:
                    root = self.get_root(folder.root_id)
                    if root is not None and root.status in ("idle", "offline"):
                        return folder.file_count
            elif not folder_rel_path:
                if root_id is None:
                    return self.total_file_count()
                root = self.get_root(root_id)
                if root is not None and (root.file_count > 0 or root.status in ("idle", "offline")):
                    return root.file_count
        body, params = self._build_query_parts(
            query=query,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=include_subfolders,
            filters=filters,
        )
        row = self._conn().execute(f"SELECT COUNT(*) AS c {body}", params).fetchone()
        return 0 if row is None else int(row["c"])

    def query_page(
        self,
        *,
        offset: int,
        limit: int,
        query: str = "",
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str | None = None,
        include_subfolders: bool = True,
        sort_column: str = "filename",
        sort_asc: bool = True,
        filters: QueryFilter | None = None,
        total_count: int | None = None,
    ) -> list[FileRecord]:
        if limit <= 0 or offset < 0:
            return []
        sort_expr = SORT_COLUMN_SQL.get(sort_column, SORT_COLUMN_SQL["filename"])
        direction = "ASC" if sort_asc else "DESC"
        id_body, params = self._build_id_query_parts(
            query=query,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=include_subfolders,
            sort_column=sort_column,
            filters=filters,
        )

        eff_dir = direction
        eff_limit = int(limit)
        eff_offset = int(offset)
        reversed_scan = False
        if total_count is not None and total_count > 0 and offset > total_count // 2:
            actual_end = min(int(total_count), int(offset) + int(limit))
            actual_limit = actual_end - int(offset)
            if actual_limit <= 0:
                return []
            eff_dir = "DESC" if direction == "ASC" else "ASC"
            eff_limit = actual_limit
            eff_offset = int(total_count) - actual_end
            reversed_scan = True

        id_sql = (
            f"SELECT f.id {id_body} ORDER BY {sort_expr} {eff_dir}, f.id {eff_dir} "
            "LIMIT ? OFFSET ?"
        )
        conn = self._conn()
        id_rows = conn.execute(id_sql, [*params, eff_limit, eff_offset]).fetchall()
        page_ids = [int(r["id"]) for r in id_rows]

        if reversed_scan:
            if not page_ids and offset < int(total_count):
                fwd_sql = (
                    f"SELECT f.id {id_body} ORDER BY {sort_expr} {direction}, f.id {direction} "
                    "LIMIT ? OFFSET ?"
                )
                id_rows = conn.execute(fwd_sql, [*params, int(limit), int(offset)]).fetchall()
                page_ids = [int(r["id"]) for r in id_rows]
            else:
                page_ids.reverse()

        if not page_ids:
            return []
        return self.query_page_by_ids(page_ids)

    def query_page_by_ids(self, page_ids: list[int]) -> list[FileRecord]:
        """Hydrate ``FileRecord`` rows in the caller-supplied ID order."""
        if not page_ids:
            return []
        by_id = self.get_files_by_ids(page_ids)
        return [by_id[fid] for fid in page_ids if fid in by_id]

    def query_sorted_id_array(
        self,
        *,
        query: str = "",
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str | None = None,
        include_subfolders: bool = True,
        sort_column: str = "filename",
        sort_asc: bool = True,
        filters: QueryFilter | None = None,
        cancel_event: threading.Event | None = None,
        batch_size: int = 50_000,
    ) -> array.array | None:
        """Stream ordered ``files.id`` values into a compact int64 ``array('q')``."""
        sort_expr = SORT_COLUMN_SQL.get(sort_column, SORT_COLUMN_SQL["filename"])
        direction = "ASC" if sort_asc else "DESC"
        id_body, params = self._build_id_query_parts(
            query=query,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=include_subfolders,
            sort_column=sort_column,
            filters=filters,
        )
        sql = f"SELECT f.id {id_body} ORDER BY {sort_expr} {direction}, f.id {direction}"
        cur = self._conn().execute(sql, params)
        ids = array.array("q")
        try:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    return None
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                ids.extend(r[0] for r in batch)
        finally:
            cur.close()
        if cancel_event is not None and cancel_event.is_set():
            return None
        return ids

    def query_file_position(
        self,
        file_id: int,
        *,
        query: str = "",
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str | None = None,
        include_subfolders: bool = True,
        sort_column: str = "filename",
        sort_asc: bool = True,
        filters: QueryFilter | None = None,
        batch_size: int = 10_000,
    ) -> int | None:
        """Return 0-based position of ``file_id`` under the exact ``query_page`` sort order."""
        target_id = int(file_id)
        sort_expr = SORT_COLUMN_SQL.get(sort_column, SORT_COLUMN_SQL["filename"])
        direction = "ASC" if sort_asc else "DESC"
        id_body, params = self._build_id_query_parts(
            query=query,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=include_subfolders,
            sort_column=sort_column,
            filters=filters,
        )
        sql = f"SELECT f.id {id_body} ORDER BY {sort_expr} {direction}, f.id {direction}"
        cur = self._conn().execute(sql, params)
        offset = 0
        try:
            while True:
                batch = cur.fetchmany(batch_size)
                if not batch:
                    break
                for idx, row in enumerate(batch):
                    if int(row[0]) == target_id:
                        return offset + idx
                offset += len(batch)
        finally:
            cur.close()
        return None

    def get_file(self, file_id: int) -> FileRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT f.*, r.path AS root_path, r.is_online AS root_online, "
                "COALESCE(um.favorite, 0) AS is_favorite, "
                "COALESCE(um.note, '') AS user_note, "
                "CASE WHEN hf.file_id IS NOT NULL THEN 1 ELSE 0 END AS is_hidden, "
                "EXISTS(SELECT 1 FROM metadata_fields mf WHERE mf.file_id = f.id AND mf.dirty = 1) AS is_dirty "
                "FROM files f JOIN roots r ON r.id = f.root_id "
                "LEFT JOIN user_meta um ON um.file_id = f.id "
                "LEFT JOIN hidden_files hf ON hf.file_id = f.id WHERE f.id = ?",
                (file_id,),
            )
            .fetchone()
        )
        return None if row is None else _row_to_file(row)

    def get_files_by_ids(self, file_ids: list[int]) -> dict[int, FileRecord]:
        if not file_ids:
            return {}
        unique_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        conn = self._conn()
        raw_rows: list[sqlite3.Row] = []
        for i in range(0, len(unique_ids), 500):
            chunk = unique_ids[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            fetched = conn.execute(
                "SELECT f.*, r.path AS root_path, r.is_online AS root_online, "
                "COALESCE(um.favorite, 0) AS is_favorite, "
                "COALESCE(um.note, '') AS user_note, "
                "CASE WHEN hf.file_id IS NOT NULL THEN 1 ELSE 0 END AS is_hidden, "
                "EXISTS(SELECT 1 FROM metadata_fields mf WHERE mf.file_id = f.id AND mf.dirty = 1) AS is_dirty "
                "FROM files f JOIN roots r ON r.id = f.root_id "
                "LEFT JOIN user_meta um ON um.file_id = f.id "
                f"LEFT JOIN hidden_files hf ON hf.file_id = f.id WHERE f.id IN ({placeholders})",
                chunk,
            ).fetchall()
            raw_rows.extend(fetched)
        if not raw_rows:
            return {}
        entries = self.get_cached_thumbnail_entries(
            [(int(r["id"]), int(r["mtime_ns"])) for r in raw_rows]
        )
        out: dict[int, FileRecord] = {}
        for r in raw_rows:
            fid = int(r["id"])
            thumb_blob, spec_blob = entries.get(fid, (None, None))
            out[fid] = _row_to_file(r, thumb_blob=thumb_blob, spectral_thumb_blob=spec_blob)
        return out

    def query_scoped_file_ids(
        self,
        *,
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str | None = None,
        include_subfolders: bool = True,
        filters: QueryFilter | None = None,
    ) -> set[int]:
        body, params = self._build_query_parts(
            query="",
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=include_subfolders,
            filters=filters,
        )
        rows = self._conn().execute(f"SELECT f.id {body}", params).fetchall()
        return {int(r["id"]) for r in rows}

    def iter_files_for_indexing(
        self,
        *,
        root_id: int | None = None,
        folder_rel_path: str = "",
        online_only: bool = True,
        batch_size: int = 500,
    ):
        conn = self._conn()
        last_id = 0
        while True:
            where = ["f.id > ?"]
            params: list[Any] = [last_id]
            if online_only:
                where.append("f.is_available = 1 AND r.is_online = 1")
            if root_id is not None:
                where.append("f.root_id = ?")
                params.append(root_id)
                if folder_rel_path:
                    where.append("f.rel_path >= ? AND f.rel_path < ?")
                    params.extend([f"{folder_rel_path}/", f"{folder_rel_path}0"])
            sql = (
                "SELECT f.*, r.path AS root_path, r.is_online AS root_online "
                "FROM files f JOIN roots r ON r.id = f.root_id "
                f"WHERE {' AND '.join(where)} ORDER BY f.id ASC LIMIT ?"
            )
            rows = conn.execute(sql, [*params, batch_size]).fetchall()
            if not rows:
                break
            batch = [_row_to_file(r) for r in rows]
            last_id = batch[-1].id
            yield batch

    def get_metadata_fields(self, file_id: int) -> list[tuple[str, str, str]]:
        rows = (
            self._conn()
            .execute(
                "SELECT field, value, source FROM metadata_fields WHERE file_id = ? ORDER BY field",
                (file_id,),
            )
            .fetchall()
        )
        return [(str(r["field"]), str(r["value"]), str(r["source"])) for r in rows]

    def get_metadata_field_rows(self, file_id: int) -> list[tuple[str, str, str, int]]:
        rows = (
            self._conn()
            .execute(
                "SELECT field, value, source, dirty FROM metadata_fields WHERE file_id = ? ORDER BY field",
                (int(file_id),),
            )
            .fetchall()
        )
        return [(str(r["field"]), str(r["value"]), str(r["source"]), int(r["dirty"])) for r in rows]

    def get_canonical_metadata(self, file_id: int) -> dict[str, tuple[str, str, bool]]:
        rec = self.get_file(int(file_id))
        if rec is None:
            return {k: ("", "embedded", False) for k in EDITABLE_FIELD_KEYS}
        return extract_canonical_from_raw(rec, self.get_metadata_field_rows(int(file_id)))

    def _refresh_file_summary_conn(self, conn: sqlite3.Connection, file_id: int) -> None:
        row = conn.execute(
            "SELECT f.*, r.path AS root_path, r.label AS root_label, r.is_online AS root_online "
            "FROM files f JOIN roots r ON r.id = f.root_id WHERE f.id = ?",
            (int(file_id),),
        ).fetchone()
        if row is None:
            return
        raw_rows = conn.execute(
            "SELECT field, value, source, dirty FROM metadata_fields WHERE file_id = ?",
            (int(file_id),),
        ).fetchall()
        raw_list = [
            (str(r["field"]), str(r["value"]), str(r["source"]), int(r["dirty"])) for r in raw_rows
        ]
        rec = _row_to_file(row)
        canonical = extract_canonical_from_raw(rec, raw_list)
        rel_parts = str(row["rel_path"]).split("/")
        fallback_pkg = rel_parts[0] if len(rel_parts) > 1 else str(row["root_label"])
        desc, pkg, emb_text = compute_file_summary_columns(
            canonical, raw_list, fallback_package=fallback_pkg
        )
        conn.execute(
            "UPDATE files SET description = ?, package = ?, embedded_text = ? WHERE id = ?",
            (desc, pkg, emb_text, int(file_id)),
        )

    def update_file_metadata(
        self,
        file_id: int,
        updates: dict[str, str],
        *,
        source: str = "user",
    ) -> FileRecord | None:
        """Persist editable metadata edits into the local library DB with ``dirty = 1``."""
        fid = int(file_id)
        rec = self.get_file(fid)
        if rec is None:
            return None
        clean_updates: dict[str, str] = {}
        for k, v in updates.items():
            if k not in FIELD_SPEC_MAP:
                continue
            clean_updates[k] = str(v).strip() if not FIELD_SPEC_MAP[k].multiline else str(v).strip()
        if not clean_updates:
            return rec

        before = extract_canonical_from_raw(rec, self.get_metadata_field_rows(fid))
        now = time.time()
        conn = self._conn()
        with conn:
            for field_key, new_val in clean_updates.items():
                cur_val = before.get(field_key, ("", "embedded", False))[0]
                is_already_dirty = before.get(field_key, ("", "embedded", False))[2]
                if new_val == cur_val and not is_already_dirty:
                    continue
                conn.execute(
                    """
                    INSERT INTO metadata_fields(file_id, field, value, source, dirty, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                    ON CONFLICT(file_id, field) DO UPDATE SET
                        value = excluded.value,
                        source = excluded.source,
                        dirty = 1,
                        updated_at = excluded.updated_at
                    """,
                    (fid, field_key, new_val, source, now),
                )
            self._refresh_file_summary_conn(conn, fid)
        self.bump_index_version()
        return self.get_file(fid)

    def batch_update_metadata(
        self,
        file_ids: list[int],
        field: str,
        op: str,
        value: str = "",
        *,
        find: str = "",
        replace_with: str = "",
        case_sensitive: bool = False,
        source: str = "user",
    ) -> int:
        if field not in FIELD_SPEC_MAP:
            raise ValueError(f"不支持编辑的字段：{field!r}")
        unique_ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        if not unique_ids:
            return 0
        # Validate transform syntax once upfront before touching DB.
        apply_field_transform("", op, value, find=find, replace_with=replace_with, case_sensitive=case_sensitive)

        now = time.time()
        changed = 0
        conn = self._conn()
        recs = self.get_files_by_ids(unique_ids)
        with conn:
            for fid in unique_ids:
                rec = recs.get(fid)
                if rec is None:
                    continue
                raw_rows = conn.execute(
                    "SELECT field, value, source, dirty FROM metadata_fields WHERE file_id = ?",
                    (fid,),
                ).fetchall()
                raw_list = [
                    (str(r["field"]), str(r["value"]), str(r["source"]), int(r["dirty"]))
                    for r in raw_rows
                ]
                canonical = extract_canonical_from_raw(rec, raw_list)
                cur_val, _cur_src, cur_dirty = canonical.get(field, ("", "embedded", False))
                new_val = apply_field_transform(
                    cur_val,
                    op,
                    value,
                    find=find,
                    replace_with=replace_with,
                    case_sensitive=case_sensitive,
                ).strip()
                if new_val == cur_val and not cur_dirty:
                    continue
                conn.execute(
                    """
                    INSERT INTO metadata_fields(file_id, field, value, source, dirty, updated_at)
                    VALUES (?, ?, ?, ?, 1, ?)
                    ON CONFLICT(file_id, field) DO UPDATE SET
                        value = excluded.value,
                        source = excluded.source,
                        dirty = 1,
                        updated_at = excluded.updated_at
                    """,
                    (fid, field, new_val, source, now),
                )
                self._refresh_file_summary_conn(conn, fid)
                changed += 1
        if changed:
            self.bump_index_version()
        return changed

    def dirty_file_ids(self) -> list[int]:
        rows = (
            self._conn()
            .execute("SELECT DISTINCT file_id FROM metadata_fields WHERE dirty = 1 ORDER BY file_id ASC")
            .fetchall()
        )
        return [int(r["file_id"]) for r in rows]

    def dirty_file_count(self) -> int:
        row = (
            self._conn()
            .execute("SELECT COUNT(DISTINCT file_id) AS c FROM metadata_fields WHERE dirty = 1")
            .fetchone()
        )
        return 0 if row is None else int(row["c"])

    def discard_dirty_metadata(self, file_ids: list[int] | None = None) -> int:
        conn = self._conn()
        with conn:
            if file_ids is None:
                affected = [
                    int(r["file_id"])
                    for r in conn.execute("SELECT DISTINCT file_id FROM metadata_fields WHERE dirty = 1").fetchall()
                ]
                if not affected:
                    return 0
                conn.execute("DELETE FROM metadata_fields WHERE dirty = 1")
            else:
                affected = list(dict.fromkeys(int(fid) for fid in file_ids))
                if not affected:
                    return 0
                for i in range(0, len(affected), 500):
                    chunk = affected[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"DELETE FROM metadata_fields WHERE dirty = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
            for fid in affected:
                self._refresh_file_summary_conn(conn, fid)
            return len(affected)

    def create_write_batch(
        self,
        batch_id: str,
        *,
        file_count: int,
        backup_dir: str,
        status: str = "pending",
    ) -> WriteBatchRecord:
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO write_batches(batch_id, created_at, file_count, backup_dir, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (batch_id, now, int(file_count), str(backup_dir), status),
            )
        return WriteBatchRecord(batch_id, now, int(file_count), str(backup_dir), status)

    def update_write_batch_status(
        self, batch_id: str, status: str, *, file_count: int | None = None
    ) -> None:
        conn = self._conn()
        with conn:
            if file_count is not None:
                conn.execute(
                    "UPDATE write_batches SET status = ?, file_count = ? WHERE batch_id = ?",
                    (status, int(file_count), batch_id),
                )
            else:
                conn.execute(
                    "UPDATE write_batches SET status = ? WHERE batch_id = ?",
                    (status, batch_id),
                )

    def list_write_batches(self, limit: int = 30) -> list[WriteBatchRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT batch_id, created_at, file_count, backup_dir, status "
                "FROM write_batches ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            )
            .fetchall()
        )
        return [
            WriteBatchRecord(
                batch_id=str(r["batch_id"]),
                created_at=float(r["created_at"]),
                file_count=int(r["file_count"]),
                backup_dir=str(r["backup_dir"]),
                status=str(r["status"]),
            )
            for r in rows
        ]

    def get_write_batch(self, batch_id: str) -> WriteBatchRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT batch_id, created_at, file_count, backup_dir, status "
                "FROM write_batches WHERE batch_id = ?",
                (batch_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return WriteBatchRecord(
            batch_id=str(row["batch_id"]),
            created_at=float(row["created_at"]),
            file_count=int(row["file_count"]),
            backup_dir=str(row["backup_dir"]),
            status=str(row["status"]),
        )

    def create_rename_batch(
        self,
        batch_id: str,
        *,
        file_count: int,
        template: str,
        manifest_path: str,
        status: str = "pending",
    ) -> RenameBatchRecord:
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                "INSERT INTO rename_batches(batch_id, created_at, file_count, template, manifest_path, status) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (batch_id, now, int(file_count), template, str(manifest_path), status),
            )
        return RenameBatchRecord(batch_id, now, int(file_count), template, str(manifest_path), status)

    def update_rename_batch_status(
        self, batch_id: str, status: str, *, file_count: int | None = None
    ) -> None:
        conn = self._conn()
        with conn:
            if file_count is not None:
                conn.execute(
                    "UPDATE rename_batches SET status = ?, file_count = ? WHERE batch_id = ?",
                    (status, int(file_count), batch_id),
                )
            else:
                conn.execute(
                    "UPDATE rename_batches SET status = ? WHERE batch_id = ?",
                    (status, batch_id),
                )

    def list_rename_batches(self, limit: int = 30) -> list[RenameBatchRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT batch_id, created_at, file_count, template, manifest_path, status "
                "FROM rename_batches ORDER BY created_at DESC LIMIT ?",
                (int(limit),),
            )
            .fetchall()
        )
        return [
            RenameBatchRecord(
                batch_id=str(r["batch_id"]),
                created_at=float(r["created_at"]),
                file_count=int(r["file_count"]),
                template=str(r["template"]),
                manifest_path=str(r["manifest_path"]),
                status=str(r["status"]),
            )
            for r in rows
        ]

    def get_rename_batch(self, batch_id: str) -> RenameBatchRecord | None:
        row = (
            self._conn()
            .execute(
                "SELECT batch_id, created_at, file_count, template, manifest_path, status "
                "FROM rename_batches WHERE batch_id = ?",
                (batch_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return RenameBatchRecord(
            batch_id=str(row["batch_id"]),
            created_at=float(row["created_at"]),
            file_count=int(row["file_count"]),
            template=str(row["template"]),
            manifest_path=str(row["manifest_path"]),
            status=str(row["status"]),
        )

    def apply_file_renames_db(self, renames: list[tuple[int, str, str]]) -> None:
        """Update ``rel_path`` and ``filename`` via two-phase SQL so cyclic swaps never collide."""
        if not renames:
            return
        now = time.time()
        conn = self._conn()
        with conn:
            for fid, _new_rel, _new_name in renames:
                conn.execute(
                    "UPDATE files SET rel_path = ? WHERE id = ?",
                    (f"__tmp_ren_{int(fid)}_{int(now * 1e6)}", int(fid)),
                )
            for fid, new_rel, new_name in renames:
                conn.execute(
                    "UPDATE files SET rel_path = ?, filename = ?, indexed_at = ? WHERE id = ?",
                    (new_rel, new_name, now, int(fid)),
                )
        self.bump_index_version()

    def sync_file_after_metadata_write(
        self,
        file_id: int,
        *,
        size_bytes: int,
        mtime_ns: int,
        bext_json: str,
        ixml_json: str,
        embedded_fields: list[tuple[str, str]],
        canonical_values: dict[str, str],
        fallback_package: str = "",
        keep_dirty: bool = False,
    ) -> None:
        """Refresh file stats, cached waveform mtime, and embedded rows after disk write/undo."""
        fid = int(file_id)
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                "DELETE FROM metadata_fields WHERE file_id = ? AND source = 'embedded'",
                (fid,),
            )
            if not keep_dirty:
                conn.execute(
                    "DELETE FROM metadata_fields WHERE file_id = ? AND dirty = 1",
                    (fid,),
                )
            if embedded_fields:
                conn.executemany(
                    """
                    INSERT INTO metadata_fields(file_id, field, value, source, dirty, updated_at)
                    VALUES (?, ?, ?, 'embedded', 0, ?)
                    ON CONFLICT(file_id, field) DO UPDATE SET
                        value = CASE WHEN metadata_fields.dirty = 0 THEN excluded.value ELSE metadata_fields.value END,
                        source = CASE WHEN metadata_fields.dirty = 0 THEN 'embedded' ELSE metadata_fields.source END,
                        updated_at = excluded.updated_at
                    """,
                    [(fid, fname, fval, now) for fname, fval in embedded_fields],
                )
            if canonical_values:
                dirty_flag = 1 if keep_dirty else 0
                src_name = "user" if keep_dirty else "embedded"
                for ck, cv in canonical_values.items():
                    if ck not in FIELD_SPEC_MAP:
                        continue
                    if not keep_dirty and not cv.strip():
                        conn.execute(
                            "DELETE FROM metadata_fields WHERE file_id = ? AND field = ?",
                            (fid, ck),
                        )
                    else:
                        conn.execute(
                            """
                            INSERT INTO metadata_fields(file_id, field, value, source, dirty, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            ON CONFLICT(file_id, field) DO UPDATE SET
                                value = excluded.value,
                                source = excluded.source,
                                dirty = excluded.dirty,
                                updated_at = excluded.updated_at
                            """,
                            (fid, ck, cv, src_name, dirty_flag, now),
                        )
            conn.execute(
                "UPDATE files SET size_bytes = ?, mtime_ns = ?, bext_json = ?, ixml_json = ?, indexed_at = ? WHERE id = ?",
                (int(size_bytes), int(mtime_ns), bext_json, ixml_json, now, fid),
            )
            conn.execute(
                "UPDATE waveform_cache SET mtime_ns = ? WHERE file_id = ?",
                (int(mtime_ns), fid),
            )
            conn.execute(
                "UPDATE waveform_thumbs SET mtime_ns = ? WHERE file_id = ?",
                (int(mtime_ns), fid),
            )
            self._refresh_file_summary_conn(conn, fid)
            if fallback_package:
                row = conn.execute("SELECT package FROM files WHERE id = ?", (fid,)).fetchone()
                if row is not None and not str(row["package"] or "").strip():
                    conn.execute("UPDATE files SET package = ? WHERE id = ?", (fallback_package, fid))
        self.bump_index_version()

    # -- waveform cache ----------------------------------------------------
    def get_cached_peaks(self, file_id: int, mtime_ns: int, points: int) -> PeakData | None:
        row = (
            self._conn()
            .execute(
                "SELECT peaks_blob FROM waveform_cache "
                "WHERE file_id = ? AND mtime_ns = ? AND points = ?",
                (file_id, mtime_ns, points),
            )
            .fetchone()
        )
        if row is None:
            return None
        try:
            return deserialize_peaks(bytes(row["peaks_blob"]))
        except Exception:
            return None

    def save_cached_peaks(
        self,
        file_id: int,
        mtime_ns: int,
        peak_data: PeakData,
        *,
        requested_points: int | None = None,
    ) -> None:
        # Only persist full-file overview peaks into waveform_cache.
        if peak_data.start_seconds > 1e-4:
            return
        if peak_data.duration_seconds > 0 and abs(peak_data.end_seconds - peak_data.duration_seconds) > 0.05:
            return
        blob = serialize_peaks(peak_data)
        thumb_blob = peaks_to_thumbnail(peak_data)
        spectral_thumb = peaks_to_thumbnail_spectral(peak_data) or None
        points_key = requested_points if requested_points is not None else peak_data.points
        conn = self._conn()
        with conn:
            conn.execute(
                """
                INSERT INTO waveform_cache(file_id, mtime_ns, peaks_blob, points)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    peaks_blob = excluded.peaks_blob,
                    points = excluded.points
                """,
                (file_id, mtime_ns, blob, points_key),
            )
            if thumb_blob:
                conn.execute(
                    """
                    INSERT INTO waveform_thumbs(file_id, mtime_ns, thumb_blob, spectral_blob)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(file_id) DO UPDATE SET
                        mtime_ns = excluded.mtime_ns,
                        thumb_blob = excluded.thumb_blob,
                        spectral_blob = CASE
                            WHEN waveform_thumbs.mtime_ns = excluded.mtime_ns
                            THEN COALESCE(excluded.spectral_blob, waveform_thumbs.spectral_blob)
                            ELSE excluded.spectral_blob
                        END
                    """,
                    (file_id, mtime_ns, thumb_blob, spectral_thumb),
                )

    def get_cached_thumbnail_entries(
        self, items: list[tuple[int, int]]
    ) -> dict[int, tuple[bytes, bytes | None]]:
        if not items:
            return {}
        expected_mtime = {int(fid): int(mtime) for fid, mtime in items}
        ids = list(expected_mtime.keys())
        conn = self._conn()
        out: dict[int, tuple[bytes, bytes | None]] = {}
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"SELECT file_id, mtime_ns, thumb_blob, spectral_blob "
                f"FROM waveform_thumbs WHERE file_id IN ({placeholders})",
                chunk,
            ).fetchall()
            for r in rows:
                fid = int(r["file_id"])
                if int(r["mtime_ns"]) == expected_mtime.get(fid):
                    spec = None if r["spectral_blob"] is None else bytes(r["spectral_blob"])
                    out[fid] = (bytes(r["thumb_blob"]), spec)
        return out

    def get_cached_thumbnails(self, items: list[tuple[int, int]]) -> dict[int, bytes]:
        return {
            fid: entry[0] for fid, entry in self.get_cached_thumbnail_entries(items).items()
        }

    def get_cached_thumbnail_entry(
        self, file_id: int, mtime_ns: int
    ) -> tuple[bytes, bytes | None] | None:
        conn = self._conn()
        row = conn.execute(
            "SELECT thumb_blob, spectral_blob FROM waveform_thumbs WHERE file_id = ? AND mtime_ns = ?",
            (file_id, mtime_ns),
        ).fetchone()
        if row is not None:
            spec = None if row["spectral_blob"] is None else bytes(row["spectral_blob"])
            return (bytes(row["thumb_blob"]), spec)
        cache_row = conn.execute(
            "SELECT peaks_blob FROM waveform_cache WHERE file_id = ? AND mtime_ns = ?",
            (file_id, mtime_ns),
        ).fetchone()
        if cache_row is None:
            return None
        try:
            data = deserialize_peaks(bytes(cache_row["peaks_blob"]))
            thumb = peaks_to_thumbnail(data)
            spec_thumb = peaks_to_thumbnail_spectral(data) or None
        except Exception:
            return None
        if thumb:
            self.save_cached_thumbnail_entries([(file_id, mtime_ns, thumb, spec_thumb)])
            return (thumb, spec_thumb)
        return None

    def get_cached_thumbnail(self, file_id: int, mtime_ns: int) -> bytes | None:
        entry = self.get_cached_thumbnail_entry(file_id, mtime_ns)
        return None if entry is None else entry[0]

    def save_cached_thumbnail_entries(
        self, items: list[tuple[int, int, bytes, bytes | None]]
    ) -> None:
        if not items:
            return
        conn = self._conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO waveform_thumbs(file_id, mtime_ns, thumb_blob, spectral_blob)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    mtime_ns = excluded.mtime_ns,
                    thumb_blob = excluded.thumb_blob,
                    spectral_blob = CASE
                        WHEN waveform_thumbs.mtime_ns = excluded.mtime_ns
                        THEN COALESCE(excluded.spectral_blob, waveform_thumbs.spectral_blob)
                        ELSE excluded.spectral_blob
                    END
                """,
                [
                    (
                        int(fid),
                        int(mtime),
                        bytes(blob),
                        None if spec is None else bytes(spec),
                    )
                    for fid, mtime, blob, spec in items
                ],
            )

    def save_cached_thumbnails(self, items: list[tuple[int, int, bytes]]) -> None:
        self.save_cached_thumbnail_entries(
            [(fid, mtime, blob, None) for fid, mtime, blob in items]
        )

    # -- R4: favorites & notes ---------------------------------------------
    def set_favorites(self, file_ids: list[int], favorite: bool) -> int:
        ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        if not ids:
            return 0
        now = time.time()
        fav_int = 1 if favorite else 0
        conn = self._conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO user_meta(file_id, favorite, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    favorite = excluded.favorite,
                    updated_at = excluded.updated_at
                """,
                [(fid, fav_int, now) for fid in ids],
            )
        self.bump_index_version()
        return len(ids)

    def toggle_favorite(self, file_id: int) -> bool:
        row = (
            self._conn()
            .execute("SELECT favorite FROM user_meta WHERE file_id = ?", (int(file_id),))
            .fetchone()
        )
        next_val = not (bool(row["favorite"]) if row is not None else False)
        self.set_favorites([file_id], next_val)
        return next_val

    def set_note(self, file_id: int, note: str) -> None:
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                """
                INSERT INTO user_meta(file_id, note, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    note = excluded.note,
                    updated_at = excluded.updated_at
                """,
                (int(file_id), note, now),
            )

    def favorite_count(self) -> int:
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*) AS c FROM user_meta um "
                "JOIN files f ON f.id = um.file_id "
                "LEFT JOIN hidden_files hf ON hf.file_id = f.id "
                "WHERE um.favorite = 1 AND hf.file_id IS NULL"
            )
            .fetchone()
        )
        return 0 if row is None else int(row["c"])

    # -- R4: recent plays --------------------------------------------------
    def record_recent_play(self, file_id: int, *, max_entries: int = 200) -> None:
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                """
                INSERT INTO recent_plays(file_id, played_at)
                VALUES (?, ?)
                ON CONFLICT(file_id) DO UPDATE SET played_at = excluded.played_at
                """,
                (int(file_id), now),
            )
            conn.execute(
                """
                DELETE FROM recent_plays WHERE file_id NOT IN (
                    SELECT file_id FROM recent_plays ORDER BY played_at DESC LIMIT ?
                )
                """,
                (int(max_entries),),
            )

    def recent_play_count(self) -> int:
        row = (
            self._conn()
            .execute(
                "SELECT COUNT(*) AS c FROM recent_plays rp "
                "JOIN files f ON f.id = rp.file_id "
                "LEFT JOIN hidden_files hf ON hf.file_id = f.id "
                "WHERE hf.file_id IS NULL"
            )
            .fetchone()
        )
        return 0 if row is None else int(row["c"])

    def clear_recent_plays(self) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM recent_plays")

    # -- R4: unified collections (folders + files) -------------------------
    def list_collections(self) -> list[CollectionRecord]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM collections ORDER BY sort_order ASC, name COLLATE NOCASE ASC, id ASC"
        ).fetchall()
        out: list[CollectionRecord] = []
        for r in rows:
            cid = int(r["id"])
            fc_row = conn.execute(
                "SELECT COUNT(*) AS c FROM collection_folders WHERE collection_id = ?", (cid,)
            ).fetchone()
            item_count = self.collection_item_count(cid)
            out.append(
                CollectionRecord(
                    id=cid,
                    name=str(r["name"]),
                    sort_order=int(r["sort_order"]),
                    folder_count=0 if fc_row is None else int(fc_row["c"]),
                    file_count=item_count,
                )
            )
        return out

    def create_collection(self, name: str) -> CollectionRecord:
        clean = name.strip()
        if not clean:
            raise ValueError("Collection name cannot be empty")
        conn = self._conn()
        row = conn.execute("SELECT id FROM collections WHERE name = ?", (clean,)).fetchone()
        if row is not None:
            raise ValueError(f"Collection already exists: {clean}")
        max_order = conn.execute("SELECT COALESCE(MAX(sort_order), 0) FROM collections").fetchone()
        next_order = (0 if max_order is None else int(max_order[0])) + 1
        with conn:
            cur = conn.execute(
                "INSERT INTO collections(name, sort_order, created_at) VALUES (?, ?, ?)",
                (clean, next_order, time.time()),
            )
            cid = int(cur.lastrowid)
        return CollectionRecord(id=cid, name=clean, sort_order=next_order, folder_count=0, file_count=0)

    def get_or_create_collection(self, name: str) -> tuple[CollectionRecord, bool]:
        clean = name.strip()
        if not clean:
            raise ValueError("Collection name cannot be empty")
        row = self._conn().execute("SELECT * FROM collections WHERE name = ?", (clean,)).fetchone()
        if row is not None:
            cid = int(row["id"])
            fc_row = (
                self._conn()
                .execute("SELECT COUNT(*) AS c FROM collection_folders WHERE collection_id = ?", (cid,))
                .fetchone()
            )
            return (
                CollectionRecord(
                    id=cid,
                    name=str(row["name"]),
                    sort_order=int(row["sort_order"]),
                    folder_count=0 if fc_row is None else int(fc_row["c"]),
                    file_count=self.collection_item_count(cid),
                ),
                False,
            )
        return self.create_collection(clean), True

    def rename_collection(self, collection_id: int, name: str) -> None:
        clean = name.strip()
        if not clean:
            raise ValueError("Collection name cannot be empty")
        conn = self._conn()
        with conn:
            conn.execute("UPDATE collections SET name = ? WHERE id = ?", (clean, int(collection_id)))

    def reorder_collections(self, ordered_collection_ids: list[int]) -> None:
        current_ids = [c.id for c in self.list_collections()]
        valid = set(current_ids)
        merged: list[int] = []
        seen: set[int] = set()
        for cid in ordered_collection_ids:
            cid_int = int(cid)
            if cid_int in valid and cid_int not in seen:
                seen.add(cid_int)
                merged.append(cid_int)
        for cid_int in current_ids:
            if cid_int not in seen:
                seen.add(cid_int)
                merged.append(cid_int)
        if not merged:
            return
        conn = self._conn()
        with conn:
            conn.executemany(
                "UPDATE collections SET sort_order = ? WHERE id = ?",
                [(pos, cid_int) for pos, cid_int in enumerate(merged, start=1)],
            )

    def delete_collection(self, collection_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM collections WHERE id = ?", (int(collection_id),))

    def list_collection_folders(self, collection_id: int) -> list[CollectionFolderRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT cf.*, r.path AS root_path FROM collection_folders cf "
                "JOIN roots r ON r.id = cf.root_id WHERE cf.collection_id = ? "
                "ORDER BY CASE WHEN cf.sort_order > 0 THEN 0 ELSE 1 END, cf.sort_order ASC, cf.label COLLATE NOCASE, cf.id ASC",
                (int(collection_id),),
            )
            .fetchall()
        )
        return [
            CollectionFolderRecord(
                id=int(r["id"]),
                collection_id=int(r["collection_id"]),
                root_id=int(r["root_id"]),
                folder_id=None if r["folder_id"] is None else int(r["folder_id"]),
                folder_rel_path=str(r["folder_rel_path"] or ""),
                label=str(r["label"] or ""),
                root_path=str(r["root_path"]),
                sort_order=int(r["sort_order"] or 0),
            )
            for r in rows
        ]

    def list_all_collection_folders(self) -> dict[int, list[CollectionFolderRecord]]:
        rows = (
            self._conn()
            .execute(
                "SELECT cf.*, r.path AS root_path FROM collection_folders cf "
                "JOIN roots r ON r.id = cf.root_id "
                "ORDER BY cf.collection_id ASC, CASE WHEN cf.sort_order > 0 THEN 0 ELSE 1 END, "
                "cf.sort_order ASC, cf.label COLLATE NOCASE, cf.id ASC"
            )
            .fetchall()
        )
        out: dict[int, list[CollectionFolderRecord]] = {}
        for r in rows:
            rec = CollectionFolderRecord(
                id=int(r["id"]),
                collection_id=int(r["collection_id"]),
                root_id=int(r["root_id"]),
                folder_id=None if r["folder_id"] is None else int(r["folder_id"]),
                folder_rel_path=str(r["folder_rel_path"] or ""),
                label=str(r["label"] or ""),
                root_path=str(r["root_path"]),
                sort_order=int(r["sort_order"] or 0),
            )
            out.setdefault(rec.collection_id, []).append(rec)
        return out

    def add_folder_to_collection(
        self,
        collection_id: int,
        root_id: int,
        folder_id: int | None,
        folder_rel_path: str,
        label: str = "",
        *,
        sort_order: int | None = None,
    ) -> None:
        clean_rel = folder_rel_path.strip("/")
        display_label = label.strip()
        if not display_label:
            if clean_rel:
                display_label = Path(clean_rel).name
            else:
                root = self.get_root(root_id)
                display_label = root.label if root is not None else f"Root {root_id}"
        conn = self._conn()
        with conn:
            if sort_order is None:
                conn.execute(
                    """
                    INSERT INTO collection_folders(collection_id, root_id, folder_id, folder_rel_path, label, sort_order, added_at)
                    VALUES (?, ?, ?, ?, ?, 0, ?)
                    ON CONFLICT(collection_id, root_id, folder_rel_path) DO UPDATE SET
                        label = excluded.label,
                        folder_id = excluded.folder_id
                    """,
                    (int(collection_id), int(root_id), folder_id, clean_rel, display_label, time.time()),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO collection_folders(collection_id, root_id, folder_id, folder_rel_path, label, sort_order, added_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection_id, root_id, folder_rel_path) DO UPDATE SET
                        label = excluded.label,
                        folder_id = excluded.folder_id,
                        sort_order = excluded.sort_order
                    """,
                    (
                        int(collection_id),
                        int(root_id),
                        folder_id,
                        clean_rel,
                        display_label,
                        int(sort_order),
                        time.time(),
                    ),
                )
        self.bump_index_version()

    def remove_folder_from_collection(self, entry_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM collection_folders WHERE id = ?", (int(entry_id),))
        self.bump_index_version()

    def move_or_reorder_collection_folder(
        self, entry_id: int, target_collection_id: int, target_index: int
    ) -> bool:
        """Reorder a folder within a collection or move it cleanly into another collection."""
        conn = self._conn()
        src = conn.execute(
            "SELECT id, collection_id, root_id, folder_rel_path FROM collection_folders WHERE id = ?",
            (int(entry_id),),
        ).fetchone()
        if src is None:
            return False
        tgt_exists = conn.execute(
            "SELECT id FROM collections WHERE id = ?", (int(target_collection_id),)
        ).fetchone()
        if tgt_exists is None:
            return False

        src_id = int(src["id"])
        src_coll_id = int(src["collection_id"])
        tgt_coll_id = int(target_collection_id)
        root_id = int(src["root_id"])
        rel_path = str(src["folder_rel_path"] or "")

        target_ids = [cf.id for cf in self.list_collection_folders(tgt_coll_id)]
        if src_coll_id == tgt_coll_id:
            target_ids = [x for x in target_ids if x != src_id]
            idx = max(0, min(int(target_index), len(target_ids)))
            target_ids.insert(idx, src_id)
            with conn:
                conn.executemany(
                    "UPDATE collection_folders SET sort_order = ? WHERE id = ?",
                    [(pos, cf_id) for pos, cf_id in enumerate(target_ids, start=1)],
                )
            return True

        dup_row = conn.execute(
            "SELECT id FROM collection_folders "
            "WHERE collection_id = ? AND root_id = ? AND folder_rel_path = ?",
            (tgt_coll_id, root_id, rel_path),
        ).fetchone()
        src_remaining = [
            cf.id for cf in self.list_collection_folders(src_coll_id) if cf.id != src_id
        ]
        with conn:
            if dup_row is not None:
                keep_id = int(dup_row["id"])
                conn.execute("DELETE FROM collection_folders WHERE id = ?", (src_id,))
                target_ids = [x for x in target_ids if x != keep_id]
                idx = max(0, min(int(target_index), len(target_ids)))
                target_ids.insert(idx, keep_id)
            else:
                conn.execute(
                    "UPDATE collection_folders SET collection_id = ? WHERE id = ?",
                    (tgt_coll_id, src_id),
                )
                idx = max(0, min(int(target_index), len(target_ids)))
                target_ids.insert(idx, src_id)
            conn.executemany(
                "UPDATE collection_folders SET sort_order = ? WHERE id = ?",
                [(pos, cf_id) for pos, cf_id in enumerate(target_ids, start=1)],
            )
            if src_remaining:
                conn.executemany(
                    "UPDATE collection_folders SET sort_order = ? WHERE id = ?",
                    [(pos, cf_id) for pos, cf_id in enumerate(src_remaining, start=1)],
                )
        self.bump_index_version()
        return True

    def add_files_to_collection(self, collection_id: int, file_ids: list[int]) -> int:
        ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        if not ids:
            return 0
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany(
                "INSERT OR IGNORE INTO collection_files(collection_id, file_id, added_at) VALUES (?, ?, ?)",
                [(int(collection_id), fid, now) for fid in ids],
            )
        self.bump_index_version()
        return len(ids)

    def remove_files_from_collection(self, collection_id: int, file_ids: list[int]) -> int:
        ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        if not ids:
            return 0
        conn = self._conn()
        with conn:
            for i in range(0, len(ids), 500):
                chunk = ids[i : i + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"DELETE FROM collection_files WHERE collection_id = ? AND file_id IN ({placeholders})",
                    [int(collection_id), *chunk],
                )
        self.bump_index_version()
        return len(ids)

    def collection_item_count(self, collection_id: int) -> int:
        cid = int(collection_id)
        conn = self._conn()
        if self.hidden_file_count() == 0:
            has_direct = (
                conn.execute(
                    "SELECT 1 FROM collection_files WHERE collection_id = ? LIMIT 1",
                    (cid,),
                ).fetchone()
                is not None
            )
            if not has_direct:
                cf_rows = conn.execute(
                    "SELECT root_id, folder_id, folder_rel_path FROM collection_folders WHERE collection_id = ?",
                    (cid,),
                ).fetchall()
                if not cf_rows:
                    return 0
                pruned = _prune_overlapping_folder_mounts(
                    [
                        (
                            int(cf["root_id"]),
                            None if cf["folder_id"] is None else int(cf["folder_id"]),
                            str(cf["folder_rel_path"] or ""),
                        )
                        for cf in cf_rows
                    ]
                )
                fast_total = 0
                can_fast = True
                for rid, fid, rp in pruned:
                    root = self.get_root(rid)
                    if root is None or root.status not in ("idle", "offline"):
                        can_fast = False
                        break
                    if not rp:
                        if root.file_count <= 0:
                            can_fast = False
                            break
                        fast_total += int(root.file_count)
                    elif fid is not None:
                        folder = self.get_folder(fid)
                        if folder is None or folder.file_count <= 0:
                            can_fast = False
                            break
                        fast_total += int(folder.file_count)
                    else:
                        can_fast = False
                        break
                if can_fast:
                    return fast_total
        return self.query_count(filters=QueryFilter(collection_id=cid))

    # -- R4: search history ------------------------------------------------
    def record_search_history(
        self,
        query: str,
        mode: str = "keyword",
        result_count: int = 0,
        *,
        max_unpinned: int = 100,
    ) -> None:
        clean = query.strip()
        if not clean:
            return
        norm_mode = "ai" if mode == "ai" else "keyword"
        now = time.time()
        conn = self._conn()
        with conn:
            conn.execute(
                """
                INSERT INTO search_history(query, mode, result_count, pinned, last_used_at)
                VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(query, mode) DO UPDATE SET
                    result_count = excluded.result_count,
                    last_used_at = excluded.last_used_at
                """,
                (clean, norm_mode, int(result_count), now),
            )
            conn.execute(
                """
                DELETE FROM search_history WHERE pinned = 0 AND id NOT IN (
                    SELECT id FROM search_history WHERE pinned = 0 ORDER BY last_used_at DESC LIMIT ?
                )
                """,
                (int(max_unpinned),),
            )

    def list_search_history(self, limit: int = 40) -> list[SearchHistoryRecord]:
        rows = (
            self._conn()
            .execute(
                "SELECT * FROM search_history ORDER BY pinned DESC, last_used_at DESC LIMIT ?",
                (int(limit),),
            )
            .fetchall()
        )
        return [
            SearchHistoryRecord(
                id=int(r["id"]),
                query=str(r["query"]),
                mode=str(r["mode"]),
                result_count=int(r["result_count"]),
                pinned=bool(r["pinned"]),
                last_used_at=float(r["last_used_at"]),
            )
            for r in rows
        ]

    def set_search_history_pinned(self, entry_id: int, pinned: bool) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE search_history SET pinned = ? WHERE id = ?",
                (1 if pinned else 0, int(entry_id)),
            )

    def delete_search_history(self, entry_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("DELETE FROM search_history WHERE id = ?", (int(entry_id),))

    def clear_search_history(self, *, keep_pinned: bool = True) -> None:
        conn = self._conn()
        with conn:
            if keep_pinned:
                conn.execute("DELETE FROM search_history WHERE pinned = 0")
            else:
                conn.execute("DELETE FROM search_history")

    # -- R4: hidden files & duplicate group queries ------------------------
    def hide_files(self, file_ids: list[int], reason: str = "duplicate") -> int:
        ids = list(dict.fromkeys(int(fid) for fid in file_ids))
        if not ids:
            return 0
        now = time.time()
        conn = self._conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO hidden_files(file_id, reason, hidden_at)
                VALUES (?, ?, ?)
                ON CONFLICT(file_id) DO UPDATE SET
                    reason = excluded.reason,
                    hidden_at = excluded.hidden_at
                """,
                [(fid, reason, now) for fid in ids],
            )
        self.bump_index_version()
        return len(ids)

    def unhide_files(self, file_ids: list[int] | None = None) -> int:
        conn = self._conn()
        with conn:
            if file_ids is None:
                cur = conn.execute("DELETE FROM hidden_files")
                removed = int(cur.rowcount)
            else:
                ids = list(dict.fromkeys(int(fid) for fid in file_ids))
                if not ids:
                    return 0
                removed = 0
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    cur = conn.execute(
                        f"DELETE FROM hidden_files WHERE file_id IN ({placeholders})",
                        chunk,
                    )
                    removed += int(cur.rowcount)
        if removed:
            self.bump_index_version()
        return removed

    def hidden_file_count(self) -> int:
        row = self._conn().execute("SELECT COUNT(*) AS c FROM hidden_files").fetchone()
        return 0 if row is None else int(row["c"])

    def duplicate_summary(self, mode: str | None = None) -> tuple[int, int, int, int]:
        """Return ``(groups, total_files, redundant_files, reclaimable_bytes)``."""
        conn = self._conn()
        where = "WHERE dg.mode = ?" if mode else ""
        params: list[Any] = [mode] if mode else []
        rows = conn.execute(
            f"""
            SELECT dg.id,
                   COUNT(dm.file_id) AS visible_count,
                   COALESCE(SUM(CASE WHEN dm.is_primary = 0 THEN 1 ELSE 0 END), 0) AS redundant_count,
                   COALESCE(SUM(CASE WHEN dm.is_primary = 0 THEN f.size_bytes ELSE 0 END), 0) AS reclaim_bytes
            FROM duplicate_groups dg
            JOIN duplicate_members dm ON dm.group_id = dg.id
            JOIN files f ON f.id = dm.file_id
            LEFT JOIN hidden_files hf ON hf.file_id = f.id
            {where} {'AND' if where else 'WHERE'} hf.file_id IS NULL
            GROUP BY dg.id
            HAVING visible_count >= 2
            """,
            params,
        ).fetchall()
        groups = len(rows)
        total_files = sum(int(r["visible_count"]) for r in rows)
        redundant_files = sum(int(r["redundant_count"]) for r in rows)
        reclaimable_bytes = sum(int(r["reclaim_bytes"]) for r in rows)
        return groups, total_files, redundant_files, reclaimable_bytes

    def list_duplicate_rows(
        self,
        mode: str = "name_duration",
        *,
        include_hidden: bool = False,
        limit: int = 2000,
    ) -> list[Any]:
        from audiosearch.library.duplicates import DuplicateRowRecord

        conn = self._conn()
        hidden_clause = "" if include_hidden else "AND hf.file_id IS NULL"
        rows = conn.execute(
            f"""
            SELECT dg.id AS group_id, dg.file_count AS group_size, dg.total_bytes AS group_bytes,
                   dm.is_primary, f.*, r.path AS root_path, r.is_online AS root_online,
                   COALESCE(um.favorite, 0) AS is_favorite,
                   COALESCE(um.note, '') AS user_note,
                   CASE WHEN hf.file_id IS NOT NULL THEN 1 ELSE 0 END AS is_hidden,
                   EXISTS(SELECT 1 FROM metadata_fields mf WHERE mf.file_id = f.id AND mf.dirty = 1) AS is_dirty
            FROM duplicate_groups dg
            JOIN duplicate_members dm ON dm.group_id = dg.id
            JOIN files f ON f.id = dm.file_id
            JOIN roots r ON r.id = f.root_id
            LEFT JOIN user_meta um ON um.file_id = f.id
            LEFT JOIN hidden_files hf ON hf.file_id = f.id
            WHERE dg.mode = ? {hidden_clause}
            ORDER BY dg.file_count DESC, dg.total_bytes DESC, dg.id ASC, dm.is_primary DESC, f.id ASC
            LIMIT ?
            """,
            (mode, int(limit)),
        ).fetchall()
        if not rows:
            return []

        # Filter out groups left with <2 visible rows when hidden items are filtered.
        group_counts: dict[int, int] = {}
        for r in rows:
            gid = int(r["group_id"])
            group_counts[gid] = group_counts.get(gid, 0) + 1
        valid_rows = [r for r in rows if include_hidden or group_counts[int(r["group_id"])] >= 2]
        if not valid_rows:
            return []

        entries = self.get_cached_thumbnail_entries(
            [(int(r["id"]), int(r["mtime_ns"])) for r in valid_rows]
        )
        group_order: dict[int, int] = {}
        out: list[DuplicateRowRecord] = []
        for r in valid_rows:
            gid = int(r["group_id"])
            if gid not in group_order:
                group_order[gid] = len(group_order) + 1
            fid = int(r["id"])
            thumb_blob, spec_blob = entries.get(fid, (None, None))
            rec = _row_to_file(r, thumb_blob=thumb_blob, spectral_thumb_blob=spec_blob)
            out.append(
                DuplicateRowRecord(
                    group_id=gid,
                    group_index=group_order[gid],
                    group_size=group_counts[gid],
                    total_bytes=int(r["group_bytes"]),
                    is_primary=bool(r["is_primary"]),
                    file=rec,
                )
            )
        return out

    def set_duplicate_primary(self, group_id: int, file_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute("UPDATE duplicate_members SET is_primary = 0 WHERE group_id = ?", (int(group_id),))
            conn.execute(
                "UPDATE duplicate_members SET is_primary = 1 WHERE group_id = ? AND file_id = ?",
                (int(group_id), int(file_id)),
            )

    def hide_all_non_primary_duplicates(self, mode: str) -> int:
        rows = (
            self._conn()
            .execute(
                """
                SELECT dm.file_id
                FROM duplicate_members dm
                JOIN duplicate_groups dg ON dg.id = dm.group_id
                LEFT JOIN hidden_files hf ON hf.file_id = dm.file_id
                WHERE dg.mode = ? AND dm.is_primary = 0 AND hf.file_id IS NULL
                """,
                (mode,),
            )
            .fetchall()
        )
        return self.hide_files([int(r["file_id"]) for r in rows], reason=f"duplicate:{mode}")


def _row_to_root(row: sqlite3.Row) -> RootRecord:
    return RootRecord(
        id=int(row["id"]),
        path=str(row["path"]),
        label=str(row["label"]),
        is_online=bool(row["is_online"]),
        file_count=int(row["file_count"]),
        last_scan_at=None if row["last_scan_at"] is None else float(row["last_scan_at"]),
        status=str(row["status"]),
    )


def _row_to_folder(row: sqlite3.Row) -> FolderRecord:
    return FolderRecord(
        id=int(row["id"]),
        root_id=int(row["root_id"]),
        parent_id=None if row["parent_id"] is None else int(row["parent_id"]),
        rel_path=str(row["rel_path"]),
        name=str(row["name"]),
        file_count=int(row["file_count"]),
    )


def _row_to_file(
    row: sqlite3.Row,
    *,
    thumb_blob: bytes | None = None,
    spectral_thumb_blob: bytes | None = None,
) -> FileRecord:
    available = bool(row["is_available"]) and bool(row["root_online"])
    keys = row.keys()
    return FileRecord(
        id=int(row["id"]),
        root_id=int(row["root_id"]),
        folder_id=None if row["folder_id"] is None else int(row["folder_id"]),
        root_path=str(row["root_path"]),
        rel_path=str(row["rel_path"]),
        filename=str(row["filename"]),
        ext=str(row["ext"]),
        size_bytes=int(row["size_bytes"]),
        mtime_ns=int(row["mtime_ns"]),
        duration_sec=float(row["duration_sec"]),
        sample_rate=int(row["sample_rate"]),
        channels=int(row["channels"]),
        bit_depth=int(row["bit_depth"]),
        package=str(row["package"]),
        description=str(row["description"]),
        embedded_text=str(row["embedded_text"]),
        bext_json=str(row["bext_json"]),
        ixml_json=str(row["ixml_json"]),
        is_available=available,
        thumb_blob=thumb_blob,
        spectral_thumb_blob=spectral_thumb_blob,
        favorite=bool(row["is_favorite"]) if "is_favorite" in keys else False,
        note=str(row["user_note"]) if "user_note" in keys else "",
        is_hidden=bool(row["is_hidden"]) if "is_hidden" in keys else False,
        is_dirty=bool(row["is_dirty"]) if "is_dirty" in keys else False,
    )
