"""Read-only background duplicate scanner for exact byte & multi-format duplicates."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from audiosearch.library.store import FileRecord, LibraryStore

DuplicateMode = Literal["exact", "name_duration"]

_SAMPLE_BYTES = 64 * 1024
_STREAM_CHUNK = 1024 * 1024
_DURATION_TOLERANCE_SEC = 0.15
_LOSSLESS_EXTS = frozenset({"wav", "flac", "aif", "aiff"})


@dataclass(frozen=True, slots=True)
class DuplicateProgress:
    mode: DuplicateMode
    stage: str
    scanned: int
    total: int
    groups_found: int
    message: str


@dataclass(frozen=True, slots=True)
class DuplicateScanReport:
    mode: DuplicateMode
    status: str
    groups_found: int
    total_files: int
    redundant_files: int
    reclaimable_bytes: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class DuplicateRowRecord:
    group_id: int
    group_index: int
    group_size: int
    total_bytes: int
    is_primary: bool
    file: FileRecord


def _fast_sample_hash(path: Path, size_bytes: int) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        if size_bytes <= _SAMPLE_BYTES * 3:
            while chunk := fh.read(_STREAM_CHUNK):
                hasher.update(chunk)
        else:
            hasher.update(fh.read(_SAMPLE_BYTES))
            mid = max(0, (size_bytes // 2) - (_SAMPLE_BYTES // 2))
            fh.seek(mid)
            hasher.update(fh.read(_SAMPLE_BYTES))
            fh.seek(max(0, size_bytes - _SAMPLE_BYTES))
            hasher.update(fh.read(_SAMPLE_BYTES))
    return hasher.hexdigest()


def _full_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_STREAM_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


class DuplicateScanner:
    """Safe, pauseable duplicate finder; never modifies or deletes source audio."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._cancel_event = threading.Event()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._pause_event.set()

    def _wait_if_paused(self, on_pause: Callable[[], None] | None = None) -> bool:
        notified = False
        while not self._pause_event.is_set():
            if not notified and on_pause is not None:
                notified = True
                on_pause()
            if self._cancel_event.wait(0.08):
                return True
        return self._cancel_event.is_set()

    def scan(
        self,
        mode: DuplicateMode = "name_duration",
        *,
        on_progress: Callable[[DuplicateProgress], None] | None = None,
    ) -> DuplicateScanReport:
        started = time.perf_counter()
        store = LibraryStore(self.db_path)
        try:
            if mode == "exact":
                groups, status = self._scan_exact(store, on_progress=on_progress)
            elif mode == "name_duration":
                groups, status = self._scan_name_duration(store, on_progress=on_progress)
            else:
                raise ValueError(f"Unsupported duplicate scan mode: {mode!r}")

            if status == "cancelled":
                return DuplicateScanReport(
                    mode=mode,
                    status="cancelled",
                    groups_found=0,
                    total_files=0,
                    redundant_files=0,
                    reclaimable_bytes=0,
                    elapsed_seconds=round(time.perf_counter() - started, 3),
                )

            now = time.time()
            total_files = 0
            redundant_files = 0
            reclaimable_bytes = 0
            conn = store._conn()
            with conn:
                conn.execute("DELETE FROM duplicate_groups WHERE mode = ?", (mode,))
                for group_key, members in groups:
                    if len(members) < 2:
                        continue
                    g_bytes = sum(int(m["size_bytes"]) for m in members)
                    cur = conn.execute(
                        "INSERT INTO duplicate_groups(mode, group_key, file_count, total_bytes, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (mode, group_key, len(members), g_bytes, now),
                    )
                    gid = int(cur.lastrowid)
                    conn.executemany(
                        "INSERT INTO duplicate_members(group_id, file_id, is_primary) VALUES (?, ?, ?)",
                        [(gid, int(m["id"]), 1 if i == 0 else 0) for i, m in enumerate(members)],
                    )
                    total_files += len(members)
                    redundant_files += len(members) - 1
                    reclaimable_bytes += sum(int(m["size_bytes"]) for m in members[1:])

            elapsed = round(time.perf_counter() - started, 3)
            if on_progress is not None:
                on_progress(
                    DuplicateProgress(
                        mode=mode,
                        stage="done",
                        scanned=total_files,
                        total=total_files,
                        groups_found=len(groups),
                        message=f"扫描完成：发现 {len(groups):,} 组（{redundant_files:,} 条冗余）",
                    )
                )
            return DuplicateScanReport(
                mode=mode,
                status="done",
                groups_found=len(groups),
                total_files=total_files,
                redundant_files=redundant_files,
                reclaimable_bytes=reclaimable_bytes,
                elapsed_seconds=elapsed,
            )
        finally:
            store.close()

    def _scan_exact(
        self,
        store: LibraryStore,
        *,
        on_progress: Callable[[DuplicateProgress], None] | None = None,
    ) -> tuple[list[tuple[str, list[dict[str, object]]]], str]:
        conn = store._conn()
        size_rows = conn.execute(
            "SELECT size_bytes, COUNT(*) AS c FROM files f JOIN roots r ON r.id = f.root_id "
            "WHERE f.is_available = 1 AND r.is_online = 1 AND f.size_bytes > 0 "
            "GROUP BY f.size_bytes HAVING c > 1"
        ).fetchall()
        candidate_sizes = [int(r["size_bytes"]) for r in size_rows]
        total_candidates = sum(int(r["c"]) for r in size_rows)
        scanned = 0
        out_groups: list[tuple[str, list[dict[str, object]]]] = []

        if on_progress is not None:
            on_progress(
                DuplicateProgress(
                    mode="exact",
                    stage="size_buckets",
                    scanned=0,
                    total=total_candidates,
                    groups_found=0,
                    message=f"同体积候选 {total_candidates:,} 条，开始校验指纹…",
                )
            )

        for i in range(0, len(candidate_sizes), 200):
            chunk = candidate_sizes[i : i + 200]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                "SELECT f.id, f.root_id, f.rel_path, f.filename, f.ext, f.size_bytes, "
                "f.description, f.is_available, r.path AS root_path, r.is_online AS root_online "
                "FROM files f JOIN roots r ON r.id = f.root_id "
                f"WHERE f.is_available = 1 AND r.is_online = 1 AND f.size_bytes IN ({placeholders})",
                chunk,
            ).fetchall()
            by_size: dict[int, list[dict[str, object]]] = {}
            for r in rows:
                by_size.setdefault(int(r["size_bytes"]), []).append(dict(r))

            for size_bytes, bucket in by_size.items():
                if self._wait_if_paused():
                    return [], "cancelled"
                sample_map: dict[str, list[dict[str, object]]] = {}
                for item in bucket:
                    scanned += 1
                    full_path = Path(str(item["root_path"])) / Path(str(item["rel_path"]))
                    if not full_path.exists():
                        continue
                    try:
                        sh = _fast_sample_hash(full_path, size_bytes)
                    except OSError:
                        continue
                    sample_map.setdefault(sh, []).append(item)

                for _sh, scands in sample_map.items():
                    if len(scands) < 2:
                        continue
                    full_map: dict[str, list[dict[str, object]]] = {}
                    for item in scands:
                        if self._wait_if_paused():
                            return [], "cancelled"
                        full_path = Path(str(item["root_path"])) / Path(str(item["rel_path"]))
                        try:
                            fh = _full_sha256(full_path)
                        except OSError:
                            continue
                        full_map.setdefault(fh, []).append(item)

                    for fh, exact_members in full_map.items():
                        if len(exact_members) < 2:
                            continue
                        exact_members.sort(
                            key=lambda m: (
                                0 if str(m["description"]).strip() else 1,
                                len(str(m["rel_path"])),
                                int(m["id"]),
                            )
                        )
                        out_groups.append((f"sha256:{fh[:16]}:{size_bytes}", exact_members))

                if on_progress is not None and (scanned % 100 == 0 or scanned == total_candidates):
                    on_progress(
                        DuplicateProgress(
                            mode="exact",
                            stage="hashing",
                            scanned=scanned,
                            total=total_candidates,
                            groups_found=len(out_groups),
                            message=f"正在比对指纹 {scanned:,}/{total_candidates:,}",
                        )
                    )

        return out_groups, "done"

    def _scan_name_duration(
        self,
        store: LibraryStore,
        *,
        on_progress: Callable[[DuplicateProgress], None] | None = None,
    ) -> tuple[list[tuple[str, list[dict[str, object]]]], str]:
        conn = store._conn()
        rows = conn.execute(
            "SELECT f.id, f.filename, f.ext, f.duration_sec, f.sample_rate, f.bit_depth, "
            "f.size_bytes, f.description, f.is_available, r.is_online AS root_online "
            "FROM files f JOIN roots r ON r.id = f.root_id"
        ).fetchall()
        total = len(rows)
        by_stem: dict[str, list[dict[str, object]]] = {}
        for idx, r in enumerate(rows, start=1):
            if idx % 2000 == 0 and self._wait_if_paused():
                return [], "cancelled"
            stem = Path(str(r["filename"])).stem.casefold().strip()
            if not stem:
                continue
            by_stem.setdefault(stem, []).append(dict(r))

        if on_progress is not None:
            on_progress(
                DuplicateProgress(
                    mode="name_duration",
                    stage="clustering",
                    scanned=total,
                    total=total,
                    groups_found=0,
                    message=f"正在聚类同名变体 ({total:,} 条)…",
                )
            )

        out_groups: list[tuple[str, list[dict[str, object]]]] = []
        for stem, bucket in by_stem.items():
            if len(bucket) < 2:
                continue
            if self._wait_if_paused():
                return [], "cancelled"
            bucket.sort(key=lambda m: float(m["duration_sec"]))
            clusters: list[list[dict[str, object]]] = []
            current: list[dict[str, object]] = [bucket[0]]
            anchor_dur = float(bucket[0]["duration_sec"])
            for item in bucket[1:]:
                dur = float(item["duration_sec"])
                if abs(dur - anchor_dur) <= _DURATION_TOLERANCE_SEC:
                    current.append(item)
                else:
                    if len(current) >= 2:
                        clusters.append(current)
                    current = [item]
                    anchor_dur = dur
            if len(current) >= 2:
                clusters.append(current)

            for c_idx, cluster in enumerate(clusters):
                cluster.sort(
                    key=lambda m: (
                        0 if (bool(m["is_available"]) and bool(m["root_online"])) else 1,
                        0 if str(m["ext"]).casefold() in _LOSSLESS_EXTS else 1,
                        -int(m["sample_rate"]),
                        -int(m["bit_depth"]),
                        0 if str(m["description"]).strip() else 1,
                        int(m["id"]),
                    )
                )
                avg_dur = sum(float(m["duration_sec"]) for m in cluster) / len(cluster)
                out_groups.append((f"{stem}@{avg_dur:.2f}s#{c_idx}", cluster))

        out_groups.sort(key=lambda g: (-len(g[1]), -sum(int(m["size_bytes"]) for m in g[1]), g[0]))
        return out_groups, "done"


__all__ = [
    "DuplicateMode",
    "DuplicateProgress",
    "DuplicateRowRecord",
    "DuplicateScanReport",
    "DuplicateScanner",
]
