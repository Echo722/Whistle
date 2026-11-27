"""Scalable incremental float16 vector store for the standalone desktop browser."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiosearch.inference.clap2_encoder import DEFAULT_CLAP2_MODEL, SMALL_CLAP2_MODEL
from audiosearch.inference.qwen3_encoder import DEFAULT_QWEN3_MODEL, DEFAULT_QWEN3_MRL_DIM

_CHUNK_ROWS = 65_536
_CUDA_TORCH_CACHE: tuple[bool, Any | None] | None = None


def _try_cuda_torch() -> Any | None:
    """Return the imported torch module when CUDA is available, else None."""
    global _CUDA_TORCH_CACHE
    if _CUDA_TORCH_CACHE is None:
        try:
            import torch

            mod = torch if torch.cuda.is_available() else None
        except Exception:
            mod = None
        _CUDA_TORCH_CACHE = (True, mod)
    return _CUDA_TORCH_CACHE[1]


@dataclass(frozen=True, slots=True)
class AiModelProfile:
    name: str
    text_model_id: str
    text_dim: int
    audio_model_id: str
    audio_dim: int
    label: str


MODEL_PROFILES: dict[str, AiModelProfile] = {
    "large": AiModelProfile(
        name="large",
        text_model_id=DEFAULT_QWEN3_MODEL,
        text_dim=DEFAULT_QWEN3_MRL_DIM,
        audio_model_id=DEFAULT_CLAP2_MODEL,
        audio_dim=960,
        label="Qwen3-0.6B + CLAP2-Large (BF16)",
    ),
    "small": AiModelProfile(
        name="small",
        text_model_id=DEFAULT_QWEN3_MODEL,
        text_dim=DEFAULT_QWEN3_MRL_DIM,
        audio_model_id=SMALL_CLAP2_MODEL,
        audio_dim=512,
        label="Qwen3-0.6B + CLAP2-Small (BF16)",
    ),
}


def resolve_profile(name: str | None) -> AiModelProfile:
    key = (name or "large").strip().lower()
    return MODEL_PROFILES.get(key, MODEL_PROFILES["large"])


def vector_dir_for_db(db_path: str | Path, profile: str = "large") -> Path:
    prof = resolve_profile(profile)
    return Path(db_path).parent / "vectors" / prof.name


@dataclass(frozen=True, slots=True)
class IndexCoverage:
    profile: str
    total_files: int
    text_files: int
    audio_files: int
    audio_windows: int

    @property
    def overall_percent(self) -> float:
        if self.total_files <= 0:
            return 0.0
        capped_text = min(max(0, self.text_files), self.total_files)
        capped_audio = min(max(0, self.audio_files), self.total_files)
        avg = (capped_text + capped_audio) / (2.0 * self.total_files)
        return min(100.0, max(0.0, avg * 100.0))


@dataclass(frozen=True, slots=True)
class AudioWindowMatch:
    file_id: int
    score: float
    start_sec: float
    end_sec: float


_INDEX_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS index_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS text_rows (
    row_idx INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    input_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS audio_rows (
    row_idx INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_text_rows_file_active ON text_rows(file_id, active);
CREATE INDEX IF NOT EXISTS idx_text_rows_active_row ON text_rows(active, row_idx);
CREATE INDEX IF NOT EXISTS idx_audio_rows_file_active ON audio_rows(file_id, active);
CREATE INDEX IF NOT EXISTS idx_audio_rows_active_row ON audio_rows(active, row_idx);
"""


def _l2_normalize_f32(vectors: np.ndarray) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return (arr / norms).astype(np.float32)


def _allowed_mask(allowed_file_ids: set[int] | None, file_ids: np.ndarray) -> np.ndarray | None:
    if allowed_file_ids is None:
        return None
    if not allowed_file_ids:
        return np.zeros(file_ids.shape, dtype=bool)
    max_fid = int(file_ids.max()) if len(file_ids) > 0 else 0
    filtered = [fid for fid in allowed_file_ids if 0 <= fid <= max_fid]
    if not filtered:
        return np.zeros(file_ids.shape, dtype=bool)
    lut = np.zeros(max_fid + 1, dtype=bool)
    lut[np.asarray(filtered, dtype=np.int64)] = True
    return lut[file_ids]


class LibraryVectorIndex:
    """Thread-safe incremental float16 binary + SQLite vector store."""

    def __init__(self, directory: str | Path, *, profile: str = "large") -> None:
        self.profile = resolve_profile(profile)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db_path = self.directory / "index.sqlite3"
        self.text_vec_path = self.directory / "text.f16"
        self.audio_vec_path = self.directory / "audio.f16"

        self._lock = threading.RLock()
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []

        self._text_cache_sig: tuple[int, int, int] | None = None
        self._text_meta_rev: tuple[int, int] | None = None
        self._text_file_ids = np.zeros(0, dtype=np.int64)
        self._text_active = np.zeros(0, dtype=bool)
        self.text_meta_cache_path = self.directory / "text_meta.npz"

        self._audio_cache_sig: tuple[int, int, int] | None = None
        self._audio_meta_rev: tuple[int, int] | None = None
        self._audio_file_ids = np.zeros(0, dtype=np.int64)
        self._audio_active = np.zeros(0, dtype=bool)
        self._audio_starts = np.zeros(0, dtype=np.float32)
        self._audio_ends = np.zeros(0, dtype=np.float32)
        self.audio_meta_cache_path = self.directory / "audio_meta.npz"

        self._write_epoch = 0
        self._coverage_cache: tuple[tuple[int, int], int, int, int] | None = None

        self._init_storage()

    def _conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 5000")
            conn.execute("PRAGMA cache_size = -65536")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = conn
            with self._lock:
                self._conns.append(conn)
        return conn

    def _init_storage(self) -> None:
        conn = self._conn()
        conn.executescript(_INDEX_SCHEMA_SQL)
        conn.execute(
            "INSERT OR IGNORE INTO index_meta(key, value) VALUES (?, ?)",
            ("profile", self.profile.name),
        )
        conn.execute(
            "INSERT OR IGNORE INTO index_meta(key, value) VALUES (?, ?)",
            ("text_dim", str(self.profile.text_dim)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO index_meta(key, value) VALUES (?, ?)",
            ("audio_dim", str(self.profile.audio_dim)),
        )
        conn.commit()
        if not self.text_vec_path.exists():
            self.text_vec_path.touch()
        if not self.audio_vec_path.exists():
            self.audio_vec_path.touch()

    def close(self) -> None:
        with self._lock:
            conns, self._conns = self._conns, []
        self._local = threading.local()
        for conn in conns:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _db_revision(self, conn: sqlite3.Connection) -> tuple[int, int]:
        row = conn.execute("PRAGMA data_version").fetchone()
        return self._write_epoch, (int(row[0]) if row is not None else 0)

    def _invalidate_mem_caches(self, *, text: bool = False, audio: bool = False) -> None:
        self._write_epoch += 1
        self._coverage_cache = None
        if text:
            self._text_cache_sig = None
            self._text_meta_rev = None
        if audio:
            self._audio_cache_sig = None
            self._audio_meta_rev = None

    # -- status & incremental maps ----------------------------------------
    def coverage(self, total_files: int) -> IndexCoverage:
        with self._lock:
            conn = self._conn()
            rev = self._db_revision(conn)
            cached = self._coverage_cache
            if cached is not None and cached[0] == rev:
                _, text_files, audio_files, audio_windows = cached
            else:
                text_files = int(
                    conn.execute(
                        "SELECT COUNT(DISTINCT file_id) FROM text_rows WHERE active = 1"
                    ).fetchone()[0]
                )
                audio_row = conn.execute(
                    "SELECT COUNT(DISTINCT file_id), COUNT(*) FROM audio_rows WHERE active = 1"
                ).fetchone()
                audio_files = int(audio_row[0])
                audio_windows = int(audio_row[1])
                self._coverage_cache = (rev, text_files, audio_files, audio_windows)
            return IndexCoverage(
                profile=self.profile.name,
                total_files=max(0, int(total_files)),
                text_files=text_files,
                audio_files=audio_files,
                audio_windows=audio_windows,
            )

    def indexed_text_map(self) -> dict[int, tuple[int, str]]:
        rows = (
            self._conn()
            .execute("SELECT file_id, mtime_ns, input_hash FROM text_rows WHERE active = 1")
            .fetchall()
        )
        return {
            int(r["file_id"]): (int(r["mtime_ns"]), str(r["input_hash"]))
            for r in rows
        }

    def indexed_audio_map(self) -> dict[int, int]:
        rows = (
            self._conn()
            .execute("SELECT DISTINCT file_id, mtime_ns FROM audio_rows WHERE active = 1")
            .fetchall()
        )
        return {int(r["file_id"]): int(r["mtime_ns"]) for r in rows}

    def deactivate_file_ids(self, file_ids: Iterable[int]) -> tuple[int, int]:
        """Mark active text & audio rows for ``file_ids`` inactive without rewriting binaries."""
        ids = sorted({int(fid) for fid in file_ids})
        if not ids:
            return 0, 0
        text_deactivated = 0
        audio_deactivated = 0
        with self._lock:
            conn = self._conn()
            with conn:
                for i in range(0, len(ids), 500):
                    chunk = ids[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    cur_t = conn.execute(
                        f"UPDATE text_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                    text_deactivated += max(0, int(cur_t.rowcount or 0))
                    cur_a = conn.execute(
                        f"UPDATE audio_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                    audio_deactivated += max(0, int(cur_a.rowcount or 0))
            if text_deactivated > 0 or audio_deactivated > 0:
                self._invalidate_mem_caches(
                    text=text_deactivated > 0,
                    audio=audio_deactivated > 0,
                )
        return text_deactivated, audio_deactivated

    def prune_missing_file_ids(self, valid_file_ids: Iterable[int]) -> tuple[int, int]:
        """Deactivate rows whose ``file_id`` is absent from ``valid_file_ids``."""
        valid_set = (
            valid_file_ids
            if isinstance(valid_file_ids, set)
            else {int(fid) for fid in valid_file_ids}
        )
        with self._lock:
            conn = self._conn()
            orphan_text = [
                int(r[0])
                for r in conn.execute("SELECT DISTINCT file_id FROM text_rows WHERE active = 1")
                if int(r[0]) not in valid_set
            ]
            orphan_audio = [
                int(r[0])
                for r in conn.execute("SELECT DISTINCT file_id FROM audio_rows WHERE active = 1")
                if int(r[0]) not in valid_set
            ]
            if not orphan_text and not orphan_audio:
                return 0, 0
            text_deactivated = 0
            audio_deactivated = 0
            with conn:
                for i in range(0, len(orphan_text), 500):
                    chunk = orphan_text[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    cur_t = conn.execute(
                        f"UPDATE text_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                    text_deactivated += max(0, int(cur_t.rowcount or 0))
                for i in range(0, len(orphan_audio), 500):
                    chunk = orphan_audio[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    cur_a = conn.execute(
                        f"UPDATE audio_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                    audio_deactivated += max(0, int(cur_a.rowcount or 0))
            if text_deactivated > 0 or audio_deactivated > 0:
                self._invalidate_mem_caches(
                    text=text_deactivated > 0,
                    audio=audio_deactivated > 0,
                )
            return text_deactivated, audio_deactivated

    # -- incremental append -----------------------------------------------
    def _reconcile_binary_tail(self, bin_path: Path, expected_rows: int, dim: int) -> int:
        expected_bytes = expected_rows * dim * 2
        actual_bytes = bin_path.stat().st_size if bin_path.exists() else 0
        if actual_bytes > expected_bytes:
            with bin_path.open("r+b") as fh:
                fh.truncate(expected_bytes)
                fh.flush()
                os.fsync(fh.fileno())
        elif actual_bytes < expected_bytes:
            raise OSError(
                f"Vector binary {bin_path} is truncated ({actual_bytes} < {expected_bytes})"
            )
        return expected_rows

    def append_text_batch(
        self,
        items: list[tuple[int, int, str]],
        vectors: np.ndarray,
    ) -> int:
        """Append normalized float16 text vectors for ``(file_id, mtime_ns, input_hash)``."""
        if not items:
            return 0
        arr = _l2_normalize_f32(vectors)
        if arr.ndim != 2 or arr.shape[0] != len(items) or arr.shape[1] != self.profile.text_dim:
            raise ValueError(
                f"Expected ({len(items)}, {self.profile.text_dim}) text vectors, got {arr.shape}"
            )
        raw_bytes = np.ascontiguousarray(arr, dtype=np.float16).tobytes()

        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT COALESCE(MAX(row_idx) + 1, 0) FROM text_rows").fetchone()
            start_idx = self._reconcile_binary_tail(
                self.text_vec_path, int(row[0]), self.profile.text_dim
            )
            with self.text_vec_path.open("ab") as fh:
                fh.write(raw_bytes)
                fh.flush()

            file_ids = list({int(fid) for fid, _, _ in items})
            with conn:
                for i in range(0, len(file_ids), 500):
                    chunk = file_ids[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE text_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                conn.executemany(
                    "INSERT INTO text_rows(row_idx, file_id, mtime_ns, input_hash, active) "
                    "VALUES (?, ?, ?, ?, 1)",
                    [
                        (start_idx + offset, int(fid), int(mtime), str(h))
                        for offset, (fid, mtime, h) in enumerate(items)
                    ],
                )
            self._invalidate_mem_caches(text=True)
            return len(items)

    def append_audio_batch(
        self,
        file_metas: list[tuple[int, int]],
        row_spans: list[tuple[int, int, float, float]],
        vectors: np.ndarray,
    ) -> int:
        """Append float16 audio window vectors for ``(file_id, mtime_ns, start, end)``."""
        if not row_spans:
            return 0
        arr = _l2_normalize_f32(vectors)
        if arr.ndim != 2 or arr.shape[0] != len(row_spans) or arr.shape[1] != self.profile.audio_dim:
            raise ValueError(
                f"Expected ({len(row_spans)}, {self.profile.audio_dim}) audio vectors, got {arr.shape}"
            )
        raw_bytes = np.ascontiguousarray(arr, dtype=np.float16).tobytes()

        with self._lock:
            conn = self._conn()
            row = conn.execute("SELECT COALESCE(MAX(row_idx) + 1, 0) FROM audio_rows").fetchone()
            start_idx = self._reconcile_binary_tail(
                self.audio_vec_path, int(row[0]), self.profile.audio_dim
            )
            with self.audio_vec_path.open("ab") as fh:
                fh.write(raw_bytes)
                fh.flush()

            file_ids = list({int(fid) for fid, _ in file_metas} | {int(s[0]) for s in row_spans})
            with conn:
                for i in range(0, len(file_ids), 500):
                    chunk = file_ids[i : i + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    conn.execute(
                        f"UPDATE audio_rows SET active = 0 WHERE active = 1 AND file_id IN ({placeholders})",
                        chunk,
                    )
                conn.executemany(
                    "INSERT INTO audio_rows(row_idx, file_id, mtime_ns, start_sec, end_sec, active) "
                    "VALUES (?, ?, ?, ?, ?, 1)",
                    [
                        (start_idx + offset, int(fid), int(mtime), float(s0), float(s1))
                        for offset, (fid, mtime, s0, s1) in enumerate(row_spans)
                    ],
                )
            self._invalidate_mem_caches(audio=True)
            return len(row_spans)

    # -- fast chunked search ----------------------------------------------
    @staticmethod
    def _save_meta_npz(cache_path: Path, **arrays: np.ndarray) -> None:
        tmp = cache_path.parent / f".{cache_path.stem}.{os.getpid()}.tmp.npz"
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, cache_path)
        except OSError:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _load_meta_npz(
        cache_path: Path, expected_sig: tuple[int, int, int], keys: tuple[str, ...]
    ) -> dict[str, np.ndarray] | None:
        if not cache_path.is_file():
            return None
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                sig_arr = data["sig"]
                if tuple(int(x) for x in sig_arr.tolist()) != expected_sig:
                    return None
                return {k: np.array(data[k], copy=True) for k in keys}
        except Exception:
            return None

    def _query_row_sig(self, conn: sqlite3.Connection, table: str) -> tuple[int, int, int]:
        cnt = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if cnt == 0:
            return (0, -1, 0)
        max_idx = int(conn.execute(f"SELECT COALESCE(MAX(row_idx), -1) FROM {table}").fetchone()[0])
        inactive = int(
            conn.execute(f"SELECT COUNT(*) FROM {table} WHERE active = 0").fetchone()[0]
        )
        return (cnt, max_idx, inactive)

    def _ensure_text_meta(self) -> tuple[np.ndarray, np.ndarray]:
        conn = self._conn()
        rev = self._db_revision(conn)
        if rev == self._text_meta_rev and self._text_cache_sig is not None:
            return self._text_file_ids, self._text_active

        sig = self._query_row_sig(conn, "text_rows")
        if sig == self._text_cache_sig:
            self._text_meta_rev = rev
            return self._text_file_ids, self._text_active

        loaded = self._load_meta_npz(self.text_meta_cache_path, sig, ("fids", "active"))
        if loaded is not None:
            self._text_file_ids = loaded["fids"]
            self._text_active = loaded["active"]
            self._text_cache_sig = sig
            self._text_meta_rev = rev
            return self._text_file_ids, self._text_active

        n = sig[1] + 1 if sig[0] > 0 else 0
        fids = np.zeros(n, dtype=np.int64)
        active = np.zeros(n, dtype=bool)
        if n > 0:
            cur = conn.execute("SELECT row_idx, file_id, active FROM text_rows ORDER BY row_idx ASC")
            try:
                while True:
                    batch = cur.fetchmany(100_000)
                    if not batch:
                        break
                    for r in batch:
                        idx = int(r[0])
                        if 0 <= idx < n:
                            fids[idx] = int(r[1])
                            active[idx] = bool(r[2])
            finally:
                cur.close()
        self._text_file_ids = fids
        self._text_active = active
        self._text_cache_sig = sig
        self._text_meta_rev = rev
        if n > 0:
            self._save_meta_npz(
                self.text_meta_cache_path,
                sig=np.asarray(sig, dtype=np.int64),
                fids=fids,
                active=active,
            )
        return fids, active

    def _ensure_audio_meta(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        conn = self._conn()
        rev = self._db_revision(conn)
        if rev == self._audio_meta_rev and self._audio_cache_sig is not None:
            return self._audio_file_ids, self._audio_active, self._audio_starts, self._audio_ends

        sig = self._query_row_sig(conn, "audio_rows")
        if sig == self._audio_cache_sig:
            self._audio_meta_rev = rev
            return self._audio_file_ids, self._audio_active, self._audio_starts, self._audio_ends

        loaded = self._load_meta_npz(
            self.audio_meta_cache_path, sig, ("fids", "active", "starts", "ends")
        )
        if loaded is not None:
            self._audio_file_ids = loaded["fids"]
            self._audio_active = loaded["active"]
            self._audio_starts = loaded["starts"]
            self._audio_ends = loaded["ends"]
            self._audio_cache_sig = sig
            self._audio_meta_rev = rev
            return self._audio_file_ids, self._audio_active, self._audio_starts, self._audio_ends

        n = sig[1] + 1 if sig[0] > 0 else 0
        fids = np.zeros(n, dtype=np.int64)
        active = np.zeros(n, dtype=bool)
        starts = np.zeros(n, dtype=np.float32)
        ends = np.zeros(n, dtype=np.float32)
        if n > 0:
            cur = conn.execute(
                "SELECT row_idx, file_id, start_sec, end_sec, active FROM audio_rows ORDER BY row_idx ASC"
            )
            try:
                while True:
                    batch = cur.fetchmany(100_000)
                    if not batch:
                        break
                    for r in batch:
                        idx = int(r[0])
                        if 0 <= idx < n:
                            fids[idx] = int(r[1])
                            starts[idx] = float(r[2])
                            ends[idx] = float(r[3])
                            active[idx] = bool(r[4])
            finally:
                cur.close()
        self._audio_file_ids = fids
        self._audio_active = active
        self._audio_starts = starts
        self._audio_ends = ends
        self._audio_cache_sig = sig
        self._audio_meta_rev = rev
        if n > 0:
            self._save_meta_npz(
                self.audio_meta_cache_path,
                sig=np.asarray(sig, dtype=np.int64),
                fids=fids,
                active=active,
                starts=starts,
                ends=ends,
            )
        return fids, active, starts, ends

    @staticmethod
    def _read_and_score_chunk(
        fh: Any,
        count: int,
        dim: int,
        q_f32: np.ndarray,
        host_buf: np.ndarray,
        torch_mod: Any | None,
        q_gpu: Any | None,
    ) -> np.ndarray:
        n_vals = count * dim
        n_bytes = n_vals * 2
        mv = memoryview(host_buf).cast("B")[:n_bytes]
        read_bytes = fh.readinto(mv)
        if read_bytes != n_bytes:
            actual_rows = (read_bytes or 0) // (dim * 2)
            chunk = host_buf[: actual_rows * dim].reshape(actual_rows, dim)
            return chunk.astype(np.float32) @ q_f32
        chunk = host_buf[:n_vals].reshape(count, dim)
        if torch_mod is not None and q_gpu is not None and count >= 4096:
            try:
                with torch_mod.inference_mode():
                    t_gpu = torch_mod.from_numpy(chunk).to(device="cuda", dtype=torch_mod.float16)
                    return torch_mod.mv(t_gpu, q_gpu).float().cpu().numpy()
            except Exception:
                pass
        return chunk.astype(np.float32) @ q_f32

    @staticmethod
    def _kth_threshold(scores_iter: Iterable[float], top_k: int) -> float:
        arr = np.fromiter(scores_iter, dtype=np.float32)
        if arr.size < top_k:
            return -1e9
        return float(np.partition(arr, arr.size - top_k)[arr.size - top_k])

    def search_text(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 400,
        min_score: float = 0.28,
        allowed_file_ids: set[int] | None = None,
        stats_out: dict[str, int] | None = None,
    ) -> list[tuple[int, float]]:
        q = _l2_normalize_f32(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.profile.text_dim:
            raise ValueError(f"Expected {self.profile.text_dim}-d query vector, got {q.shape[0]}")

        top_k = max(1, int(top_k))
        with self._lock:
            fids, active = self._ensure_text_meta()
            n_rows = min(
                len(fids),
                (self.text_vec_path.stat().st_size // (self.profile.text_dim * 2))
                if self.text_vec_path.exists()
                else 0,
            )
            if n_rows <= 0:
                if stats_out is not None:
                    stats_out["matched_rows"] = 0
                return []

            allowed_arr = _allowed_mask(allowed_file_ids, fids[:n_rows])
            best_scores: dict[int, float] = {}
            dim = self.profile.text_dim
            base_min = float(min_score)
            eff_min = base_min
            matched_rows = 0

            torch_mod = _try_cuda_torch() if n_rows >= 8192 else None
            q_gpu = None
            if torch_mod is not None:
                try:
                    q_gpu = torch_mod.from_numpy(q).to(device="cuda", dtype=torch_mod.float16)
                except Exception:
                    torch_mod = None
                    q_gpu = None

            host_buf = np.empty(_CHUNK_ROWS * dim, dtype=np.float16)
            prune_cap = max(top_k * 4, 2000)

            with self.text_vec_path.open("rb") as fh:
                for start in range(0, n_rows, _CHUNK_ROWS):
                    count = min(_CHUNK_ROWS, n_rows - start)
                    scores = self._read_and_score_chunk(
                        fh, count, dim, q, host_buf, torch_mod, q_gpu
                    )
                    if scores.shape[0] <= 0:
                        break
                    c_len = scores.shape[0]
                    valid = active[start : start + c_len]
                    if allowed_arr is not None:
                        valid = valid & allowed_arr[start : start + c_len]
                    if stats_out is not None:
                        matched_rows += int(np.count_nonzero(valid & (scores >= base_min)))

                    pos = np.flatnonzero(valid & (scores >= eff_min))
                    if pos.size == 0:
                        continue
                    if pos.size > 1:
                        pos = pos[np.argsort(-scores[pos], kind="mergesort")]
                    for p in pos:
                        sc = float(scores[p])
                        if sc < eff_min:
                            break
                        fid = int(fids[start + int(p)])
                        prev = best_scores.get(fid)
                        if prev is None or sc > prev:
                            best_scores[fid] = sc
                    if len(best_scores) >= top_k:
                        kth = self._kth_threshold(best_scores.values(), top_k)
                        if kth > eff_min:
                            eff_min = kth
                        if len(best_scores) > prune_cap:
                            trimmed = sorted(
                                best_scores.items(), key=lambda kv: (-kv[1], kv[0])
                            )[:top_k]
                            best_scores = dict(trimmed)

            if stats_out is not None:
                stats_out["matched_rows"] = matched_rows
            ordered = sorted(best_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            return ordered[:top_k]

    def search_audio(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 400,
        min_score: float = 0.16,
        allowed_file_ids: set[int] | None = None,
        exclude_file_ids: set[int] | None = None,
        stats_out: dict[str, int] | None = None,
    ) -> list[AudioWindowMatch]:
        q = _l2_normalize_f32(np.asarray(query_vec, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.profile.audio_dim:
            raise ValueError(f"Expected {self.profile.audio_dim}-d query vector, got {q.shape[0]}")

        top_k = max(1, int(top_k))
        with self._lock:
            fids, active, starts, ends = self._ensure_audio_meta()
            n_rows = min(
                len(fids),
                (self.audio_vec_path.stat().st_size // (self.profile.audio_dim * 2))
                if self.audio_vec_path.exists()
                else 0,
            )
            if n_rows <= 0:
                if stats_out is not None:
                    stats_out["matched_rows"] = 0
                return []

            allowed_arr = _allowed_mask(allowed_file_ids, fids[:n_rows])
            excluded = exclude_file_ids or set()
            best_by_file: dict[int, AudioWindowMatch] = {}
            dim = self.profile.audio_dim
            base_min = float(min_score)
            eff_min = base_min
            matched_rows = 0

            torch_mod = _try_cuda_torch() if n_rows >= 8192 else None
            q_gpu = None
            if torch_mod is not None:
                try:
                    q_gpu = torch_mod.from_numpy(q).to(device="cuda", dtype=torch_mod.float16)
                except Exception:
                    torch_mod = None
                    q_gpu = None

            host_buf = np.empty(_CHUNK_ROWS * dim, dtype=np.float16)
            prune_cap = max(top_k * 4, 2000)

            with self.audio_vec_path.open("rb") as fh:
                for start in range(0, n_rows, _CHUNK_ROWS):
                    count = min(_CHUNK_ROWS, n_rows - start)
                    scores = self._read_and_score_chunk(
                        fh, count, dim, q, host_buf, torch_mod, q_gpu
                    )
                    if scores.shape[0] <= 0:
                        break
                    c_len = scores.shape[0]
                    valid = active[start : start + c_len]
                    if allowed_arr is not None:
                        valid = valid & allowed_arr[start : start + c_len]
                    if stats_out is not None:
                        matched_rows += int(np.count_nonzero(valid & (scores >= base_min)))

                    pos = np.flatnonzero(valid & (scores >= eff_min))
                    if pos.size == 0:
                        continue
                    if pos.size > 1:
                        pos = pos[np.argsort(-scores[pos], kind="mergesort")]
                    for p in pos:
                        sc = float(scores[p])
                        if sc < eff_min:
                            break
                        row_i = start + int(p)
                        fid = int(fids[row_i])
                        if fid in excluded:
                            continue
                        prev = best_by_file.get(fid)
                        if prev is None or sc > prev.score:
                            best_by_file[fid] = AudioWindowMatch(
                                file_id=fid,
                                score=sc,
                                start_sec=float(starts[row_i]),
                                end_sec=float(ends[row_i]),
                            )
                    if len(best_by_file) >= top_k:
                        kth = self._kth_threshold((m.score for m in best_by_file.values()), top_k)
                        if kth > eff_min:
                            eff_min = kth
                        if len(best_by_file) > prune_cap:
                            trimmed = sorted(
                                best_by_file.items(), key=lambda kv: (-kv[1].score, kv[0])
                            )[:top_k]
                            best_by_file = dict(trimmed)

            if stats_out is not None:
                stats_out["matched_rows"] = matched_rows
            ordered = sorted(best_by_file.values(), key=lambda m: (-m.score, m.file_id))
            return ordered[:top_k]

    def get_file_audio_vector(self, file_id: int) -> np.ndarray | None:
        """Return the mean L2-normalized audio embedding for ``file_id`` if indexed."""
        with self._lock:
            rows = (
                self._conn()
                .execute(
                    "SELECT row_idx FROM audio_rows WHERE file_id = ? AND active = 1 ORDER BY row_idx ASC",
                    (int(file_id),),
                )
                .fetchall()
            )
            if not rows or not self.audio_vec_path.exists():
                return None
            dim = self.profile.audio_dim
            row_bytes = dim * 2
            file_bytes = self.audio_vec_path.stat().st_size
            vecs: list[np.ndarray] = []
            with self.audio_vec_path.open("rb") as fh:
                for r in rows:
                    idx = int(r["row_idx"])
                    offset = idx * row_bytes
                    if offset + row_bytes > file_bytes:
                        continue
                    fh.seek(offset)
                    buf = fh.read(row_bytes)
                    if len(buf) == row_bytes:
                        vecs.append(np.frombuffer(buf, dtype=np.float16).astype(np.float32))
            if not vecs:
                return None
            mean_vec = np.mean(np.stack(vecs, axis=0), axis=0)
            return _l2_normalize_f32(mean_vec)


__all__ = [
    "MODEL_PROFILES",
    "AiModelProfile",
    "AudioWindowMatch",
    "IndexCoverage",
    "LibraryVectorIndex",
    "resolve_profile",
    "vector_dir_for_db",
]
