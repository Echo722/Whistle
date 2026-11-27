"""Background incremental text & audio vector indexer."""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiosearch.ai.vector_store import LibraryVectorIndex, vector_dir_for_db
from audiosearch.inference.clap2_encoder import Clap2Encoder, Clap2EncoderConfig
from audiosearch.inference.qwen3_encoder import Qwen3TextEncoder, Qwen3TextEncoderConfig
from audiosearch.library.store import FileRecord, LibraryStore

WINDOW_SECONDS = 10.0
HOP_SECONDS = 5.0
MAX_WINDOWS_PER_FILE = 24
MIN_CLAP2_AUDIO_SEC = 0.25


def _has_whistle_model_snapshot(cache_root: Path) -> bool:
    """Return True when ``cache_root`` contains a Whistle text or audio model snapshot."""
    hub = cache_root / "hub"
    return (
        (hub / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots").is_dir()
        or (hub / "models--laion--clap2-large-audioxl-s1B" / "snapshots").is_dir()
    )


def resolve_hf_cache_root(
    db_path: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path | None:
    """Find local HF cache (explicit -> AUDIOSEARCH_MODELS_DIR -> snapshot -> hub -> HF_HOME)."""
    import sys

    if cache_root is not None:
        return Path(cache_root)
    env_models = os.environ.get("AUDIOSEARCH_MODELS_DIR", "").strip()
    if env_models:
        base = Path(env_models).expanduser()
        if (base / "hub").is_dir():
            return base
        if (base / "hf" / "hub").is_dir():
            return base / "hf"
        return base

    candidates: list[Path] = []
    if db_path is not None:
        candidates.append(Path(db_path).resolve().parent / "models" / "hf")
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates.append(exe_dir / "models" / "hf")
        candidates.append(exe_dir / "data" / "models" / "hf")
    candidates.append(Path(".local") / "models" / "hf")

    env_hf = os.environ.get("HF_HOME", "").strip()
    snapshot_candidates = [*candidates, *([Path(env_hf)] if env_hf else [])]
    for cand in snapshot_candidates:
        if _has_whistle_model_snapshot(cand):
            return cand
    for cand in candidates:
        if (cand / "hub").is_dir():
            return cand
    if env_hf:
        return Path(env_hf)
    return None


def build_file_meta_text(rec: FileRecord) -> str:
    """Build deterministic compact metadata string for Qwen3 text encoding."""
    stem = Path(rec.filename).stem
    folder_part = ""
    rel_parent = Path(rec.rel_path).parent
    if str(rel_parent) not in ("", "."):
        folder_part = " / ".join(rel_parent.parts[-3:])
    parts = [
        stem,
        rec.package.strip(),
        folder_part,
        rec.description.strip(),
        rec.embedded_text.strip()[:300] if rec.embedded_text else "",
    ]
    return " | ".join(p for p in parts if p)


def text_input_hash(meta_text: str) -> str:
    return "sha256:" + hashlib.sha256(meta_text.encode("utf-8")).hexdigest()


def plan_audio_windows(
    duration_sec: float,
    *,
    window_sec: float = WINDOW_SECONDS,
    hop_sec: float = HOP_SECONDS,
    max_windows: int = MAX_WINDOWS_PER_FILE,
) -> list[tuple[float, float]]:
    """Slice file duration into ``[start, end)`` windows bounded by ``max_windows``."""
    dur = float(duration_sec or 0.0)
    if dur <= 0.0:
        return []
    if dur <= window_sec:
        return [(0.0, dur)]

    # Moderate length: dense overlapping sliding windows covering the full clip.
    if dur <= window_sec + hop_sec * (max_windows - 1):
        windows: list[tuple[float, float]] = []
        t = 0.0
        while t + window_sec < dur - 1e-4 and len(windows) < max_windows:
            windows.append((round(t, 4), round(t + window_sec, 4)))
            t += hop_sec
        if not windows or dur - windows[-1][1] > 1.0:
            tail_start = max(0.0, dur - window_sec)
            windows.append((round(tail_start, 4), round(dur, 4)))
        elif windows[-1][1] < dur:
            last_s0 = windows[-1][0]
            if dur - last_s0 <= window_sec + 1e-4:
                windows[-1] = (last_s0, round(dur, 4))
        return windows[:max_windows]

    # Long file (e.g., 2–7 minute ambience): evenly spaced representative windows.
    span = max(0.0, dur - window_sec)
    if max_windows <= 1:
        return [(0.0, round(min(dur, window_sec), 4))]
    step = span / float(max_windows - 1)
    return [
        (round(i * step, 4), round(min(dur, i * step + window_sec), 4))
        for i in range(max_windows)
    ]


def _read_windows_from_handle(
    handle: Any,
    windows: list[tuple[float, float]],
) -> tuple[list[np.ndarray], int]:
    sr = int(handle.samplerate)
    total_frames = int(len(handle))
    if sr <= 0 or total_frames <= 0:
        return [], sr
    min_samples = max(160, int(round(MIN_CLAP2_AUDIO_SEC * sr)))
    segments: list[np.ndarray] = []
    for s0, s1 in windows:
        f0 = max(0, min(int(round(s0 * sr)), total_frames - 1))
        f1 = max(f0 + 1, min(int(round(s1 * sr)), total_frames))
        handle.seek(f0)
        frames = handle.read(f1 - f0, dtype="float32", always_2d=True)
        if frames.size == 0:
            mono = np.zeros(min_samples, dtype=np.float32)
        elif frames.shape[1] == 1:
            mono = frames[:, 0]
        else:
            mono = frames.mean(axis=1)
        mono = np.ascontiguousarray(mono, dtype=np.float32)
        if mono.shape[0] < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[: mono.shape[0]] = mono
            mono = padded
        segments.append(mono)
    return segments, sr


def _decode_windows_streaming_partial(
    path: Path,
    windows: list[tuple[float, float]],
) -> tuple[list[np.ndarray], int]:
    """Sequential chunk fallback for files whose decoder errors on seek or near EOF."""
    from audiosearch.audioio.decode import open_audio_file

    with open_audio_file(path) as handle:
        sr = int(handle.samplerate)
        if sr <= 0:
            return [], sr
        chunks: list[np.ndarray] = []
        while True:
            try:
                block = handle.read(4096, dtype="float32", always_2d=True)
            except Exception:
                break
            if block.size == 0:
                break
            mono_block = block[:, 0] if block.shape[1] == 1 else block.mean(axis=1)
            chunks.append(np.ascontiguousarray(mono_block, dtype=np.float32))

    if not chunks:
        raise RuntimeError(f"No readable audio frames in {path}")

    full_mono = np.concatenate(chunks)
    total_frames = int(full_mono.shape[0])
    min_samples = max(160, int(round(MIN_CLAP2_AUDIO_SEC * sr)))
    segments: list[np.ndarray] = []
    for s0, s1 in windows:
        f0 = max(0, min(int(round(s0 * sr)), total_frames - 1))
        f1 = max(f0 + 1, min(int(round(s1 * sr)), total_frames))
        mono = np.ascontiguousarray(full_mono[f0:f1], dtype=np.float32)
        if mono.shape[0] < min_samples:
            padded = np.zeros(min_samples, dtype=np.float32)
            padded[: mono.shape[0]] = mono
            mono = padded
        segments.append(mono)
    return segments, sr


def decode_audio_windows_mono(
    path: Path,
    windows: list[tuple[float, float]],
) -> tuple[list[np.ndarray], int]:
    """Decode specified ``[start, end)`` windows as contiguous mono float32 waveforms."""
    from audiosearch.audioio.decode import open_audio_file, open_pedalboard_audio

    try:
        with open_audio_file(path) as handle:
            return _read_windows_from_handle(handle, windows)
    except Exception as primary_exc:
        try:
            with open_pedalboard_audio(path) as pb_handle:
                return _read_windows_from_handle(pb_handle, windows)
        except Exception:
            pass
        try:
            return _decode_windows_streaming_partial(path, windows)
        except Exception:
            raise primary_exc from None


@dataclass(frozen=True, slots=True)
class IndexProgress:
    phase: str
    status: str
    total_files: int
    text_done: int
    audio_done: int
    encoded_text: int
    encoded_audio: int
    reused: int
    failed: int
    items_per_sec: float
    message: str


@dataclass(frozen=True, slots=True)
class IndexReport:
    profile: str
    status: str
    total_files: int
    encoded_text: int
    encoded_audio: int
    reused_text: int
    reused_audio: int
    audio_windows_added: int
    failed: int
    elapsed_seconds: float
    pruned_text: int = 0
    pruned_audio: int = 0


class VectorIndexer:
    """Pauseable, cancellable incremental indexer writing into ``LibraryVectorIndex``."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        profile: str = "large",
        vector_dir: str | Path | None = None,
        hf_cache_root: str | Path | None = None,
        text_encoder: Any | None = None,
        audio_encoder: Any | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.profile = profile
        self.vector_dir = Path(vector_dir) if vector_dir is not None else vector_dir_for_db(self.db_path, profile)
        self.hf_cache_root = resolve_hf_cache_root(self.db_path, hf_cache_root)
        self._injected_text = text_encoder
        self._injected_audio = audio_encoder

        self._pause_event = threading.Event()
        self._pause_event.set()  # set == running
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

    def _wait_if_paused(self, report_cb: Callable[[], None] | None = None) -> bool:
        """Block while paused; return True if cancelled."""
        notified = False
        while not self._pause_event.is_set():
            if not notified and report_cb is not None:
                notified = True
                report_cb()
            if self._cancel_event.wait(0.1):
                return True
        return self._cancel_event.is_set()

    def run(
        self,
        *,
        root_id: int | None = None,
        folder_rel_path: str = "",
        limit: int | None = None,
        text_only: bool = False,
        audio_only: bool = False,
        text_batch_size: int = 64,
        audio_window_batch_size: int = 64,
        prefetch_workers: int | None = None,
        on_progress: Callable[[IndexProgress], None] | None = None,
    ) -> IndexReport:
        started = time.perf_counter()
        store = LibraryStore(self.db_path)
        index = LibraryVectorIndex(self.vector_dir, profile=self.profile)
        prof = index.profile

        encoded_text = 0
        encoded_audio = 0
        reused_text = 0
        reused_audio = 0
        windows_added = 0
        failed = 0
        pruned_text = 0
        pruned_audio = 0
        status = "done"

        try:
            if root_id is None and not folder_rel_path and limit is None:
                pruned_text, pruned_audio = index.prune_missing_file_ids(store.all_file_ids())

            all_records: list[FileRecord] = []
            for chunk in store.iter_files_for_indexing(
                root_id=root_id,
                folder_rel_path=folder_rel_path,
                online_only=True,
                batch_size=500,
            ):
                all_records.extend(chunk)
                if limit is not None and len(all_records) >= limit:
                    all_records = all_records[:limit]
                    break

            total_files = len(all_records)
            text_done = 0
            audio_done = 0

            def emit(phase: str, st: str, msg: str, t0: float, count_done: int) -> None:
                if on_progress is None:
                    return
                dt = max(0.001, time.perf_counter() - t0)
                on_progress(
                    IndexProgress(
                        phase=phase,
                        status=st,
                        total_files=total_files,
                        text_done=text_done,
                        audio_done=audio_done,
                        encoded_text=encoded_text,
                        encoded_audio=encoded_audio,
                        reused=reused_text + reused_audio,
                        failed=failed,
                        items_per_sec=count_done / dt,
                        message=msg,
                    )
                )

            # -- Phase 1: Text Embeddings (Qwen3-0.6B MRL 512d) -------------
            if not audio_only and total_files > 0:
                existing_text = index.indexed_text_map()
                text_todo: list[tuple[FileRecord, str, str]] = []
                for rec in all_records:
                    mtxt = build_file_meta_text(rec)
                    h = text_input_hash(mtxt)
                    prev = existing_text.get(rec.id)
                    if prev is not None and prev == (rec.mtime_ns, h):
                        reused_text += 1
                    else:
                        text_todo.append((rec, mtxt, h))
                text_done = reused_text

                if text_todo:
                    t_phase = time.perf_counter()
                    encoder = self._injected_text or Qwen3TextEncoder(
                        Qwen3TextEncoderConfig(
                            model_id=prof.text_model_id,
                            mrl_dim=prof.text_dim,
                            batch_size=text_batch_size,
                            cache_root=self.hf_cache_root,
                        )
                    )
                    emit("text", "indexing", "正在加载文本向量模型…", t_phase, 0)
                    encoder.load()
                    try:
                        for start in range(0, len(text_todo), text_batch_size):
                            if self._wait_if_paused(
                                lambda: emit("text", "paused", "AI 建库已暂停（文本阶段）", t_phase, encoded_text)
                            ):
                                status = "cancelled"
                                break
                            batch = text_todo[start : start + text_batch_size]
                            vecs = encoder.encode_documents(
                                [item[1] for item in batch], batch_size=text_batch_size
                            )
                            index.append_text_batch(
                                [(item[0].id, item[0].mtime_ns, item[2]) for item in batch],
                                vecs,
                            )
                            encoded_text += len(batch)
                            text_done += len(batch)
                            emit(
                                "text",
                                "indexing",
                                f"文本建库 {text_done}/{total_files}",
                                t_phase,
                                encoded_text,
                            )
                    finally:
                        if self._injected_text is None and hasattr(encoder, "close"):
                            encoder.close()
                else:
                    emit("text", "indexing", f"文本索引已全部复用 ({reused_text}/{total_files})", started, 0)

            if self._cancel_event.is_set():
                status = "cancelled"

            # -- Phase 2: Audio Embeddings (CLAP2-Large BF16) ---------------
            if status != "cancelled" and not text_only and total_files > 0:
                existing_audio = index.indexed_audio_map()
                audio_todo: list[FileRecord] = []
                for rec in all_records:
                    if rec.sample_rate <= 0 or rec.channels <= 0:
                        continue
                    prev_mtime = existing_audio.get(rec.id)
                    if prev_mtime is not None and prev_mtime == rec.mtime_ns:
                        reused_audio += 1
                    else:
                        audio_todo.append(rec)
                audio_done = reused_audio

                if audio_todo:
                    t_phase = time.perf_counter()
                    a_encoder = self._injected_audio or Clap2Encoder(
                        Clap2EncoderConfig(
                            model_id=prof.audio_model_id,
                            revision="latest",
                            precision="auto",
                            batch_size=audio_window_batch_size,
                            cache_root=self.hf_cache_root,
                        )
                    )
                    emit("audio", "indexing", f"正在加载音频模型 {prof.audio_model_id}…", t_phase, 0)
                    a_encoder.load()
                    workers = max(
                        1,
                        int(prefetch_workers)
                        if prefetch_workers
                        else min(8, max(2, (os.cpu_count() or 4) - 4)),
                    )
                    use_prepared = hasattr(a_encoder, "prepare_audio_item") and hasattr(
                        a_encoder, "encode_prepared"
                    )
                    prev_threads: int | None = None
                    if use_prepared:
                        try:
                            import torch

                            prev_threads = torch.get_num_threads()
                            torch.set_num_threads(max(1, (os.cpu_count() or 4) // workers))
                        except ImportError:
                            prev_threads = None
                    executor = ThreadPoolExecutor(
                        max_workers=workers, thread_name_prefix="ai-audio-prefetch"
                    )

                    def load_one(rec: FileRecord) -> tuple[Any, ...]:
                        """Worker: decode + (optionally) CPU front end, off the GPU thread."""
                        wins = plan_audio_windows(rec.duration_sec)
                        if not wins:
                            return rec, None, None
                        try:
                            segs, sr = decode_audio_windows_mono(rec.full_path, wins)
                            if not segs or len(segs) != len(wins):
                                return rec, None, None
                            if use_prepared:
                                items: list[Any] = [a_encoder.prepare_audio_item(w, sr) for w in segs]
                            else:
                                items = [(w, sr) for w in segs]
                        except Exception:
                            return rec, None, None
                        return rec, wins, items

                    try:
                        pending_files: list[tuple[int, int]] = []
                        pending_spans: list[tuple[int, int, float, float]] = []
                        pending_items: list[Any] = []
                        last_flush = [time.perf_counter()]

                        def flush_audio() -> None:
                            nonlocal encoded_audio, audio_done, windows_added
                            if not pending_spans:
                                return
                            if use_prepared:
                                vecs = a_encoder.encode_prepared(
                                    pending_items, batch_size=audio_window_batch_size
                                )
                            else:
                                vecs = a_encoder.encode_audio_items(
                                    pending_items, batch_size=audio_window_batch_size
                                )
                            last_flush[0] = time.perf_counter()
                            index.append_audio_batch(pending_files, pending_spans, vecs)
                            encoded_audio += len(pending_files)
                            audio_done += len(pending_files)
                            windows_added += len(pending_spans)
                            pending_files.clear()
                            pending_spans.clear()
                            pending_items.clear()
                            emit(
                                "audio",
                                "indexing",
                                f"声波建库 {audio_done}/{total_files}",
                                t_phase,
                                encoded_audio,
                            )

                        todo_iter = iter(audio_todo)
                        inflight: deque[Future[tuple[Any, ...]]] = deque()
                        max_inflight = workers * 4

                        def submit_more() -> None:
                            while len(inflight) < max_inflight:
                                nxt = next(todo_iter, None)
                                if nxt is None:
                                    return
                                inflight.append(executor.submit(load_one, nxt))

                        submit_more()
                        while inflight:
                            if self._wait_if_paused(
                                lambda: emit("audio", "paused", "AI 建库已暂停（声波阶段）", t_phase, encoded_audio)
                            ):
                                status = "cancelled"
                                break
                            rec, wins, items = inflight.popleft().result()
                            submit_more()
                            if wins is None or items is None:
                                failed += 1
                                continue
                            pending_files.append((rec.id, rec.mtime_ns))
                            for (s0, s1), item in zip(wins, items, strict=True):
                                pending_spans.append((rec.id, rec.mtime_ns, s0, s1))
                                pending_items.append(item)
                            if (
                                len(pending_items) >= audio_window_batch_size
                                or time.perf_counter() - last_flush[0] >= 2.0
                            ):
                                flush_audio()

                        if status != "cancelled":
                            flush_audio()
                    finally:
                        executor.shutdown(wait=False, cancel_futures=True)
                        if prev_threads is not None:
                            try:
                                import torch

                                torch.set_num_threads(prev_threads)
                            except ImportError:
                                pass
                        if self._injected_audio is None and hasattr(a_encoder, "close"):
                            a_encoder.close()
                else:
                    emit("audio", "indexing", f"声波索引已全部复用 ({reused_audio}/{total_files})", started, 0)

            elapsed = time.perf_counter() - started
            if on_progress is not None:
                on_progress(
                    IndexProgress(
                        phase="done" if status == "done" else status,
                        status=status,
                        total_files=total_files,
                        text_done=text_done,
                        audio_done=audio_done,
                        encoded_text=encoded_text,
                        encoded_audio=encoded_audio,
                        reused=reused_text + reused_audio,
                        failed=failed,
                        items_per_sec=(encoded_text + encoded_audio) / max(0.001, elapsed),
                        message="AI 建库完成" if status == "done" else "AI 建库已取消",
                    )
                )
            return IndexReport(
                profile=self.profile,
                status=status,
                total_files=total_files,
                encoded_text=encoded_text,
                encoded_audio=encoded_audio,
                reused_text=reused_text,
                reused_audio=reused_audio,
                audio_windows_added=windows_added,
                failed=failed,
                elapsed_seconds=round(elapsed, 3),
                pruned_text=pruned_text,
                pruned_audio=pruned_audio,
            )
        finally:
            index.close()
            store.close()


__all__ = [
    "HOP_SECONDS",
    "MAX_WINDOWS_PER_FILE",
    "WINDOW_SECONDS",
    "IndexProgress",
    "IndexReport",
    "VectorIndexer",
    "build_file_meta_text",
    "decode_audio_windows_mono",
    "plan_audio_windows",
    "resolve_hf_cache_root",
    "text_input_hash",
]
