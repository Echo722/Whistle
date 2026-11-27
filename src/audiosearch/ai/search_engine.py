"""R3 AI semantic search & Audio-to-Audio similarity engine."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audiosearch.ai.indexer import decode_audio_windows_mono, resolve_hf_cache_root
from audiosearch.ai.query_rules import AiQueryPlan, analyze_ai_query
from audiosearch.ai.vector_store import AudioWindowMatch, LibraryVectorIndex
from audiosearch.inference.clap2_encoder import Clap2Encoder, Clap2EncoderConfig
from audiosearch.inference.qwen3_encoder import Qwen3TextEncoder, Qwen3TextEncoderConfig
from audiosearch.library.store import FileRecord, LibraryStore, QueryFilter

DEFAULT_TEXT_MIN_SCORE = 0.24
DEFAULT_AUDIO_MIN_SCORE = 0.16
DEFAULT_SIMILAR_MIN_SCORE = 0.55
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class AiSearchHit:
    file_id: int
    rank: int
    score: float
    relevance_pct: int
    hit_reason: str
    text_score: float | None = None
    audio_score: float | None = None
    hit_start_sec: float | None = None
    hit_end_sec: float | None = None


@dataclass(frozen=True, slots=True)
class AiSearchResponse:
    mode: str  # "semantic" | "similar"
    query_text: str
    hits: tuple[AiSearchHit, ...]
    records: tuple[FileRecord, ...]
    total_indexed_text: int
    total_indexed_audio: int
    elapsed_ms: float
    audio_prompt: str = ""
    exclusions: tuple[str, ...] = ()
    note: str = ""
    seed_file_id: int | None = None
    seed_label: str = ""
    seed_range: tuple[float, float] | None = None
    total_candidates: int = 0
    has_more: bool = False
    result_limit: int = 500


def _fmt_window_clock(seconds: float) -> str:
    sec = max(0.0, float(seconds))
    mins, rem = divmod(sec, 60.0)
    return f"{int(mins)}:{int(rem):02d}"


def _matches_exclusions(rec: FileRecord, exclusions: tuple[str, ...]) -> bool:
    if not exclusions:
        return False
    haystack = " ".join(
        part
        for part in (
            rec.filename,
            rec.description,
            rec.package,
            rec.rel_path,
            rec.embedded_text,
        )
        if part
    ).casefold()
    return any(term in haystack for term in exclusions)


def _same_folder_or_package(rec: FileRecord, seed: FileRecord) -> bool:
    if rec.root_id == seed.root_id and rec.folder_id == seed.folder_id:
        return True
    seed_pkg = seed.package.strip().casefold()
    rec_pkg = rec.package.strip().casefold()
    return bool(seed_pkg and rec_pkg == seed_pkg)


class AiSearchEngine:
    """Thread-safe dual-tower semantic search + CLAP2 audio similarity engine."""

    def __init__(
        self,
        store: LibraryStore,
        index: LibraryVectorIndex,
        *,
        hf_cache_root: str | Path | None = None,
        text_encoder: Any | None = None,
        audio_encoder: Any | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.hf_cache_root = resolve_hf_cache_root(store.db_path, hf_cache_root)
        self._lock = threading.RLock()
        self._injected_text = text_encoder is not None
        self._injected_audio = audio_encoder is not None
        self._text_encoder = text_encoder
        self._audio_encoder = audio_encoder
        self._text_loaded = False
        self._audio_loaded = False
        self._text_query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._audio_prompt_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_cache_max = 32

    def _get_text_encoder(self):
        with self._lock:
            if self._text_encoder is None:
                prof = self.index.profile
                self._text_encoder = Qwen3TextEncoder(
                    Qwen3TextEncoderConfig(
                        model_id=prof.text_model_id,
                        mrl_dim=prof.text_dim,
                        cache_root=self.hf_cache_root,
                    )
                )
            if not self._text_loaded:
                self._text_encoder.load()
                self._text_loaded = True
            return self._text_encoder

    def _get_audio_encoder(self):
        with self._lock:
            if self._audio_encoder is None:
                prof = self.index.profile
                self._audio_encoder = Clap2Encoder(
                    Clap2EncoderConfig(
                        model_id=prof.audio_model_id,
                        revision="latest",
                        precision="auto",
                        cache_root=self.hf_cache_root,
                    )
                )
            if not self._audio_loaded:
                self._audio_encoder.load()
                self._audio_loaded = True
            return self._audio_encoder

    def _encode_cached(
        self,
        cache: OrderedDict[str, np.ndarray],
        key: str,
        encode_fn: Any,
    ) -> np.ndarray:
        """Encode ``key`` once under ``self._lock`` and cache in bounded LRU."""
        with self._lock:
            cached = cache.get(key)
            if cached is not None:
                cache.move_to_end(key)
                return cached
            vec = np.asarray(encode_fn(key), dtype=np.float32)
            cache[key] = vec
            if len(cache) > self._query_cache_max:
                cache.popitem(last=False)
            return vec

    def _encode_text_query(self, query: str) -> np.ndarray:
        return self._encode_cached(
            self._text_query_cache,
            query,
            lambda q: self._get_text_encoder().encode_queries([q])[0],
        )

    def _encode_audio_prompt(self, prompt: str) -> np.ndarray:
        return self._encode_cached(
            self._audio_prompt_cache,
            prompt,
            lambda p: self._get_audio_encoder().encode_texts([p])[0],
        )

    def _encode_seed_audio(self, path: Path, win: tuple[float, float]) -> np.ndarray:
        segs, sr = decode_audio_windows_mono(path, [win])
        with self._lock:
            a_enc = self._get_audio_encoder()
            return np.asarray(a_enc.encode_audio_items([(segs[0], sr)], batch_size=1)[0], dtype=np.float32)

    def close(self) -> None:
        with self._lock:
            self._text_query_cache.clear()
            self._audio_prompt_cache.clear()
            if not self._injected_text and self._text_encoder is not None:
                if hasattr(self._text_encoder, "close"):
                    self._text_encoder.close()
                self._text_encoder = None
                self._text_loaded = False
            if not self._injected_audio and self._audio_encoder is not None:
                if hasattr(self._audio_encoder, "close"):
                    self._audio_encoder.close()
                self._audio_encoder = None
                self._audio_loaded = False

    def _resolve_scope(
        self,
        *,
        root_id: int | None,
        folder_id: int | None,
        folder_rel_path: str,
        filters: QueryFilter | None = None,
    ) -> set[int] | None:
        has_filter = filters is not None and not filters.is_default()
        has_hidden = self.store.has_hidden_files()
        effective_root = root_id
        if (
            root_id is not None
            and folder_id is None
            and not folder_rel_path
            and self.store._is_sole_root(root_id)
        ):
            effective_root = None
        if (
            effective_root is None
            and folder_id is None
            and not folder_rel_path
            and not has_filter
            and not has_hidden
        ):
            return None
        return self.store.query_scoped_file_ids(
            root_id=effective_root,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            include_subfolders=True,
            filters=filters,
        )

    def _search_text_index(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        min_score: float,
        allowed_file_ids: set[int] | None,
        stats_out: dict[str, int] | None = None,
    ) -> list[tuple[int, float]]:
        try:
            return self.index.search_text(
                query_vec,
                top_k=top_k,
                min_score=min_score,
                allowed_file_ids=allowed_file_ids,
                stats_out=stats_out,
            )
        except TypeError:
            return self.index.search_text(
                query_vec,
                top_k=top_k,
                min_score=min_score,
                allowed_file_ids=allowed_file_ids,
            )

    def _search_audio_index(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int,
        min_score: float,
        allowed_file_ids: set[int] | None,
        exclude_file_ids: set[int] | None = None,
        stats_out: dict[str, int] | None = None,
    ) -> list[AudioWindowMatch]:
        kwargs: dict[str, Any] = {
            "top_k": top_k,
            "min_score": min_score,
            "allowed_file_ids": allowed_file_ids,
        }
        if exclude_file_ids is not None:
            kwargs["exclude_file_ids"] = exclude_file_ids
        try:
            return self.index.search_audio(query_vec, **kwargs, stats_out=stats_out)
        except TypeError:
            return self.index.search_audio(query_vec, **kwargs)

    def search_semantic(
        self,
        query: str,
        *,
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str = "",
        filters: QueryFilter | None = None,
        limit: int = 500,
        candidate_pool: int | None = None,
        text_min_score: float = DEFAULT_TEXT_MIN_SCORE,
        audio_min_score: float = DEFAULT_AUDIO_MIN_SCORE,
    ) -> AiSearchResponse:
        t0 = time.perf_counter()
        eff_limit = max(1, min(int(limit), 5000))
        eff_pool = (
            max(eff_limit, int(candidate_pool))
            if candidate_pool is not None
            else max(eff_limit * 2, 1000)
        )
        cov = self.index.coverage(self.store.total_file_count())
        plan: AiQueryPlan = analyze_ai_query(query)

        if not plan.raw_query:
            return AiSearchResponse(
                mode="semantic",
                query_text="",
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=0.0,
                result_limit=eff_limit,
            )

        if cov.text_files == 0 and cov.audio_files == 0:
            return AiSearchResponse(
                mode="semantic",
                query_text=plan.raw_query,
                hits=(),
                records=(),
                total_indexed_text=0,
                total_indexed_audio=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                audio_prompt=plan.audio_prompt,
                exclusions=plan.hard_exclusions,
                note="AI 向量索引为空：请先点击右下角「AI 建库」建立本地向量索引。",
                result_limit=eff_limit,
            )

        if not plan.positive_text_query and not plan.audio_prompt:
            return AiSearchResponse(
                mode="semantic",
                query_text=plan.raw_query,
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                exclusions=plan.hard_exclusions,
                note="请至少输入一个正向描述词（当前仅包含排除条件）。",
                result_limit=eff_limit,
            )

        scoped_ids = self._resolve_scope(
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            filters=filters,
        )
        if scoped_ids is not None and not scoped_ids:
            return AiSearchResponse(
                mode="semantic",
                query_text=plan.raw_query,
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                audio_prompt=plan.audio_prompt,
                exclusions=plan.hard_exclusions,
                note="当前范围或筛选条件下无可检索素材。",
                result_limit=eff_limit,
            )

        text_matches: list[tuple[int, float]] = []
        text_stats: dict[str, int] = {}
        if plan.positive_text_query and cov.text_files > 0:
            q_vec = self._encode_text_query(plan.positive_text_query)
            text_matches = self._search_text_index(
                q_vec,
                top_k=eff_pool,
                min_score=text_min_score,
                allowed_file_ids=scoped_ids,
                stats_out=text_stats,
            )

        audio_matches: list[AudioWindowMatch] = []
        audio_stats: dict[str, int] = {}
        if plan.audio_prompt and cov.audio_files > 0:
            a_vec = self._encode_audio_prompt(plan.audio_prompt)
            audio_matches = self._search_audio_index(
                a_vec,
                top_k=eff_pool,
                min_score=audio_min_score,
                allowed_file_ids=scoped_ids,
                stats_out=audio_stats,
            )

        text_ranks = {fid: (idx + 1, sc) for idx, (fid, sc) in enumerate(text_matches)}
        audio_ranks = {m.file_id: (idx + 1, m) for idx, m in enumerate(audio_matches)}
        candidate_ids = list(dict.fromkeys([*text_ranks.keys(), *audio_ranks.keys()]))
        records_map = self.store.get_files_by_ids(candidate_ids)

        top1_audio = audio_matches[0].score if audio_matches else 0.0
        comp_floor = max(0.23, top1_audio * 0.88)
        max_rrf = (plan.text_weight + plan.audio_weight) / (DEFAULT_RRF_K + 1.0)

        scored_entries: list[tuple[float, int, FileRecord, float | None, AudioWindowMatch | None, bool]] = []
        for fid in candidate_ids:
            rec = records_map.get(fid)
            if rec is None:
                continue
            if _matches_exclusions(rec, plan.hard_exclusions):
                continue

            t_entry = text_ranks.get(fid)
            a_entry = audio_ranks.get(fid)
            rrf = 0.0
            if t_entry is not None:
                rrf += plan.text_weight / (DEFAULT_RRF_K + t_entry[0])
            if a_entry is not None:
                rrf += plan.audio_weight / (DEFAULT_RRF_K + a_entry[0])

            compensated = False
            if (
                a_entry is not None
                and not rec.description.strip()
                and a_entry[1].score >= comp_floor
                and (
                    t_entry is None
                    or t_entry[0] > a_entry[0] * 2
                    or t_entry[1] < 0.42
                )
            ):
                rrf += (0.65 * plan.text_weight) / (DEFAULT_RRF_K + a_entry[0])
                compensated = True

            if rrf <= 0.0:
                continue
            scored_entries.append(
                (
                    rrf,
                    fid,
                    rec,
                    t_entry[1] if t_entry is not None else None,
                    a_entry[1] if a_entry is not None else None,
                    compensated,
                )
            )

        scored_entries.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored_entries[:eff_limit]
        has_more = (
            len(scored_entries) > eff_limit
            or (len(text_matches) >= eff_pool)
            or (len(audio_matches) >= eff_pool)
        )

        hits: list[AiSearchHit] = []
        records: list[FileRecord] = []
        for rank, (rrf, fid, rec, t_score, a_match, compensated) in enumerate(chosen, start=1):
            is_long = rec.duration_sec > 10.0 and a_match is not None
            hit_start = a_match.start_sec if is_long and a_match is not None else None
            hit_end = a_match.end_sec if is_long and a_match is not None else None

            if compensated:
                base_reason = "声波高置信(无描述)"
            elif t_score is not None and a_match is not None:
                base_reason = "文本+声波"
            elif a_match is not None:
                base_reason = "声波匹配"
            else:
                base_reason = "文本语义"

            if hit_start is not None and hit_end is not None:
                reason = f"{base_reason} · {_fmt_window_clock(hit_start)}-{_fmt_window_clock(hit_end)}"
            else:
                reason = base_reason

            t_conf = (t_score - 0.20) / 0.35 if t_score is not None else 0.0
            a_conf = (a_match.score - 0.16) / 0.20 if a_match is not None else 0.0
            raw_conf = min(1.0, max(0.25, max(t_conf, a_conf)))
            rrf_ratio = min(1.0, rrf / max(max_rrf, 1e-6))
            pct = min(99, max(12, int(round((0.5 * rrf_ratio + 0.5 * raw_conf) * 98.0))))
            hits.append(
                AiSearchHit(
                    file_id=fid,
                    rank=rank,
                    score=rrf,
                    relevance_pct=pct,
                    hit_reason=reason,
                    text_score=t_score,
                    audio_score=a_match.score if a_match is not None else None,
                    hit_start_sec=hit_start,
                    hit_end_sec=hit_end,
                )
            )
            records.append(rec)

        notes: list[str] = []
        if cov.audio_files == 0:
            notes.append("声波向量尚未建库（仅文本检索）")
        elif cov.audio_files < cov.total_files:
            notes.append(f"声波已索引 {cov.audio_files:,}/{cov.total_files:,} 条")
        if plan.audio_prompt:
            notes.append(f"声波提示：{plan.audio_prompt}")
        if plan.hard_exclusions:
            notes.append(f"硬排除：{'、'.join(plan.hard_exclusions)}")

        return AiSearchResponse(
            mode="semantic",
            query_text=plan.raw_query,
            hits=tuple(hits),
            records=tuple(records),
            total_indexed_text=cov.text_files,
            total_indexed_audio=cov.audio_files,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            audio_prompt=plan.audio_prompt,
            exclusions=plan.hard_exclusions,
            note=" | ".join(notes),
            total_candidates=len(scored_entries),
            has_more=has_more,
            result_limit=eff_limit,
        )

    def find_similar_file(
        self,
        file_id: int,
        *,
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str = "",
        filters: QueryFilter | None = None,
        exclude_same_folder_or_package: bool = False,
        limit: int = 500,
        min_score: float = DEFAULT_SIMILAR_MIN_SCORE,
    ) -> AiSearchResponse:
        seed = self.store.get_file(file_id)
        if seed is None:
            cov = self.index.coverage(self.store.total_file_count())
            return AiSearchResponse(
                mode="similar",
                query_text="",
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=0.0,
                note="未找到参考音频记录。",
            )
        return self._find_similar_common(
            seed=seed,
            seed_range=None,
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            filters=filters,
            exclude_same_folder_or_package=exclude_same_folder_or_package,
            limit=limit,
            min_score=min_score,
        )

    def find_similar_range(
        self,
        file_id: int,
        start_sec: float,
        end_sec: float,
        *,
        root_id: int | None = None,
        folder_id: int | None = None,
        folder_rel_path: str = "",
        filters: QueryFilter | None = None,
        exclude_same_folder_or_package: bool = False,
        limit: int = 500,
        min_score: float = DEFAULT_SIMILAR_MIN_SCORE,
    ) -> AiSearchResponse:
        seed = self.store.get_file(file_id)
        if seed is None:
            cov = self.index.coverage(self.store.total_file_count())
            return AiSearchResponse(
                mode="similar",
                query_text="",
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=0.0,
                note="未找到参考音频记录。",
            )
        s0 = max(0.0, float(start_sec))
        s1 = max(s0 + 0.05, float(end_sec))
        return self._find_similar_common(
            seed=seed,
            seed_range=(s0, s1),
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            filters=filters,
            exclude_same_folder_or_package=exclude_same_folder_or_package,
            limit=limit,
            min_score=min_score,
        )

    def _find_similar_common(
        self,
        *,
        seed: FileRecord,
        seed_range: tuple[float, float] | None,
        root_id: int | None,
        folder_id: int | None,
        folder_rel_path: str,
        filters: QueryFilter | None,
        exclude_same_folder_or_package: bool,
        limit: int,
        min_score: float,
    ) -> AiSearchResponse:
        t0 = time.perf_counter()
        cov = self.index.coverage(self.store.total_file_count())
        seed_label = (
            f"{seed.filename} [{_fmt_window_clock(seed_range[0])}-{_fmt_window_clock(seed_range[1])}]"
            if seed_range is not None
            else seed.filename
        )
        if cov.audio_files == 0:
            return AiSearchResponse(
                mode="similar",
                query_text=seed_label,
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=0,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                seed_file_id=seed.id,
                seed_label=seed_label,
                seed_range=seed_range,
                note="声波向量索引为空：请先点击右下角「AI 建库」建立声波索引后再找同类。",
            )

        scoped_ids = self._resolve_scope(
            root_id=root_id,
            folder_id=folder_id,
            folder_rel_path=folder_rel_path,
            filters=filters,
        )
        if scoped_ids is not None and not (scoped_ids - {seed.id}):
            return AiSearchResponse(
                mode="similar",
                query_text=seed_label,
                hits=(),
                records=(),
                total_indexed_text=cov.text_files,
                total_indexed_audio=cov.audio_files,
                elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                seed_file_id=seed.id,
                seed_label=seed_label,
                seed_range=seed_range,
                note="当前范围或筛选条件下无可对比素材。",
            )

        seed_vec = None
        if seed_range is None:
            seed_vec = self.index.get_file_audio_vector(seed.id)

        if seed_vec is None:
            if not seed.is_available or not seed.full_path.exists():
                return AiSearchResponse(
                    mode="similar",
                    query_text=seed_label,
                    hits=(),
                    records=(),
                    total_indexed_text=cov.text_files,
                    total_indexed_audio=cov.audio_files,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    seed_file_id=seed.id,
                    seed_label=seed_label,
                    seed_range=seed_range,
                    note="参考音频离线或文件不存在，无法编码特征。",
                )
            win = seed_range if seed_range is not None else (0.0, min(10.0, max(0.25, seed.duration_sec)))
            seed_vec = self._encode_seed_audio(seed.full_path, win)

        eff_limit = max(1, min(int(limit), 5000))
        fetch_k = min(15000, max(eff_limit * 3, 1000))
        sim_stats: dict[str, int] = {}
        matches = self._search_audio_index(
            seed_vec,
            top_k=fetch_k,
            min_score=min_score,
            allowed_file_ids=scoped_ids,
            exclude_file_ids={seed.id},
            stats_out=sim_stats,
        )
        records_map = self.store.get_files_by_ids([m.file_id for m in matches])

        hits: list[AiSearchHit] = []
        records: list[FileRecord] = []
        has_extra_valid = False
        for match in matches:
            rec = records_map.get(match.file_id)
            if rec is None:
                continue
            if exclude_same_folder_or_package and _same_folder_or_package(rec, seed):
                continue
            if len(hits) >= eff_limit:
                has_extra_valid = True
                break
            pct = min(99, max(1, int(round(match.score * 100.0))))
            is_long = rec.duration_sec > 10.0
            hit_start = match.start_sec if is_long else None
            hit_end = match.end_sec if is_long else None
            reason = f"音色相似 {pct}%"
            if hit_start is not None and hit_end is not None:
                reason += f" · {_fmt_window_clock(hit_start)}-{_fmt_window_clock(hit_end)}"
            hits.append(
                AiSearchHit(
                    file_id=rec.id,
                    rank=len(hits) + 1,
                    score=match.score,
                    relevance_pct=pct,
                    hit_reason=reason,
                    audio_score=match.score,
                    hit_start_sec=hit_start,
                    hit_end_sec=hit_end,
                )
            )
            records.append(rec)

        return AiSearchResponse(
            mode="similar",
            query_text=seed_label,
            hits=tuple(hits),
            records=tuple(records),
            total_indexed_text=cov.text_files,
            total_indexed_audio=cov.audio_files,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            seed_file_id=seed.id,
            seed_label=seed_label,
            seed_range=seed_range,
            note=f"参考：{seed_label} · 已索引声波 {cov.audio_files:,} 条",
            total_candidates=len(matches),
            has_more=has_extra_valid or len(matches) >= fetch_k,
            result_limit=eff_limit,
        )


__all__ = [
    "DEFAULT_AUDIO_MIN_SCORE",
    "DEFAULT_SIMILAR_MIN_SCORE",
    "DEFAULT_TEXT_MIN_SCORE",
    "AiSearchEngine",
    "AiSearchHit",
    "AiSearchResponse",
]
